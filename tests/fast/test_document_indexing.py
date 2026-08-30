"""Focused deterministic coverage for the Task 3B document index runtime."""

import asyncio
import sys
from datetime import datetime
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from langley.infrastructure.models import DocumentIndexJob
from langley.knowledge.document_indexing import (
    DOCUMENT_INDEX_REPRESENTATION,
    DocumentIndexChunk,
    DocumentIndexClaim,
    DocumentIndexConfiguration,
    DocumentIndexFailure,
    DocumentIndexRuntime,
    interrupt_document_index_job,
    mark_document_index_running,
)
from langley.knowledge.embedding_runtime import KnowledgeEmbeddingRuntime
from langley.knowledge.index_build import KnowledgeIndexBuildRuntime
from langley.settings import Settings


def _configuration() -> DocumentIndexConfiguration:
    return DocumentIndexConfiguration(
        model="model",
        revision="revision",
        dimension=2,
        representation=DOCUMENT_INDEX_REPRESENTATION,
        qdrant_url="http://qdrant.invalid",
    )


def _claim() -> DocumentIndexClaim:
    return DocumentIndexClaim(
        job_id=7,
        document_version_id=11,
        attempt_no=2,
        target_chunk_revision=3,
        knowledge_base_id=13,
        user_id=17,
        model="model",
        revision="revision",
        dimension=2,
        representation=DOCUMENT_INDEX_REPRESENTATION,
    )


def _chunks() -> tuple[DocumentIndexChunk, ...]:
    return (
        DocumentIndexChunk(
            id=101,
            ordinal=1,
            content="exact body",
            heading_path=("Root", "Leaf"),
            source_regions=({"kind": "text_span", "start_byte": 0, "end_byte": 10},),
        ),
    )


class _Embedding:
    def __init__(self, *, failure: Exception | None = None) -> None:
        self.failure = failure
        self.contents: list[str] | None = None

    def encode_documents(self, contents: list[str], **_: object) -> list[list[float]]:
        self.contents = contents
        if self.failure is not None:
            raise self.failure
        return [[1.0, 0.0] for _ in contents]


class _RuntimeHarness(DocumentIndexRuntime):
    def __init__(
        self,
        embedding: _Embedding,
        *,
        snapshot_failure: DocumentIndexFailure | None = None,
        publication_failure: bool = False,
        verified_count: int = 1,
    ) -> None:
        super().__init__(
            cast(object, None),  # type: ignore[arg-type]
            _configuration(),
            cast(KnowledgeEmbeddingRuntime, embedding),
        )
        self.snapshot_failure = snapshot_failure
        self.publication_failure = publication_failure
        self.verified_count = verified_count
        self.events: list[str] = []
        self.failed: DocumentIndexFailure | None = None
        self.indexed_chunk_revision: int | None = 1

    async def _load_snapshot(self, claim: DocumentIndexClaim):
        del claim
        self.events.append("snapshot")
        if self.snapshot_failure is not None:
            raise self.snapshot_failure
        return _chunks(), "a" * 64

    async def _publication_barrier(
        self, claim: DocumentIndexClaim, expected_chunk_set_sha256: str
    ) -> None:
        del claim, expected_chunk_set_sha256
        self.indexed_chunk_revision = None
        self.events.append("barrier")

    async def _publish_vectors(self, claim, chunks, vectors) -> None:
        del claim, chunks, vectors
        assert self.events[-1] == "barrier"
        assert self.indexed_chunk_revision is None
        self.events.append("publish")
        if self.publication_failure:
            raise RuntimeError("controlled partial write")

    async def _advance_to_verifying(self, claim, expected_chunk_set_sha256) -> None:
        del claim, expected_chunk_set_sha256
        self.events.append("verifying")

    async def _count_vectors(self, claim: DocumentIndexClaim) -> int:
        del claim
        self.events.append("count")
        return self.verified_count

    async def _complete(self, claim, expected_chunk_set_sha256) -> None:
        del expected_chunk_set_sha256
        self.indexed_chunk_revision = claim.target_chunk_revision
        self.events.append("complete")

    async def _cleanup_vectors(self, claim: DocumentIndexClaim) -> None:
        del claim
        self.events.append("cleanup")

    async def _fail(self, job_id: int, failure: DocumentIndexFailure) -> None:
        del job_id
        self.failed = failure
        self.events.append("failed")


def test_success_uses_source_context_and_crosses_barrier_before_qdrant() -> None:
    embedding = _Embedding()
    runtime = _RuntimeHarness(embedding)

    asyncio.run(runtime.execute(_claim()))

    assert embedding.contents == ["Root\nLeaf\n\nexact body"]
    assert runtime.events == [
        "snapshot",
        "barrier",
        "publish",
        "verifying",
        "count",
        "complete",
    ]
    assert runtime.failed is None
    assert runtime.indexed_chunk_revision == _claim().target_chunk_revision


def test_stale_source_and_embedding_failure_never_touch_qdrant() -> None:
    stale = _RuntimeHarness(
        _Embedding(),
        snapshot_failure=DocumentIndexFailure("SOURCE_CHUNKS_CHANGED", "safe"),
    )
    asyncio.run(stale.execute(_claim()))
    assert stale.events == ["snapshot", "failed"]
    assert stale.failed is not None
    assert stale.failed.code == "SOURCE_CHUNKS_CHANGED"

    embedding = _RuntimeHarness(_Embedding(failure=RuntimeError("private detail")))
    asyncio.run(embedding.execute(_claim()))
    assert embedding.events == ["snapshot", "failed"]
    assert embedding.failed is not None
    assert embedding.failed.code == "EMBEDDING_FAILED"


@pytest.mark.parametrize(
    ("publication_failure", "verified_count", "expected_code"),
    [
        (True, 1, "INDEX_PUBLICATION_FAILED"),
        (False, 0, "INDEX_VERIFICATION_FAILED"),
    ],
)
def test_post_barrier_failure_cleans_document_scope_and_never_completes(
    publication_failure: bool, verified_count: int, expected_code: str
) -> None:
    runtime = _RuntimeHarness(
        _Embedding(),
        publication_failure=publication_failure,
        verified_count=verified_count,
    )

    asyncio.run(runtime.execute(_claim()))

    assert "barrier" in runtime.events
    assert "cleanup" in runtime.events
    assert "complete" not in runtime.events
    assert runtime.failed is not None
    assert runtime.failed.code == expected_code
    assert runtime.indexed_chunk_revision is None


def test_restart_interrupts_running_but_leaves_pending_unchanged() -> None:
    now = datetime(2026, 8, 30, 12, 0)
    running = DocumentIndexJob(status="PENDING")
    pending = DocumentIndexJob(status="PENDING")

    mark_document_index_running(running, now=now)
    interrupt_document_index_job(running, now=now)
    interrupt_document_index_job(pending, now=now)

    assert (running.status, running.stage, running.error_code) == (
        "INTERRUPTED",
        "EMBEDDING",
        "INDEX_INTERRUPTED",
    )
    assert pending.status == "PENDING"
    assert pending.stage is None
    assert pending.error_code is None


def test_one_embedding_runtime_is_shared_by_build_document_and_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loads: list[tuple[str, str, str]] = []

    class _Model:
        device = "cpu"

        def __init__(self, model: str, *, revision: str, device: str) -> None:
            loads.append((model, revision, device))

        def encode_document(self, contents: list[str], **_: object) -> np.ndarray:
            return np.asarray([[3.0, 4.0] for _ in contents], dtype=np.float32)

        def encode_query(self, contents: list[str], **_: object) -> np.ndarray:
            assert contents == ["query"]
            return np.asarray([[3.0, 4.0]], dtype=np.float32)

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=_Model),
    )
    shared = KnowledgeEmbeddingRuntime(device="cpu")
    settings = Settings(knowledge_embedding_device="cpu")
    legacy = KnowledgeIndexBuildRuntime(
        cast(object, None),  # type: ignore[arg-type]
        settings,
        embedding_runtime=shared,
    )
    document = DocumentIndexRuntime(
        cast(object, None),  # type: ignore[arg-type]
        _configuration(),
        shared,
    )

    shared.encode_documents(
        ["document"], model="model", revision="revision", dimension=2
    )
    shared.encode_query("query", model="model", revision="revision", dimension=2)

    assert loads == [("model", "revision", "cpu")]
    assert legacy.embedding_runtime is shared
    assert document.embedding_runtime is shared
