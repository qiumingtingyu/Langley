"""Deterministic Task 5.2 dense-retrieval behavior without DB or model runtime."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import HTTPException

import langley.knowledge.retrieval as retrieval
from langley.api.knowledge import _raise_retrieval_error
from langley.infrastructure.models import KnowledgeBase
from langley.knowledge.index_build import (
    DenseSearchHit,
    IndexBuildFailure,
    KnowledgeIndexBuildRuntime,
    _normalize_query_embedding,
)
from langley.knowledge.retrieval import (
    ActiveRetrievalContext,
    IndexNotReadyError,
    RetrievalEmbeddingInvalidError,
    RetrievalEmbeddingUnavailableError,
    RetrievalGenerationChangedError,
    RetrievalIndexInconsistentError,
    RetrievalQdrantUnavailableError,
    _AuthoritativeChunk,
    _context_from_knowledge_base,
    retrieve_dense,
)
from langley.settings import Settings


def _context() -> ActiveRetrievalContext:
    return ActiveRetrievalContext(
        knowledge_base_id=4,
        user_id=7,
        generation_id="11111111-1111-4111-8111-111111111111",
        model="active-model",
        revision="a" * 40,
        dimension=2,
        representation="content_only",
        chunk_snapshot_sha256="a" * 64,
    )


def _chunk(chunk_id: int, ordinal: int) -> _AuthoritativeChunk:
    return _AuthoritativeChunk(
        knowledge_chunk_id=chunk_id,
        chunk_ordinal=ordinal,
        content=f"content-{chunk_id}",
        heading_path=("Heading",),
        source_regions=({"kind": "text", "start": 0, "end": 1},),
        document_id=9,
        document_version_id=10,
        source_display_name="source.md",
        source_sha256="b" * 64,
    )


class _FakeRuntime:
    def __init__(
        self,
        *,
        query_vector: list[float] | Exception = [0.6, 0.8],
        search_hits: tuple[DenseSearchHit, ...] | Exception = (),
    ) -> None:
        self.query_vector = query_vector
        self.search_hits = search_hits
        self.encode_calls: list[tuple[object, ...]] = []
        self.search_calls: list[tuple[object, ...]] = []

    async def encode_query(self, query: str, **kwargs: object) -> list[float]:
        self.encode_calls.append((query, kwargs))
        if isinstance(self.query_vector, Exception):
            raise self.query_vector
        return self.query_vector

    async def search_dense(
        self, vector: list[float], **kwargs: object
    ) -> tuple[DenseSearchHit, ...]:
        self.search_calls.append((vector, kwargs))
        if isinstance(self.search_hits, Exception):
            raise self.search_hits
        return self.search_hits


def _patch_reads(
    monkeypatch: pytest.MonkeyPatch,
    *,
    final_chunks: tuple[_AuthoritativeChunk, ...] = (),
    final_error: Exception | None = None,
) -> list[tuple[int, ...]]:
    calls: list[tuple[int, ...]] = []

    async def initial(*args: object, **kwargs: object) -> ActiveRetrievalContext:
        return _context()

    async def final(
        *args: object, returned_chunk_ids: tuple[int, ...], **kwargs: object
    ):
        calls.append(returned_chunk_ids)
        if final_error is not None:
            raise final_error
        return final_chunks

    monkeypatch.setattr(retrieval, "_read_active_context", initial)
    monkeypatch.setattr(retrieval, "_final_revalidate", final)
    return calls


def test_retrieval_preserves_exact_query_active_config_and_qdrant_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_calls = _patch_reads(
        monkeypatch, final_chunks=(_chunk(8, 2), _chunk(20, 1), _chunk(42, 3))
    )
    runtime = _FakeRuntime(
        search_hits=(
            DenseSearchHit(20, 0.9),
            DenseSearchHit(8, 0.8),
            DenseSearchHit(42, 0.7),
        )
    )

    result = asyncio.run(
        retrieve_dense(
            None,  # type: ignore[arg-type]
            runtime,  # type: ignore[arg-type]
            user_id=7,
            knowledge_base_id=4,
            query="  exact original query  ",
            top_k=3,
        )
    )

    assert runtime.encode_calls == [
        (
            "  exact original query  ",
            {
                "model": "active-model",
                "revision": "a" * 40,
                "dimension": 2,
                "representation": "content_only",
            },
        )
    ]
    assert runtime.search_calls == [
        (
            [0.6, 0.8],
            {
                "user_id": 7,
                "knowledge_base_id": 4,
                "generation_id": "11111111-1111-4111-8111-111111111111",
                "top_k": 3,
                "dimension": 2,
            },
        )
    ]
    assert final_calls == [(20, 8, 42)]
    assert [(hit.rank, hit.knowledge_chunk_id) for hit in result.hits] == [
        (1, 20),
        (2, 8),
        (3, 42),
    ]


def test_non_ready_rejects_before_embedding_or_qdrant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _FakeRuntime(search_hits=(DenseSearchHit(20, 0.9),))

    async def initial(*args: object, **kwargs: object) -> ActiveRetrievalContext:
        raise IndexNotReadyError()

    monkeypatch.setattr(retrieval, "_read_active_context", initial)

    with pytest.raises(IndexNotReadyError):
        asyncio.run(
            retrieve_dense(
                None,  # type: ignore[arg-type]
                runtime,  # type: ignore[arg-type]
                user_id=7,
                knowledge_base_id=4,
                query="query",
                top_k=1,
            )
        )
    assert not runtime.encode_calls
    assert not runtime.search_calls


@pytest.mark.parametrize(
    "failure, expected",
    [
        (
            IndexBuildFailure("INVALID_EMBEDDING", "invalid"),
            RetrievalEmbeddingInvalidError,
        ),
        (
            RuntimeError("configured device unavailable"),
            RetrievalEmbeddingUnavailableError,
        ),
    ],
)
def test_embedding_failures_do_not_search(
    monkeypatch: pytest.MonkeyPatch, failure: Exception, expected: type[Exception]
) -> None:
    _patch_reads(monkeypatch)
    runtime = _FakeRuntime(query_vector=failure)

    with pytest.raises(expected):
        asyncio.run(
            retrieve_dense(
                None,  # type: ignore[arg-type]
                runtime,  # type: ignore[arg-type]
                user_id=7,
                knowledge_base_id=4,
                query="query",
                top_k=1,
            )
        )
    assert len(runtime.encode_calls) == 1
    assert not runtime.search_calls


def test_duplicate_qdrant_result_fails_closed_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_calls = _patch_reads(monkeypatch)
    runtime = _FakeRuntime(
        search_hits=(DenseSearchHit(20, 0.9), DenseSearchHit(20, 0.8))
    )

    with pytest.raises(RetrievalIndexInconsistentError):
        asyncio.run(
            retrieve_dense(
                None,  # type: ignore[arg-type]
                runtime,  # type: ignore[arg-type]
                user_id=7,
                knowledge_base_id=4,
                query="query",
                top_k=2,
            )
        )
    assert len(runtime.encode_calls) == len(runtime.search_calls) == 1
    assert final_calls == [(20, 20)]


def test_qdrant_result_count_above_top_k_fails_closed_without_partial_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_calls = _patch_reads(monkeypatch)
    runtime = _FakeRuntime(
        search_hits=(DenseSearchHit(20, 0.9), DenseSearchHit(8, 0.8))
    )

    with pytest.raises(RetrievalIndexInconsistentError):
        asyncio.run(
            retrieve_dense(
                None,  # type: ignore[arg-type]
                runtime,  # type: ignore[arg-type]
                user_id=7,
                knowledge_base_id=4,
                query="query",
                top_k=1,
            )
        )
    assert len(runtime.search_calls) == 1
    assert final_calls == [(20, 8)]


def test_zero_qdrant_result_obeys_final_generation_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_reads(monkeypatch, final_error=RetrievalGenerationChangedError())
    runtime = _FakeRuntime(search_hits=())

    with pytest.raises(RetrievalGenerationChangedError):
        asyncio.run(
            retrieve_dense(
                None,  # type: ignore[arg-type]
                runtime,  # type: ignore[arg-type]
                user_id=7,
                knowledge_base_id=4,
                query="query",
                top_k=1,
            )
        )


def test_zero_qdrant_result_with_same_generation_is_inconsistent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_reads(monkeypatch)
    runtime = _FakeRuntime(search_hits=())

    with pytest.raises(RetrievalIndexInconsistentError):
        asyncio.run(
            retrieve_dense(
                None,  # type: ignore[arg-type]
                runtime,  # type: ignore[arg-type]
                user_id=7,
                knowledge_base_id=4,
                query="query",
                top_k=1,
            )
        )


def test_qdrant_runtime_failure_is_not_retried_after_final_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_calls = _patch_reads(monkeypatch)
    runtime = _FakeRuntime(search_hits=RuntimeError("qdrant unavailable"))

    with pytest.raises(RetrievalQdrantUnavailableError):
        asyncio.run(
            retrieve_dense(
                None,  # type: ignore[arg-type]
                runtime,  # type: ignore[arg-type]
                user_id=7,
                knowledge_base_id=4,
                query="query",
                top_k=1,
            )
        )
    assert len(runtime.search_calls) == 1
    assert final_calls == [()]


@pytest.mark.parametrize(
    "values",
    [
        np.asarray([[1.0, 2.0]], dtype=np.float64),
        np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
        np.asarray([[np.nan, 1.0]], dtype=np.float32),
        np.asarray([[np.inf, 1.0]], dtype=np.float32),
        np.asarray([[0.0, 0.0]], dtype=np.float32),
    ],
)
def test_malformed_query_embedding_fails_closed(values: np.ndarray) -> None:
    with pytest.raises(IndexBuildFailure, match="INVALID_EMBEDDING"):
        _normalize_query_embedding(values, dimension=2)


def test_query_embedding_is_float32_and_l2_normalized() -> None:
    assert _normalize_query_embedding(
        np.asarray([[3.0, 4.0]], dtype=np.float32), dimension=2
    ) == pytest.approx([0.6, 0.8])


def test_qdrant_search_uses_exact_scope_filters_without_threshold() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.kwargs: dict[str, object] | None = None
            self.closed = False

        async def query_points(self, **kwargs: object) -> SimpleNamespace:
            self.kwargs = kwargs
            return SimpleNamespace(
                points=[SimpleNamespace(payload={"knowledge_chunk_id": 20}, score=0.9)]
            )

        async def close(self) -> None:
            self.closed = True

    class Runtime(KnowledgeIndexBuildRuntime):
        def __init__(self, client: FakeClient) -> None:
            super().__init__(None, Settings(knowledge_embedding_dimension=2))  # type: ignore[arg-type]
            self.client = client

        async def _qdrant_client(self) -> FakeClient:
            return self.client

        @staticmethod
        async def _require_collection(client: FakeClient, *, dimension: int) -> None:
            assert dimension == 2

    client = FakeClient()
    runtime = Runtime(client)
    hits = asyncio.run(
        runtime.search_dense(
            [0.6, 0.8],
            user_id=7,
            knowledge_base_id=4,
            generation_id="generation-7",
            top_k=3,
            dimension=2,
        )
    )

    assert hits == (DenseSearchHit(20, 0.9),)
    assert client.closed
    assert client.kwargs is not None
    assert client.kwargs["limit"] == 3
    assert client.kwargs["with_vectors"] is False
    assert "score_threshold" not in client.kwargs
    conditions = client.kwargs["query_filter"].must
    assert [(condition.key, condition.match.value) for condition in conditions] == [
        ("user_id", 7),
        ("knowledge_base_id", 4),
        ("generation_id", "generation-7"),
    ]


def test_qdrant_retrieval_missing_collection_does_not_create_one() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.create_called = False
            self.query_called = False
            self.closed = False

        async def collection_exists(self, collection_name: str) -> bool:
            return False

        async def create_collection(self, **kwargs: object) -> None:
            self.create_called = True

        async def query_points(self, **kwargs: object) -> SimpleNamespace:
            self.query_called = True
            return SimpleNamespace(points=[])

        async def close(self) -> None:
            self.closed = True

    class Runtime(KnowledgeIndexBuildRuntime):
        def __init__(self, client: FakeClient) -> None:
            super().__init__(None, Settings(knowledge_embedding_dimension=2))  # type: ignore[arg-type]
            self.client = client

        async def _qdrant_client(self) -> FakeClient:
            return self.client

    client = FakeClient()
    with pytest.raises(IndexBuildFailure, match="INDEX_COLLECTION_MISSING"):
        asyncio.run(
            Runtime(client).search_dense(
                [0.6, 0.8],
                user_id=7,
                knowledge_base_id=4,
                generation_id="generation",
                top_k=1,
                dimension=2,
            )
        )
    assert client.closed
    assert not client.create_called
    assert not client.query_called


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("active_generation_id", None),
        ("active_generation_id", "not-a-uuid"),
        ("active_embedding_model", None),
        ("active_embedding_model", "  "),
        ("active_embedding_revision", None),
        ("active_embedding_revision", "r" * 40),
        ("active_embedding_dimension", 0),
        ("active_embedding_dimension", -1),
        ("active_embedding_representation", "unsupported"),
        ("active_chunk_snapshot_sha256", None),
        ("active_chunk_snapshot_sha256", "g" * 64),
    ],
)
def test_corrupted_ready_active_facts_fail_closed(field: str, value: object) -> None:
    knowledge_base = KnowledgeBase(
        id=4,
        user_id=7,
        name="Ready",
        index_status="READY",
        active_generation_id="11111111-1111-4111-8111-111111111111",
        active_embedding_model="BAAI/bge-m3",
        active_embedding_revision="a" * 40,
        active_embedding_dimension=1024,
        active_embedding_representation="content_only",
        active_chunk_snapshot_sha256="b" * 64,
    )
    setattr(knowledge_base, field, value)

    with pytest.raises(RetrievalIndexInconsistentError):
        _context_from_knowledge_base(knowledge_base, user_id=7, final_read=False)


@pytest.mark.parametrize(
    "error, expected_status",
    [
        (IndexNotReadyError(), 409),
        (RetrievalGenerationChangedError(), 409),
        (RetrievalIndexInconsistentError(), 500),
        (RetrievalEmbeddingInvalidError(), 500),
        (RetrievalEmbeddingUnavailableError(), 503),
        (RetrievalQdrantUnavailableError(), 503),
    ],
)
def test_error_mapping(error: Exception, expected_status: int) -> None:
    with pytest.raises(HTTPException) as raised:
        _raise_retrieval_error(error)  # type: ignore[arg-type]
    assert raised.value.status_code == expected_status
