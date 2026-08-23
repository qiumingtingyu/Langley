"""One controlled, evidence-writing Task 5.1/5.2 real BGE-M3 retrieval smoke.

This is deliberately invoked directly, never by the ordinary test suite.
"""

import asyncio
import os
from argparse import Namespace
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic, sleep
from typing import Any

from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels
from sqlalchemy.ext.asyncio import async_sessionmaker

from langley.business_time import utc_now
from langley.infrastructure.database import (
    create_database_engine,
    create_session_factory,
    dispose_database_engine,
)
from langley.infrastructure.models import (
    Document,
    DocumentVersion,
    KnowledgeBase,
    KnowledgeChunk,
    User,
)
from langley.knowledge.index_build import (
    COLLECTION_NAME,
    KnowledgeIndexBuildRuntime,
    _normalize_embedding_rows,
)
from langley.main import create_app
from langley.settings import Settings

_EVIDENCE_PATH = Path(".review/slice6/task05/evidence/real-bge-retrieval-smoke.txt")


class CaptureBgeRuntime(KnowledgeIndexBuildRuntime):
    """Production-equivalent encoder with its actual resolved device observed."""

    def __init__(self, session_factory: async_sessionmaker, settings: Settings) -> None:
        super().__init__(session_factory, settings)
        self.actual_embedding_device: str | None = None

    def _encode_documents(self, contents: list[str], job: Any) -> list[list[float]]:
        import torch
        from sentence_transformers import SentenceTransformer

        if self.settings.knowledge_embedding_device != "cuda:0":
            raise RuntimeError("configured device is not cuda:0")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; CPU fallback is prohibited")
        model = SentenceTransformer(
            job.embedding_model,
            revision=job.embedding_revision,
            device=self.settings.knowledge_embedding_device,
        )
        self.actual_embedding_device = str(model.device)
        if self.actual_embedding_device != "cuda:0":
            raise RuntimeError("SentenceTransformer did not resolve cuda:0")
        values = model.encode_document(
            contents, convert_to_numpy=True, show_progress_bar=False
        )
        return _normalize_embedding_rows(
            values, row_count=len(contents), dimension=job.embedding_dimension
        )


async def _seed(session_factory: async_sessionmaker) -> tuple[int, int]:
    async with session_factory() as session, session.begin():
        now = utc_now()
        session.add(User(id=1, created_at=now))
        await session.flush()
        knowledge_base = KnowledgeBase(
            user_id=1,
            name="Task 5.1 real BGE smoke",
            created_at=now,
        )
        session.add(knowledge_base)
        await session.flush()
        document = Document(
            knowledge_base_id=knowledge_base.id,
            name="real-bge-smoke",
            created_at=now,
        )
        session.add(document)
        await session.flush()
        version = DocumentVersion(
            document_id=document.id,
            source_filename="real-bge-smoke.md",
            source_media_type="text/markdown",
            source_sha256="b" * 64,
            source_size_bytes=35,
            storage_key="manual/real-bge-smoke.md",
            chunk_max_chars=1200,
            created_at=now,
        )
        session.add(version)
        await session.flush()
        session.add(
            KnowledgeChunk(
                document_version_id=version.id,
                ordinal=1,
                content="BGE-M3 CUDA smoke document content.",
                heading_path=[],
                source_regions=[{"kind": "text", "start": 0, "end": 36}],
                created_at=now,
            )
        )
        await session.flush()
        return knowledge_base.id, 1


def _write_evidence(values: dict[str, object]) -> None:
    _EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _EVIDENCE_PATH.write_text(
        "\n".join(f"{key}: {value}" for key, value in values.items()) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    test_database_url = os.environ["LANGLEY_TEST_DATABASE_URL"]
    settings = Settings(database_url=test_database_url, local_user_id=1)
    observed: dict[str, object] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "configured_device": settings.knowledge_embedding_device,
        "model": settings.knowledge_embedding_model,
        "revision": settings.knowledge_embedding_revision,
        "result": "FAIL",
    }
    engine = create_database_engine(settings.database_url)
    session_factory = create_session_factory(engine)
    try:
        config = Config("alembic.ini")
        config.cmd_opts = Namespace(x=["use_test_database=true"])
        command.upgrade(config, "head")
        knowledge_base_id, chunk_count = asyncio.run(_seed(session_factory))
        observed["knowledge_base_id"] = knowledge_base_id
        observed["controlled_chunk_count"] = chunk_count
        runtime = CaptureBgeRuntime(session_factory, settings)
        app = create_app(settings, knowledge_index_runtime=runtime)
        with TestClient(app) as client:
            accepted = client.post(
                f"/api/knowledge-bases/{knowledge_base_id}/index-build"
            )
            observed["http_post_status"] = accepted.status_code
            if accepted.status_code != 202:
                raise RuntimeError(f"build admission failed: {accepted.text}")
            job_id = accepted.json()["job_id"]
            observed["job_id"] = job_id
            immediate = client.get(
                f"/api/knowledge-bases/{knowledge_base_id}/index-status"
            ).json()
            observed["status_immediately_after_202"] = immediate["index_status"]
            observed["http_202_before_final_ready"] = (
                immediate["index_status"] != "READY"
            )
            lifecycle = [
                f"{immediate['index_status']}/{immediate['latest_job']['status']}"
            ]
            deadline = monotonic() + 600
            while monotonic() < deadline:
                current = client.get(
                    f"/api/knowledge-bases/{knowledge_base_id}/index-status"
                ).json()
                latest = current["latest_job"]
                point = (
                    f"{current['index_status']}/{latest['status']}/{latest['stage']}"
                )
                if lifecycle[-1] != point:
                    lifecycle.append(point)
                if latest["status"] in {"SUCCEEDED", "FAILED", "INTERRUPTED"}:
                    break
                sleep(0.25)
            else:
                raise RuntimeError("timed out waiting for build terminal state")
            if latest["status"] != "SUCCEEDED" or current["index_status"] != "READY":
                raise RuntimeError(f"build terminal state: {latest} / {current}")
            retrieval_response = client.post(
                f"/api/knowledge-bases/{knowledge_base_id}/retrieval",
                json={
                    "query": "BGE-M3 CUDA smoke document content.",
                    "top_k": 1,
                },
            )
            observed["retrieval_http_status"] = retrieval_response.status_code
            retrieval_payload = retrieval_response.json()
            observed["retrieval_hit_count"] = len(retrieval_payload.get("hits", []))
            if retrieval_response.status_code != 200:
                raise RuntimeError(f"retrieval failed: {retrieval_response.text}")
            if not 1 <= len(retrieval_payload["hits"]) <= 1:
                raise RuntimeError(f"retrieval hit count: {retrieval_payload}")
            retrieval_hit = retrieval_payload["hits"][0]
            observed["retrieval_first_chunk_id"] = retrieval_hit["knowledge_chunk_id"]
            if (
                retrieval_hit["rank"] != 1
                or retrieval_hit["content"] != "BGE-M3 CUDA smoke document content."
                or retrieval_hit["source_display_name"] != "real-bge-smoke"
                or not retrieval_hit["source_sha256"] == "b" * 64
            ):
                raise RuntimeError(
                    f"retrieval hit is not authoritative: {retrieval_hit}"
                )
        observed["actual_embedding_device"] = runtime.actual_embedding_device
        observed["observed_lifecycle"] = " -> ".join(lifecycle)
        observed["processed_chunk_count"] = latest["processed_chunk_count"]
        observed["total_chunk_count"] = latest["total_chunk_count"]
        observed["final_job_status"] = latest["status"]
        observed["final_index_status"] = current["index_status"]
        observed["active_generation_id"] = "unavailable"
        observed["generation_id"] = "unavailable"

        async def inspect_qdrant() -> tuple[int, set[str], str]:
            client = AsyncQdrantClient(url=settings.qdrant_url)
            try:
                async with session_factory() as session:
                    knowledge_base = await session.get(KnowledgeBase, knowledge_base_id)
                    assert knowledge_base is not None
                    generation_id = knowledge_base.active_generation_id
                    assert generation_id is not None
                result = await client.count(
                    collection_name=COLLECTION_NAME,
                    count_filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key="generation_id",
                                match=qmodels.MatchValue(value=generation_id),
                            )
                        ]
                    ),
                    exact=True,
                )
                records, _ = await client.scroll(
                    collection_name=COLLECTION_NAME,
                    scroll_filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key="generation_id",
                                match=qmodels.MatchValue(value=generation_id),
                            )
                        ]
                    ),
                    with_payload=True,
                    limit=1,
                )
                assert records
                return result.count, set(records[0].payload), generation_id
            finally:
                await client.close()

        point_count, payload_fields, generation_id = asyncio.run(inspect_qdrant())
        observed["generation_id"] = generation_id
        observed["active_generation_id"] = generation_id
        observed["qdrant_generation_point_count"] = point_count
        observed["qdrant_payload_fields"] = ",".join(sorted(payload_fields))
        if point_count != chunk_count:
            raise RuntimeError(
                "Qdrant generation count does not match controlled chunks"
            )
        required = {
            "knowledge_chunk_id",
            "knowledge_base_id",
            "document_version_id",
            "user_id",
            "generation_id",
        }
        if payload_fields != required:
            raise RuntimeError(f"unexpected Qdrant payload: {payload_fields}")
        observed["result"] = "PASS"
    except Exception as error:
        observed["failure"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        _write_evidence(observed)
        asyncio.run(dispose_database_engine(engine))


if __name__ == "__main__":
    main()
