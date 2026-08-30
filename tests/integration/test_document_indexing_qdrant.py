"""Focused real-Qdrant document replacement proof for Task 3B."""

import asyncio
from typing import cast

from qdrant_client import AsyncQdrantClient

from langley.knowledge.document_indexing import (
    DOCUMENT_INDEX_COLLECTION,
    DOCUMENT_INDEX_REPRESENTATION,
    DocumentIndexChunk,
    DocumentIndexClaim,
    DocumentIndexConfiguration,
    DocumentIndexRuntime,
)
from langley.knowledge.embedding_runtime import KnowledgeEmbeddingRuntime
from langley.settings import Settings


def _claim(*, version_id: int) -> DocumentIndexClaim:
    settings = Settings()
    return DocumentIndexClaim(
        job_id=version_id,
        document_version_id=version_id,
        attempt_no=1,
        target_chunk_revision=1,
        knowledge_base_id=9_300_001,
        user_id=9_300_002,
        model=settings.knowledge_embedding_model,
        revision=settings.knowledge_embedding_revision,
        dimension=settings.knowledge_embedding_dimension,
        representation=DOCUMENT_INDEX_REPRESENTATION,
    )


def _chunk(*, chunk_id: int, content: str) -> DocumentIndexChunk:
    return DocumentIndexChunk(
        id=chunk_id,
        ordinal=1,
        content=content,
        heading_path=(),
        source_regions=(
            {"kind": "text_span", "start_byte": 0, "end_byte": len(content)},
        ),
    )


def test_real_qdrant_replaces_only_one_document_without_generation_payload() -> None:
    async def scenario() -> None:
        settings = Settings()
        configuration = DocumentIndexConfiguration.from_settings(settings)
        runtime = DocumentIndexRuntime(
            cast(object, None),  # type: ignore[arg-type]
            configuration,
            cast(KnowledgeEmbeddingRuntime, object()),
        )
        claim_a = _claim(version_id=9_300_011)
        claim_b = _claim(version_id=9_300_012)
        old_a = _chunk(chunk_id=9_300_101, content="old A")
        new_a = _chunk(chunk_id=9_300_102, content="new A")
        chunk_b = _chunk(chunk_id=9_300_201, content="stable B")
        vector = [1.0] + [0.0] * (settings.knowledge_embedding_dimension - 1)
        client = AsyncQdrantClient(url=settings.qdrant_url)
        try:
            await runtime._publish_vectors(claim_a, (old_a,), [vector])
            await runtime._publish_vectors(claim_b, (chunk_b,), [vector])
            before_b, _ = await client.scroll(
                collection_name=DOCUMENT_INDEX_COLLECTION,
                scroll_filter=runtime._scope_filter(claim_b),
                limit=10,
                with_payload=True,
                with_vectors=False,
            )
            await runtime._publish_vectors(claim_a, (new_a,), [vector])
            after_a, _ = await client.scroll(
                collection_name=DOCUMENT_INDEX_COLLECTION,
                scroll_filter=runtime._scope_filter(claim_a),
                limit=10,
                with_payload=True,
                with_vectors=False,
            )
            after_b, _ = await client.scroll(
                collection_name=DOCUMENT_INDEX_COLLECTION,
                scroll_filter=runtime._scope_filter(claim_b),
                limit=10,
                with_payload=True,
                with_vectors=False,
            )

            assert [point.id for point in after_a] == [new_a.id]
            assert [point.id for point in before_b] == [chunk_b.id]
            assert [point.id for point in after_b] == [chunk_b.id]
            for point in (*after_a, *after_b):
                assert point.payload is not None
                assert set(point.payload) == {
                    "knowledge_chunk_id",
                    "knowledge_base_id",
                    "document_version_id",
                    "user_id",
                }
                assert "generation_id" not in point.payload
        finally:
            await runtime._cleanup_vectors(claim_a)
            await runtime._cleanup_vectors(claim_b)
            await client.close()

    asyncio.run(scenario())
