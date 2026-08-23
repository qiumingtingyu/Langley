"""Focused local-Qdrant verification for Task 5.2 dense query filters."""

from __future__ import annotations

import asyncio
from uuid import uuid4

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels

from langley.knowledge.index_build import COLLECTION_NAME, KnowledgeIndexBuildRuntime
from langley.settings import Settings


def test_local_qdrant_dense_search_filters_scope_and_preserves_order() -> None:
    async def scenario() -> None:
        settings = Settings()
        runtime = KnowledgeIndexBuildRuntime(None, settings)  # type: ignore[arg-type]
        client = AsyncQdrantClient(url=settings.qdrant_url)
        point_ids = [str(uuid4()) for _ in range(5)]
        user_id = 987_654_321
        knowledge_base_id = 765_432_109
        generation_id = "task52-local-qdrant-generation"
        query_vector = [1.0] + [0.0] * (settings.knowledge_embedding_dimension - 1)
        try:
            await runtime._ensure_collection(
                client, dimension=settings.knowledge_embedding_dimension
            )
            await client.upsert(
                collection_name=COLLECTION_NAME,
                wait=True,
                points=[
                    qmodels.PointStruct(
                        id=point_ids[0],
                        vector=query_vector,
                        payload={
                            "knowledge_chunk_id": 101,
                            "user_id": user_id,
                            "knowledge_base_id": knowledge_base_id,
                            "generation_id": generation_id,
                        },
                    ),
                    qmodels.PointStruct(
                        id=point_ids[1],
                        vector=[0.8, 0.6]
                        + [0.0] * (settings.knowledge_embedding_dimension - 2),
                        payload={
                            "knowledge_chunk_id": 102,
                            "user_id": user_id,
                            "knowledge_base_id": knowledge_base_id,
                            "generation_id": generation_id,
                        },
                    ),
                    qmodels.PointStruct(
                        id=point_ids[2],
                        vector=query_vector,
                        payload={
                            "knowledge_chunk_id": 201,
                            "user_id": user_id + 1,
                            "knowledge_base_id": knowledge_base_id,
                            "generation_id": generation_id,
                        },
                    ),
                    qmodels.PointStruct(
                        id=point_ids[3],
                        vector=query_vector,
                        payload={
                            "knowledge_chunk_id": 202,
                            "user_id": user_id,
                            "knowledge_base_id": knowledge_base_id + 1,
                            "generation_id": generation_id,
                        },
                    ),
                    qmodels.PointStruct(
                        id=point_ids[4],
                        vector=query_vector,
                        payload={
                            "knowledge_chunk_id": 203,
                            "user_id": user_id,
                            "knowledge_base_id": knowledge_base_id,
                            "generation_id": "other-generation",
                        },
                    ),
                ],
            )
            hits = await runtime.search_dense(
                query_vector,
                user_id=user_id,
                knowledge_base_id=knowledge_base_id,
                generation_id=generation_id,
                top_k=2,
                dimension=settings.knowledge_embedding_dimension,
            )
            assert [(hit.knowledge_chunk_id, hit.score) for hit in hits] == [
                (101, 1.0),
                (102, 0.8),
            ]
        finally:
            try:
                await client.delete(
                    collection_name=COLLECTION_NAME,
                    points_selector=qmodels.PointIdsList(points=point_ids),
                    wait=True,
                )
            finally:
                await client.close()

    asyncio.run(scenario())
