"""Focused local-Qdrant verification for Task 5.2 dense query filters."""

from __future__ import annotations

import asyncio
from argparse import Namespace
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels

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
from langley.knowledge.index_build import COLLECTION_NAME, KnowledgeIndexBuildRuntime
from langley.knowledge.retrieval import retrieve_dense
from langley.settings import Settings


@pytest.fixture
def migrated_database(test_database_url: str, reset_database) -> str:
    reset_database()
    config = Config("alembic.ini")
    config.cmd_opts = Namespace(x=["use_test_database=true"])
    command.upgrade(config, "head")
    return test_database_url


def test_local_qdrant_dense_search_filters_scope_and_preserves_order() -> None:
    async def scenario() -> None:
        settings = Settings()
        runtime = KnowledgeIndexBuildRuntime(None, settings)  # type: ignore[arg-type]
        client = AsyncQdrantClient(url=settings.qdrant_url)
        point_ids = [str(uuid4()) for _ in range(5)]
        user_id = 987_654_321
        knowledge_base_id = 765_432_109
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
                        },
                    ),
                    qmodels.PointStruct(
                        id=point_ids[2],
                        vector=query_vector,
                        payload={
                            "knowledge_chunk_id": 201,
                            "user_id": user_id + 1,
                            "knowledge_base_id": knowledge_base_id,
                        },
                    ),
                    qmodels.PointStruct(
                        id=point_ids[3],
                        vector=query_vector,
                        payload={
                            "knowledge_chunk_id": 202,
                            "user_id": user_id,
                            "knowledge_base_id": knowledge_base_id + 1,
                        },
                    ),
                    qmodels.PointStruct(
                        id=point_ids[4],
                        vector=[-1.0]
                        + [0.0] * (settings.knowledge_embedding_dimension - 1),
                        payload={
                            "knowledge_chunk_id": 203,
                            "user_id": user_id,
                            "knowledge_base_id": knowledge_base_id,
                        },
                    ),
                ],
            )
            hits = await runtime.search_dense(
                query_vector,
                user_id=user_id,
                knowledge_base_id=knowledge_base_id,
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


def test_real_qdrant_candidates_are_filtered_by_mysql_document_readiness(
    migrated_database: str,
) -> None:
    async def scenario() -> None:
        settings = Settings()
        engine = create_database_engine(migrated_database)
        factory = create_session_factory(engine)
        user_id = 9_400_001
        knowledge_base_id = 9_400_002
        ready_chunk_id = 9_400_101
        stale_chunk_id = 9_400_102

        class Runtime(KnowledgeIndexBuildRuntime):
            async def encode_query(
                self, *args: object, **kwargs: object
            ) -> list[float]:
                del args, kwargs
                return [1.0] + [0.0] * (settings.knowledge_embedding_dimension - 1)

        runtime = Runtime(factory, settings)
        client = AsyncQdrantClient(url=settings.qdrant_url)
        vector = [1.0] + [0.0] * (settings.knowledge_embedding_dimension - 1)
        try:
            async with factory() as session, session.begin():
                now = utc_now()
                session.add(User(id=user_id, created_at=now))
                await session.flush()
                knowledge_base = KnowledgeBase(
                    id=knowledge_base_id,
                    user_id=user_id,
                    name="Qdrant authority",
                    index_status="READY",
                    active_embedding_model=settings.knowledge_embedding_model,
                    active_embedding_revision=settings.knowledge_embedding_revision,
                    active_embedding_dimension=settings.knowledge_embedding_dimension,
                    active_embedding_representation=(
                        settings.knowledge_embedding_representation
                    ),
                    created_at=now,
                )
                session.add(knowledge_base)
                await session.flush()
                for offset, indexed_revision in enumerate((1, None), start=1):
                    document = Document(
                        id=knowledge_base_id + offset,
                        knowledge_base_id=knowledge_base_id,
                        name=f"Document {offset}",
                        created_at=now,
                    )
                    session.add(document)
                    await session.flush()
                    version = DocumentVersion(
                        id=knowledge_base_id + 10 + offset,
                        document_id=document.id,
                        source_filename=f"document-{offset}.md",
                        source_media_type="text/markdown",
                        source_sha256=f"{offset:064x}",
                        source_size_bytes=1,
                        storage_key=f"fixture/qdrant-{offset}.md",
                        chunk_max_chars=100,
                        chunk_revision=1,
                        chunk_set_sha256=f"{offset + 10:064x}",
                        indexed_chunk_revision=indexed_revision,
                        created_at=now,
                    )
                    session.add(version)
                    await session.flush()
                    session.add(
                        KnowledgeChunk(
                            id=(ready_chunk_id if offset == 1 else stale_chunk_id),
                            document_version_id=version.id,
                            ordinal=1,
                            content=(
                                "ready evidence" if offset == 1 else "stale evidence"
                            ),
                            heading_path=[],
                            source_regions=[{"kind": "text", "start": 0, "end": 1}],
                            created_at=now,
                        )
                    )

            await runtime._ensure_collection(
                client, dimension=settings.knowledge_embedding_dimension
            )
            await client.upsert(
                collection_name=COLLECTION_NAME,
                wait=True,
                points=[
                    qmodels.PointStruct(
                        id=stale_chunk_id,
                        vector=vector,
                        payload={
                            "knowledge_chunk_id": stale_chunk_id,
                            "user_id": user_id,
                            "knowledge_base_id": knowledge_base_id,
                        },
                    ),
                    qmodels.PointStruct(
                        id=ready_chunk_id,
                        vector=[0.9, 0.1]
                        + [0.0] * (settings.knowledge_embedding_dimension - 2),
                        payload={
                            "knowledge_chunk_id": ready_chunk_id,
                            "user_id": user_id,
                            "knowledge_base_id": knowledge_base_id,
                        },
                    ),
                ],
            )
            result = await retrieve_dense(
                factory,
                runtime,
                user_id=user_id,
                knowledge_base_id=knowledge_base_id,
                query="query",
                top_k=2,
            )
            assert [hit.knowledge_chunk_id for hit in result.hits] == [ready_chunk_id]
        finally:
            try:
                await client.delete(
                    collection_name=COLLECTION_NAME,
                    points_selector=qmodels.PointIdsList(
                        points=[ready_chunk_id, stale_chunk_id]
                    ),
                    wait=True,
                )
            finally:
                await client.close()
                await dispose_database_engine(engine)

    asyncio.run(scenario())
