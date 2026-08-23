"""Serial real-MySQL coverage for Task 5.2 production dense retrieval."""

from __future__ import annotations

import asyncio
from argparse import Namespace
from dataclasses import dataclass

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import delete, select

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
    DenseSearchHit,
    current_knowledge_chunks_statement,
)
from langley.knowledge.retrieval import (
    KnowledgeBaseRetrievalNotFoundError,
    RetrievalIndexInconsistentError,
    retrieve_dense,
)


@pytest.fixture
def migrated_database(test_database_url: str, reset_database) -> str:
    reset_database()
    config = Config("alembic.ini")
    config.cmd_opts = Namespace(x=["use_test_database=true"])
    command.upgrade(config, "head")
    return test_database_url


@dataclass(frozen=True)
class _Seed:
    knowledge_base_id: int
    other_knowledge_base_id: int
    first_chunk_id: int
    second_chunk_id: int
    other_chunk_id: int


class _Runtime:
    def __init__(self, hits: tuple[DenseSearchHit, ...]) -> None:
        self.hits = hits
        self.encode_call_count = 0
        self.search_call_count = 0

    async def encode_query(self, query: str, **kwargs: object) -> list[float]:
        self.encode_call_count += 1
        assert query == "exact query"
        assert kwargs == {
            "model": "BAAI/bge-m3",
            "revision": "a" * 40,
            "dimension": 2,
            "representation": "content_only",
        }
        return [0.6, 0.8]

    async def search_dense(
        self, vector: list[float], **kwargs: object
    ) -> tuple[DenseSearchHit, ...]:
        self.search_call_count += 1
        assert vector == [0.6, 0.8]
        assert kwargs["user_id"] == 1
        assert kwargs["generation_id"] == "11111111-1111-4111-8111-111111111111"
        return self.hits


async def _seed(database_url: str) -> _Seed:
    engine = create_database_engine(database_url)
    factory = create_session_factory(engine)
    try:
        async with factory() as session, session.begin():
            now = utc_now()
            session.add_all([User(id=1, created_at=now), User(id=2, created_at=now)])
            await session.flush()
            bases = [
                KnowledgeBase(
                    user_id=1,
                    name="Retrieval",
                    index_status="READY",
                    active_generation_id="11111111-1111-4111-8111-111111111111",
                    active_embedding_model="BAAI/bge-m3",
                    active_embedding_revision="a" * 40,
                    active_embedding_dimension=2,
                    active_embedding_representation="content_only",
                    active_chunk_snapshot_sha256="a" * 64,
                    created_at=now,
                ),
                KnowledgeBase(
                    user_id=1,
                    name="Other",
                    index_status="READY",
                    active_generation_id="22222222-2222-4222-8222-222222222222",
                    active_embedding_model="BAAI/bge-m3",
                    active_embedding_revision="a" * 40,
                    active_embedding_dimension=2,
                    active_embedding_representation="content_only",
                    active_chunk_snapshot_sha256="b" * 64,
                    created_at=now,
                ),
            ]
            session.add_all(bases)
            await session.flush()
            documents = [
                Document(knowledge_base_id=base.id, name=base.name, created_at=now)
                for base in bases
            ]
            session.add_all(documents)
            await session.flush()
            versions = [
                DocumentVersion(
                    document_id=document.id,
                    source_filename=f"{document.name}.md",
                    source_media_type="text/markdown",
                    source_sha256="c" * 64,
                    source_size_bytes=1,
                    storage_key=f"fixture/{document.id}.md",
                    chunk_max_chars=100,
                    created_at=now,
                )
                for document in documents
            ]
            session.add_all(versions)
            await session.flush()
            chunks = [
                KnowledgeChunk(
                    document_version_id=versions[0].id,
                    ordinal=1,
                    content="first",
                    heading_path=["First"],
                    source_regions=[{"kind": "text", "start": 0, "end": 1}],
                    created_at=now,
                ),
                KnowledgeChunk(
                    document_version_id=versions[0].id,
                    ordinal=2,
                    content="second",
                    heading_path=["Second"],
                    source_regions=[{"kind": "text", "start": 1, "end": 2}],
                    created_at=now,
                ),
                KnowledgeChunk(
                    document_version_id=versions[1].id,
                    ordinal=1,
                    content="other",
                    heading_path=[],
                    source_regions=[{"kind": "text", "start": 0, "end": 1}],
                    created_at=now,
                ),
            ]
            session.add_all(chunks)
            await session.flush()
            return _Seed(
                knowledge_base_id=bases[0].id,
                other_knowledge_base_id=bases[1].id,
                first_chunk_id=chunks[0].id,
                second_chunk_id=chunks[1].id,
                other_chunk_id=chunks[2].id,
            )
    finally:
        await dispose_database_engine(engine)


def test_real_mysql_retrieval_uses_authoritative_rows_and_qdrant_order(
    migrated_database: str,
) -> None:
    async def scenario() -> None:
        seed = await _seed(migrated_database)
        engine = create_database_engine(migrated_database)
        try:
            runtime = _Runtime(
                (
                    DenseSearchHit(seed.second_chunk_id, 0.9),
                    DenseSearchHit(seed.first_chunk_id, 0.8),
                )
            )
            result = await retrieve_dense(
                create_session_factory(engine),
                runtime,  # type: ignore[arg-type]
                user_id=1,
                knowledge_base_id=seed.knowledge_base_id,
                query="exact query",
                top_k=2,
            )
            assert [
                (hit.rank, hit.knowledge_chunk_id, hit.content) for hit in result.hits
            ] == [
                (1, seed.second_chunk_id, "second"),
                (2, seed.first_chunk_id, "first"),
            ]
            assert runtime.encode_call_count == runtime.search_call_count == 1
        finally:
            await dispose_database_engine(engine)

    asyncio.run(scenario())


def test_real_mysql_retrieval_rejects_not_owned_and_other_kb_chunks(
    migrated_database: str,
) -> None:
    async def scenario() -> None:
        seed = await _seed(migrated_database)
        engine = create_database_engine(migrated_database)
        try:
            factory = create_session_factory(engine)
            not_owned_runtime = _Runtime(())
            with pytest.raises(KnowledgeBaseRetrievalNotFoundError):
                await retrieve_dense(
                    factory,
                    not_owned_runtime,  # type: ignore[arg-type]
                    user_id=2,
                    knowledge_base_id=seed.knowledge_base_id,
                    query="exact query",
                    top_k=1,
                )
            assert not_owned_runtime.encode_call_count == 0
            wrong_chunk_runtime = _Runtime((DenseSearchHit(seed.other_chunk_id, 0.9),))
            with pytest.raises(RetrievalIndexInconsistentError):
                await retrieve_dense(
                    factory,
                    wrong_chunk_runtime,  # type: ignore[arg-type]
                    user_id=1,
                    knowledge_base_id=seed.knowledge_base_id,
                    query="exact query",
                    top_k=1,
                )
            assert (
                wrong_chunk_runtime.encode_call_count
                == wrong_chunk_runtime.search_call_count
                == 1
            )
        finally:
            await dispose_database_engine(engine)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("active_generation_id", None),
        ("active_embedding_model", None),
        ("active_embedding_revision", "r" * 40),
        ("active_embedding_dimension", 0),
        ("active_embedding_representation", "unsupported"),
        ("active_chunk_snapshot_sha256", "g" * 64),
    ],
)
def test_real_mysql_corrupted_ready_active_facts_fail_closed(
    migrated_database: str, field: str, value: object
) -> None:
    async def scenario() -> None:
        seed = await _seed(migrated_database)
        engine = create_database_engine(migrated_database)
        try:
            factory = create_session_factory(engine)
            async with factory() as session, session.begin():
                knowledge_base = await session.get(
                    KnowledgeBase, seed.knowledge_base_id
                )
                assert knowledge_base is not None
                setattr(knowledge_base, field, value)
            runtime = _Runtime(())
            with pytest.raises(RetrievalIndexInconsistentError):
                await retrieve_dense(
                    factory,
                    runtime,  # type: ignore[arg-type]
                    user_id=1,
                    knowledge_base_id=seed.knowledge_base_id,
                    query="exact query",
                    top_k=1,
                )
            assert runtime.encode_call_count == runtime.search_call_count == 0
        finally:
            await dispose_database_engine(engine)

    asyncio.run(scenario())


def test_real_mysql_final_read_uses_one_repeatable_read_snapshot(
    migrated_database: str,
) -> None:
    async def scenario() -> None:
        seed = await _seed(migrated_database)
        engine = create_database_engine(migrated_database)
        factory = create_session_factory(engine)
        first_read_complete = asyncio.Event()
        mutation_committed = asyncio.Event()
        observed: dict[str, object] = {}

        async def final_reader() -> None:
            async with factory() as session:
                async with session.begin():
                    knowledge_base = await session.scalar(
                        select(KnowledgeBase).where(
                            KnowledgeBase.id == seed.knowledge_base_id
                        )
                    )
                    assert knowledge_base is not None
                    observed["status"] = knowledge_base.index_status
                    first_read_complete.set()
                    await mutation_committed.wait()
                    rows = (
                        await session.scalars(
                            current_knowledge_chunks_statement(
                                seed.knowledge_base_id
                            ).where(KnowledgeChunk.id == seed.first_chunk_id)
                        )
                    ).all()
                    observed["chunk_ids"] = [row.id for row in rows]

        async def mutate_after_snapshot() -> None:
            await first_read_complete.wait()
            async with factory() as session, session.begin():
                knowledge_base = await session.get(
                    KnowledgeBase, seed.knowledge_base_id
                )
                assert knowledge_base is not None
                knowledge_base.index_status = "STALE"
                await session.execute(
                    delete(KnowledgeChunk).where(
                        KnowledgeChunk.id == seed.first_chunk_id
                    )
                )
            mutation_committed.set()

        try:
            await asyncio.gather(final_reader(), mutate_after_snapshot())
            assert observed == {"status": "READY", "chunk_ids": [seed.first_chunk_id]}
        finally:
            await dispose_database_engine(engine)

    asyncio.run(scenario())
