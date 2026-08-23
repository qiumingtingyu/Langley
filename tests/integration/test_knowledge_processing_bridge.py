"""Real MySQL coverage for the explicit Knowledge processing bridge."""

import asyncio
from argparse import Namespace
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from langley.bootstrap import bootstrap_local_user
from langley.infrastructure.database import (
    create_database_engine,
    create_session_factory,
    dispose_database_engine,
)
from langley.infrastructure.local_file_storage import LocalFileStorage
from langley.infrastructure.models import (
    Document,
    DocumentVersion,
    KnowledgeBase,
    KnowledgeChunk,
)
from langley.knowledge.chunking import CandidateChunk, ChunkingConfig
from langley.knowledge.commands import (
    DocumentAdmissionConflictError,
    DocumentRebuildConflictError,
    _materialize_chunk_rows,
    _replace_document_version_chunks,
    create_initial_document,
    create_knowledge_base,
    load_document_source_ref,
    rebuild_document_version_chunks,
)
from langley.knowledge.contracts import TextSpanRegion
from langley.main import create_app
from langley.settings import Settings


@pytest.fixture
def migrated_database(test_database_url: str, reset_database) -> str:
    reset_database()
    config = Config("alembic.ini")
    config.cmd_opts = Namespace(x=["use_test_database=true"])
    command.upgrade(config, "head")
    return test_database_url


def _settings(database_url: str, storage_root: Path, user_id: int = 1) -> Settings:
    return Settings(
        environment="test",
        database_url=database_url,
        knowledge_storage_root=storage_root,
        local_user_id=user_id,
    )


def _bootstrap(database_url: str, storage_root: Path, user_id: int = 1) -> None:
    assert asyncio.run(
        bootstrap_local_user(_settings(database_url, storage_root, user_id))
    )


def test_chunks_api_exposes_explicit_processing_and_pagination(
    migrated_database: str, tmp_path: Path
) -> None:
    storage_root = tmp_path / "knowledge"
    _bootstrap(migrated_database, storage_root, 1)
    _bootstrap(migrated_database, storage_root, 2)
    with TestClient(create_app(_settings(migrated_database, storage_root))) as client:
        base = client.post("/api/knowledge-bases", json={"name": "Bridge"}).json()
        uploaded = client.post(
            f"/api/knowledge-bases/{base['id']}/documents",
            files={
                "file": (
                    "bridge.md",
                    b"# One\nFirst complete content.\n"
                    b"# Two\nSecond complete content.\n",
                    "text/markdown",
                )
            },
        ).json()
        version_id = uploaded["source"]["document_version_id"]

        before = client.get(f"/api/document-versions/{version_id}/chunks")
        assert before.status_code == 200
        assert before.json() == {
            "document_version_id": version_id,
            "successful_chunk_max_chars": None,
            "suggested_chunk_max_chars": 1200,
            "chunk_count": 0,
            "offset": 0,
            "limit": 50,
            "chunks": [],
        }
        assert (
            client.post(
                f"/api/document-versions/{version_id}/chunks/rebuild", json={}
            ).status_code
            == 422
        )
        assert (
            client.post(
                f"/api/document-versions/{version_id}/chunks/rebuild",
                json={"max_chunk_chars": 0},
            ).status_code
            == 422
        )
        rebuilt = client.post(
            f"/api/document-versions/{version_id}/chunks/rebuild",
            json={"max_chunk_chars": 1200},
        )
        assert rebuilt.status_code == 200
        assert rebuilt.json() == {
            "document_version_id": version_id,
            "successful_chunk_max_chars": 1200,
            "chunk_count": 2,
            "resulting_index_status": "CHUNKED",
        }
        page = client.get(
            f"/api/document-versions/{version_id}/chunks?offset=1&limit=1"
        )
        assert page.status_code == 200
        assert page.json()["chunk_count"] == 2
        assert page.json()["successful_chunk_max_chars"] == 1200
        assert page.json()["chunks"] == [
            {
                "ordinal": 2,
                "content": "Second complete content.\n",
                "heading_path": ["Two"],
                "source_regions": [
                    {"kind": "text_span", "start_byte": 36, "end_byte": 61}
                ],
            }
        ]
        assert (
            client.get(
                f"/api/document-versions/{version_id}/chunks?limit=101"
            ).status_code
            == 422
        )

    with TestClient(create_app(_settings(migrated_database, storage_root, 2))) as other:
        assert (
            other.get(f"/api/document-versions/{version_id}/chunks").status_code == 404
        )
        assert (
            other.post(
                f"/api/document-versions/{version_id}/chunks/rebuild",
                json={"max_chunk_chars": 1200},
            ).status_code
            == 404
        )


def test_zero_chunk_success_and_index_status_transitions(
    migrated_database: str, tmp_path: Path
) -> None:
    async def scenario() -> None:
        engine = create_database_engine(migrated_database)
        try:
            factory = create_session_factory(engine)
            storage = LocalFileStorage(tmp_path / "sources")
            async with factory() as session:
                base = await create_knowledge_base(session, user_id=1, name="Bridge")
            version = await create_initial_document(
                factory,
                storage,
                user_id=1,
                knowledge_base_id=base.id,
                name="Empty chunks",
                source_filename="empty.md",
                source_media_type="text/markdown",
                source_bytes=b"# Empty\n\n   \n",
            )
            expected = (
                ("CHUNKED", None, "CHUNKED"),
                ("READY", "generation-ready", "STALE"),
                ("STALE", "generation-stale", "STALE"),
                ("FAILED", None, "CHUNKED"),
                ("FAILED", "generation-failed", "STALE"),
            )
            for status, generation, resulting_status in expected:
                async with factory() as session, session.begin():
                    current = await session.get(
                        KnowledgeBase, base.id, with_for_update=True
                    )
                    assert current is not None
                    current.index_status = status
                    current.active_generation_id = generation
                result = await rebuild_document_version_chunks(
                    session_factory=factory,
                    file_storage=storage,
                    user_id=1,
                    document_version_id=version.id,
                    config=ChunkingConfig(max_chunk_chars=333),
                )
                assert result.chunk_count == 0
                assert result.index_status == resulting_status
                async with factory() as session:
                    current = await session.get(DocumentVersion, version.id)
                    assert current is not None
                    assert current.chunk_max_chars == 333
                    assert (
                        await session.scalar(
                            select(KnowledgeBase.index_status).where(
                                KnowledgeBase.id == base.id
                            )
                        )
                        == resulting_status
                    )
        finally:
            await dispose_database_engine(engine)

    _bootstrap(migrated_database, tmp_path / "sources")
    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("status", "active_generation_id", "expected_status"),
    [
        ("READY", "active-ready", "STALE"),
        ("STALE", "active-stale", "STALE"),
        ("CHUNKED", None, "CHUNKED"),
        ("FAILED", "active-failed", "STALE"),
        ("FAILED", None, "CHUNKED"),
    ],
)
def test_upload_invalidates_current_index_readiness(
    migrated_database: str,
    tmp_path: Path,
    status: str,
    active_generation_id: str | None,
    expected_status: str,
) -> None:
    async def scenario() -> None:
        engine = create_database_engine(migrated_database)
        try:
            factory = create_session_factory(engine)
            storage = LocalFileStorage(tmp_path / "sources")
            async with factory() as session:
                base = await create_knowledge_base(session, user_id=1, name="Bridge")
            async with factory() as session, session.begin():
                current = await session.get(
                    KnowledgeBase, base.id, with_for_update=True
                )
                assert current is not None
                current.index_status = status
                current.active_generation_id = active_generation_id
            version = await create_initial_document(
                factory,
                storage,
                user_id=1,
                knowledge_base_id=base.id,
                name="New source",
                source_filename="new-source.md",
                source_media_type="text/markdown",
                source_bytes=b"# New\nUnprocessed content.\n",
            )
            async with factory() as session:
                current = await session.get(KnowledgeBase, base.id)
                persisted = await session.get(DocumentVersion, version.id)
                assert current is not None
                assert persisted is not None
                assert persisted.chunk_max_chars is None
                assert current.index_status == expected_status
                assert current.active_generation_id == active_generation_id
        finally:
            await dispose_database_engine(engine)

    _bootstrap(migrated_database, tmp_path / "sources")
    asyncio.run(scenario())


def test_indexing_rejects_upload_publication_without_document_facts(
    migrated_database: str, tmp_path: Path
) -> None:
    async def scenario() -> None:
        engine = create_database_engine(migrated_database)
        try:
            factory = create_session_factory(engine)
            storage = LocalFileStorage(tmp_path / "sources")
            async with factory() as session:
                base = await create_knowledge_base(session, user_id=1, name="Bridge")
            async with factory() as session, session.begin():
                current = await session.get(
                    KnowledgeBase, base.id, with_for_update=True
                )
                assert current is not None
                current.index_status = "INDEXING"
                current.building_generation_id = "building"
            with pytest.raises(
                DocumentAdmissionConflictError, match="KNOWLEDGE_BASE_INDEXING"
            ):
                await create_initial_document(
                    factory,
                    storage,
                    user_id=1,
                    knowledge_base_id=base.id,
                    name="Rejected source",
                    source_filename="rejected.md",
                    source_media_type="text/markdown",
                    source_bytes=b"# Rejected\nContent.\n",
                )
            async with factory() as session:
                current = await session.get(KnowledgeBase, base.id)
                assert current is not None
                assert current.index_status == "INDEXING"
                assert current.building_generation_id == "building"
                assert (await session.scalar(select(Document.id))) is None
                assert (await session.scalar(select(DocumentVersion.id))) is None
        finally:
            await dispose_database_engine(engine)

    _bootstrap(migrated_database, tmp_path / "sources")
    asyncio.run(scenario())


def test_indexing_and_failed_replacement_preserve_current_facts(
    migrated_database: str, tmp_path: Path
) -> None:
    async def scenario() -> None:
        engine = create_database_engine(migrated_database)
        try:
            factory = create_session_factory(engine)
            storage = LocalFileStorage(tmp_path / "sources")
            async with factory() as session:
                base = await create_knowledge_base(session, user_id=1, name="Bridge")
            version = await create_initial_document(
                factory,
                storage,
                user_id=1,
                knowledge_base_id=base.id,
                name="Existing chunks",
                source_filename="existing.md",
                source_media_type="text/markdown",
                source_bytes=b"# Existing\nOld complete content.\n",
            )
            await rebuild_document_version_chunks(
                session_factory=factory,
                file_storage=storage,
                user_id=1,
                document_version_id=version.id,
                config=ChunkingConfig(max_chunk_chars=111),
            )
            source_ref = await load_document_source_ref(
                factory, user_id=1, document_version_id=version.id
            )
            async with factory() as session, session.begin():
                current = await session.get(
                    KnowledgeBase, base.id, with_for_update=True
                )
                assert current is not None
                current.index_status = "INDEXING"
                current.building_generation_id = "building"
            with pytest.raises(DocumentRebuildConflictError):
                await rebuild_document_version_chunks(
                    session_factory=factory,
                    file_storage=storage,
                    user_id=1,
                    document_version_id=version.id,
                    config=ChunkingConfig(max_chunk_chars=222),
                )
            async with factory() as session:
                current = await session.get(DocumentVersion, version.id)
                assert current is not None
                assert current.chunk_max_chars == 111
                assert (
                    await session.scalar(
                        select(KnowledgeBase.index_status).where(
                            KnowledgeBase.id == base.id
                        )
                    )
                    == "INDEXING"
                )

            async with factory() as session, session.begin():
                current = await session.get(
                    KnowledgeBase, base.id, with_for_update=True
                )
                assert current is not None
                current.index_status = "CHUNKED"
                current.building_generation_id = None
            duplicate_rows = _materialize_chunk_rows(
                version.id,
                (
                    CandidateChunk(1, "new", (), (TextSpanRegion(1, 4),)),
                    CandidateChunk(1, "duplicate", (), (TextSpanRegion(1, 4),)),
                ),
            )
            with pytest.raises(IntegrityError):
                await _replace_document_version_chunks(
                    session_factory=factory,
                    user_id=1,
                    source_ref=source_ref,
                    prepared_rows=duplicate_rows,
                    successful_chunk_max_chars=222,
                )
            async with factory() as session:
                current = await session.get(DocumentVersion, version.id)
                assert current is not None
                assert current.chunk_max_chars == 111
                chunks = (
                    await session.scalars(
                        select(KnowledgeChunk)
                        .where(KnowledgeChunk.document_version_id == version.id)
                        .order_by(KnowledgeChunk.ordinal)
                    )
                ).all()
                assert [chunk.content for chunk in chunks] == [
                    "Old complete content.\n"
                ]
                assert (
                    await session.scalar(
                        select(KnowledgeBase.index_status).where(
                            KnowledgeBase.id == base.id
                        )
                    )
                    == "CHUNKED"
                )
        finally:
            await dispose_database_engine(engine)

    _bootstrap(migrated_database, tmp_path / "sources")
    asyncio.run(scenario())
