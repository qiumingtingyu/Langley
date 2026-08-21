"""Real MySQL and filesystem coverage for the Slice 6 Task 1 foundation."""

import asyncio
from argparse import Namespace
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import func, select, text
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from langley.infrastructure.database import (
    create_database_engine,
    create_session_factory,
    dispose_database_engine,
)
from langley.infrastructure.local_file_storage import LocalFileStorage
from langley.infrastructure.models import Document, DocumentVersion, User
from langley.knowledge.commands import (
    KnowledgeBaseNotFoundError,
    SourceIntegrityError,
    create_initial_document,
    create_knowledge_base,
    find_unreferenced_local_sources,
    load_document_source_ref,
    read_verified_source,
)
from langley.knowledge.contracts import StoredSource

_TIMESTAMP = datetime(2026, 8, 21, 0, 0)


@pytest.fixture
def migrated_database(test_database_url: str, reset_database) -> str:
    """Reset only langley_test and migrate it to the current real head."""
    reset_database()
    config = Config("alembic.ini")
    config.cmd_opts = Namespace(x=["use_test_database=true"])
    command.upgrade(config, "head")
    return test_database_url


async def _session_factory_for(
    database_url: str,
) -> tuple[object, async_sessionmaker[AsyncSession]]:
    engine = create_database_engine(database_url)
    return engine, create_session_factory(engine)


async def _seed_user(
    session_factory: async_sessionmaker[AsyncSession], user_id: int
) -> None:
    async with session_factory() as session:
        async with session.begin():
            session.add(User(id=user_id, created_at=_TIMESTAMP))


async def _count_rows(
    session_factory: async_sessionmaker[AsyncSession],
    model: type[Document] | type[DocumentVersion],
) -> int:
    async with session_factory() as session:
        return (await session.scalar(select(func.count()).select_from(model))) or 0


def test_schema_constraints_and_machine_collation(migrated_database: str) -> None:
    """MySQL enforces Task 1 facts without encoding Markdown-only capability."""

    async def verify() -> None:
        engine = create_database_engine(migrated_database)
        try:
            async with engine.begin() as connection:
                await connection.execute(
                    text("INSERT INTO users (id, created_at) VALUES (1, :created_at)"),
                    {"created_at": _TIMESTAMP},
                )
                with pytest.raises(DBAPIError):
                    await connection.execute(
                        text(
                            "INSERT INTO knowledge_bases (user_id, name, created_at) "
                            "VALUES (1, '   ', :created_at)"
                        ),
                        {"created_at": _TIMESTAMP},
                    )
                await connection.execute(
                    text(
                        "INSERT INTO knowledge_bases (id, user_id, name, created_at) "
                        "VALUES (1, 1, 'same name', :created_at), "
                        "(2, 1, 'same name', :created_at)"
                    ),
                    {"created_at": _TIMESTAMP},
                )
                await connection.execute(
                    text(
                        "INSERT INTO documents "
                        "(id, knowledge_base_id, name, created_at) "
                        "VALUES (1, 1, 'document', :created_at), "
                        "(2, 2, 'document', :created_at)"
                    ),
                    {"created_at": _TIMESTAMP},
                )
                await connection.execute(
                    text(
                        "INSERT INTO document_versions "
                        "(document_id, source_filename, source_media_type, "
                        "source_sha256, "
                        "source_size_bytes, storage_key, created_at) VALUES "
                        "(1, 'one.md', 'application/pdf', :sha256, 1, "
                        "'key-one', :created_at), "
                        "(2, 'two.md', 'application/pdf', :sha256, 1, "
                        "'key-two', :created_at)"
                    ),
                    {"sha256": "a" * 64, "created_at": _TIMESTAMP},
                )
                with pytest.raises(DBAPIError):
                    await connection.execute(
                        text(
                            "INSERT INTO document_versions "
                            "(document_id, source_filename, source_media_type, "
                            "source_sha256, "
                            "source_size_bytes, storage_key, created_at) VALUES "
                            "(2, 'three.md', 'application/pdf', :sha256, 1, "
                            "'key-one', :created_at)"
                        ),
                        {"sha256": "b" * 64, "created_at": _TIMESTAMP},
                    )
            async with engine.connect() as connection:
                binary_count = await connection.scalar(
                    text(
                        "SELECT COUNT(*) FROM information_schema.columns "
                        "WHERE table_schema = DATABASE() "
                        "AND table_name = 'document_versions' "
                        "AND column_name IN "
                        "('source_media_type', 'source_sha256', 'storage_key') "
                        "AND collation_name = 'utf8mb4_0900_bin'"
                    )
                )
                assert binary_count == 3
        finally:
            await dispose_database_engine(engine)

    asyncio.run(verify())


def test_initial_admission_is_owned_canonical_and_exact(
    migrated_database: str, tmp_path: Path
) -> None:
    async def verify() -> None:
        engine, session_factory = await _session_factory_for(migrated_database)
        try:
            await _seed_user(session_factory, 1)
            async with session_factory() as session:
                knowledge_base = await create_knowledge_base(
                    session, user_id=1, name="Study notes"
                )
            storage = LocalFileStorage(tmp_path / "knowledge")
            source_bytes = b"# Notes\n\nExact bytes.\n"
            version = await create_initial_document(
                session_factory,
                storage,
                user_id=1,
                knowledge_base_id=knowledge_base.id,
                name="Notes",
                source_filename="../../notes.md",
                source_media_type=" Text/Markdown ",
                source_bytes=source_bytes,
            )

            source_ref = await load_document_source_ref(
                session_factory, user_id=1, document_version_id=version.id
            )
            assert source_ref.source_media_type == "text/markdown"
            assert await read_verified_source(storage, source_ref) == source_bytes
            assert source_ref.storage_key.startswith("users/1/sources/")
            assert "notes.md" not in source_ref.storage_key
        finally:
            await dispose_database_engine(engine)

    asyncio.run(verify())


def test_wrong_user_cannot_admit_into_another_users_knowledge_base(
    migrated_database: str, tmp_path: Path
) -> None:
    async def verify() -> None:
        engine, session_factory = await _session_factory_for(migrated_database)
        try:
            await _seed_user(session_factory, 1)
            await _seed_user(session_factory, 2)
            async with session_factory() as session:
                knowledge_base = await create_knowledge_base(
                    session, user_id=1, name="Private"
                )
            with pytest.raises(KnowledgeBaseNotFoundError):
                await create_initial_document(
                    session_factory,
                    LocalFileStorage(tmp_path / "knowledge"),
                    user_id=2,
                    knowledge_base_id=knowledge_base.id,
                    name="No access",
                    source_filename="no-access.md",
                    source_media_type="text/markdown",
                    source_bytes=b"# No access\n",
                )
        finally:
            await dispose_database_engine(engine)

    asyncio.run(verify())


def test_unsupported_media_type_is_rejected_before_file_finalization(
    migrated_database: str, tmp_path: Path
) -> None:
    async def verify() -> None:
        engine, session_factory = await _session_factory_for(migrated_database)
        try:
            await _seed_user(session_factory, 1)
            async with session_factory() as session:
                knowledge_base = await create_knowledge_base(
                    session, user_id=1, name="Foundation"
                )
            storage_root = tmp_path / "knowledge"
            with pytest.raises(ValueError, match="unsupported source_media_type"):
                await create_initial_document(
                    session_factory,
                    LocalFileStorage(storage_root),
                    user_id=1,
                    knowledge_base_id=knowledge_base.id,
                    name="Unsupported",
                    source_filename="unsupported.pdf",
                    source_media_type="application/pdf",
                    source_bytes=b"not a PDF parser input",
                )
            assert not storage_root.exists()
            assert await _count_rows(session_factory, Document) == 0
            assert await _count_rows(session_factory, DocumentVersion) == 0
        finally:
            await dispose_database_engine(engine)

    asyncio.run(verify())


def test_finalization_failure_creates_no_document_or_version(
    migrated_database: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def verify() -> None:
        engine, session_factory = await _session_factory_for(migrated_database)
        try:
            await _seed_user(session_factory, 1)
            async with session_factory() as session:
                knowledge_base = await create_knowledge_base(
                    session, user_id=1, name="Foundation"
                )

            def fail_replace(source: Path, destination: Path) -> None:
                del source, destination
                raise OSError("rename failed")

            monkeypatch.setattr(
                "langley.infrastructure.local_file_storage.os.replace", fail_replace
            )
            with pytest.raises(OSError, match="rename failed"):
                await create_initial_document(
                    session_factory,
                    LocalFileStorage(tmp_path / "knowledge"),
                    user_id=1,
                    knowledge_base_id=knowledge_base.id,
                    name="Failed finalization",
                    source_filename="failed.md",
                    source_media_type="text/markdown",
                    source_bytes=b"# Not admitted\n",
                )
            assert await _count_rows(session_factory, Document) == 0
            assert await _count_rows(session_factory, DocumentVersion) == 0
        finally:
            await dispose_database_engine(engine)

    asyncio.run(verify())


@dataclass
class _ConflictingStorage:
    """Return a conflicting DB key after preserving a real finalized source."""

    storage: LocalFileStorage
    conflicting_storage_key: str
    finalized_source: StoredSource | None = None

    async def store_source(self, user_id: int, source_bytes: bytes) -> StoredSource:
        self.finalized_source = await self.storage.store_source(user_id, source_bytes)
        return StoredSource(
            storage_key=self.conflicting_storage_key,
            sha256=self.finalized_source.sha256,
            size_bytes=self.finalized_source.size_bytes,
        )

    async def read_source(self, storage_key: str) -> bytes:
        return await self.storage.read_source(storage_key)


def test_db_admission_failure_leaves_finalized_orphan_without_business_rows(
    migrated_database: str, tmp_path: Path
) -> None:
    async def verify() -> None:
        engine, session_factory = await _session_factory_for(migrated_database)
        try:
            await _seed_user(session_factory, 1)
            async with session_factory() as session:
                knowledge_base = await create_knowledge_base(
                    session, user_id=1, name="Foundation"
                )
            storage = LocalFileStorage(tmp_path / "knowledge")
            existing = await create_initial_document(
                session_factory,
                storage,
                user_id=1,
                knowledge_base_id=knowledge_base.id,
                name="Existing",
                source_filename="existing.md",
                source_media_type="text/markdown",
                source_bytes=b"# Existing\n",
            )
            conflicting_storage = _ConflictingStorage(storage, existing.storage_key)

            with pytest.raises(IntegrityError):
                await create_initial_document(
                    session_factory,
                    conflicting_storage,
                    user_id=1,
                    knowledge_base_id=knowledge_base.id,
                    name="Failed",
                    source_filename="failed.md",
                    source_media_type="text/markdown",
                    source_bytes=b"# Finalized orphan\n",
                )

            assert conflicting_storage.finalized_source is not None
            assert (
                await storage.read_source(
                    conflicting_storage.finalized_source.storage_key
                )
                == b"# Finalized orphan\n"
            )
            assert await _count_rows(session_factory, Document) == 1
            assert await _count_rows(session_factory, DocumentVersion) == 1
        finally:
            await dispose_database_engine(engine)

    asyncio.run(verify())


def test_integrity_and_orphan_diagnostic_are_non_destructive(
    migrated_database: str, tmp_path: Path
) -> None:
    async def verify() -> None:
        engine, session_factory = await _session_factory_for(migrated_database)
        try:
            await _seed_user(session_factory, 1)
            async with session_factory() as session:
                knowledge_base = await create_knowledge_base(
                    session, user_id=1, name="Foundation"
                )
            storage = LocalFileStorage(tmp_path / "knowledge")
            version = await create_initial_document(
                session_factory,
                storage,
                user_id=1,
                knowledge_base_id=knowledge_base.id,
                name="Tracked",
                source_filename="tracked.md",
                source_media_type="text/markdown",
                source_bytes=b"# Tracked\n",
            )
            orphan = await storage.store_source(1, b"# Orphan\n")

            assert await find_unreferenced_local_sources(session_factory, storage) == (
                orphan.storage_key,
            )
            assert await storage.read_source(orphan.storage_key) == b"# Orphan\n"

            source_ref = await load_document_source_ref(
                session_factory, user_id=1, document_version_id=version.id
            )
            tracked_path = tmp_path / "knowledge" / Path(source_ref.storage_key)
            tracked_path.write_bytes(b"# Changed\n")
            with pytest.raises(SourceIntegrityError):
                await read_verified_source(storage, source_ref)
        finally:
            await dispose_database_engine(engine)

    asyncio.run(verify())
