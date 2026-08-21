"""Real MySQL evidence for Task 2.4 atomic current-chunk replacement."""

import asyncio
from argparse import Namespace
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import delete, select, update
from sqlalchemy.exc import IntegrityError

import langley.knowledge.commands as knowledge_commands
from langley.business_time import utc_now
from langley.infrastructure.database import (
    create_database_engine,
    create_session_factory,
    dispose_database_engine,
)
from langley.infrastructure.local_file_storage import LocalFileStorage
from langley.infrastructure.models import DocumentVersion, KnowledgeChunk, User
from langley.knowledge.chunking import (
    CandidateChunk,
    ChunkingConfig,
    build_candidate_chunks,
)
from langley.knowledge.commands import (
    DocumentVersionNotFoundError,
    SourceIntegrityError,
    _materialize_chunk_rows,
    _replace_document_version_chunks,
    create_initial_document,
    create_knowledge_base,
    load_document_source_ref,
    rebuild_document_version_chunks,
)
from langley.knowledge.contracts import TextSpanRegion, encode_source_region
from langley.knowledge.markdown import parse_markdown

_SOURCE = b"# A\none two three four five six seven eight nine ten eleven twelve\n"


@pytest.fixture
def migrated_database(test_database_url: str, reset_database) -> str:
    reset_database()
    config = Config("alembic.ini")
    config.cmd_opts = Namespace(x=["use_test_database=true"])
    command.upgrade(config, "head")
    return test_database_url


async def _chunk_facts(factory, document_version_id: int) -> list[tuple[object, ...]]:
    async with factory() as session:
        rows = (
            await session.scalars(
                select(KnowledgeChunk)
                .where(KnowledgeChunk.document_version_id == document_version_id)
                .order_by(KnowledgeChunk.ordinal)
            )
        ).all()
    return [
        (row.ordinal, row.content, row.heading_path, row.source_regions) for row in rows
    ]


def _candidate_facts(source: bytes, config: ChunkingConfig) -> list[tuple[object, ...]]:
    return [
        (
            candidate.ordinal,
            candidate.content,
            list(candidate.heading_path),
            [encode_source_region(region) for region in candidate.source_regions],
        )
        for candidate in build_candidate_chunks(parse_markdown(source), config)
    ]


async def _seed_committed_chunks(factory, storage: LocalFileStorage) -> DocumentVersion:
    async with factory() as session, session.begin():
        session.add(User(id=1, created_at=utc_now()))
    async with factory() as session:
        base = await create_knowledge_base(session, user_id=1, name="Base")
    version = await create_initial_document(
        factory,
        storage,
        user_id=1,
        knowledge_base_id=base.id,
        name="Doc",
        source_filename="doc.md",
        source_media_type="text/markdown",
        source_bytes=_SOURCE,
    )
    await rebuild_document_version_chunks(
        session_factory=factory,
        file_storage=storage,
        user_id=1,
        document_version_id=version.id,
        config=ChunkingConfig(max_chunk_chars=64),
    )
    return version


def test_reader_sees_complete_old_set_until_writer_commits(
    migrated_database: str, tmp_path: Path
) -> None:
    async def verify() -> None:
        engine = create_database_engine(migrated_database)
        try:
            factory = create_session_factory(engine)
            async with factory() as session, session.begin():
                session.add(User(id=1, created_at=utc_now()))
            async with factory() as session:
                base = await create_knowledge_base(session, user_id=1, name="Base")
            storage = LocalFileStorage(tmp_path / "sources")
            version = await create_initial_document(
                factory,
                storage,
                user_id=1,
                knowledge_base_id=base.id,
                name="Doc",
                source_filename="doc.md",
                source_media_type="text/markdown",
                source_bytes=b"# A\nold\n",
            )
            await rebuild_document_version_chunks(
                session_factory=factory,
                file_storage=storage,
                user_id=1,
                document_version_id=version.id,
                config=ChunkingConfig(),
            )
            async with factory() as writer:
                async with writer.begin():
                    await writer.scalar(
                        select(DocumentVersion)
                        .where(DocumentVersion.id == version.id)
                        .with_for_update()
                    )
                    await writer.execute(
                        delete(KnowledgeChunk).where(
                            KnowledgeChunk.document_version_id == version.id
                        )
                    )
                    writer.add(
                        KnowledgeChunk(
                            document_version_id=version.id,
                            ordinal=1,
                            content="new\n",
                            heading_path=["A"],
                            source_regions=[
                                {"kind": "text_span", "start_byte": 4, "end_byte": 8}
                            ],
                            created_at=utc_now(),
                        )
                    )
                    await writer.flush()
                    async with factory() as reader:
                        assert (
                            await reader.scalars(select(KnowledgeChunk.content))
                        ).all() == ["old\n"]
                async with factory() as reader:
                    assert (
                        await reader.scalars(select(KnowledgeChunk.content))
                    ).all() == ["new\n"]
        finally:
            await dispose_database_engine(engine)

    asyncio.run(verify())


def test_rebuild_replaces_old_set_with_complete_different_set_and_is_deterministic(
    migrated_database: str, tmp_path: Path
) -> None:
    """R1, R2, R11, and R12 use real rows and distinct config-derived sets."""

    async def verify() -> None:
        engine = create_database_engine(migrated_database)
        try:
            factory = create_session_factory(engine)
            async with factory() as session, session.begin():
                session.add(User(id=1, created_at=utc_now()))
            async with factory() as session:
                base = await create_knowledge_base(session, user_id=1, name="Base")
            storage = LocalFileStorage(tmp_path / "sources")
            version = await create_initial_document(
                factory,
                storage,
                user_id=1,
                knowledge_base_id=base.id,
                name="Doc",
                source_filename="doc.md",
                source_media_type="text/markdown",
                source_bytes=_SOURCE,
            )
            config_a = ChunkingConfig(max_chunk_chars=64)
            config_b = ChunkingConfig(max_chunk_chars=20)
            expected_a = _candidate_facts(_SOURCE, config_a)
            expected_b = _candidate_facts(_SOURCE, config_b)
            assert expected_a != expected_b

            await rebuild_document_version_chunks(
                session_factory=factory,
                file_storage=storage,
                user_id=1,
                document_version_id=version.id,
                config=config_a,
            )
            assert await _chunk_facts(factory, version.id) == expected_a

            await rebuild_document_version_chunks(
                session_factory=factory,
                file_storage=storage,
                user_id=1,
                document_version_id=version.id,
                config=config_b,
            )
            assert await _chunk_facts(factory, version.id) == expected_b

            await rebuild_document_version_chunks(
                session_factory=factory,
                file_storage=storage,
                user_id=1,
                document_version_id=version.id,
                config=config_b,
            )
            assert await _chunk_facts(factory, version.id) == expected_b
        finally:
            await dispose_database_engine(engine)

    asyncio.run(verify())


def test_phase_b_and_source_verification_failures_preserve_old_set(
    migrated_database: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R4/R5: no replacement transaction runs after a build or source failure."""

    async def verify() -> None:
        engine = create_database_engine(migrated_database)
        try:
            factory = create_session_factory(engine)
            async with factory() as session, session.begin():
                session.add(User(id=1, created_at=utc_now()))
            async with factory() as session:
                base = await create_knowledge_base(session, user_id=1, name="Base")
            storage = LocalFileStorage(tmp_path / "sources")
            version = await create_initial_document(
                factory,
                storage,
                user_id=1,
                knowledge_base_id=base.id,
                name="Doc",
                source_filename="doc.md",
                source_media_type="text/markdown",
                source_bytes=_SOURCE,
            )
            await rebuild_document_version_chunks(
                session_factory=factory,
                file_storage=storage,
                user_id=1,
                document_version_id=version.id,
                config=ChunkingConfig(max_chunk_chars=64),
            )
            old_facts = await _chunk_facts(factory, version.id)

            def fail_parse(_: bytes):
                raise ValueError("parser failed after Phase A")

            monkeypatch.setattr(knowledge_commands, "parse_markdown", fail_parse)
            with pytest.raises(ValueError, match="parser failed"):
                await rebuild_document_version_chunks(
                    session_factory=factory,
                    file_storage=storage,
                    user_id=1,
                    document_version_id=version.id,
                    config=ChunkingConfig(max_chunk_chars=20),
                )
            assert await _chunk_facts(factory, version.id) == old_facts
            monkeypatch.undo()

            source_ref = await load_document_source_ref(
                factory, user_id=1, document_version_id=version.id
            )
            source_path = storage._path_for_storage_key(source_ref.storage_key)
            source_path.unlink()
            with pytest.raises(SourceIntegrityError, match="missing"):
                await rebuild_document_version_chunks(
                    session_factory=factory,
                    file_storage=storage,
                    user_id=1,
                    document_version_id=version.id,
                    config=ChunkingConfig(max_chunk_chars=20),
                )
            assert await _chunk_facts(factory, version.id) == old_facts

            source_path.write_bytes(_SOURCE.replace(b"one", b"two", 1))
            with pytest.raises(SourceIntegrityError, match="hash"):
                await rebuild_document_version_chunks(
                    session_factory=factory,
                    file_storage=storage,
                    user_id=1,
                    document_version_id=version.id,
                    config=ChunkingConfig(max_chunk_chars=20),
                )
            assert await _chunk_facts(factory, version.id) == old_facts
        finally:
            await dispose_database_engine(engine)

    asyncio.run(verify())


def test_zero_candidate_rebuild_deletes_a_complete_old_set(
    migrated_database: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R3: zero candidates remain a valid atomic replacement of an old set."""

    async def verify() -> None:
        engine = create_database_engine(migrated_database)
        try:
            factory = create_session_factory(engine)
            async with factory() as session, session.begin():
                session.add(User(id=1, created_at=utc_now()))
            async with factory() as session:
                base = await create_knowledge_base(session, user_id=1, name="Base")
            storage = LocalFileStorage(tmp_path / "sources")
            version = await create_initial_document(
                factory,
                storage,
                user_id=1,
                knowledge_base_id=base.id,
                name="Doc",
                source_filename="doc.md",
                source_media_type="text/markdown",
                source_bytes=_SOURCE,
            )
            await rebuild_document_version_chunks(
                session_factory=factory,
                file_storage=storage,
                user_id=1,
                document_version_id=version.id,
                config=ChunkingConfig(max_chunk_chars=64),
            )
            assert await _chunk_facts(factory, version.id) != []

            monkeypatch.setattr(
                knowledge_commands,
                "build_candidate_chunks",
                lambda _parsed, _config: (),
            )

            await rebuild_document_version_chunks(
                session_factory=factory,
                file_storage=storage,
                user_id=1,
                document_version_id=version.id,
                config=ChunkingConfig(),
            )
            assert await _chunk_facts(factory, version.id) == []
        finally:
            await dispose_database_engine(engine)

    asyncio.run(verify())


def test_same_version_concurrent_distinct_rebuilds_leave_one_complete_set(
    migrated_database: str, tmp_path: Path
) -> None:
    """R7: concurrent Phase C transactions serialize on DocumentVersion FOR UPDATE."""

    async def verify() -> None:
        engine = create_database_engine(migrated_database)
        try:
            factory = create_session_factory(engine)
            async with factory() as session, session.begin():
                session.add(User(id=1, created_at=utc_now()))
            async with factory() as session:
                base = await create_knowledge_base(session, user_id=1, name="Base")
            storage = LocalFileStorage(tmp_path / "sources")
            version = await create_initial_document(
                factory,
                storage,
                user_id=1,
                knowledge_base_id=base.id,
                name="Doc",
                source_filename="doc.md",
                source_media_type="text/markdown",
                source_bytes=_SOURCE,
            )
            config_a = ChunkingConfig(max_chunk_chars=64)
            config_b = ChunkingConfig(max_chunk_chars=20)
            expected_a = _candidate_facts(_SOURCE, config_a)
            expected_b = _candidate_facts(_SOURCE, config_b)
            assert expected_a != expected_b

            await asyncio.gather(
                rebuild_document_version_chunks(
                    session_factory=factory,
                    file_storage=storage,
                    user_id=1,
                    document_version_id=version.id,
                    config=config_a,
                ),
                rebuild_document_version_chunks(
                    session_factory=factory,
                    file_storage=storage,
                    user_id=1,
                    document_version_id=version.id,
                    config=config_b,
                ),
            )
            assert await _chunk_facts(factory, version.id) in (expected_a, expected_b)
        finally:
            await dispose_database_engine(engine)

    asyncio.run(verify())


def test_insert_failure_after_delete_rolls_back_to_the_complete_old_set(
    migrated_database: str, tmp_path: Path
) -> None:
    """R6: MySQL uniqueness failure rolls back the DELETE and all replacement rows."""

    async def verify() -> None:
        engine = create_database_engine(migrated_database)
        try:
            factory = create_session_factory(engine)
            storage = LocalFileStorage(tmp_path / "sources")
            version = await _seed_committed_chunks(factory, storage)
            old_facts = await _chunk_facts(factory, version.id)
            source_ref = await load_document_source_ref(
                factory, user_id=1, document_version_id=version.id
            )
            duplicate_rows = _materialize_chunk_rows(
                version.id,
                (
                    CandidateChunk(1, "new", (), (TextSpanRegion(4, 7),)),
                    CandidateChunk(1, "set", (), (TextSpanRegion(4, 7),)),
                ),
            )

            with pytest.raises(IntegrityError):
                await _replace_document_version_chunks(
                    session_factory=factory,
                    user_id=1,
                    source_ref=source_ref,
                    prepared_rows=duplicate_rows,
                )

            assert await _chunk_facts(factory, version.id) == old_facts
        finally:
            await dispose_database_engine(engine)

    asyncio.run(verify())


def test_wrong_user_rebuild_preserves_the_complete_old_set(
    migrated_database: str, tmp_path: Path
) -> None:
    """R9: ownership rejection occurs before a replacement transaction mutates."""

    async def verify() -> None:
        engine = create_database_engine(migrated_database)
        try:
            factory = create_session_factory(engine)
            storage = LocalFileStorage(tmp_path / "sources")
            version = await _seed_committed_chunks(factory, storage)
            old_facts = await _chunk_facts(factory, version.id)

            with pytest.raises(DocumentVersionNotFoundError):
                await rebuild_document_version_chunks(
                    session_factory=factory,
                    file_storage=storage,
                    user_id=2,
                    document_version_id=version.id,
                    config=ChunkingConfig(max_chunk_chars=20),
                )

            assert await _chunk_facts(factory, version.id) == old_facts
        finally:
            await dispose_database_engine(engine)

    asyncio.run(verify())


def test_phase_c_source_identity_mismatch_preserves_the_complete_old_set(
    migrated_database: str, tmp_path: Path
) -> None:
    """R10: immutable source revalidation rejects direct test-side DB drift."""

    async def verify() -> None:
        engine = create_database_engine(migrated_database)
        try:
            factory = create_session_factory(engine)
            storage = LocalFileStorage(tmp_path / "sources")
            version = await _seed_committed_chunks(factory, storage)
            old_facts = await _chunk_facts(factory, version.id)
            source_ref = await load_document_source_ref(
                factory, user_id=1, document_version_id=version.id
            )
            async with factory() as session, session.begin():
                await session.execute(
                    update(DocumentVersion)
                    .where(DocumentVersion.id == version.id)
                    .values(source_size_bytes=source_ref.source_size_bytes + 1)
                )

            with pytest.raises(RuntimeError, match="source identity"):
                await _replace_document_version_chunks(
                    session_factory=factory,
                    user_id=1,
                    source_ref=source_ref,
                    prepared_rows=[],
                )

            assert await _chunk_facts(factory, version.id) == old_facts
        finally:
            await dispose_database_engine(engine)

    asyncio.run(verify())
