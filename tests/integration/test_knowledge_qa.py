"""Focused real-MySQL durability and atomicity for Task 6 Knowledge QA."""

import asyncio
from argparse import Namespace
from datetime import datetime

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError

from langley.answer_execution import _commit_success, _start_running
from langley.answering.grounding import GroundingPolicy
from langley.answering.knowledge_qa import CitationDraft
from langley.bootstrap import bootstrap_local_user
from langley.conversation_commands import (
    ClientRequestIdReusedError,
    admit_new_question,
    admit_regenerate,
    admit_retry,
)
from langley.conversations import create_conversation, get_conversation_messages
from langley.infrastructure.database import (
    create_database_engine,
    create_session_factory,
    dispose_database_engine,
)
from langley.infrastructure.local_file_storage import LocalFileStorage
from langley.infrastructure.models import (
    Document,
    DocumentVersion,
    KnowledgeChunk,
    Message,
    MessageCitation,
    Run,
    User,
)
from langley.knowledge.chunking import ChunkingConfig
from langley.knowledge.commands import (
    KnowledgeBaseNotFoundError,
    create_initial_document,
    create_knowledge_base,
    load_document_source_ref,
    read_verified_source,
    rebuild_document_version_chunks,
)
from langley.settings import Settings

_TIME = datetime(2026, 8, 24, 0, 0)


@pytest.fixture
def migrated_database(test_database_url: str, reset_database) -> str:
    reset_database()
    config = Config("alembic.ini")
    config.cmd_opts = Namespace(x=["use_test_database=true"])
    command.upgrade(config, "head")
    return test_database_url


def test_grounding_policy_empty_database_migration_round_trip(
    test_database_url: str, reset_database
) -> None:
    del test_database_url
    reset_database()
    config = Config("alembic.ini")
    config.cmd_opts = Namespace(x=["use_test_database=true"])

    command.upgrade(config, "head")
    command.downgrade(config, "0008_message_citations")
    command.upgrade(config, "head")


def _settings(database_url: str) -> Settings:
    return Settings(environment="test", database_url=database_url, local_user_id=1)


async def _factory(database_url: str):
    engine = create_database_engine(database_url)
    return engine, create_session_factory(engine)


async def _owned_conversation_and_kb(
    database_url: str,
) -> tuple[object, object, int, int]:
    engine, factory = await _factory(database_url)
    assert await bootstrap_local_user(_settings(database_url))
    async with factory() as session:
        conversation = await create_conversation(session, user_id=1, title=None)
        knowledge_base = await create_knowledge_base(session, user_id=1, name="Notes")
    return engine, factory, conversation.id, knowledge_base.id


def test_knowledge_selector_persists_and_retry_regenerate_inherit(
    migrated_database: str,
) -> None:
    async def verify() -> None:
        (
            engine,
            factory,
            conversation_id,
            knowledge_base_id,
        ) = await _owned_conversation_and_kb(migrated_database)
        try:
            async with factory() as session:
                admitted = await admit_new_question(
                    session,
                    user_id=1,
                    conversation_id=conversation_id,
                    content="question",
                    client_request_id="new",
                    knowledge_base_id=knowledge_base_id,
                    grounding_policy=GroundingPolicy.REQUIRED,
                )
            assert admitted.run.knowledge_base_id == knowledge_base_id
            assert admitted.run.grounding_policy == "REQUIRED"
            await _start_running(
                factory, conversation_id=conversation_id, run_id=admitted.run.id
            )
            async with factory() as session:
                async with session.begin():
                    run = await session.get(Run, admitted.run.id, with_for_update=True)
                    assert run is not None
                    run.status = "FAILED"
                    run.finished_at = _TIME
                    run.error_code = "TEST"
                    run.updated_at = _TIME
            async with factory() as session:
                retried = await admit_retry(
                    session,
                    user_id=1,
                    conversation_id=conversation_id,
                    client_request_id="retry",
                )
            assert retried.run.knowledge_base_id == knowledge_base_id
            assert retried.run.grounding_policy == "REQUIRED"
            await _start_running(
                factory, conversation_id=conversation_id, run_id=retried.run.id
            )
            await _commit_success(
                factory,
                conversation_id=conversation_id,
                run_id=retried.run.id,
                content="answer",
            )
            async with factory() as session:
                regenerated = await admit_regenerate(
                    session,
                    user_id=1,
                    conversation_id=conversation_id,
                    client_request_id="regenerate",
                )
            assert regenerated.run.knowledge_base_id == knowledge_base_id
            assert regenerated.run.grounding_policy == "REQUIRED"
        finally:
            await dispose_database_engine(engine)

    asyncio.run(verify())


def test_required_policy_needs_kb_and_is_part_of_exact_replay_identity(
    migrated_database: str,
) -> None:
    async def verify() -> None:
        (
            engine,
            factory,
            conversation_id,
            knowledge_base_id,
        ) = await _owned_conversation_and_kb(migrated_database)
        try:
            async with factory() as session:
                with pytest.raises(ValueError, match="needs a knowledge base"):
                    await admit_new_question(
                        session,
                        user_id=1,
                        conversation_id=conversation_id,
                        content="question",
                        client_request_id="required-without-kb",
                        grounding_policy=GroundingPolicy.REQUIRED,
                    )
            async with factory() as session:
                first = await admit_new_question(
                    session,
                    user_id=1,
                    conversation_id=conversation_id,
                    content="question",
                    client_request_id="policy-identity",
                    knowledge_base_id=knowledge_base_id,
                    grounding_policy=GroundingPolicy.AUTO,
                )
            async with factory() as session:
                replay = await admit_new_question(
                    session,
                    user_id=1,
                    conversation_id=conversation_id,
                    content="question",
                    client_request_id="policy-identity",
                    knowledge_base_id=knowledge_base_id,
                    grounding_policy=GroundingPolicy.AUTO,
                )
            assert replay.is_replay is True
            assert replay.run.id == first.run.id
            async with factory() as session:
                with pytest.raises(ClientRequestIdReusedError):
                    await admit_new_question(
                        session,
                        user_id=1,
                        conversation_id=conversation_id,
                        content="question",
                        client_request_id="policy-identity",
                        knowledge_base_id=knowledge_base_id,
                        grounding_policy=GroundingPolicy.REQUIRED,
                    )
        finally:
            await dispose_database_engine(engine)

    asyncio.run(verify())


def test_grounding_policy_schema_uses_binary_collation_and_auto_default(
    migrated_database: str,
) -> None:
    async def verify() -> None:
        engine = create_database_engine(migrated_database)
        try:
            async with engine.connect() as connection:
                row = (
                    (
                        await connection.execute(
                            text(
                                "SELECT IS_NULLABLE, COLUMN_DEFAULT, COLLATION_NAME "
                                "FROM information_schema.COLUMNS "
                                "WHERE TABLE_SCHEMA = DATABASE() "
                                "AND TABLE_NAME = 'runs' "
                                "AND COLUMN_NAME = 'grounding_policy'"
                            )
                        )
                    )
                    .mappings()
                    .one()
                )
            assert row["IS_NULLABLE"] == "NO"
            assert row["COLUMN_DEFAULT"] == "AUTO"
            assert row["COLLATION_NAME"] == "utf8mb4_0900_bin"
        finally:
            await dispose_database_engine(engine)

    asyncio.run(verify())


def test_migration_downgrade_refuses_to_erase_required_policy(
    migrated_database: str,
) -> None:
    async def admit_required() -> None:
        (
            engine,
            factory,
            conversation_id,
            knowledge_base_id,
        ) = await _owned_conversation_and_kb(migrated_database)
        try:
            async with factory() as session:
                await admit_new_question(
                    session,
                    user_id=1,
                    conversation_id=conversation_id,
                    content="question",
                    client_request_id="required-before-downgrade",
                    knowledge_base_id=knowledge_base_id,
                    grounding_policy=GroundingPolicy.REQUIRED,
                )
        finally:
            await dispose_database_engine(engine)

    asyncio.run(admit_required())
    config = Config("alembic.ini")
    config.cmd_opts = Namespace(x=["use_test_database=true"])
    with pytest.raises(RuntimeError, match="REQUIRED Runs exist"):
        command.downgrade(config, "0008_message_citations")


def test_unowned_knowledge_base_cannot_be_admitted(migrated_database: str) -> None:
    async def verify() -> None:
        engine, factory, _, knowledge_base_id = await _owned_conversation_and_kb(
            migrated_database
        )
        try:
            async with factory() as session:
                async with session.begin():
                    session.add(User(id=2, created_at=_TIME))
            async with factory() as session:
                conversation = await create_conversation(session, user_id=2, title=None)
            async with factory() as session:
                with pytest.raises(KnowledgeBaseNotFoundError):
                    await admit_new_question(
                        session,
                        user_id=2,
                        conversation_id=conversation.id,
                        content="question",
                        client_request_id="unowned",
                        knowledge_base_id=knowledge_base_id,
                    )
        finally:
            await dispose_database_engine(engine)

    asyncio.run(verify())


def test_new_question_replay_includes_nullable_knowledge_selector(
    migrated_database: str,
) -> None:
    async def verify() -> None:
        (
            engine,
            factory,
            conversation_id,
            knowledge_base_id,
        ) = await _owned_conversation_and_kb(migrated_database)
        try:
            async with factory() as session:
                other = await create_knowledge_base(
                    session, user_id=1, name="Other notes"
                )
            async with factory() as session:
                first = await admit_new_question(
                    session,
                    user_id=1,
                    conversation_id=conversation_id,
                    content="question",
                    client_request_id="same-kb",
                    knowledge_base_id=knowledge_base_id,
                )
            async with factory() as session:
                replay = await admit_new_question(
                    session,
                    user_id=1,
                    conversation_id=conversation_id,
                    content="question",
                    client_request_id="same-kb",
                    knowledge_base_id=knowledge_base_id,
                )
            assert replay.is_replay is True
            assert replay.run.id == first.run.id

            for selector in (other.id, None):
                async with factory() as session:
                    with pytest.raises(ClientRequestIdReusedError):
                        await admit_new_question(
                            session,
                            user_id=1,
                            conversation_id=conversation_id,
                            content="question",
                            client_request_id="same-kb",
                            knowledge_base_id=selector,
                        )
        finally:
            await dispose_database_engine(engine)

    asyncio.run(verify())


def test_success_commits_assistant_citation_and_run_together(
    migrated_database: str,
) -> None:
    async def verify() -> None:
        (
            engine,
            factory,
            conversation_id,
            knowledge_base_id,
        ) = await _owned_conversation_and_kb(migrated_database)
        try:
            async with factory() as session:
                async with session.begin():
                    document = Document(
                        knowledge_base_id=knowledge_base_id,
                        name="notes.md",
                        created_at=_TIME,
                    )
                    session.add(document)
                    await session.flush()
                    version = DocumentVersion(
                        document_id=document.id,
                        source_filename="notes.md",
                        source_media_type="text/markdown",
                        source_sha256="a" * 64,
                        source_size_bytes=1,
                        storage_key="users/1/sources/test",
                        chunk_max_chars=1200,
                        created_at=_TIME,
                    )
                    session.add(version)
                    await session.flush()
            async with factory() as session:
                admitted = await admit_new_question(
                    session,
                    user_id=1,
                    conversation_id=conversation_id,
                    content="question",
                    client_request_id="success",
                    knowledge_base_id=knowledge_base_id,
                )
            await _start_running(
                factory, conversation_id=conversation_id, run_id=admitted.run.id
            )
            await _commit_success(
                factory,
                conversation_id=conversation_id,
                run_id=admitted.run.id,
                content="answer [K1]",
                citation_drafts=(
                    CitationDraft(
                        evidence_handle=1,
                        document_version_id=version.id,
                        evidence_text="authoritative evidence",
                        source_display_name_snapshot="notes.md",
                        heading_path_snapshot=("Heading",),
                        source_regions_snapshot=(
                            {"kind": "text", "start_byte": 0, "end_byte": 1},
                        ),
                    ),
                ),
            )
            async with factory() as session:
                run = await session.get(Run, admitted.run.id)
                citations = list((await session.scalars(select(MessageCitation))).all())
                result = await get_conversation_messages(session, 1, conversation_id)
            assert run is not None and run.status == "SUCCEEDED"
            citation_ids = [
                (citation.document_version_id, citation.evidence_handle)
                for citation in citations
            ]
            assert citation_ids == [(version.id, 1)]
            assert result is not None
            citation_read = result[3][citations[0].message_id][0]
            assert citation_read.source_display_name == "notes.md"

            async with factory() as session:
                invalid = await admit_new_question(
                    session,
                    user_id=1,
                    conversation_id=conversation_id,
                    content="next question",
                    client_request_id="rollback",
                    knowledge_base_id=knowledge_base_id,
                )
            await _start_running(
                factory, conversation_id=conversation_id, run_id=invalid.run.id
            )
            with pytest.raises(IntegrityError):
                await _commit_success(
                    factory,
                    conversation_id=conversation_id,
                    run_id=invalid.run.id,
                    content="invalid citation",
                    citation_drafts=(
                        CitationDraft(
                            evidence_handle=1,
                            document_version_id=999999,
                            evidence_text="evidence",
                            source_display_name_snapshot="notes.md",
                            heading_path_snapshot=(),
                            source_regions_snapshot=(
                                {"kind": "text", "start_byte": 0, "end_byte": 1},
                            ),
                        ),
                    ),
                )
            async with factory() as session:
                failed_run = await session.get(Run, invalid.run.id)
                assistants = list(
                    (
                        await session.scalars(
                            select(Message).where(Message.run_id == invalid.run.id)
                        )
                    ).all()
                )
            assert failed_run is not None and failed_run.status == "RUNNING"
            assert assistants == []
        finally:
            await dispose_database_engine(engine)

    asyncio.run(verify())


def test_rechunk_preserves_historical_citation_snapshot_and_provenance(
    migrated_database: str, tmp_path
) -> None:
    async def verify() -> None:
        storage = LocalFileStorage(tmp_path / "sources")
        (
            engine,
            factory,
            conversation_id,
            knowledge_base_id,
        ) = await _owned_conversation_and_kb(migrated_database)
        try:
            source_bytes = b"# Heading\nAuthoritative evidence block for citation.\n"
            version = await create_initial_document(
                factory,
                storage,
                user_id=1,
                knowledge_base_id=knowledge_base_id,
                name="Notes",
                source_filename="notes.md",
                source_media_type="text/markdown",
                source_bytes=source_bytes,
            )
            await rebuild_document_version_chunks(
                session_factory=factory,
                file_storage=storage,
                user_id=1,
                document_version_id=version.id,
                config=ChunkingConfig(max_chunk_chars=1200),
            )
            async with factory() as session:
                old_chunk_ids = list(
                    (
                        await session.scalars(
                            select(KnowledgeChunk.id).where(
                                KnowledgeChunk.document_version_id == version.id
                            )
                        )
                    ).all()
                )
                chunk = await session.get(KnowledgeChunk, old_chunk_ids[0])
                assert chunk is not None
            async with factory() as session:
                admitted = await admit_new_question(
                    session,
                    user_id=1,
                    conversation_id=conversation_id,
                    content="question",
                    client_request_id="citation",
                    knowledge_base_id=knowledge_base_id,
                )
            await _start_running(
                factory, conversation_id=conversation_id, run_id=admitted.run.id
            )
            draft = CitationDraft(
                evidence_handle=1,
                document_version_id=version.id,
                evidence_text=chunk.content,
                source_display_name_snapshot="notes.md",
                heading_path_snapshot=tuple(chunk.heading_path),
                source_regions_snapshot=tuple(chunk.source_regions),
            )
            await _commit_success(
                factory,
                conversation_id=conversation_id,
                run_id=admitted.run.id,
                content="answer [K1]",
                citation_drafts=(draft,),
            )

            await rebuild_document_version_chunks(
                session_factory=factory,
                file_storage=storage,
                user_id=1,
                document_version_id=version.id,
                config=ChunkingConfig(max_chunk_chars=16),
            )
            async with factory() as session:
                new_chunk_ids = list(
                    (
                        await session.scalars(
                            select(KnowledgeChunk.id).where(
                                KnowledgeChunk.document_version_id == version.id
                            )
                        )
                    ).all()
                )
                citations = list((await session.scalars(select(MessageCitation))).all())
                result = await get_conversation_messages(session, 1, conversation_id)
            assert set(old_chunk_ids).isdisjoint(new_chunk_ids)
            assert len(citations) == 1
            citation = citations[0]
            assert (
                citation.evidence_handle,
                citation.document_version_id,
                citation.evidence_text,
                citation.source_display_name_snapshot,
                citation.heading_path_snapshot,
                citation.source_regions_snapshot,
            ) == (
                1,
                version.id,
                draft.evidence_text,
                draft.source_display_name_snapshot,
                list(draft.heading_path_snapshot),
                list(draft.source_regions_snapshot),
            )
            assert result is not None
            read = result[3][citation.message_id]
            assert [(item.evidence_handle, item.evidence_text) for item in read] == [
                (1, draft.evidence_text)
            ]
            source_ref = await load_document_source_ref(
                factory, user_id=1, document_version_id=version.id
            )
            verified_source = await read_verified_source(storage, source_ref)
            region = citation.source_regions_snapshot[0]
            assert (
                verified_source[region["start_byte"] : region["end_byte"]].decode(
                    "utf-8"
                )
                == citation.evidence_text
            )
        finally:
            await dispose_database_engine(engine)

    asyncio.run(verify())
