"""Real MySQL scope, published-read snapshot and durable read citations."""

import asyncio
from argparse import Namespace

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from langley.answer_execution import _commit_success, _start_running
from langley.answering.knowledge_evidence import KnowledgeEvidenceSession
from langley.answering.knowledge_qa import validated_answer_completion
from langley.business_time import utc_now
from langley.conversation_commands import admit_new_question
from langley.conversations import create_conversation, get_conversation_messages
from langley.infrastructure.database import (
    create_database_engine,
    create_session_factory,
)
from langley.infrastructure.local_file_storage import LocalFileStorage
from langley.infrastructure.models import (
    Document,
    DocumentProcessingJob,
    DocumentVersion,
    KnowledgeBase,
    KnowledgeChunk,
    User,
)
from langley.knowledge.chunking import ChunkingConfig
from langley.knowledge.commands import (
    create_initial_document,
    rebuild_document_version_chunks,
)
from langley.knowledge.contracts import (
    KnowledgeLocator,
    KnowledgePerceptionError,
    PdfPageRegion,
)
from langley.knowledge.perception import inspect_knowledge, read_knowledge


@pytest.fixture
def migrated_database(test_database_url, reset_database):
    reset_database()
    config = Config("alembic.ini")
    config.cmd_opts = Namespace(x=["use_test_database=true"])
    command.upgrade(config, "head")
    return test_database_url


async def seed(factory, storage):
    async with factory() as session, session.begin():
        session.add_all(
            [User(id=1, created_at=utc_now()), User(id=2, created_at=utc_now())]
        )
        await session.flush()
        session.add_all(
            [
                KnowledgeBase(id=7, user_id=1, name="Selected", created_at=utc_now()),
                KnowledgeBase(id=8, user_id=1, name="Other", created_at=utc_now()),
                KnowledgeBase(id=9, user_id=2, name="Foreign", created_at=utc_now()),
            ]
        )
    versions = []
    for user, kb in ((1, 7), (1, 8), (2, 9)):
        versions.append(
            await create_initial_document(
                factory,
                storage,
                user_id=user,
                knowledge_base_id=kb,
                name=f"notes-{kb}",
                source_filename="notes.md",
                source_media_type="text/markdown",
                source_bytes=b"# Root\n## Exact\nsource-native body\n",
            )
        )
    async with factory() as session, session.begin():
        pdf = Document(knowledge_base_id=7, name="book.pdf", created_at=utc_now())
        session.add(pdf)
        await session.flush()
        version = DocumentVersion(
            document_id=pdf.id,
            source_filename="book.pdf",
            source_media_type="application/pdf",
            source_sha256="a" * 64,
            source_size_bytes=3,
            storage_key="synthetic/pdf",
            chunk_revision=1,
            chunk_set_sha256="b" * 64,
            created_at=utc_now(),
        )
        session.add(version)
        await session.flush()
        session.add_all(pdf_chunks(version.id))
        # Latest failed attempt must not hide authoritative published content.
        session.add(
            DocumentProcessingJob(
                document_version_id=version.id,
                attempt_no=2,
                status="FAILED",
                stage="PARSING",
                recipe_id="synthetic",
                error_code="PDF_PARSE_FAILED",
                error_message="fixture",
                created_at=utc_now(),
                started_at=utc_now(),
                finished_at=utc_now(),
            )
        )
    return versions, pdf.id, version.id


def pdf_chunks(version_id, suffix=""):
    return [
        KnowledgeChunk(
            document_version_id=version_id,
            ordinal=index,
            content=content + suffix,
            heading_path=["Transactions", "MVCC"],
            source_regions=[{"kind": "pdf_page", "page_start": start, "page_end": end}],
            created_at=utc_now(),
        )
        for index, (start, end, content) in enumerate(
            ((45, 48, "first"), (49, 52, "second"), (52, 54, "third")), 1
        )
    ]


async def read(factory, storage, document_id, **fields):
    return await read_knowledge(
        factory,
        storage,
        user_id=1,
        knowledge_base_id=7,
        locator=KnowledgeLocator(document_id=document_id, **fields),
        max_content_bytes=16384,
    )


def test_owned_catalog_pass_through_and_temporary_version_policy(
    migrated_database, tmp_path
):
    async def scenario():
        engine = create_database_engine(migrated_database)
        factory = create_session_factory(engine)
        storage = LocalFileStorage(tmp_path / "sources")
        try:
            versions, pdf_id, _ = await seed(factory, storage)
            catalog = await inspect_knowledge(
                factory, storage, user_id=1, knowledge_base_id=7, document_id=None
            )
            assert {d["locator"]["document_id"] for d in catalog["documents"]} == {
                versions[0].document_id,
                pdf_id,
            }
            for entry in catalog["documents"]:
                assert "document_id" not in entry
                result = await read_knowledge(
                    factory,
                    storage,
                    user_id=1,
                    knowledge_base_id=7,
                    locator=KnowledgeLocator.model_validate(entry["locator"]),
                    max_content_bytes=16384,
                )
                assert result.content
            for version in versions[1:]:
                for operation in ("inspect", "read"):
                    with pytest.raises(KnowledgePerceptionError) as error:
                        if operation == "inspect":
                            await inspect_knowledge(
                                factory,
                                storage,
                                user_id=1,
                                knowledge_base_id=7,
                                document_id=version.document_id,
                            )
                        else:
                            await read(
                                factory, storage, version.document_id, kind="document"
                            )
                    assert error.value.code == "DOCUMENT_NOT_AVAILABLE"
            with pytest.raises(KnowledgePerceptionError):
                await inspect_knowledge(
                    factory, storage, user_id=2, knowledge_base_id=7, document_id=None
                )
            async with factory() as session, session.begin():
                session.add(
                    DocumentVersion(
                        document_id=versions[0].document_id,
                        source_filename="second.md",
                        source_media_type="text/markdown",
                        source_sha256="c" * 64,
                        source_size_bytes=1,
                        storage_key="synthetic/ambiguous",
                        created_at=utc_now(),
                    )
                )
            with pytest.raises(KnowledgePerceptionError) as error:
                await read(factory, storage, versions[0].document_id, kind="document")
            assert error.value.code == "KNOWLEDGE_READ_UNAVAILABLE"
            catalog = await inspect_knowledge(
                factory, storage, user_id=1, knowledge_base_id=7, document_id=None
            )
            ambiguous = next(d for d in catalog["documents"] if d["label"] == "notes-7")
            assert not ambiguous["available"] and "locator" not in ambiguous
            assert "document_id" not in ambiguous
        finally:
            await engine.dispose()

    asyncio.run(scenario())


def test_pdf_pages_published_snapshot_and_no_heading_authority(
    migrated_database, tmp_path
):
    async def scenario():
        engine = create_database_engine(migrated_database)
        factory = create_session_factory(engine)
        storage = LocalFileStorage(tmp_path / "sources")
        try:
            _, pdf_id, version_id = await seed(factory, storage)
            outline = await inspect_knowledge(
                factory, storage, user_id=1, knowledge_base_id=7, document_id=pdf_id
            )
            assert len(outline["detected_regions"]) == 3 and "page_count" not in outline
            for entry in outline["detected_regions"]:
                result = await read_knowledge(
                    factory,
                    storage,
                    user_id=1,
                    knowledge_base_id=7,
                    locator=KnowledgeLocator.model_validate(entry["locator"]),
                    max_content_bytes=16384,
                )
                assert result.heading_path == ()
            result = await read(
                factory, storage, pdf_id, kind="pages", page_start=50, page_end=50
            )
            assert result.content == "second" and result.source_regions == (
                PdfPageRegion(49, 52),
            )
            assert result.heading_path == ()
            with pytest.raises(KnowledgePerceptionError) as error:
                await read(
                    factory, storage, pdf_id, kind="pages", page_start=100, page_end=100
                )
            assert error.value.code == "LOCATION_NOT_FOUND"
            with pytest.raises(KnowledgePerceptionError) as error:
                await read(
                    factory,
                    storage,
                    pdf_id,
                    kind="heading",
                    heading_path=["Transactions"],
                )
            assert error.value.code == "INVALID_LOCATOR"
            with pytest.raises(KnowledgePerceptionError) as error:
                await read_knowledge(
                    factory,
                    storage,
                    user_id=1,
                    knowledge_base_id=7,
                    locator=KnowledgeLocator(document_id=pdf_id, kind="document"),
                    max_content_bytes=5,
                )
            assert error.value.code == "KNOWLEDGE_READ_SCOPE_TOO_LARGE"

            class InterleavingSession(AsyncSession):
                async def execute(self, statement, *args, **kwargs):
                    rows = await super().execute(statement, *args, **kwargs)
                    if "octet_length" in str(statement):
                        async with factory() as writer, writer.begin():
                            await writer.execute(
                                delete(KnowledgeChunk).where(
                                    KnowledgeChunk.document_version_id == version_id
                                )
                            )
                            writer.add_all(pdf_chunks(version_id, " new"))
                            version = await writer.get(DocumentVersion, version_id)
                            version.chunk_revision = 2
                            version.chunk_set_sha256 = "d" * 64
                    return rows

            interleaving = async_sessionmaker(
                engine, class_=InterleavingSession, expire_on_commit=False
            )
            old = await read(interleaving, storage, pdf_id, kind="document")
            assert old.content == "first\n\nsecond\n\nthird"
            current = await read(factory, storage, pdf_id, kind="document")
            assert current.content == "first new\n\nsecond new\n\nthird new"
            assert old.heading_path == current.heading_path == ()
            async with factory() as session, session.begin():
                await session.execute(
                    delete(KnowledgeChunk).where(
                        KnowledgeChunk.document_version_id == version_id
                    )
                )
            with pytest.raises(KnowledgePerceptionError) as error:
                await read(factory, storage, pdf_id, kind="document")
            assert error.value.code == "KNOWLEDGE_READ_UNAVAILABLE"
        finally:
            await engine.dispose()

    asyncio.run(scenario())


def test_direct_read_citations_survive_rename_rechunk_and_pdf_republication(
    migrated_database, tmp_path
):
    async def scenario():
        engine = create_database_engine(migrated_database)
        factory = create_session_factory(engine)
        storage = LocalFileStorage(tmp_path / "sources")
        try:
            versions, pdf_id, pdf_version = await seed(factory, storage)
            outline = await inspect_knowledge(
                factory,
                storage,
                user_id=1,
                knowledge_base_id=7,
                document_id=versions[0].document_id,
            )
            markdown = await read_knowledge(
                factory,
                storage,
                user_id=1,
                knowledge_base_id=7,
                locator=KnowledgeLocator.model_validate(
                    outline["headings"][1]["locator"]
                ),
                max_content_bytes=16384,
            )
            pdf = await read(
                factory, storage, pdf_id, kind="pages", page_start=50, page_end=50
            )
            evidence = KnowledgeEvidenceSession()
            evidence.register_read(markdown)
            evidence.register_read(pdf)
            answer = validated_answer_completion(
                "answer [K1] [K2]",
                evidence,
                requires_citation=True,
                abstention_control_token=None,
            )
            async with factory() as session:
                conversation = await create_conversation(session, user_id=1, title=None)
                admission = await admit_new_question(
                    session,
                    user_id=1,
                    conversation_id=conversation.id,
                    content="read sources",
                    client_request_id="perception-citation",
                    knowledge_base_id=7,
                )
            await _start_running(
                factory, conversation_id=conversation.id, run_id=admission.run.id
            )
            await _commit_success(
                factory,
                conversation_id=conversation.id,
                run_id=admission.run.id,
                content=answer.content,
                citation_drafts=answer.citations,
            )
            await rebuild_document_version_chunks(
                session_factory=factory,
                file_storage=storage,
                user_id=1,
                document_version_id=versions[0].id,
                config=ChunkingConfig(max_chunk_chars=6),
            )
            async with factory() as session, session.begin():
                document = await session.get(Document, versions[0].document_id)
                document.name = "renamed"
                await session.execute(
                    delete(KnowledgeChunk).where(
                        KnowledgeChunk.document_version_id == pdf_version
                    )
                )
                session.add_all(pdf_chunks(pdf_version, " replaced"))
            async with factory() as session:
                messages = await get_conversation_messages(session, 1, conversation.id)
            citations = next(iter(messages[3].values()))
            assert [c.evidence_text for c in citations] == [markdown.content, "second"]
            assert citations[0].source_display_name == "notes-7"
            assert citations[0].heading_path == ["Root", "Exact"]
            assert citations[1].heading_path == []
            assert citations[1].source_regions == [
                {"kind": "pdf_page", "page_start": 49, "page_end": 52}
            ]
            assert [c.document_version_id for c in citations] == [
                versions[0].id,
                pdf_version,
            ]
        finally:
            await engine.dispose()

    asyncio.run(scenario())
