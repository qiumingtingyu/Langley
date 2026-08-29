"""Serial real-MySQL coverage for durable document processing attempts."""

import asyncio
from argparse import Namespace
from datetime import timedelta

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import func, select
from sqlalchemy.exc import DBAPIError, IntegrityError

from langley.business_time import utc_now
from langley.infrastructure.database import (
    create_database_engine,
    create_session_factory,
    dispose_database_engine,
)
from langley.infrastructure.models import (
    Document,
    DocumentProcessingJob,
    DocumentVersion,
    KnowledgeBase,
    KnowledgeChunk,
    User,
)
from langley.knowledge.document_processing import (
    PDF_PROCESSING_RECIPE_ID,
    DocumentProcessingAdmissionError,
    DocumentProcessingErrorCode,
    DocumentProcessingStage,
    admit_document_processing_attempt,
    advance_document_processing_attempt_stage,
    fail_document_processing_attempt,
    reconcile_interrupted_document_processing_jobs,
    start_document_processing_attempt,
    succeed_document_processing_attempt,
)
from langley.knowledge.reads import read_document_processing_status


@pytest.fixture
def migrated_database(test_database_url: str, reset_database) -> str:
    reset_database()
    config = Config("alembic.ini")
    config.cmd_opts = Namespace(x=["use_test_database=true"])
    command.upgrade(config, "head")
    return test_database_url


async def _seed_versions(
    session_factory, count: int, *, media_type: str = "application/pdf"
) -> tuple[int, ...]:
    async with session_factory() as session, session.begin():
        now = utc_now()
        session.add_all([User(id=1, created_at=now), User(id=2, created_at=now)])
        await session.flush()
        knowledge_base = KnowledgeBase(user_id=1, name="PDF fixture", created_at=now)
        session.add(knowledge_base)
        await session.flush()
        version_ids: list[int] = []
        for ordinal in range(1, count + 1):
            suffix = "pdf" if media_type == "application/pdf" else "md"
            document = Document(
                knowledge_base_id=knowledge_base.id,
                name=f"fixture-{ordinal}",
                created_at=now,
            )
            session.add(document)
            await session.flush()
            version = DocumentVersion(
                document_id=document.id,
                source_filename=f"fixture-{ordinal}.{suffix}",
                source_media_type=media_type,
                source_sha256=f"{ordinal:x}" * 64,
                source_size_bytes=128,
                storage_key=f"fixture/{ordinal}.pdf",
                chunk_max_chars=None,
                created_at=now,
            )
            session.add(version)
            await session.flush()
            version_ids.append(version.id)
        return tuple(version_ids)


def test_retry_history_and_published_chunk_read_semantics(
    migrated_database: str,
) -> None:
    async def run() -> None:
        engine = create_database_engine(migrated_database)
        session_factory = create_session_factory(engine)
        try:
            (version_id,) = await _seed_versions(session_factory, 1)
            first = await admit_document_processing_attempt(
                session_factory, user_id=1, document_version_id=version_id
            )
            with pytest.raises(DocumentProcessingAdmissionError) as active_error:
                await admit_document_processing_attempt(
                    session_factory, user_id=1, document_version_id=version_id
                )
            assert active_error.value.code == "DOCUMENT_PROCESSING_ACTIVE"

            await start_document_processing_attempt(
                session_factory, job_id=first.job_id
            )
            with pytest.raises(DocumentProcessingAdmissionError) as running_error:
                await admit_document_processing_attempt(
                    session_factory, user_id=1, document_version_id=version_id
                )
            assert running_error.value.code == "DOCUMENT_PROCESSING_ACTIVE"
            await advance_document_processing_attempt_stage(
                session_factory,
                job_id=first.job_id,
                stage=DocumentProcessingStage.PUBLISHING,
            )
            await succeed_document_processing_attempt(
                session_factory, job_id=first.job_id
            )
            with pytest.raises(IntegrityError):
                async with session_factory() as session, session.begin():
                    session.add(
                        DocumentProcessingJob(
                            document_version_id=version_id,
                            attempt_no=1,
                            status="PENDING",
                            stage=None,
                            recipe_id=PDF_PROCESSING_RECIPE_ID,
                            error_code=None,
                            error_message=None,
                            created_at=utc_now(),
                            started_at=None,
                            finished_at=None,
                        )
                    )
            async with session_factory() as session, session.begin():
                session.add(
                    KnowledgeChunk(
                        document_version_id=version_id,
                        ordinal=1,
                        content="Published content remains authoritative.",
                        heading_path=[],
                        source_regions=[
                            {"kind": "pdf_page", "page_start": 1, "page_end": 1}
                        ],
                        created_at=utc_now(),
                    )
                )

            second = await admit_document_processing_attempt(
                session_factory, user_id=1, document_version_id=version_id
            )
            assert (first.attempt_no, second.attempt_no) == (1, 2)
            await start_document_processing_attempt(
                session_factory, job_id=second.job_id
            )
            await advance_document_processing_attempt_stage(
                session_factory,
                job_id=second.job_id,
                stage=DocumentProcessingStage.PARSING,
            )
            await fail_document_processing_attempt(
                session_factory,
                job_id=second.job_id,
                error_code=DocumentProcessingErrorCode.PDF_PARSE_FAILED,
                safe_error_message="PDF 结构无法解析。",
            )

            async with session_factory() as session:
                read = await read_document_processing_status(
                    session, user_id=1, document_version_id=version_id
                )
                assert read is not None and read.latest_attempt is not None
                assert read.latest_attempt.id == second.job_id
                assert read.latest_attempt.status == "FAILED"
                assert read.latest_attempt.stage == "PARSING"
                assert read.published_chunks_exist is True
                assert (
                    await read_document_processing_status(
                        session, user_id=2, document_version_id=version_id
                    )
                    is None
                )
                jobs = (
                    await session.scalars(
                        select(DocumentProcessingJob)
                        .where(DocumentProcessingJob.document_version_id == version_id)
                        .order_by(DocumentProcessingJob.attempt_no)
                    )
                ).all()
                assert [job.status for job in jobs] == ["SUCCEEDED", "FAILED"]
            third = await admit_document_processing_attempt(
                session_factory, user_id=1, document_version_id=version_id
            )
            assert third.attempt_no == 3
        finally:
            await dispose_database_engine(engine)

    asyncio.run(run())


def test_non_pdf_source_is_rejected_without_creating_attempt(
    migrated_database: str,
) -> None:
    async def run() -> None:
        engine = create_database_engine(migrated_database)
        session_factory = create_session_factory(engine)
        try:
            (version_id,) = await _seed_versions(
                session_factory, 1, media_type="text/markdown"
            )
            with pytest.raises(DocumentProcessingAdmissionError) as error:
                await admit_document_processing_attempt(
                    session_factory, user_id=1, document_version_id=version_id
                )
            assert error.value.code == "DOCUMENT_SOURCE_TYPE_UNSUPPORTED"
            async with session_factory() as session:
                job_count = await session.scalar(
                    select(func.count()).select_from(DocumentProcessingJob)
                )
                assert job_count == 0
        finally:
            await dispose_database_engine(engine)

    asyncio.run(run())


def test_database_rejects_processing_timestamps_before_creation(
    migrated_database: str,
) -> None:
    async def run() -> None:
        engine = create_database_engine(migrated_database)
        session_factory = create_session_factory(engine)
        try:
            (version_id,) = await _seed_versions(session_factory, 1)
            created_at = utc_now()
            invalid_jobs = (
                DocumentProcessingJob(
                    document_version_id=version_id,
                    attempt_no=1,
                    status="RUNNING",
                    stage="VERIFYING_SOURCE",
                    recipe_id=PDF_PROCESSING_RECIPE_ID,
                    error_code=None,
                    error_message=None,
                    created_at=created_at,
                    started_at=created_at - timedelta(microseconds=1),
                    finished_at=None,
                ),
                DocumentProcessingJob(
                    document_version_id=version_id,
                    attempt_no=1,
                    status="INTERRUPTED",
                    stage=None,
                    recipe_id=PDF_PROCESSING_RECIPE_ID,
                    error_code="PROCESS_INTERRUPTED",
                    error_message=None,
                    created_at=created_at,
                    started_at=None,
                    finished_at=created_at - timedelta(microseconds=1),
                ),
            )
            for job in invalid_jobs:
                with pytest.raises(DBAPIError):
                    async with session_factory() as session, session.begin():
                        session.add(job)
        finally:
            await dispose_database_engine(engine)

    asyncio.run(run())


def test_concurrent_admission_allows_exactly_one_active_attempt(
    migrated_database: str,
) -> None:
    async def run() -> None:
        engine = create_database_engine(migrated_database)
        session_factory = create_session_factory(engine)
        try:
            (version_id,) = await _seed_versions(session_factory, 1)

            async def attempt() -> str:
                try:
                    await admit_document_processing_attempt(
                        session_factory, user_id=1, document_version_id=version_id
                    )
                except DocumentProcessingAdmissionError as error:
                    return error.code
                return "ADMITTED"

            outcomes = await asyncio.gather(attempt(), attempt())
            assert sorted(outcomes) == ["ADMITTED", "DOCUMENT_PROCESSING_ACTIVE"]
            async with session_factory() as session:
                jobs = (await session.scalars(select(DocumentProcessingJob))).all()
                assert len(jobs) == 1
                assert (jobs[0].attempt_no, jobs[0].status) == (1, "PENDING")
        finally:
            await dispose_database_engine(engine)

    asyncio.run(run())


def test_restart_repair_interrupts_only_active_attempts_and_retains_stage(
    migrated_database: str,
) -> None:
    async def run() -> None:
        engine = create_database_engine(migrated_database)
        session_factory = create_session_factory(engine)
        try:
            pending_version, running_version, terminal_version = await _seed_versions(
                session_factory, 3
            )
            pending = await admit_document_processing_attempt(
                session_factory,
                user_id=1,
                document_version_id=pending_version,
            )
            running = await admit_document_processing_attempt(
                session_factory,
                user_id=1,
                document_version_id=running_version,
            )
            await start_document_processing_attempt(
                session_factory, job_id=running.job_id
            )
            await advance_document_processing_attempt_stage(
                session_factory,
                job_id=running.job_id,
                stage=DocumentProcessingStage.PARSING,
            )
            terminal = await admit_document_processing_attempt(
                session_factory,
                user_id=1,
                document_version_id=terminal_version,
            )
            await start_document_processing_attempt(
                session_factory, job_id=terminal.job_id
            )
            await advance_document_processing_attempt_stage(
                session_factory,
                job_id=terminal.job_id,
                stage=DocumentProcessingStage.PUBLISHING,
            )
            await succeed_document_processing_attempt(
                session_factory, job_id=terminal.job_id
            )

            repaired = await reconcile_interrupted_document_processing_jobs(
                session_factory
            )
            assert repaired == (pending.job_id, running.job_id)
            retry = await admit_document_processing_attempt(
                session_factory,
                user_id=1,
                document_version_id=pending_version,
            )
            assert retry.attempt_no == 2
            async with session_factory() as session:
                rows = {
                    job.id: job
                    for job in (
                        await session.scalars(
                            select(DocumentProcessingJob).order_by(
                                DocumentProcessingJob.id
                            )
                        )
                    ).all()
                }
                assert (
                    rows[pending.job_id].status,
                    rows[pending.job_id].stage,
                    rows[pending.job_id].started_at,
                ) == ("INTERRUPTED", None, None)
                assert (
                    rows[running.job_id].status,
                    rows[running.job_id].stage,
                ) == ("INTERRUPTED", "PARSING")
                assert rows[running.job_id].started_at is not None
                assert rows[running.job_id].finished_at is not None
                assert rows[running.job_id].error_code == "PROCESS_INTERRUPTED"
                assert rows[terminal.job_id].status == "SUCCEEDED"
        finally:
            await dispose_database_engine(engine)

    asyncio.run(run())
