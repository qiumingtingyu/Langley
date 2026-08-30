"""Serial real-MySQL coverage for durable document processing attempts."""

import asyncio
import sys
from argparse import Namespace
from datetime import timedelta
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import func, select
from sqlalchemy.exc import DBAPIError, IntegrityError
from structlog.testing import capture_logs

from langley.business_time import utc_now
from langley.infrastructure.database import (
    create_database_engine,
    create_session_factory,
    dispose_database_engine,
)
from langley.infrastructure.local_file_storage import LocalFileStorage
from langley.infrastructure.models import (
    Document,
    DocumentProcessingJob,
    DocumentVersion,
    KnowledgeBase,
    KnowledgeChunk,
    KnowledgeIndexJob,
    User,
)
from langley.knowledge.chunking import CandidateChunk
from langley.knowledge.commands import create_initial_pdf_document
from langley.knowledge.contracts import PdfPageRegion
from langley.knowledge.document_processing import (
    PDF_PROCESSING_RECIPE_ID,
    DocumentProcessingAdmissionError,
    DocumentProcessingErrorCode,
    DocumentProcessingStage,
    admit_document_processing_attempt,
    advance_document_processing_attempt_stage,
    claim_next_document_processing_attempt,
    fail_document_processing_attempt,
    reconcile_interrupted_document_processing_jobs,
    start_document_processing_attempt,
    succeed_document_processing_attempt,
)
from langley.knowledge.index_build import IndexBuildAdmissionError, admit_index_build
from langley.knowledge.pdf_processing import (
    DocumentProcessingRuntime,
    PersistentPdfWorker,
    publish_pdf_processing_result,
)
from langley.knowledge.pdf_processing_result import PdfProcessingResult
from langley.knowledge.reads import read_document_processing_status
from langley.settings import Settings


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


def test_processing_and_index_admission_reject_each_other(
    migrated_database: str,
) -> None:
    async def run() -> None:
        engine = create_database_engine(migrated_database)
        session_factory = create_session_factory(engine)
        try:
            (version_id,) = await _seed_versions(session_factory, 1)
            async with session_factory() as session, session.begin():
                knowledge_base = await session.scalar(
                    select(KnowledgeBase)
                    .join(Document, Document.knowledge_base_id == KnowledgeBase.id)
                    .join(DocumentVersion, DocumentVersion.document_id == Document.id)
                    .where(DocumentVersion.id == version_id)
                    .with_for_update()
                )
                assert knowledge_base is not None
                knowledge_base.index_status = "INDEXING"
                knowledge_base_id = knowledge_base.id
            with pytest.raises(DocumentProcessingAdmissionError) as indexing:
                await admit_document_processing_attempt(
                    session_factory, user_id=1, document_version_id=version_id
                )
            assert indexing.value.code == "KNOWLEDGE_BASE_INDEXING"

            async with session_factory() as session, session.begin():
                knowledge_base = await session.get(
                    KnowledgeBase, knowledge_base_id, with_for_update=True
                )
                assert knowledge_base is not None
                knowledge_base.index_status = "CHUNKED"
            await admit_document_processing_attempt(
                session_factory, user_id=1, document_version_id=version_id
            )
            with pytest.raises(IndexBuildAdmissionError) as processing:
                await admit_index_build(
                    session_factory,
                    user_id=1,
                    knowledge_base_id=knowledge_base_id,
                    settings=Settings(),
                )
            assert processing.value.code == "KNOWLEDGE_BASE_DOCUMENTS_PROCESSING"
        finally:
            await dispose_database_engine(engine)

    asyncio.run(run())


def test_restart_repair_preserves_pending_and_interrupts_running(
    migrated_database: str, tmp_path: Path
) -> None:
    async def run() -> None:
        engine = create_database_engine(migrated_database)
        session_factory = create_session_factory(engine)
        storage = LocalFileStorage(tmp_path / "restart-storage")
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
            stale_result = await storage.prepare_processing_result_path(
                running.job_id, running.attempt_no
            )
            stale_result.write_text("stale", encoding="ascii")
            stale_result.with_suffix(".json.tmp").write_text(
                "partial", encoding="ascii"
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
            await storage.confirm_no_processing_worker()
            await storage.cleanup_stale_processing_artifacts()
            assert repaired == (running.job_id,)
            assert not stale_result.exists()
            assert not stale_result.with_suffix(".json.tmp").exists()
            assert not stale_result.parent.exists()
            with pytest.raises(DocumentProcessingAdmissionError) as pending_active:
                await admit_document_processing_attempt(
                    session_factory,
                    user_id=1,
                    document_version_id=pending_version,
                )
            assert pending_active.value.code == "DOCUMENT_PROCESSING_ACTIVE"
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
                ) == ("PENDING", None, None)
                assert rows[pending.job_id].finished_at is None
                assert rows[pending.job_id].error_code is None
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


def test_pdf_admission_atomically_persists_version_and_pending_job(
    migrated_database: str, tmp_path: Path
) -> None:
    async def run() -> None:
        engine = create_database_engine(migrated_database)
        session_factory = create_session_factory(engine)
        storage = LocalFileStorage(tmp_path / "sources")
        try:
            async with session_factory() as session, session.begin():
                now = utc_now()
                session.add(User(id=1, created_at=now))
                await session.flush()
                knowledge_base = KnowledgeBase(
                    user_id=1, name="PDF admission", created_at=now
                )
                session.add(knowledge_base)
                await session.flush()
                knowledge_base_id = knowledge_base.id
            admission = await create_initial_pdf_document(
                session_factory,
                storage,
                user_id=1,
                knowledge_base_id=knowledge_base_id,
                name="fixture",
                source_filename="fixture.pdf",
                source_media_type="application/pdf",
                source_bytes=b"%PDF-1.4 controlled fixture",
            )
            async with session_factory() as session:
                version = await session.get(DocumentVersion, admission.version.id)
                job = await session.get(
                    DocumentProcessingJob, admission.processing.job_id
                )
                assert version is not None and job is not None
                assert version.source_media_type == "application/pdf"
                assert (
                    job.document_version_id,
                    job.attempt_no,
                    job.status,
                    job.stage,
                    job.recipe_id,
                ) == (
                    version.id,
                    1,
                    "PENDING",
                    None,
                    PDF_PROCESSING_RECIPE_ID,
                )
        finally:
            await dispose_database_engine(engine)

    asyncio.run(run())


def test_oldest_pending_claim_is_short_and_enters_verifying_source(
    migrated_database: str,
) -> None:
    async def run() -> None:
        engine = create_database_engine(migrated_database)
        session_factory = create_session_factory(engine)
        try:
            first_version, second_version = await _seed_versions(session_factory, 2)
            first = await admit_document_processing_attempt(
                session_factory, user_id=1, document_version_id=first_version
            )
            await admit_document_processing_attempt(
                session_factory, user_id=1, document_version_id=second_version
            )
            claim = await claim_next_document_processing_attempt(session_factory)
            assert claim is not None and claim.job_id == first.job_id
            async with session_factory() as session:
                job = await session.get(DocumentProcessingJob, claim.job_id)
                assert job is not None
                assert (job.status, job.stage) == ("RUNNING", "VERIFYING_SOURCE")
                assert job.started_at is not None
        finally:
            await dispose_database_engine(engine)

    asyncio.run(run())


def test_worker_traceback_is_logged_while_database_error_remains_safe(
    migrated_database: str, tmp_path: Path
) -> None:
    async def run() -> None:
        engine = create_database_engine(migrated_database)
        session_factory = create_session_factory(engine)
        storage = LocalFileStorage(tmp_path / "diagnostic-storage")
        try:
            async with session_factory() as session, session.begin():
                now = utc_now()
                session.add(User(id=1, created_at=now))
                await session.flush()
                knowledge_base = KnowledgeBase(
                    user_id=1, name="diagnostic", created_at=now
                )
                session.add(knowledge_base)
                await session.flush()
                knowledge_base_id = knowledge_base.id
            admission = await create_initial_pdf_document(
                session_factory,
                storage,
                user_id=1,
                knowledge_base_id=knowledge_base_id,
                name="controlled",
                source_filename="controlled.pdf",
                source_media_type="application/pdf",
                source_bytes=b"%PDF-1.4 controlled technical failure",
            )
            claim = await claim_next_document_processing_attempt(session_factory)
            assert claim is not None and claim.job_id == admission.processing.job_id
            behavior_path = tmp_path / "behavior"
            pid_log_path = tmp_path / "pids"
            done_path = tmp_path / "done"
            behavior_path.write_text("failure", encoding="ascii")
            worker = PersistentPdfWorker(
                tokenizer_id="unused",
                tokenizer_revision="unused",
                worker_marker_path=storage.processing_worker_marker_path(),
                command=(
                    sys.executable,
                    str(Path("tests/fixtures/pdf_persistent_fake_worker.py").resolve()),
                    "--behavior-path",
                    str(behavior_path),
                    "--pid-log-path",
                    str(pid_log_path),
                    "--done-path",
                    str(done_path),
                ),
            )
            runtime = DocumentProcessingRuntime(
                session_factory,
                storage,
                timeout_seconds=2,
                worker=worker,
            )
            with capture_logs() as logs:
                await runtime.execute(claim)
            await runtime.stop()
            async with session_factory() as session:
                job = await session.get(DocumentProcessingJob, claim.job_id)
                assert job is not None
                assert (job.status, job.stage, job.error_code) == (
                    "FAILED",
                    "PARSING",
                    "PDF_PARSE_FAILED",
                )
                assert job.error_message == "PDF 结构无法解析。"
                assert "controlled failure" not in job.error_message
                assert "Traceback" not in job.error_message
            diagnostics = "\n".join(str(entry.get("diagnostic", "")) for entry in logs)
            assert "RuntimeError: controlled failure" in diagnostics
            assert any(entry.get("job_id") == claim.job_id for entry in logs)
        finally:
            await dispose_database_engine(engine)

    asyncio.run(run())


def test_atomic_pdf_publication_replaces_all_or_preserves_old_chunks(
    migrated_database: str,
) -> None:
    async def run() -> None:
        engine = create_database_engine(migrated_database)
        session_factory = create_session_factory(engine)
        try:
            (version_id,) = await _seed_versions(session_factory, 1)
            async with session_factory() as session, session.begin():
                session.add(
                    KnowledgeChunk(
                        document_version_id=version_id,
                        ordinal=1,
                        content="old usable chunk",
                        heading_path=[],
                        source_regions=[
                            {"kind": "pdf_page", "page_start": 1, "page_end": 1}
                        ],
                        created_at=utc_now(),
                    )
                )
            first = await admit_document_processing_attempt(
                session_factory, user_id=1, document_version_id=version_id
            )
            claim = await claim_next_document_processing_attempt(session_factory)
            assert claim is not None and claim.job_id == first.job_id
            await advance_document_processing_attempt_stage(
                session_factory,
                job_id=claim.job_id,
                stage=DocumentProcessingStage.PUBLISHING,
            )
            result = PdfProcessingResult(
                page_count=2,
                candidates=(
                    CandidateChunk(1, "new one", ("A",), (PdfPageRegion(1, 1),)),
                    CandidateChunk(2, "new two", ("B",), (PdfPageRegion(2, 2),)),
                ),
            )
            await publish_pdf_processing_result(
                session_factory, claim=claim, result=result
            )
            async with session_factory() as session:
                chunks = (
                    await session.scalars(
                        select(KnowledgeChunk)
                        .where(KnowledgeChunk.document_version_id == version_id)
                        .order_by(KnowledgeChunk.ordinal)
                    )
                ).all()
                job = await session.get(DocumentProcessingJob, claim.job_id)
                assert [chunk.content for chunk in chunks] == ["new one", "new two"]
                assert job is not None and job.status == "SUCCEEDED"
                assert (
                    await session.scalar(
                        select(func.count()).select_from(KnowledgeIndexJob)
                    )
                ) == 0

            second = await admit_document_processing_attempt(
                session_factory, user_id=1, document_version_id=version_id
            )
            failed_claim = await claim_next_document_processing_attempt(session_factory)
            assert failed_claim is not None and failed_claim.job_id == second.job_id
            await advance_document_processing_attempt_stage(
                session_factory,
                job_id=failed_claim.job_id,
                stage=DocumentProcessingStage.PUBLISHING,
            )
            duplicate_ordinals = PdfProcessingResult(
                page_count=1,
                candidates=(
                    CandidateChunk(1, "bad one", (), (PdfPageRegion(1, 1),)),
                    CandidateChunk(1, "bad two", (), (PdfPageRegion(1, 1),)),
                ),
            )
            published_facts = [
                (
                    chunk.ordinal,
                    chunk.content,
                    chunk.heading_path,
                    chunk.source_regions,
                )
                for chunk in chunks
            ]
            with pytest.raises(ValueError, match="invalid chunk ordinal"):
                await publish_pdf_processing_result(
                    session_factory,
                    claim=failed_claim,
                    result=duplicate_ordinals,
                )
            await fail_document_processing_attempt(
                session_factory,
                job_id=failed_claim.job_id,
                error_code=DocumentProcessingErrorCode.PUBLICATION_FAILED,
            )
            async with session_factory() as session:
                chunks = (
                    await session.scalars(
                        select(KnowledgeChunk)
                        .where(KnowledgeChunk.document_version_id == version_id)
                        .order_by(KnowledgeChunk.ordinal)
                    )
                ).all()
                job = await session.get(DocumentProcessingJob, failed_claim.job_id)
                assert [
                    (
                        chunk.ordinal,
                        chunk.content,
                        chunk.heading_path,
                        chunk.source_regions,
                    )
                    for chunk in chunks
                ] == published_facts
                assert job is not None and (
                    job.status,
                    job.error_code,
                ) == ("FAILED", "PUBLICATION_FAILED")
        finally:
            await dispose_database_engine(engine)

    asyncio.run(run())
