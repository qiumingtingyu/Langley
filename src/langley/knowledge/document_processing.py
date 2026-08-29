"""Durable lifecycle primitives for immutable-document processing attempts."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from langley.business_time import utc_now
from langley.infrastructure.models import (
    Document,
    DocumentProcessingJob,
    DocumentVersion,
    KnowledgeBase,
)

PDF_PROCESSING_RECIPE_ID = "pdf_docling_hybrid512_v1"
_INTERRUPTED_MESSAGE = "应用重启时中断了尚未完成的文档处理。"


class DocumentProcessingStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"


class DocumentProcessingStage(StrEnum):
    VERIFYING_SOURCE = "VERIFYING_SOURCE"
    PARSING = "PARSING"
    CHUNKING = "CHUNKING"
    VALIDATING = "VALIDATING"
    PUBLISHING = "PUBLISHING"


class DocumentProcessingErrorCode(StrEnum):
    SOURCE_MISSING = "SOURCE_MISSING"
    SOURCE_INTEGRITY_MISMATCH = "SOURCE_INTEGRITY_MISMATCH"
    PDF_PROCESS_TIMEOUT = "PDF_PROCESS_TIMEOUT"
    PDF_PROCESS_RESOURCE_LIMIT = "PDF_PROCESS_RESOURCE_LIMIT"
    PDF_PARSE_FAILED = "PDF_PARSE_FAILED"
    PDF_CHUNKING_FAILED = "PDF_CHUNKING_FAILED"
    PDF_OUTPUT_INVALID = "PDF_OUTPUT_INVALID"
    SOURCE_CHANGED_DURING_PROCESSING = "SOURCE_CHANGED_DURING_PROCESSING"
    PUBLICATION_FAILED = "PUBLICATION_FAILED"
    PROCESS_INTERRUPTED = "PROCESS_INTERRUPTED"


_STAGE_ORDER = {stage: index for index, stage in enumerate(DocumentProcessingStage)}


class DocumentProcessingAdmissionError(Exception):
    """A processing attempt cannot be admitted for the requested version."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class DocumentProcessingStateError(Exception):
    """A lifecycle transition is invalid for the current durable state."""


@dataclass(frozen=True)
class DocumentProcessingAdmission:
    job_id: int
    document_version_id: int
    attempt_no: int


def _require_status(
    job: DocumentProcessingJob, expected: DocumentProcessingStatus
) -> None:
    if job.status != expected:
        raise DocumentProcessingStateError(
            f"expected {expected.value}, found {job.status}"
        )


def _validated_safe_error_message(value: str | None) -> str | None:
    if value is None:
        return None
    if not value.strip() or len(value) > 255 or "\n" in value or "\r" in value:
        raise ValueError("invalid safe error message")
    return value


def mark_document_processing_running(
    job: DocumentProcessingJob, *, now: datetime
) -> None:
    """Start one pending attempt at its first durable stage."""

    _require_status(job, DocumentProcessingStatus.PENDING)
    job.status = DocumentProcessingStatus.RUNNING
    job.stage = DocumentProcessingStage.VERIFYING_SOURCE
    job.started_at = now


def advance_document_processing_stage(
    job: DocumentProcessingJob, stage: DocumentProcessingStage
) -> None:
    """Advance a running attempt monotonically, allowing deliberate stage skips."""

    _require_status(job, DocumentProcessingStatus.RUNNING)
    if job.stage is None:
        raise DocumentProcessingStateError("running attempt has no stage")
    current = DocumentProcessingStage(job.stage)
    if _STAGE_ORDER[stage] <= _STAGE_ORDER[current]:
        raise DocumentProcessingStateError("processing stage must move forward")
    job.stage = stage


def mark_document_processing_succeeded(
    job: DocumentProcessingJob, *, now: datetime
) -> None:
    """Finish only after the publication stage has completed."""

    _require_status(job, DocumentProcessingStatus.RUNNING)
    if job.stage != DocumentProcessingStage.PUBLISHING:
        raise DocumentProcessingStateError("success requires PUBLISHING stage")
    job.status = DocumentProcessingStatus.SUCCEEDED
    job.finished_at = now


def mark_document_processing_failed(
    job: DocumentProcessingJob,
    error_code: DocumentProcessingErrorCode,
    *,
    safe_error_message: str | None,
    now: datetime,
) -> None:
    """Fail a running attempt while retaining its last durable stage."""

    _require_status(job, DocumentProcessingStatus.RUNNING)
    if error_code is DocumentProcessingErrorCode.PROCESS_INTERRUPTED:
        raise ValueError("PROCESS_INTERRUPTED is reserved for restart repair")
    validated_message = _validated_safe_error_message(safe_error_message)
    job.status = DocumentProcessingStatus.FAILED
    job.error_code = error_code
    job.error_message = validated_message
    job.finished_at = now


def mark_document_processing_interrupted(
    job: DocumentProcessingJob, *, now: datetime
) -> None:
    """Repair one active attempt without inventing an automatic retry."""

    if job.status not in {
        DocumentProcessingStatus.PENDING,
        DocumentProcessingStatus.RUNNING,
    }:
        raise DocumentProcessingStateError("only active attempts can be interrupted")
    job.status = DocumentProcessingStatus.INTERRUPTED
    job.error_code = DocumentProcessingErrorCode.PROCESS_INTERRUPTED
    job.error_message = _INTERRUPTED_MESSAGE
    job.finished_at = now


async def admit_document_processing_attempt(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    user_id: int,
    document_version_id: int,
) -> DocumentProcessingAdmission:
    """Atomically admit one active attempt for an owned immutable version."""

    async with session_factory() as session, session.begin():
        version = await session.scalar(
            select(DocumentVersion)
            .join(Document, Document.id == DocumentVersion.document_id)
            .join(KnowledgeBase, KnowledgeBase.id == Document.knowledge_base_id)
            .where(
                DocumentVersion.id == document_version_id,
                KnowledgeBase.user_id == user_id,
            )
            .with_for_update()
        )
        if version is None:
            raise DocumentProcessingAdmissionError("DOCUMENT_VERSION_NOT_FOUND")
        if version.source_media_type != "application/pdf":
            raise DocumentProcessingAdmissionError("DOCUMENT_SOURCE_TYPE_UNSUPPORTED")
        latest = await session.scalar(
            select(DocumentProcessingJob)
            .where(DocumentProcessingJob.document_version_id == document_version_id)
            .order_by(
                DocumentProcessingJob.attempt_no.desc(),
                DocumentProcessingJob.id.desc(),
            )
            .limit(1)
            .with_for_update()
        )
        if latest is not None and latest.status in {
            DocumentProcessingStatus.PENDING,
            DocumentProcessingStatus.RUNNING,
        }:
            raise DocumentProcessingAdmissionError("DOCUMENT_PROCESSING_ACTIVE")
        attempt_no = 1 if latest is None else latest.attempt_no + 1
        job = DocumentProcessingJob(
            document_version_id=document_version_id,
            attempt_no=attempt_no,
            status=DocumentProcessingStatus.PENDING,
            stage=None,
            recipe_id=PDF_PROCESSING_RECIPE_ID,
            error_code=None,
            error_message=None,
            created_at=utc_now(),
            started_at=None,
            finished_at=None,
        )
        session.add(job)
        await session.flush()
        return DocumentProcessingAdmission(
            job_id=job.id,
            document_version_id=document_version_id,
            attempt_no=attempt_no,
        )


async def _locked_job(session: AsyncSession, job_id: int) -> DocumentProcessingJob:
    job = await session.scalar(
        select(DocumentProcessingJob)
        .where(DocumentProcessingJob.id == job_id)
        .with_for_update()
    )
    if job is None:
        raise DocumentProcessingStateError("document processing job not found")
    return job


async def start_document_processing_attempt(
    session_factory: async_sessionmaker[AsyncSession], *, job_id: int
) -> None:
    async with session_factory() as session, session.begin():
        mark_document_processing_running(
            await _locked_job(session, job_id), now=utc_now()
        )


async def advance_document_processing_attempt_stage(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    job_id: int,
    stage: DocumentProcessingStage,
) -> None:
    async with session_factory() as session, session.begin():
        advance_document_processing_stage(await _locked_job(session, job_id), stage)


async def succeed_document_processing_attempt(
    session_factory: async_sessionmaker[AsyncSession], *, job_id: int
) -> None:
    async with session_factory() as session, session.begin():
        mark_document_processing_succeeded(
            await _locked_job(session, job_id), now=utc_now()
        )


async def fail_document_processing_attempt(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    job_id: int,
    error_code: DocumentProcessingErrorCode,
    safe_error_message: str | None = None,
) -> None:
    async with session_factory() as session, session.begin():
        mark_document_processing_failed(
            await _locked_job(session, job_id),
            error_code,
            safe_error_message=safe_error_message,
            now=utc_now(),
        )


async def reconcile_interrupted_document_processing_jobs(
    session_factory: async_sessionmaker[AsyncSession],
) -> tuple[int, ...]:
    """Mark pre-restart active attempts interrupted; never retry automatically."""

    async with session_factory() as session, session.begin():
        jobs = tuple(
            (
                await session.scalars(
                    select(DocumentProcessingJob)
                    .where(
                        DocumentProcessingJob.status.in_(
                            (
                                DocumentProcessingStatus.PENDING,
                                DocumentProcessingStatus.RUNNING,
                            )
                        )
                    )
                    .order_by(DocumentProcessingJob.id.asc())
                    .with_for_update()
                )
            ).all()
        )
        now = utc_now()
        for job in jobs:
            mark_document_processing_interrupted(job, now=now)
        return tuple(job.id for job in jobs)
