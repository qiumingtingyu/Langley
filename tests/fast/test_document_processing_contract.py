"""Focused deterministic coverage for the durable processing lifecycle."""

from datetime import UTC, datetime, timedelta

import pytest

from langley.infrastructure.models import DocumentProcessingJob
from langley.knowledge.document_processing import (
    PDF_PROCESSING_RECIPE_ID,
    DocumentProcessingErrorCode,
    DocumentProcessingStage,
    DocumentProcessingStateError,
    advance_document_processing_stage,
    mark_document_processing_failed,
    mark_document_processing_interrupted,
    mark_document_processing_running,
    mark_document_processing_succeeded,
)

_CREATED = datetime(2026, 8, 29, tzinfo=UTC)
_STARTED = _CREATED + timedelta(seconds=1)
_FINISHED = _STARTED + timedelta(seconds=2)


def _pending_job() -> DocumentProcessingJob:
    return DocumentProcessingJob(
        id=1,
        document_version_id=2,
        attempt_no=1,
        status="PENDING",
        stage=None,
        recipe_id=PDF_PROCESSING_RECIPE_ID,
        error_code=None,
        error_message=None,
        created_at=_CREATED,
        started_at=None,
        finished_at=None,
    )


def test_running_and_stage_progress_are_monotonic() -> None:
    job = _pending_job()
    mark_document_processing_running(job, now=_STARTED)
    assert (job.status, job.stage, job.started_at) == (
        "RUNNING",
        "VERIFYING_SOURCE",
        _STARTED,
    )

    advance_document_processing_stage(job, DocumentProcessingStage.CHUNKING)
    assert job.stage == "CHUNKING"
    for stage in (
        DocumentProcessingStage.CHUNKING,
        DocumentProcessingStage.PARSING,
    ):
        with pytest.raises(DocumentProcessingStateError):
            advance_document_processing_stage(job, stage)


def test_success_requires_publishing_and_terminal_state_cannot_revive() -> None:
    job = _pending_job()
    mark_document_processing_running(job, now=_STARTED)
    with pytest.raises(DocumentProcessingStateError):
        mark_document_processing_succeeded(job, now=_FINISHED)
    advance_document_processing_stage(job, DocumentProcessingStage.PUBLISHING)
    mark_document_processing_succeeded(job, now=_FINISHED)
    assert (job.status, job.stage, job.finished_at) == (
        "SUCCEEDED",
        "PUBLISHING",
        _FINISHED,
    )
    with pytest.raises(DocumentProcessingStateError):
        mark_document_processing_running(job, now=_FINISHED)
    with pytest.raises(DocumentProcessingStateError):
        mark_document_processing_failed(
            job,
            DocumentProcessingErrorCode.PDF_PARSE_FAILED,
            safe_error_message=None,
            now=_FINISHED,
        )


def test_failure_retains_stage_and_records_only_stable_safe_error() -> None:
    job = _pending_job()
    mark_document_processing_running(job, now=_STARTED)
    advance_document_processing_stage(job, DocumentProcessingStage.PARSING)
    mark_document_processing_failed(
        job,
        DocumentProcessingErrorCode.PDF_PARSE_FAILED,
        safe_error_message="PDF 结构无法解析。",
        now=_FINISHED,
    )
    assert (job.status, job.stage, job.error_code, job.error_message) == (
        "FAILED",
        "PARSING",
        "PDF_PARSE_FAILED",
        "PDF 结构无法解析。",
    )
    assert job.finished_at == _FINISHED

    for message in ("", "line one\nline two", "x" * 256):
        other = _pending_job()
        mark_document_processing_running(other, now=_STARTED)
        with pytest.raises(ValueError):
            mark_document_processing_failed(
                other,
                DocumentProcessingErrorCode.PDF_PARSE_FAILED,
                safe_error_message=message,
                now=_FINISHED,
            )
        assert (other.status, other.finished_at) == ("RUNNING", None)
    reserved = _pending_job()
    mark_document_processing_running(reserved, now=_STARTED)
    with pytest.raises(ValueError):
        mark_document_processing_failed(
            reserved,
            DocumentProcessingErrorCode.PROCESS_INTERRUPTED,
            safe_error_message=None,
            now=_FINISHED,
        )


def test_restart_repair_preserves_pending_or_running_progress_shape() -> None:
    pending = _pending_job()
    mark_document_processing_interrupted(pending, now=_FINISHED)
    assert (pending.status, pending.stage, pending.started_at) == (
        "INTERRUPTED",
        None,
        None,
    )
    assert (pending.error_code, pending.finished_at) == (
        "PROCESS_INTERRUPTED",
        _FINISHED,
    )

    running = _pending_job()
    mark_document_processing_running(running, now=_STARTED)
    advance_document_processing_stage(running, DocumentProcessingStage.VALIDATING)
    mark_document_processing_interrupted(running, now=_FINISHED)
    assert (running.status, running.stage, running.started_at) == (
        "INTERRUPTED",
        "VALIDATING",
        _STARTED,
    )
    with pytest.raises(DocumentProcessingStateError):
        mark_document_processing_interrupted(running, now=_FINISHED)
