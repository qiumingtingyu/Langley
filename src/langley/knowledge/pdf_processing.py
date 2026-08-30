"""Persistent local PDF worker orchestration and atomic publication."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from langley.business_time import utc_now
from langley.infrastructure.local_file_storage import (
    LocalFileStorage,
    process_is_alive,
)
from langley.infrastructure.models import (
    DocumentProcessingJob,
    DocumentVersion,
    KnowledgeBase,
    KnowledgeChunk,
)
from langley.knowledge.contracts import DocumentSourceRef, encode_source_region
from langley.knowledge.document_indexing import (
    DocumentIndexConfiguration,
    publish_document_chunks,
)
from langley.knowledge.document_processing import (
    PDF_PROCESSING_TOKENIZER_ID,
    PDF_PROCESSING_TOKENIZER_REVISION,
    DocumentProcessingClaim,
    DocumentProcessingErrorCode,
    DocumentProcessingStage,
    DocumentProcessingStateError,
    advance_document_processing_attempt_stage,
    claim_next_document_processing_attempt,
    fail_document_processing_attempt,
    mark_document_processing_succeeded,
)
from langley.knowledge.pdf_processing_result import (
    PdfProcessingResult,
    PdfProcessingResultInvalid,
    load_pdf_processing_result,
)

logger = structlog.get_logger(__name__)

_WORKER_ERROR_CODES = {
    DocumentProcessingErrorCode.PDF_PROCESS_RESOURCE_LIMIT,
    DocumentProcessingErrorCode.PDF_PARSE_FAILED,
    DocumentProcessingErrorCode.PDF_CHUNKING_FAILED,
    DocumentProcessingErrorCode.PDF_OUTPUT_INVALID,
    DocumentProcessingErrorCode.SOURCE_CHANGED_DURING_PROCESSING,
}
_WINDOWS_RESOURCE_EXIT_CODES = {0xC0000017, 0xC000009A}
_CONTROL_LINE_LIMIT = 4096
_TERMINATE_GRACE_SECONDS = 5.0
_WORKER_EOF_GRACE_SECONDS = 0.25


class PdfSubprocessFailure(Exception):
    """One stable subprocess failure after the child has been reaped."""

    def __init__(self, error_code: DocumentProcessingErrorCode, *, pid: int | None):
        super().__init__(error_code.value)
        self.error_code = error_code
        self.pid = pid


class PdfPublicationFailure(Exception):
    """One publication precondition failed without exposing partial chunks."""

    def __init__(self, error_code: DocumentProcessingErrorCode):
        super().__init__(error_code.value)
        self.error_code = error_code


def _positive_control_integer(value: object) -> int | None:
    if type(value) is not int or value <= 0:
        return None
    return value


def _nonnegative_control_number(value: object) -> float | None:
    if type(value) is int:
        return float(value) if value >= 0 else None
    if type(value) is float:
        return value if value >= 0 else None
    return None


async def _terminate_and_reap(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        await process.wait()
        return
    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=_TERMINATE_GRACE_SECONDS)
    except TimeoutError:
        process.kill()
        await process.wait()


def _wait_for_windows_process_exit(pid: int, timeout_seconds: float) -> bool:
    """Wait a bounded interval for one exact Windows PID to exit."""
    import ctypes
    from ctypes import wintypes

    synchronize = 0x00100000
    wait_object_0 = 0x00000000
    wait_timeout = 0x00000102
    error_access_denied = 5
    error_invalid_parameter = 87

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    open_process = kernel32.OpenProcess
    open_process.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    open_process.restype = wintypes.HANDLE
    wait_for_single_object = kernel32.WaitForSingleObject
    wait_for_single_object.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    wait_for_single_object.restype = wintypes.DWORD
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL

    handle = open_process(synchronize, False, pid)
    if not handle:
        error_code = ctypes.get_last_error()
        if error_code == error_invalid_parameter:
            return True
        if error_code == error_access_denied:
            return False
        raise ctypes.WinError(error_code)
    try:
        timeout_ms = max(0, round(timeout_seconds * 1000))
        wait_result = wait_for_single_object(handle, timeout_ms)
        if wait_result == wait_object_0:
            return True
        if wait_result == wait_timeout:
            return False
        raise ctypes.WinError(ctypes.get_last_error())
    finally:
        close_handle(handle)


def _force_terminate_windows_process(pid: int, timeout_seconds: float) -> None:
    """Force one exact Windows PID to terminate and confirm bounded exit."""
    import ctypes
    from ctypes import wintypes

    process_terminate = 0x0001
    synchronize = 0x00100000
    wait_object_0 = 0x00000000
    error_invalid_parameter = 87

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    open_process = kernel32.OpenProcess
    open_process.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    open_process.restype = wintypes.HANDLE
    terminate_process = kernel32.TerminateProcess
    terminate_process.argtypes = (wintypes.HANDLE, wintypes.UINT)
    terminate_process.restype = wintypes.BOOL
    wait_for_single_object = kernel32.WaitForSingleObject
    wait_for_single_object.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    wait_for_single_object.restype = wintypes.DWORD
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL

    handle = open_process(process_terminate | synchronize, False, pid)
    if not handle:
        error_code = ctypes.get_last_error()
        if error_code == error_invalid_parameter:
            return
        raise ctypes.WinError(error_code)
    try:
        if not terminate_process(handle, 1):
            error_code = ctypes.get_last_error()
            if not process_is_alive(pid):
                return
            raise ctypes.WinError(error_code)
        timeout_ms = max(0, round(timeout_seconds * 1000))
        if wait_for_single_object(handle, timeout_ms) != wait_object_0:
            raise RuntimeError("PDF worker did not terminate within the deadline")
    finally:
        close_handle(handle)


@dataclass(frozen=True)
class PdfWorkerCompletion:
    pid: int
    page_count: int
    chunk_count: int
    parse_ms: float
    chunk_ms: float
    total_ms: float


class PersistentPdfWorker:
    """One lazy local JSONL worker whose large data path remains filesystem staging."""

    def __init__(
        self,
        *,
        tokenizer_id: str,
        tokenizer_revision: str,
        worker_marker_path: Path,
        command: Sequence[str] | None = None,
    ) -> None:
        self._command = (
            tuple(command)
            if command is not None
            else (
                sys.executable,
                "-m",
                "langley.knowledge.pdf_processing_worker",
                "--persistent",
                "--tokenizer-id",
                tokenizer_id,
                "--tokenizer-revision",
                tokenizer_revision,
                "--worker-marker-path",
                str(worker_marker_path),
            )
        )
        self._worker_marker_path = worker_marker_path
        self._process: asyncio.subprocess.Process | None = None
        self._worker_pid: int | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._stderr_context: dict[str, object] = {}
        self._lock = asyncio.Lock()

    @property
    def pid(self) -> int | None:
        return self._worker_pid

    @property
    def launcher_pid(self) -> int | None:
        process = self._process
        return process.pid if process is not None else None

    async def process(
        self,
        command: dict[str, object],
        *,
        timeout_seconds: float,
        on_chunking: Callable[[], Awaitable[None]],
    ) -> PdfWorkerCompletion:
        async with self._lock:
            process = await self._ensure_started()
            worker_pid = self._worker_pid
            assert worker_pid is not None
            self._stderr_context = {
                "job_id": command["job_id"],
                "attempt_no": command["attempt_no"],
                "document_version_id": command["document_version_id"],
                "stage": "PARSING",
            }
            try:
                assert process.stdin is not None
                process.stdin.write(
                    (json.dumps(command, separators=(",", ":")) + "\n").encode("utf-8")
                )
                await process.stdin.drain()
                async with asyncio.timeout(timeout_seconds):
                    return await self._read_job_events(
                        process,
                        worker_pid=worker_pid,
                        command=command,
                        on_chunking=on_chunking,
                    )
            except TimeoutError as error:
                await self._terminate()
                raise PdfSubprocessFailure(
                    DocumentProcessingErrorCode.PDF_PROCESS_TIMEOUT, pid=worker_pid
                ) from error
            except asyncio.CancelledError:
                await self._terminate()
                raise
            except PdfSubprocessFailure as error:
                if error.error_code in {
                    DocumentProcessingErrorCode.PDF_PROCESS_RESOURCE_LIMIT,
                    DocumentProcessingErrorCode.PDF_PROCESS_WORKER_EXITED,
                    DocumentProcessingErrorCode.PDF_OUTPUT_INVALID,
                }:
                    await self._terminate()
                raise
            except (BrokenPipeError, ConnectionError, OSError) as error:
                await self._terminate()
                raise PdfSubprocessFailure(
                    DocumentProcessingErrorCode.PDF_PROCESS_WORKER_EXITED,
                    pid=worker_pid,
                ) from error
            except Exception as error:
                await self._terminate()
                raise PdfSubprocessFailure(
                    DocumentProcessingErrorCode.PDF_OUTPUT_INVALID, pid=worker_pid
                ) from error
            finally:
                self._stderr_context = {}

    async def stop(self) -> None:
        async with self._lock:
            process = self._process
            if process is None:
                return
            worker_pid = self._worker_pid
            if process.returncode is None and process.stdin is not None:
                try:
                    process.stdin.write(b'{"command":"shutdown"}\n')
                    await process.stdin.drain()
                    process.stdin.close()
                    await asyncio.wait_for(
                        process.wait(), timeout=_TERMINATE_GRACE_SECONDS
                    )
                except (BrokenPipeError, ConnectionError, OSError, TimeoutError):
                    await self._terminate()
                    return
            else:
                await process.wait()
            if (
                os.name == "nt"
                and worker_pid is not None
                and not await asyncio.to_thread(
                    _wait_for_windows_process_exit, worker_pid, 0
                )
            ):
                await self._terminate()
                return
            await self._discard_reaped(process, worker_pid)

    async def _ensure_started(self) -> asyncio.subprocess.Process:
        if self._process is not None:
            if (
                self._process.returncode is None
                and self._worker_pid is not None
                and await asyncio.to_thread(process_is_alive, self._worker_pid)
            ):
                return self._process
            await self._terminate()
        creationflags = 0
        if os.name == "nt":
            import subprocess

            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
        try:
            process = await asyncio.create_subprocess_exec(
                *self._command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                creationflags=creationflags,
                limit=_CONTROL_LINE_LIMIT,
            )
        except OSError as error:
            raise PdfSubprocessFailure(
                DocumentProcessingErrorCode.PDF_PROCESS_LAUNCH_FAILED, pid=None
            ) from error
        self._process = process
        self._worker_pid = None
        self._stderr_task = asyncio.create_task(self._drain_worker_stderr(process))
        try:
            event = await asyncio.wait_for(
                self._read_control_event(process), timeout=_TERMINATE_GRACE_SECONDS
            )
            worker_pid = event.get("pid") if event is not None else None
            if type(worker_pid) is int and worker_pid > 0:
                self._worker_pid = worker_pid
            if (
                event is None
                or set(event) != {"event", "pid"}
                or event.get("event") != "ready"
                or type(worker_pid) is not int
                or worker_pid <= 0
                or not await asyncio.to_thread(process_is_alive, worker_pid)
            ):
                raise ValueError("invalid worker readiness event")
            await asyncio.to_thread(self._write_marker_for_pid, worker_pid)
        except Exception as error:
            failure_pid = self._worker_pid
            await self._terminate()
            raise PdfSubprocessFailure(
                DocumentProcessingErrorCode.PDF_PROCESS_LAUNCH_FAILED,
                pid=failure_pid,
            ) from error
        logger.info(
            "knowledge.document_processing.worker_started",
            launcher_pid=process.pid,
            worker_pid=worker_pid,
        )
        return process

    async def _read_job_events(
        self,
        process: asyncio.subprocess.Process,
        *,
        worker_pid: int,
        command: dict[str, object],
        on_chunking: Callable[[], Awaitable[None]],
    ) -> PdfWorkerCompletion:
        identity = {
            "job_id": command["job_id"],
            "attempt_no": command["attempt_no"],
            "document_version_id": command["document_version_id"],
            "pid": worker_pid,
        }
        parsing_events = 0
        chunking_events = 0
        while True:
            event = await self._read_control_event(process)
            if event is None:
                return_code = (
                    process.returncode & 0xFFFFFFFF
                    if process.returncode is not None
                    else None
                )
                error_code = (
                    DocumentProcessingErrorCode.PDF_PROCESS_RESOURCE_LIMIT
                    if return_code is not None
                    and return_code in _WINDOWS_RESOURCE_EXIT_CODES
                    else DocumentProcessingErrorCode.PDF_PROCESS_WORKER_EXITED
                )
                raise PdfSubprocessFailure(error_code, pid=worker_pid)
            if set(event) == {"event", "stage", *identity}:
                if any(event[key] != value for key, value in identity.items()):
                    raise PdfSubprocessFailure(
                        DocumentProcessingErrorCode.PDF_OUTPUT_INVALID, pid=worker_pid
                    )
                if event["event"] != "stage":
                    raise PdfSubprocessFailure(
                        DocumentProcessingErrorCode.PDF_OUTPUT_INVALID, pid=worker_pid
                    )
                if event["stage"] == "PARSING":
                    parsing_events += 1
                    if parsing_events != 1 or chunking_events:
                        raise PdfSubprocessFailure(
                            DocumentProcessingErrorCode.PDF_OUTPUT_INVALID,
                            pid=worker_pid,
                        )
                    continue
                if event["stage"] == "CHUNKING":
                    chunking_events += 1
                    if parsing_events != 1 or chunking_events != 1:
                        raise PdfSubprocessFailure(
                            DocumentProcessingErrorCode.PDF_OUTPUT_INVALID,
                            pid=worker_pid,
                        )
                    self._stderr_context["stage"] = "CHUNKING"
                    await on_chunking()
                    continue
                raise PdfSubprocessFailure(
                    DocumentProcessingErrorCode.PDF_OUTPUT_INVALID, pid=worker_pid
                )
            if set(event) == {"event", "error_code", *identity}:
                if any(event[key] != value for key, value in identity.items()):
                    raise PdfSubprocessFailure(
                        DocumentProcessingErrorCode.PDF_OUTPUT_INVALID, pid=worker_pid
                    )
                error_code_value = event["error_code"]
                if not isinstance(error_code_value, str):
                    raise PdfSubprocessFailure(
                        DocumentProcessingErrorCode.PDF_OUTPUT_INVALID, pid=worker_pid
                    )
                try:
                    error_code = DocumentProcessingErrorCode(error_code_value)
                except ValueError as error:
                    raise PdfSubprocessFailure(
                        DocumentProcessingErrorCode.PDF_OUTPUT_INVALID, pid=worker_pid
                    ) from error
                if event["event"] != "error" or error_code not in _WORKER_ERROR_CODES:
                    raise PdfSubprocessFailure(
                        DocumentProcessingErrorCode.PDF_OUTPUT_INVALID, pid=worker_pid
                    )
                raise PdfSubprocessFailure(error_code, pid=worker_pid)
            completed_keys = {
                "event",
                "page_count",
                "chunk_count",
                "parse_ms",
                "chunk_ms",
                "total_ms",
                *identity,
            }
            if set(event) == completed_keys:
                page_count = _positive_control_integer(event["page_count"])
                chunk_count = _positive_control_integer(event["chunk_count"])
                parse_ms = _nonnegative_control_number(event["parse_ms"])
                chunk_ms = _nonnegative_control_number(event["chunk_ms"])
                total_ms = _nonnegative_control_number(event["total_ms"])
                if (
                    event["event"] != "completed"
                    or any(event[key] != value for key, value in identity.items())
                    or parsing_events != 1
                    or chunking_events != 1
                    or page_count is None
                    or chunk_count is None
                    or parse_ms is None
                    or chunk_ms is None
                    or total_ms is None
                ):
                    raise PdfSubprocessFailure(
                        DocumentProcessingErrorCode.PDF_OUTPUT_INVALID, pid=worker_pid
                    )
                return PdfWorkerCompletion(
                    pid=worker_pid,
                    page_count=page_count,
                    chunk_count=chunk_count,
                    parse_ms=parse_ms,
                    chunk_ms=chunk_ms,
                    total_ms=total_ms,
                )
            raise PdfSubprocessFailure(
                DocumentProcessingErrorCode.PDF_OUTPUT_INVALID, pid=worker_pid
            )

    @staticmethod
    async def _read_control_event(
        process: asyncio.subprocess.Process,
    ) -> dict[str, object] | None:
        assert process.stdout is not None
        line = await process.stdout.readline()
        if not line:
            return None
        if len(line) > _CONTROL_LINE_LIMIT:
            raise ValueError("worker control line too large")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("worker control event must be an object")
        return value

    async def _drain_worker_stderr(self, process: asyncio.subprocess.Process) -> None:
        if process.stderr is None:
            return
        while line := await process.stderr.readline():
            diagnostic = line.decode("utf-8", errors="replace").rstrip()
            if not diagnostic:
                continue
            logger.error(
                "knowledge.document_processing.worker_diagnostic",
                **self._stderr_context,
                launcher_pid=process.pid,
                worker_pid=self._worker_pid,
                diagnostic=diagnostic[:8192],
            )

    async def _terminate(self) -> None:
        process = self._process
        if process is None:
            return
        worker_pid = self._worker_pid
        if process.stdin is not None:
            process.stdin.close()
        if os.name == "nt" and worker_pid is not None:
            worker_exited = await asyncio.to_thread(
                _wait_for_windows_process_exit,
                worker_pid,
                _WORKER_EOF_GRACE_SECONDS,
            )
            if not worker_exited:
                await asyncio.to_thread(
                    _force_terminate_windows_process,
                    worker_pid,
                    _TERMINATE_GRACE_SECONDS,
                )
            if await asyncio.to_thread(process_is_alive, worker_pid):
                raise RuntimeError("PDF worker remained alive after termination")
        if process.returncode is None:
            try:
                await asyncio.wait_for(
                    process.wait(), timeout=_WORKER_EOF_GRACE_SECONDS
                )
            except TimeoutError:
                await _terminate_and_reap(process)
        else:
            await process.wait()
        await self._discard_reaped(process, worker_pid)

    async def _discard_reaped(
        self, process: asyncio.subprocess.Process, worker_pid: int | None
    ) -> None:
        if process.returncode is None:
            await process.wait()
        stderr_task, self._stderr_task = self._stderr_task, None
        if stderr_task is not None:
            await asyncio.gather(stderr_task, return_exceptions=True)
        if self._process is process:
            self._process = None
            self._worker_pid = None
        if worker_pid is not None:
            await asyncio.to_thread(self._remove_marker_for_pid, worker_pid)

    def _write_marker_for_pid(self, pid: int) -> None:
        self._worker_marker_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self._worker_marker_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps({"pid": pid}, separators=(",", ":")),
            encoding="ascii",
            newline="\n",
        )
        os.replace(temporary_path, self._worker_marker_path)

    def _remove_marker_for_pid(self, pid: int) -> None:
        try:
            value = json.loads(self._worker_marker_path.read_text(encoding="ascii"))
            if value == {"pid": pid}:
                self._worker_marker_path.unlink(missing_ok=True)
        except (OSError, json.JSONDecodeError):
            pass


def _verify_local_source(path: Path, source_ref: DocumentSourceRef) -> None:
    try:
        if path.stat().st_size != source_ref.source_size_bytes:
            raise PdfPublicationFailure(
                DocumentProcessingErrorCode.SOURCE_INTEGRITY_MISMATCH
            )
        digest = sha256()
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
        if digest.hexdigest() != source_ref.source_sha256:
            raise PdfPublicationFailure(
                DocumentProcessingErrorCode.SOURCE_INTEGRITY_MISMATCH
            )
    except FileNotFoundError as error:
        raise PdfPublicationFailure(
            DocumentProcessingErrorCode.SOURCE_MISSING
        ) from error


def _source_identity_matches(
    version: DocumentVersion, source_ref: DocumentSourceRef
) -> bool:
    return (
        version.id == source_ref.document_version_id
        and version.storage_key == source_ref.storage_key
        and version.source_media_type == source_ref.source_media_type
        and version.source_sha256 == source_ref.source_sha256
        and version.source_size_bytes == source_ref.source_size_bytes
    )


async def publish_pdf_processing_result(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    claim: DocumentProcessingClaim,
    result: PdfProcessingResult,
    index_configuration: DocumentIndexConfiguration | None = None,
) -> bool:
    """Atomically replace all PDF chunks and complete the exact RUNNING attempt."""
    rows = [
        KnowledgeChunk(
            document_version_id=claim.document_version_id,
            ordinal=candidate.ordinal,
            content=candidate.content,
            heading_path=list(candidate.heading_path),
            source_regions=[
                encode_source_region(region) for region in candidate.source_regions
            ],
            created_at=utc_now(),
        )
        for candidate in result.candidates
    ]
    async with session_factory() as session, session.begin():
        knowledge_base = await session.scalar(
            select(KnowledgeBase)
            .where(
                KnowledgeBase.id == claim.knowledge_base_id,
                KnowledgeBase.user_id == claim.user_id,
            )
            .with_for_update()
        )
        if knowledge_base is None:
            raise PdfPublicationFailure(
                DocumentProcessingErrorCode.SOURCE_CHANGED_DURING_PROCESSING
            )
        if knowledge_base.index_status == "INDEXING":
            raise PdfPublicationFailure(DocumentProcessingErrorCode.PUBLICATION_FAILED)
        job = await session.get(
            DocumentProcessingJob, claim.job_id, with_for_update=True
        )
        version = await session.get(
            DocumentVersion, claim.document_version_id, with_for_update=True
        )
        if version is None or not _source_identity_matches(version, claim.source_ref):
            raise PdfPublicationFailure(
                DocumentProcessingErrorCode.SOURCE_CHANGED_DURING_PROCESSING
            )
        if (
            job is None
            or job.status != "RUNNING"
            or job.stage != "PUBLISHING"
            or job.document_version_id != claim.document_version_id
            or job.attempt_no != claim.attempt_no
            or job.recipe_id != claim.recipe_id
        ):
            raise PdfPublicationFailure(DocumentProcessingErrorCode.PUBLICATION_FAILED)
        publication = await publish_document_chunks(
            session,
            knowledge_base=knowledge_base,
            version=version,
            prepared_rows=rows,
            configuration=(index_configuration or _configured_index_defaults()),
        )
        mark_document_processing_succeeded(job, now=utc_now())
        await session.flush()
        return publication.job_created


class DocumentProcessingRuntime:
    """Execute one claimed PDF attempt without retaining database resources."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        file_storage: LocalFileStorage,
        *,
        timeout_seconds: float,
        index_configuration: DocumentIndexConfiguration | None = None,
        index_wake: Callable[[], None] | None = None,
        worker: PersistentPdfWorker | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._file_storage = file_storage
        self._timeout_seconds = timeout_seconds
        self._index_configuration = index_configuration or _configured_index_defaults()
        self._index_wake = index_wake or _do_not_wake
        self._worker = worker or PersistentPdfWorker(
            tokenizer_id=PDF_PROCESSING_TOKENIZER_ID,
            tokenizer_revision=PDF_PROCESSING_TOKENIZER_REVISION,
            worker_marker_path=file_storage.processing_worker_marker_path(),
        )

    @property
    def worker_pid(self) -> int | None:
        return self._worker.pid

    async def stop(self) -> None:
        await self._worker.stop()

    async def execute(self, claim: DocumentProcessingClaim) -> None:
        started = perf_counter()
        current_stage = DocumentProcessingStage.VERIFYING_SOURCE
        stage_started = started
        metadata = {
            "job_id": claim.job_id,
            "document_version_id": claim.document_version_id,
            "attempt_no": claim.attempt_no,
        }
        logger.info(
            "knowledge.document_processing.started",
            **metadata,
            stage=current_stage.value,
        )
        logger.info(
            "knowledge.document_processing.stage_started",
            **metadata,
            stage=current_stage.value,
        )

        async def advance(stage: DocumentProcessingStage) -> None:
            nonlocal current_stage, stage_started
            now = perf_counter()
            logger.info(
                "knowledge.document_processing.stage_completed",
                **metadata,
                stage=current_stage.value,
                duration_ms=round((now - stage_started) * 1000, 3),
            )
            await advance_document_processing_attempt_stage(
                self._session_factory, job_id=claim.job_id, stage=stage
            )
            current_stage = stage
            stage_started = perf_counter()
            logger.info(
                "knowledge.document_processing.stage_started",
                **metadata,
                stage=current_stage.value,
            )

        error_code: DocumentProcessingErrorCode | None = None
        failure_worker_pid: int | None = None
        result_path: Path | None = None
        try:
            if claim.source_ref.source_media_type != "application/pdf":
                raise PdfPublicationFailure(
                    DocumentProcessingErrorCode.SOURCE_INTEGRITY_MISMATCH
                )
            source_path = self._file_storage.source_path(claim.source_ref.storage_key)
            await asyncio.to_thread(_verify_local_source, source_path, claim.source_ref)
            await advance(DocumentProcessingStage.PARSING)
            result_path = await self._file_storage.prepare_processing_result_path(
                claim.job_id, claim.attempt_no
            )
            completion = await self._worker.process(
                {
                    "command": "process",
                    "job_id": claim.job_id,
                    "attempt_no": claim.attempt_no,
                    "document_version_id": claim.document_version_id,
                    "source_path": str(source_path),
                    "source_sha256": claim.source_ref.source_sha256,
                    "source_size_bytes": claim.source_ref.source_size_bytes,
                    "recipe_id": claim.recipe_id,
                    "staging_path": str(result_path),
                },
                timeout_seconds=self._timeout_seconds,
                on_chunking=lambda: advance(DocumentProcessingStage.CHUNKING),
            )
            logger.info(
                "knowledge.document_processing.worker_completed",
                **metadata,
                worker_pid=completion.pid,
                parse_ms=completion.parse_ms,
                chunk_ms=completion.chunk_ms,
                worker_total_ms=completion.total_ms,
            )
            await advance(DocumentProcessingStage.VALIDATING)
            result = await asyncio.to_thread(
                load_pdf_processing_result,
                result_path,
                expected_recipe_id=claim.recipe_id,
                expected_job_id=claim.job_id,
                expected_attempt_no=claim.attempt_no,
                expected_document_version_id=claim.document_version_id,
                expected_source_sha256=claim.source_ref.source_sha256,
                expected_source_size_bytes=claim.source_ref.source_size_bytes,
            )
            try:
                await asyncio.to_thread(
                    _verify_local_source, source_path, claim.source_ref
                )
            except PdfPublicationFailure as error:
                raise PdfPublicationFailure(
                    DocumentProcessingErrorCode.SOURCE_CHANGED_DURING_PROCESSING
                ) from error
            await advance(DocumentProcessingStage.PUBLISHING)
            index_job_created = await publish_pdf_processing_result(
                self._session_factory,
                claim=claim,
                result=result,
                index_configuration=self._index_configuration,
            )
            if index_job_created:
                self._index_wake()
            finished = perf_counter()
            logger.info(
                "knowledge.document_processing.stage_completed",
                **metadata,
                stage=current_stage.value,
                duration_ms=round((finished - stage_started) * 1000, 3),
            )
            logger.info(
                "knowledge.document_processing.succeeded",
                **metadata,
                stage=current_stage.value,
                duration_ms=round((finished - started) * 1000, 3),
                page_count=result.page_count,
                chunk_count=len(result.candidates),
            )
            return
        except asyncio.CancelledError:
            raise
        except PdfSubprocessFailure as error:
            error_code = error.error_code
            failure_worker_pid = error.pid
            logger.error(
                "knowledge.document_processing.worker_failed",
                **metadata,
                stage=current_stage.value,
                error_code=error.error_code.value,
                worker_pid=error.pid,
                worker_timeout=(
                    error.error_code is DocumentProcessingErrorCode.PDF_PROCESS_TIMEOUT
                ),
                worker_crash=(
                    error.error_code
                    is DocumentProcessingErrorCode.PDF_PROCESS_WORKER_EXITED
                ),
                worker_launch_failed=(
                    error.error_code
                    is DocumentProcessingErrorCode.PDF_PROCESS_LAUNCH_FAILED
                ),
            )
        except PdfProcessingResultInvalid:
            error_code = DocumentProcessingErrorCode.PDF_OUTPUT_INVALID
        except PdfPublicationFailure as error:
            error_code = error.error_code
        except (OSError, DocumentProcessingStateError):
            error_code = DocumentProcessingErrorCode.PUBLICATION_FAILED
        except Exception:
            error_code = DocumentProcessingErrorCode.PUBLICATION_FAILED
            logger.exception(
                "knowledge.document_processing.unexpected_failure", **metadata
            )
        finally:
            if result_path is not None:
                try:
                    await self._file_storage.cleanup_processing_attempt(
                        claim.job_id, claim.attempt_no
                    )
                except OSError:
                    logger.warning(
                        "knowledge.document_processing.staging_cleanup_failed",
                        **metadata,
                    )

        assert error_code is not None
        try:
            await fail_document_processing_attempt(
                self._session_factory,
                job_id=claim.job_id,
                error_code=error_code,
                safe_error_message=_safe_error_message(error_code),
            )
        except DocumentProcessingStateError:
            logger.warning(
                "knowledge.document_processing.failure_state_changed",
                **metadata,
                error_code=error_code.value,
            )
            return
        logger.error(
            "knowledge.document_processing.failed",
            **metadata,
            stage=current_stage.value,
            duration_ms=round((perf_counter() - started) * 1000, 3),
            error_code=error_code.value,
            worker_pid=failure_worker_pid,
        )


def _safe_error_message(error_code: DocumentProcessingErrorCode) -> str:
    return {
        DocumentProcessingErrorCode.SOURCE_MISSING: "原始 PDF 文件不存在。",
        DocumentProcessingErrorCode.SOURCE_INTEGRITY_MISMATCH: (
            "原始 PDF 完整性校验失败。"
        ),
        DocumentProcessingErrorCode.PDF_PROCESS_TIMEOUT: "PDF 处理超时。",
        DocumentProcessingErrorCode.PDF_PROCESS_RESOURCE_LIMIT: (
            "PDF 处理超出本机资源限制。"
        ),
        DocumentProcessingErrorCode.PDF_PROCESS_LAUNCH_FAILED: (
            "PDF 处理进程无法启动。"
        ),
        DocumentProcessingErrorCode.PDF_PROCESS_WORKER_EXITED: (
            "PDF 处理进程意外退出。"
        ),
        DocumentProcessingErrorCode.PDF_PARSE_FAILED: "PDF 结构无法解析。",
        DocumentProcessingErrorCode.PDF_CHUNKING_FAILED: "PDF 文本分块失败。",
        DocumentProcessingErrorCode.PDF_OUTPUT_INVALID: "PDF 处理结果未通过校验。",
        DocumentProcessingErrorCode.SOURCE_CHANGED_DURING_PROCESSING: (
            "处理期间原始 PDF 发生变化。"
        ),
        DocumentProcessingErrorCode.PUBLICATION_FAILED: "PDF 处理结果发布失败。",
    }[error_code]


class DocumentProcessingDispatcher:
    """One application-local serial dispatcher over durable MySQL PENDING jobs."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        runtime: DocumentProcessingRuntime,
    ) -> None:
        self._session_factory = session_factory
        self._runtime = runtime
        self._wake = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._task is not None:
            raise RuntimeError("document processing dispatcher already started")
        self._wake.set()
        self._task = asyncio.create_task(self._run())

    def wake(self) -> None:
        self._wake.set()

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        stop = getattr(self._runtime, "stop", None)
        if stop is not None:
            await stop()

    async def _run(self) -> None:
        while True:
            try:
                claim = await claim_next_document_processing_attempt(
                    self._session_factory
                )
                if claim is not None:
                    await self._runtime.execute(claim)
                    continue
                self._wake.clear()
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=2.0)
                except TimeoutError:
                    pass
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("knowledge.document_processing.dispatch_failed")
                await asyncio.sleep(2.0)


def _configured_index_defaults() -> DocumentIndexConfiguration:
    """Resolve direct runtime calls from the environment-backed settings."""

    from langley.settings import Settings

    return DocumentIndexConfiguration.from_settings(Settings())


def _do_not_wake() -> None:
    return None
