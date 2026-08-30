"""Focused deterministic coverage for PDF staging, child, and dispatcher seams."""

import asyncio
import json
import os
import sys
from collections import deque
from pathlib import Path

import pytest
from structlog.testing import capture_logs

import langley.knowledge.pdf_processing as processing_module
from langley.infrastructure.local_file_storage import LocalFileStorage, process_is_alive
from langley.knowledge.chunking import CandidateChunk
from langley.knowledge.contracts import DocumentSourceRef, PdfPageRegion
from langley.knowledge.document_indexing import DocumentIndexConfiguration
from langley.knowledge.document_processing import (
    PDF_PROCESSING_RECIPE_ID,
    PDF_PROCESSING_TOKENIZER_ID,
    PDF_PROCESSING_TOKENIZER_REVISION,
    DocumentProcessingClaim,
    DocumentProcessingErrorCode,
)
from langley.knowledge.pdf_processing import (
    DocumentProcessingDispatcher,
    PdfSubprocessFailure,
    PersistentPdfWorker,
)
from langley.knowledge.pdf_processing_result import (
    PdfProcessingResultInvalid,
    load_pdf_processing_result,
    write_pdf_processing_result,
)
from langley.settings import Settings


def _write_result(path: Path) -> None:
    write_pdf_processing_result(
        path,
        recipe_id=PDF_PROCESSING_RECIPE_ID,
        job_id=7,
        attempt_no=2,
        document_version_id=11,
        source_sha256="a" * 64,
        source_size_bytes=100,
        page_count=3,
        candidates=(
            CandidateChunk(
                ordinal=1,
                content="Raw source evidence.",
                heading_path=("Heading",),
                source_regions=(PdfPageRegion(1, 2),),
            ),
        ),
    )


def _load_result(path: Path):
    return load_pdf_processing_result(
        path,
        expected_recipe_id=PDF_PROCESSING_RECIPE_ID,
        expected_job_id=7,
        expected_attempt_no=2,
        expected_document_version_id=11,
        expected_source_sha256="a" * 64,
        expected_source_size_bytes=100,
    )


def test_staging_schema_round_trip_and_attempt_identity_are_strict(
    tmp_path: Path,
) -> None:
    result_path = tmp_path / "result.json"
    _write_result(result_path)
    result = _load_result(result_path)
    assert result.page_count == 3
    assert result.candidates == (
        CandidateChunk(
            ordinal=1,
            content="Raw source evidence.",
            heading_path=("Heading",),
            source_regions=(PdfPageRegion(1, 2),),
        ),
    )

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["job_id"] = 8
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PdfProcessingResultInvalid):
        _load_result(result_path)


@pytest.mark.parametrize(
    ("path", "invalid_value"),
    [
        (("job_id",), 7.0),
        (("attempt_no",), True),
        (("document_version_id",), 11.0),
        (("source_size_bytes",), 100.0),
        (("page_count",), 3.0),
        (("chunks", 0, "ordinal"), True),
        (("chunks", 0, "source_regions", 0, "page_start"), 1.0),
        (("chunks", 0, "source_regions", 0, "page_end"), True),
    ],
)
def test_staging_rejects_non_integer_numeric_values(
    tmp_path: Path, path: tuple[object, ...], invalid_value: object
) -> None:
    result_path = tmp_path / "result.json"
    _write_result(result_path)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    target = payload
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = invalid_value
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PdfProcessingResultInvalid):
        _load_result(result_path)


def test_staging_rejects_out_of_range_pages_and_non_deterministic_ordinals(
    tmp_path: Path,
) -> None:
    result_path = tmp_path / "result.json"
    _write_result(result_path)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["chunks"][0]["ordinal"] = 2
    payload["chunks"][0]["source_regions"][0]["page_end"] = 4
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PdfProcessingResultInvalid):
        _load_result(result_path)


def test_startup_cleanup_refuses_to_delete_while_marked_worker_is_alive(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        storage = LocalFileStorage(tmp_path / "storage")
        result_path = await storage.prepare_processing_result_path(7, 2)
        result_path.write_text("active", encoding="ascii")
        marker_path = storage.processing_worker_marker_path()
        marker_path.write_text(json.dumps({"pid": os.getpid()}), encoding="ascii")
        with pytest.raises(RuntimeError, match="still running"):
            await storage.confirm_no_processing_worker()
        assert result_path.exists()

    asyncio.run(run())


def _persistent_command(tmp_path: Path) -> tuple[tuple[str, ...], Path, Path, Path]:
    behavior_path = tmp_path / "behavior"
    pid_log_path = tmp_path / "pids"
    done_path = tmp_path / "done"
    command = (
        sys.executable,
        str(Path("tests/fixtures/pdf_persistent_fake_worker.py").resolve()),
        "--behavior-path",
        str(behavior_path),
        "--pid-log-path",
        str(pid_log_path),
        "--done-path",
        str(done_path),
    )
    return command, behavior_path, pid_log_path, done_path


def _persistent_job(job_id: int, staging_path: Path) -> dict[str, object]:
    return {
        "command": "process",
        "job_id": job_id,
        "attempt_no": 1,
        "document_version_id": job_id + 100,
        "source_path": str(Path(__file__).resolve()),
        "source_sha256": "a" * 64,
        "source_size_bytes": 100,
        "recipe_id": PDF_PROCESSING_RECIPE_ID,
        "staging_path": str(staging_path),
    }


def test_document_processing_runtime_uses_frozen_recipe_tokenizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def fake_worker_factory(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(processing_module, "PersistentPdfWorker", fake_worker_factory)
    mutable_settings = Settings(
        knowledge_embedding_model="mutable-index-model",
        knowledge_embedding_revision="mutable-index-revision",
    )

    processing_module.DocumentProcessingRuntime(
        object(),
        LocalFileStorage(tmp_path / "storage"),
        timeout_seconds=1,
        index_configuration=DocumentIndexConfiguration.from_settings(mutable_settings),
        index_wake=lambda: None,
    )

    assert PDF_PROCESSING_TOKENIZER_ID == "BAAI/bge-m3"
    assert (
        PDF_PROCESSING_TOKENIZER_REVISION == "5617a9f61b028005a4858fdac845db406aefb181"
    )
    assert captured["tokenizer_id"] == PDF_PROCESSING_TOKENIZER_ID
    assert captured["tokenizer_revision"] == PDF_PROCESSING_TOKENIZER_REVISION
    assert captured["tokenizer_id"] != mutable_settings.knowledge_embedding_model
    assert (
        captured["tokenizer_revision"] != mutable_settings.knowledge_embedding_revision
    )


def test_persistent_worker_reuses_pid_and_keeps_large_data_in_staging(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        command, behavior_path, pid_log_path, _ = _persistent_command(tmp_path)
        behavior_path.write_text("normal", encoding="ascii")
        worker = PersistentPdfWorker(
            tokenizer_id="unused",
            tokenizer_revision="unused",
            worker_marker_path=tmp_path / "worker.json",
            command=command,
        )
        stages: list[int] = []
        first_path = tmp_path / "first.json"
        second_path = tmp_path / "second.json"
        first = await worker.process(
            _persistent_job(1, first_path),
            timeout_seconds=2,
            on_chunking=lambda: _record_stage(stages, 1),
        )
        second = await worker.process(
            _persistent_job(2, second_path),
            timeout_seconds=2,
            on_chunking=lambda: _record_stage(stages, 2),
        )
        assert first.pid == second.pid == worker.pid
        assert pid_log_path.read_text(encoding="ascii").splitlines() == [
            str(first.pid),
            str(first.pid),
        ]
        assert first_path.stat().st_size == second_path.stat().st_size == 100_000
        assert stages == [1, 2]
        assert process_is_alive(first.pid)
        await worker.stop()
        assert not process_is_alive(first.pid)

    asyncio.run(run())


async def _record_stage(stages: list[int], value: int) -> None:
    stages.append(value)


def test_persistent_worker_timeout_reaps_then_next_job_gets_new_pid(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        command, behavior_path, _, done_path = _persistent_command(tmp_path)
        behavior_path.write_text("timeout", encoding="ascii")
        worker = PersistentPdfWorker(
            tokenizer_id="unused",
            tokenizer_revision="unused",
            worker_marker_path=tmp_path / "worker.json",
            command=command,
        )
        with pytest.raises(PdfSubprocessFailure) as timeout:
            await worker.process(
                _persistent_job(1, tmp_path / "timeout.json"),
                timeout_seconds=0.3,
                on_chunking=lambda: _record_stage([], 1),
            )
        assert (
            timeout.value.error_code is DocumentProcessingErrorCode.PDF_PROCESS_TIMEOUT
        )
        first_pid = timeout.value.pid
        assert first_pid is not None and worker.pid is None
        assert not process_is_alive(first_pid)
        assert not done_path.exists()

        behavior_path.write_text("normal", encoding="ascii")
        second = await worker.process(
            _persistent_job(2, tmp_path / "next.json"),
            timeout_seconds=2,
            on_chunking=lambda: _record_stage([], 2),
        )
        assert second.pid != first_pid
        await worker.stop()

    asyncio.run(run())


def test_persistent_worker_unexpected_exit_is_not_parse_failure(tmp_path: Path) -> None:
    async def run() -> None:
        command, behavior_path, _, _ = _persistent_command(tmp_path)
        behavior_path.write_text("crash", encoding="ascii")
        worker = PersistentPdfWorker(
            tokenizer_id="unused",
            tokenizer_revision="unused",
            worker_marker_path=tmp_path / "worker.json",
            command=command,
        )
        with pytest.raises(PdfSubprocessFailure) as failure:
            await worker.process(
                _persistent_job(1, tmp_path / "crash.json"),
                timeout_seconds=2,
                on_chunking=lambda: _record_stage([], 1),
            )
        assert (
            failure.value.error_code
            is DocumentProcessingErrorCode.PDF_PROCESS_WORKER_EXITED
        )
        assert worker.pid is None

    asyncio.run(run())


def test_persistent_worker_launch_failure_is_not_parse_failure(tmp_path: Path) -> None:
    async def run() -> None:
        worker = PersistentPdfWorker(
            tokenizer_id="unused",
            tokenizer_revision="unused",
            worker_marker_path=tmp_path / "worker.json",
            command=(str(tmp_path / "missing-worker.exe"),),
        )
        with pytest.raises(PdfSubprocessFailure) as failure:
            await worker.process(
                _persistent_job(1, tmp_path / "missing.json"),
                timeout_seconds=2,
                on_chunking=lambda: _record_stage([], 1),
            )
        assert (
            failure.value.error_code
            is DocumentProcessingErrorCode.PDF_PROCESS_LAUNCH_FAILED
        )
        assert failure.value.pid is None

    asyncio.run(run())


def test_worker_technical_traceback_is_logged_but_safe_error_is_stable(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        command, behavior_path, _, _ = _persistent_command(tmp_path)
        behavior_path.write_text("failure", encoding="ascii")
        worker = PersistentPdfWorker(
            tokenizer_id="unused",
            tokenizer_revision="unused",
            worker_marker_path=tmp_path / "worker.json",
            command=command,
        )
        with capture_logs() as logs:
            with pytest.raises(PdfSubprocessFailure) as failure:
                await worker.process(
                    _persistent_job(7, tmp_path / "failure.json"),
                    timeout_seconds=2,
                    on_chunking=lambda: _record_stage([], 1),
                )
        assert failure.value.error_code is DocumentProcessingErrorCode.PDF_PARSE_FAILED
        diagnostics = "\n".join(str(entry.get("diagnostic", "")) for entry in logs)
        assert "RuntimeError" in diagnostics
        assert "controlled failure" in diagnostics
        diagnostic_entries = [
            entry
            for entry in logs
            if entry.get("event") == "knowledge.document_processing.worker_diagnostic"
        ]
        assert diagnostic_entries
        assert all(entry.get("job_id") == 7 for entry in diagnostic_entries)
        await worker.stop()

    asyncio.run(run())


def test_dispatcher_never_executes_more_than_one_local_claim(monkeypatch) -> None:
    source_ref = DocumentSourceRef(
        document_version_id=1,
        storage_key="users/1/sources/00000000000000000000000000000001/source",
        source_media_type="application/pdf",
        source_sha256="a" * 64,
        source_size_bytes=10,
    )
    claims = deque(
        DocumentProcessingClaim(
            job_id=job_id,
            document_version_id=job_id,
            knowledge_base_id=1,
            user_id=1,
            attempt_no=1,
            recipe_id=PDF_PROCESSING_RECIPE_ID,
            source_ref=source_ref,
        )
        for job_id in (1, 2)
    )
    release_first = asyncio.Event()
    second_started = asyncio.Event()
    active = 0
    maximum_active = 0

    async def fake_claim(_session_factory):
        return claims.popleft() if claims else None

    class Runtime:
        async def execute(self, claim) -> None:
            nonlocal active, maximum_active
            active += 1
            maximum_active = max(maximum_active, active)
            if claim.job_id == 1:
                await release_first.wait()
            else:
                second_started.set()
            active -= 1

    monkeypatch.setattr(
        processing_module, "claim_next_document_processing_attempt", fake_claim
    )

    async def run() -> None:
        dispatcher = DocumentProcessingDispatcher(object(), Runtime())
        dispatcher.start()
        await asyncio.sleep(0)
        assert not second_started.is_set()
        release_first.set()
        await asyncio.wait_for(second_started.wait(), timeout=1)
        await dispatcher.stop()
        assert maximum_active == 1

    asyncio.run(run())
