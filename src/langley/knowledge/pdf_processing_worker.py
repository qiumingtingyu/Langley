"""DB-free persistent Docling worker for serial detached processing attempts."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, TypedDict

from langley.knowledge.chunking import CandidateChunk
from langley.knowledge.pdf_processing_result import (
    coalesce_pdf_page_regions,
    write_pdf_processing_result,
)


def _emit(value: dict[str, object]) -> None:
    print(json.dumps(value, separators=(",", ":")), flush=True)


def _source_matches(path: Path, expected_size: int, expected_sha256: str) -> bool:
    try:
        if path.stat().st_size != expected_size:
            return False
        digest = sha256()
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
        return digest.hexdigest() == expected_sha256
    except OSError:
        return False


def _converter():
    accelerator = import_module("docling.datamodel.accelerator_options")
    base_models = import_module("docling.datamodel.base_models")
    pipeline_options = import_module("docling.datamodel.pipeline_options")
    converter_module = import_module("docling.document_converter")
    accelerator_device = accelerator.AcceleratorDevice
    accelerator_options = accelerator.AcceleratorOptions
    input_format = base_models.InputFormat
    heading_options = pipeline_options.HeadingHierarchyOptions
    pdf_options = pipeline_options.PdfPipelineOptions

    options = pdf_options(
        do_ocr=False,
        do_table_structure=True,
        generate_page_images=False,
        generate_picture_images=False,
        enable_remote_services=False,
        accelerator_options=accelerator_options(
            num_threads=4, device=accelerator_device.CPU
        ),
        heading_hierarchy_options=heading_options(
            enabled=True,
            use_bookmarks=True,
            use_numbering=True,
            use_style=True,
            use_font_style=True,
            max_level=6,
        ),
    )
    options.table_structure_options.do_cell_matching = True
    return converter_module.DocumentConverter(
        allowed_formats=[input_format.PDF],
        format_options={
            input_format.PDF: converter_module.PdfFormatOption(pipeline_options=options)
        },
    )


def _persistent_chunker(tokenizer_id: str, tokenizer_revision: str):
    from transformers import AutoTokenizer

    chunking = import_module("docling.chunking")
    tokenizer_module = import_module(
        "docling_core.transforms.chunker.tokenizer.huggingface"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id,
        revision=tokenizer_revision,
    )
    return chunking.HybridChunker(
        tokenizer=tokenizer_module.HuggingFaceTokenizer(
            tokenizer=tokenizer, max_tokens=512
        ),
        merge_peers=True,
        repeat_table_header=True,
        omit_header_on_overflow=False,
    )


def _chunk_with_runtime(document: Any, chunker: Any) -> tuple[CandidateChunk, ...]:
    candidates: list[CandidateChunk] = []
    for ordinal, chunk in enumerate(chunker.chunk(document), start=1):
        pages = {
            int(provenance.page_no)
            for item in chunk.meta.doc_items
            for provenance in (item.prov or ())
        }
        candidates.append(
            CandidateChunk(
                ordinal=ordinal,
                content=chunk.text.strip(),
                heading_path=tuple(chunk.meta.headings or ()),
                source_regions=coalesce_pdf_page_regions(pages),
            )
        )
    return tuple(candidates)


class _WarmPdfRuntime:
    """Lazy worker-local caches without retaining per-document mutable state."""

    def __init__(self, tokenizer_id: str, tokenizer_revision: str) -> None:
        self._tokenizer_id = tokenizer_id
        self._tokenizer_revision = tokenizer_revision
        self._converter_instance: Any | None = None
        self._chunker_instance: Any | None = None

    def converter(self) -> Any:
        if self._converter_instance is None:
            self._converter_instance = _converter()
        return self._converter_instance

    def chunker(self) -> Any:
        if self._chunker_instance is None:
            self._chunker_instance = _persistent_chunker(
                self._tokenizer_id, self._tokenizer_revision
            )
        return self._chunker_instance


_PERSISTENT_PROCESS_KEYS = {
    "command",
    "job_id",
    "attempt_no",
    "document_version_id",
    "source_path",
    "source_sha256",
    "source_size_bytes",
    "recipe_id",
    "staging_path",
}


class _PersistentProcessCommand(TypedDict):
    command: Literal["process"]
    job_id: int
    attempt_no: int
    document_version_id: int
    source_path: str
    source_sha256: str
    source_size_bytes: int
    recipe_id: str
    staging_path: str


def _required_positive_integer(value: dict[object, object], key: str) -> int:
    candidate = value.get(key)
    if type(candidate) is not int or candidate <= 0:
        raise ValueError("invalid worker numeric identity")
    return candidate


def _required_string(value: dict[object, object], key: str) -> str:
    candidate = value.get(key)
    if not isinstance(candidate, str) or not candidate:
        raise ValueError("invalid worker string identity")
    return candidate


def _validated_persistent_command(value: object) -> _PersistentProcessCommand:
    if not isinstance(value, dict) or set(value) != _PERSISTENT_PROCESS_KEYS:
        raise ValueError("invalid worker command")
    if value.get("command") != "process":
        raise ValueError("invalid worker command")
    return _PersistentProcessCommand(
        command="process",
        job_id=_required_positive_integer(value, "job_id"),
        attempt_no=_required_positive_integer(value, "attempt_no"),
        document_version_id=_required_positive_integer(value, "document_version_id"),
        source_path=_required_string(value, "source_path"),
        source_sha256=_required_string(value, "source_sha256"),
        source_size_bytes=_required_positive_integer(value, "source_size_bytes"),
        recipe_id=_required_string(value, "recipe_id"),
        staging_path=_required_string(value, "staging_path"),
    )


def _job_event(
    command: _PersistentProcessCommand, **values: object
) -> dict[str, object]:
    return {
        **values,
        "job_id": command["job_id"],
        "attempt_no": command["attempt_no"],
        "document_version_id": command["document_version_id"],
        "pid": os.getpid(),
    }


def _diagnose_worker_exception(
    command: _PersistentProcessCommand, *, stage: str, error: BaseException
) -> None:
    print(
        json.dumps(
            _job_event(
                command,
                event="worker_exception",
                stage=stage,
                exception_type=type(error).__name__,
            ),
            separators=(",", ":"),
        ),
        file=sys.stderr,
        flush=True,
    )
    traceback.print_exception(type(error), error, error.__traceback__, file=sys.stderr)
    sys.stderr.flush()


def _process_persistent_command(
    command: _PersistentProcessCommand, runtime: _WarmPdfRuntime
) -> None:
    started = perf_counter()
    source_path = Path(command["source_path"])
    source_size_bytes = command["source_size_bytes"]
    source_sha256 = command["source_sha256"]
    if not _source_matches(source_path, source_size_bytes, source_sha256):
        _emit(
            _job_event(
                command,
                event="error",
                error_code="SOURCE_CHANGED_DURING_PROCESSING",
            )
        )
        return

    _emit(_job_event(command, event="stage", stage="PARSING"))
    parse_started = perf_counter()
    try:
        document = (
            runtime.converter().convert(source_path, raises_on_error=True).document
        )
    except MemoryError as error:
        _diagnose_worker_exception(command, stage="PARSING", error=error)
        _emit(
            _job_event(
                command,
                event="error",
                error_code="PDF_PROCESS_RESOURCE_LIMIT",
            )
        )
        return
    except Exception as error:
        _diagnose_worker_exception(command, stage="PARSING", error=error)
        _emit(_job_event(command, event="error", error_code="PDF_PARSE_FAILED"))
        return
    parse_ms = round((perf_counter() - parse_started) * 1000, 3)

    _emit(_job_event(command, event="stage", stage="CHUNKING"))
    chunk_started = perf_counter()
    try:
        candidates = _chunk_with_runtime(document, runtime.chunker())
    except MemoryError as error:
        _diagnose_worker_exception(command, stage="CHUNKING", error=error)
        _emit(
            _job_event(
                command,
                event="error",
                error_code="PDF_PROCESS_RESOURCE_LIMIT",
            )
        )
        return
    except Exception as error:
        _diagnose_worker_exception(command, stage="CHUNKING", error=error)
        _emit(_job_event(command, event="error", error_code="PDF_CHUNKING_FAILED"))
        return
    chunk_ms = round((perf_counter() - chunk_started) * 1000, 3)

    try:
        write_pdf_processing_result(
            Path(command["staging_path"]),
            recipe_id=command["recipe_id"],
            job_id=command["job_id"],
            attempt_no=command["attempt_no"],
            document_version_id=command["document_version_id"],
            source_sha256=source_sha256,
            source_size_bytes=source_size_bytes,
            page_count=len(document.pages),
            candidates=candidates,
        )
    except Exception as error:
        _diagnose_worker_exception(command, stage="CHUNKING", error=error)
        _emit(_job_event(command, event="error", error_code="PDF_OUTPUT_INVALID"))
        return
    _emit(
        _job_event(
            command,
            event="completed",
            page_count=len(document.pages),
            chunk_count=len(candidates),
            parse_ms=parse_ms,
            chunk_ms=chunk_ms,
            total_ms=round((perf_counter() - started) * 1000, 3),
        )
    )


def _write_worker_marker(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps({"pid": os.getpid()}, separators=(",", ":")),
        encoding="ascii",
        newline="\n",
    )
    os.replace(temporary, path)


def _remove_own_worker_marker(path: Path) -> None:
    try:
        if json.loads(path.read_text(encoding="ascii")) == {"pid": os.getpid()}:
            path.unlink(missing_ok=True)
    except (OSError, json.JSONDecodeError):
        pass


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--persistent", action="store_true", required=True)
    parser.add_argument("--tokenizer-id", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--worker-marker-path", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    runtime = _WarmPdfRuntime(arguments.tokenizer_id, arguments.tokenizer_revision)
    try:
        _write_worker_marker(arguments.worker_marker_path)
        _emit({"event": "ready", "pid": os.getpid()})
        for line in sys.stdin:
            try:
                value = json.loads(line)
                if value == {"command": "shutdown"}:
                    return 0
                command = _validated_persistent_command(value)
            except Exception as error:
                print(
                    json.dumps(
                        {
                            "event": "worker_protocol_error",
                            "exception_type": type(error).__name__,
                        },
                        separators=(",", ":"),
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                return 2
            _process_persistent_command(command, runtime)
        return 0
    finally:
        _remove_own_worker_marker(arguments.worker_marker_path)


if __name__ == "__main__":
    raise SystemExit(main())
