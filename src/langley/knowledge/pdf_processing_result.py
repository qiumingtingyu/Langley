"""Langley-owned detached schema for PDF parser/chunker staging results."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from langley.knowledge.chunking import CandidateChunk
from langley.knowledge.contracts import (
    PdfPageRegion,
    encode_source_region,
    validate_heading_path,
    validate_source_regions,
)

PDF_PROCESSING_RESULT_SCHEMA_VERSION = "pdf_processing_result_v1"
_MAX_RESULT_BYTES = 64 * 1024 * 1024
_RESULT_KEYS = {
    "schema_version",
    "recipe_id",
    "job_id",
    "attempt_no",
    "document_version_id",
    "source_sha256",
    "source_size_bytes",
    "page_count",
    "chunks",
}
_CHUNK_KEYS = {"ordinal", "content", "heading_path", "source_regions"}


class PdfProcessingResultInvalid(ValueError):
    """A detached worker result does not satisfy the Langley boundary."""


@dataclass(frozen=True)
class PdfProcessingResult:
    page_count: int
    candidates: tuple[CandidateChunk, ...]


def coalesce_pdf_page_regions(page_numbers: set[int]) -> tuple[PdfPageRegion, ...]:
    """Convert unique 1-based pages into deterministic contiguous ranges."""
    if not page_numbers or any(
        type(page) is not int or page < 1 for page in page_numbers
    ):
        raise ValueError("invalid PDF pages")
    ordered = sorted(page_numbers)
    ranges: list[PdfPageRegion] = []
    start = end = ordered[0]
    for page in ordered[1:]:
        if page == end + 1:
            end = page
            continue
        ranges.append(PdfPageRegion(page_start=start, page_end=end))
        start = end = page
    ranges.append(PdfPageRegion(page_start=start, page_end=end))
    return tuple(ranges)


def write_pdf_processing_result(
    output_path: Path,
    *,
    recipe_id: str,
    job_id: int,
    attempt_no: int,
    document_version_id: int,
    source_sha256: str,
    source_size_bytes: int,
    page_count: int,
    candidates: tuple[CandidateChunk, ...],
) -> None:
    """Atomically write one JSON result without serializing Docling objects."""
    payload = {
        "schema_version": PDF_PROCESSING_RESULT_SCHEMA_VERSION,
        "recipe_id": recipe_id,
        "job_id": job_id,
        "attempt_no": attempt_no,
        "document_version_id": document_version_id,
        "source_sha256": source_sha256,
        "source_size_bytes": source_size_bytes,
        "page_count": page_count,
        "chunks": [
            {
                "ordinal": candidate.ordinal,
                "content": candidate.content,
                "heading_path": list(candidate.heading_path),
                "source_regions": [
                    encode_source_region(region) for region in candidate.source_regions
                ],
            }
            for candidate in candidates
        ],
    }
    temporary_path = output_path.with_suffix(".json.tmp")
    with temporary_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary_path, output_path)


def load_pdf_processing_result(
    result_path: Path,
    *,
    expected_recipe_id: str,
    expected_job_id: int,
    expected_attempt_no: int,
    expected_document_version_id: int,
    expected_source_sha256: str,
    expected_source_size_bytes: int,
) -> PdfProcessingResult:
    """Load and strictly validate one job/source-bound staging result."""
    try:
        size = result_path.stat().st_size
        if not 0 < size <= _MAX_RESULT_BYTES:
            raise PdfProcessingResultInvalid("invalid result size")
        value = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PdfProcessingResultInvalid("result cannot be read") from error
    if not isinstance(value, dict) or set(value) != _RESULT_KEYS:
        raise PdfProcessingResultInvalid("invalid result object")
    expected = {
        "schema_version": PDF_PROCESSING_RESULT_SCHEMA_VERSION,
        "recipe_id": expected_recipe_id,
        "job_id": expected_job_id,
        "attempt_no": expected_attempt_no,
        "document_version_id": expected_document_version_id,
        "source_sha256": expected_source_sha256,
        "source_size_bytes": expected_source_size_bytes,
    }
    for key in ("job_id", "attempt_no", "document_version_id", "source_size_bytes"):
        if type(value[key]) is not int or value[key] <= 0:
            raise PdfProcessingResultInvalid("result numeric identity mismatch")
    if any(value[key] != expected_value for key, expected_value in expected.items()):
        raise PdfProcessingResultInvalid("result identity mismatch")
    page_count = value["page_count"]
    chunks = value["chunks"]
    if type(page_count) is not int or page_count < 1:
        raise PdfProcessingResultInvalid("invalid page count")
    if not isinstance(chunks, list) or not chunks:
        raise PdfProcessingResultInvalid("invalid chunks")

    candidates: list[CandidateChunk] = []
    for expected_ordinal, chunk in enumerate(chunks, start=1):
        try:
            if not isinstance(chunk, dict) or set(chunk) != _CHUNK_KEYS:
                raise ValueError
            ordinal = chunk["ordinal"]
            content = chunk["content"]
            if ordinal != expected_ordinal or type(ordinal) is not int:
                raise ValueError
            if not isinstance(content, str) or not content.strip():
                raise ValueError
            heading_path = tuple(validate_heading_path(chunk["heading_path"]))
            regions = tuple(validate_source_regions(chunk["source_regions"]))
            pdf_regions = tuple(
                region for region in regions if isinstance(region, PdfPageRegion)
            )
            if len(pdf_regions) != len(regions):
                raise ValueError
            if any(region.page_end > page_count for region in pdf_regions):
                raise ValueError
        except (KeyError, TypeError, ValueError) as error:
            raise PdfProcessingResultInvalid(
                f"invalid chunk at ordinal {expected_ordinal}"
            ) from error
        candidates.append(
            CandidateChunk(
                ordinal=ordinal,
                content=content,
                heading_path=heading_path,
                source_regions=pdf_regions,
            )
        )
    return PdfProcessingResult(page_count=page_count, candidates=tuple(candidates))
