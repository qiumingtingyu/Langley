"""Pure, deterministic integrity validation for the Task 3 Golden fixture set."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

DATASET_ROOT = Path(__file__).parents[1] / "fixtures" / "knowledge" / "retrieval"
ALLOWED_LANGUAGES = {"en", "ja", "zh"}
ALLOWED_SOURCE_FAMILIES = {
    "fastapi_docs",
    "langley_project_fixture",
    "scifact",
    "xiaolin_personal_markdown",
}
APPROVED_CASE_IDS = {
    "fastapi-002",
    "scifact-001",
    "scifact-002",
    "scifact-003",
    "scifact-004",
    "scifact-005",
    "scifact-006",
    "xiaolin-cn-001",
    "xiaolin-cn-002",
    "xiaolin-cn-003",
    "xiaolin-co-001",
    "xiaolin-co-002",
    "xiaolin-os-001",
}


def _require_string(value: object, label: str, *, nonblank: bool = False) -> str:
    if not isinstance(value, str):
        raise AssertionError(f"{label} must be a string")
    if nonblank and not value.strip():
        raise AssertionError(f"{label} must be nonblank")
    return value


def _safe_document_path(root: Path, value: object, document_key: str) -> Path:
    relative = _require_string(value, f"document {document_key}.path", nonblank=True)
    pure_path = PurePosixPath(relative)
    if (
        pure_path.is_absolute()
        or ".." in pure_path.parts
        or pure_path.parts[:1] != ("documents",)
    ):
        raise AssertionError(f"document {document_key} has unsafe path {relative!r}")
    if pure_path.suffix != ".md":
        raise AssertionError(f"document {document_key} must reference a Markdown file")
    root_resolved = root.resolve()
    resolved = (root / Path(*pure_path.parts)).resolve()
    if not resolved.is_relative_to(root_resolved / "documents"):
        raise AssertionError(f"document {document_key} escapes documents directory")
    return resolved


def _validate_document_provenance(
    provenance: object, document_key: str
) -> dict[str, Any]:
    if not isinstance(provenance, dict):
        raise AssertionError(f"document {document_key}.provenance must be an object")
    family = _require_string(
        provenance.get("source_family"),
        f"document {document_key}.provenance.source_family",
    )
    _require_string(
        provenance.get("origin"),
        f"document {document_key}.provenance.origin",
        nonblank=True,
    )
    if family not in ALLOWED_SOURCE_FAMILIES:
        raise AssertionError(
            f"document {document_key} has unknown source family {family!r}"
        )
    if family in {"scifact", "fastapi_docs"}:
        upstream = provenance.get("upstream")
        if not isinstance(upstream, dict):
            raise AssertionError(
                f"document {document_key} requires upstream provenance"
            )
        _require_string(
            upstream.get("repository"),
            f"document {document_key}.provenance.upstream.repository",
            nonblank=True,
        )
        _require_string(
            upstream.get("revision"),
            f"document {document_key}.provenance.upstream.revision",
            nonblank=True,
        )
        _require_string(
            upstream.get("content_license"),
            f"document {document_key}.provenance.upstream.content_license",
            nonblank=True,
        )
    return provenance


def validate_dataset(root: Path = DATASET_ROOT) -> None:
    """Validate only local fixture bytes and manifest semantics; no network or DB."""
    payload = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError("dataset root must be an object")
    if payload.get("format_version") != 1:
        raise AssertionError("dataset format_version must be 1")
    _require_string(payload.get("name"), "dataset name", nonblank=True)
    _require_string(payload.get("approval"), "dataset approval", nonblank=True)
    documents = payload.get("documents")
    cases = payload.get("cases")
    if not isinstance(documents, list) or not documents:
        raise AssertionError("dataset documents must be a non-empty list")
    if not isinstance(cases, list) or not cases:
        raise AssertionError("dataset cases must be a non-empty list")

    document_paths: set[Path] = set()
    documents_by_key: dict[str, Path] = {}
    document_families: dict[str, str] = {}
    for document in documents:
        if not isinstance(document, dict):
            raise AssertionError("each document must be an object")
        key = _require_string(document.get("key"), "document key", nonblank=True)
        if key in documents_by_key:
            raise AssertionError(f"duplicate document key {key!r}")
        if document.get("language") not in ALLOWED_LANGUAGES:
            raise AssertionError(f"document {key} has unsupported language")
        path = _safe_document_path(root, document.get("path"), key)
        if not path.is_file():
            raise AssertionError(f"document {key} is missing: {path}")
        source_bytes = path.read_bytes()
        try:
            source_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise AssertionError(f"document {key} is not strict UTF-8") from error
        if b"\r" in source_bytes:
            raise AssertionError(f"document {key} is not LF-only")
        expected_sha256 = _require_string(
            document.get("sha256"), f"document {key}.sha256", nonblank=True
        )
        actual_sha256 = hashlib.sha256(source_bytes).hexdigest()
        if actual_sha256 != expected_sha256:
            raise AssertionError(f"document {key} SHA-256 mismatch: {actual_sha256}")
        document_provenance = _validate_document_provenance(
            document.get("provenance"), key
        )
        document_paths.add(path)
        documents_by_key[key] = path
        document_families[key] = _require_string(
            document_provenance["source_family"],
            f"document {key}.provenance.source_family",
        )

    actual_paths = set((root / "documents").glob("*.md"))
    if actual_paths != document_paths:
        raise AssertionError(
            "dataset document paths do not exactly match documents/*.md"
        )

    case_ids: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise AssertionError("each case must be an object")
        case_id = _require_string(case.get("case_id"), "case_id", nonblank=True)
        if case_id in case_ids:
            raise AssertionError(f"duplicate case_id {case_id!r}")
        case_ids.add(case_id)
        _require_string(case.get("query"), f"case {case_id}.query", nonblank=True)
        document_key = _require_string(
            case.get("document_key"), f"case {case_id}.document_key", nonblank=True
        )
        if document_key not in documents_by_key:
            raise AssertionError(
                f"case {case_id} references missing document {document_key!r}"
            )
        provenance = case.get("provenance")
        if not isinstance(provenance, dict):
            raise AssertionError(f"case {case_id}.provenance must be an object")
        family = _require_string(
            provenance.get("source_family"),
            f"case {case_id}.provenance.source_family",
            nonblank=True,
        )
        if family not in ALLOWED_SOURCE_FAMILIES:
            raise AssertionError(f"case {case_id} has unknown source family {family!r}")
        if family != document_families[document_key]:
            raise AssertionError(
                f"case {case_id} source family does not match {document_key}"
            )
        if "source_query_id" in provenance and not isinstance(
            provenance["source_query_id"], int
        ):
            raise AssertionError(
                f"case {case_id}.provenance.source_query_id must be an integer"
            )
        evidence = case.get("evidence")
        if not isinstance(evidence, dict):
            raise AssertionError(f"case {case_id}.evidence must be an object")
        start_byte = evidence.get("start_byte")
        end_byte = evidence.get("end_byte")
        if isinstance(start_byte, bool) or not isinstance(start_byte, int):
            raise AssertionError(
                f"case {case_id}.evidence.start_byte must be an integer"
            )
        if isinstance(end_byte, bool) or not isinstance(end_byte, int):
            raise AssertionError(f"case {case_id}.evidence.end_byte must be an integer")
        source_bytes = documents_by_key[document_key].read_bytes()
        if not 0 <= start_byte < end_byte <= len(source_bytes):
            raise AssertionError(f"case {case_id} has invalid evidence bounds")
        evidence_text = _require_string(
            evidence.get("text"), f"case {case_id}.evidence.text", nonblank=True
        )
        try:
            actual_text = source_bytes[start_byte:end_byte].decode("utf-8")
        except UnicodeDecodeError as error:
            raise AssertionError(
                f"case {case_id} evidence does not align to UTF-8 boundaries"
            ) from error
        if actual_text != evidence_text:
            raise AssertionError(
                f"case {case_id} evidence text does not match source bytes"
            )

    if case_ids != APPROVED_CASE_IDS:
        raise AssertionError(f"approved case set mismatch: {sorted(case_ids)}")


def test_retrieval_golden_dataset_is_valid() -> None:
    validate_dataset()


def test_retrieval_golden_dataset_has_approved_composition() -> None:
    payload = json.loads((DATASET_ROOT / "dataset.json").read_text(encoding="utf-8"))
    assert len(payload["documents"]) == 40
    assert len(payload["cases"]) == 13
    families = [case["provenance"]["source_family"] for case in payload["cases"]]
    assert families.count("scifact") == 6
    assert families.count("xiaolin_personal_markdown") == 6
    assert families.count("fastapi_docs") == 1
    assert families.count("langley_project_fixture") == 0
