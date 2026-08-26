"""Deterministic validation for the frozen Knowledge Casebook V1 assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ANSWERABLE = "ANSWERABLE"
_UNANSWERABLE = "UNANSWERABLE"
_ANSWER_TASKS = {"DIRECT_FACT", "EXPLANATION", "COMPARISON", "PROCEDURE"}
_KNOWLEDGE_SEARCH_EXPECTATIONS = {"REQUIRED", "FORBIDDEN"}
_QUERY_ORIGINS = {
    "HUMAN_AUTHORED",
    "UPSTREAM_CLAIM_NARROWED",
    "UPSTREAM_CLAIM_TRANSLATED",
}
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_EXPECTED_DOCUMENT_COUNT = 40
_EXPECTED_CASE_COUNT = 25


class CasebookValidationError(ValueError):
    """A frozen Casebook or Corpus Manifest invariant was violated."""


@dataclass(frozen=True)
class CasebookValidationSummary:
    """Small deterministic success result suitable for a CLI gate."""

    casebook_id: str
    case_count: int
    document_count: int
    supporting_evidence_count: int
    near_miss_evidence_count: int


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CasebookValidationError(f"cannot read {label} {path}: {error}") from error
    return _object(value, label)


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CasebookValidationError(f"{label} must be an object")
    return value


def _array(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise CasebookValidationError(f"{label} must be an array")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CasebookValidationError(f"{label} must be a nonblank string")
    return value


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CasebookValidationError(f"{label} must be an integer")
    return value


def _require_exact_keys(
    value: dict[str, Any], *, required: set[str], optional: set[str], label: str
) -> None:
    keys = set(value)
    missing = sorted(required - keys)
    unexpected = sorted(keys - required - optional)
    if missing:
        raise CasebookValidationError(f"{label} is missing keys: {', '.join(missing)}")
    if unexpected:
        raise CasebookValidationError(
            f"{label} has unexpected keys: {', '.join(unexpected)}"
        )


def _validate_manifest(
    manifest: dict[str, Any], *, corpus_root: Path
) -> tuple[str, dict[str, bytes]]:
    _require_exact_keys(
        manifest,
        required={"manifest_id", "status", "document_count", "documents"},
        optional=set(),
        label="Corpus Manifest",
    )
    manifest_id = _string(manifest["manifest_id"], "manifest_id")
    if manifest["status"] != "FROZEN":
        raise CasebookValidationError("Corpus Manifest status must be FROZEN")
    documents = _array(manifest["documents"], "documents")
    document_count = _integer(manifest["document_count"], "document_count")
    if document_count != len(documents) or document_count != _EXPECTED_DOCUMENT_COUNT:
        raise CasebookValidationError(
            f"Corpus Manifest must contain exactly {_EXPECTED_DOCUMENT_COUNT} documents"
        )

    resolved_root = corpus_root.resolve(strict=True)
    source_by_key: dict[str, bytes] = {}
    seen_paths: set[Path] = set()
    for index, raw_document in enumerate(documents):
        document = _object(raw_document, f"documents[{index}]")
        _require_exact_keys(
            document,
            required={"key", "path", "sha256", "language", "provenance"},
            optional=set(),
            label=f"documents[{index}]",
        )
        key = _string(document["key"], f"documents[{index}].key")
        if key in source_by_key:
            raise CasebookValidationError(f"duplicate document key {key!r}")
        relative_path = Path(_string(document["path"], f"document {key}.path"))
        if relative_path.is_absolute():
            raise CasebookValidationError(f"document {key} path must be relative")
        try:
            source_path = (resolved_root / relative_path).resolve(strict=True)
            source_path.relative_to(resolved_root)
        except (OSError, ValueError) as error:
            raise CasebookValidationError(
                f"document {key} path is missing or outside the corpus root"
            ) from error
        if source_path in seen_paths:
            raise CasebookValidationError(f"duplicate document path {relative_path}")
        seen_paths.add(source_path)

        expected_sha = _string(document["sha256"], f"document {key}.sha256")
        if _SHA256_PATTERN.fullmatch(expected_sha) is None:
            raise CasebookValidationError(f"document {key} has invalid SHA-256")
        _string(document["language"], f"document {key}.language")
        _object(document["provenance"], f"document {key}.provenance")
        try:
            source_bytes = source_path.read_bytes()
        except OSError as error:
            raise CasebookValidationError(
                f"cannot read document {key}: {error}"
            ) from error
        actual_sha = hashlib.sha256(source_bytes).hexdigest()
        if actual_sha != expected_sha:
            raise CasebookValidationError(
                f"document {key} SHA-256 mismatch: expected {expected_sha}, "
                f"got {actual_sha}"
            )
        try:
            source_bytes.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise CasebookValidationError(
                f"document {key} is not strict UTF-8"
            ) from error
        source_by_key[key] = source_bytes

    return manifest_id, source_by_key


def _validate_evidence(
    raw_evidence: object,
    *,
    label: str,
    source_by_key: dict[str, bytes],
) -> tuple[str, str]:
    evidence = _object(raw_evidence, label)
    _require_exact_keys(
        evidence,
        required={
            "evidence_id",
            "document_key",
            "start_byte",
            "end_byte",
            "evidence_text",
        },
        optional=set(),
        label=label,
    )
    evidence_id = _string(evidence["evidence_id"], f"{label}.evidence_id")
    document_key = _string(evidence["document_key"], f"{label}.document_key")
    source_bytes = source_by_key.get(document_key)
    if source_bytes is None:
        raise CasebookValidationError(
            f"{label} references unknown document key {document_key!r}"
        )
    start = _integer(evidence["start_byte"], f"{label}.start_byte")
    end = _integer(evidence["end_byte"], f"{label}.end_byte")
    if start < 0 or end <= start or end > len(source_bytes):
        raise CasebookValidationError(f"{label} has an invalid byte span")
    expected_text = _string(evidence["evidence_text"], f"{label}.evidence_text")
    try:
        actual_text = source_bytes[start:end].decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise CasebookValidationError(
            f"{label} byte span is not strict UTF-8"
        ) from error
    if actual_text != expected_text:
        raise CasebookValidationError(
            f"{label} evidence_text does not match source bytes"
        )
    return evidence_id, document_key


def _validate_case(
    raw_case: object,
    *,
    index: int,
    source_by_key: dict[str, bytes],
) -> tuple[str, int, int]:
    case = _object(raw_case, f"cases[{index}]")
    _require_exact_keys(
        case,
        required={
            "case_id",
            "query",
            "query_language",
            "answerability",
            "answer_task",
            "supporting_evidence",
            "required_answer_points",
            "case_provenance",
        },
        optional={
            "near_miss_evidence",
            "reference_answer",
            "evidence_gap",
            "agent_expectation",
        },
        label=f"cases[{index}]",
    )
    case_id = _string(case["case_id"], f"cases[{index}].case_id")
    _string(case["query"], f"case {case_id}.query")
    _string(case["query_language"], f"case {case_id}.query_language")
    answerability = _string(case["answerability"], f"case {case_id}.answerability")
    if answerability not in {_ANSWERABLE, _UNANSWERABLE}:
        raise CasebookValidationError(f"case {case_id} has invalid answerability")
    answer_task = _string(case["answer_task"], f"case {case_id}.answer_task")
    if answer_task not in _ANSWER_TASKS:
        raise CasebookValidationError(f"case {case_id} has invalid answer_task")

    provenance = _object(case["case_provenance"], f"case {case_id}.case_provenance")
    _require_exact_keys(
        provenance,
        required={"query_origin"},
        optional={"upstream_claim_id"},
        label=f"case {case_id}.case_provenance",
    )
    query_origin = _string(provenance["query_origin"], f"case {case_id}.query_origin")
    if query_origin not in _QUERY_ORIGINS:
        raise CasebookValidationError(
            f"case {case_id} has unsupported query_origin {query_origin!r}"
        )
    if "upstream_claim_id" in provenance:
        _integer(provenance["upstream_claim_id"], f"case {case_id}.upstream_claim_id")

    expectation = case.get("agent_expectation")
    if expectation is not None:
        expectation_object = _object(expectation, f"case {case_id}.agent_expectation")
        _require_exact_keys(
            expectation_object,
            required={"knowledge_search"},
            optional=set(),
            label=f"case {case_id}.agent_expectation",
        )
        if expectation_object["knowledge_search"] not in (
            _KNOWLEDGE_SEARCH_EXPECTATIONS
        ):
            raise CasebookValidationError(
                f"case {case_id} has invalid knowledge_search expectation"
            )

    supporting = _array(
        case["supporting_evidence"], f"case {case_id}.supporting_evidence"
    )
    near_miss = _array(
        case.get("near_miss_evidence", []), f"case {case_id}.near_miss_evidence"
    )
    supporting_ids: set[str] = set()
    all_evidence_ids: set[str] = set()
    for evidence_index, raw_evidence in enumerate(supporting):
        evidence_id, _ = _validate_evidence(
            raw_evidence,
            label=f"case {case_id}.supporting_evidence[{evidence_index}]",
            source_by_key=source_by_key,
        )
        if evidence_id in all_evidence_ids:
            raise CasebookValidationError(
                f"case {case_id} has duplicate evidence_id {evidence_id!r}"
            )
        all_evidence_ids.add(evidence_id)
        supporting_ids.add(evidence_id)
    for evidence_index, raw_evidence in enumerate(near_miss):
        evidence_id, _ = _validate_evidence(
            raw_evidence,
            label=f"case {case_id}.near_miss_evidence[{evidence_index}]",
            source_by_key=source_by_key,
        )
        if evidence_id in all_evidence_ids:
            raise CasebookValidationError(
                f"case {case_id} has duplicate evidence_id {evidence_id!r}"
            )
        all_evidence_ids.add(evidence_id)

    points = _array(
        case["required_answer_points"], f"case {case_id}.required_answer_points"
    )
    seen_point_ids: set[str] = set()
    used_supporting_ids: set[str] = set()
    for point_index, raw_point in enumerate(points):
        point = _object(
            raw_point, f"case {case_id}.required_answer_points[{point_index}]"
        )
        _require_exact_keys(
            point,
            required={"point_id", "text", "evidence_ids"},
            optional=set(),
            label=f"case {case_id}.required_answer_points[{point_index}]",
        )
        point_id = _string(point["point_id"], f"case {case_id}.point_id")
        if point_id in seen_point_ids:
            raise CasebookValidationError(
                f"case {case_id} has duplicate point_id {point_id!r}"
            )
        seen_point_ids.add(point_id)
        _string(point["text"], f"case {case_id} point {point_id}.text")
        evidence_ids = _array(
            point["evidence_ids"], f"case {case_id} point {point_id}.evidence_ids"
        )
        if not evidence_ids:
            raise CasebookValidationError(
                f"case {case_id} point {point_id} needs supporting evidence"
            )
        for raw_evidence_id in evidence_ids:
            evidence_id = _string(
                raw_evidence_id,
                f"case {case_id} point {point_id}.evidence_ids item",
            )
            if evidence_id not in supporting_ids:
                raise CasebookValidationError(
                    f"case {case_id} point {point_id} references non-supporting "
                    f"evidence {evidence_id!r}"
                )
            used_supporting_ids.add(evidence_id)

    if answerability == _ANSWERABLE:
        if not supporting or not points:
            raise CasebookValidationError(
                f"ANSWERABLE case {case_id} needs evidence and answer points"
            )
        _string(case.get("reference_answer"), f"case {case_id}.reference_answer")
        if "evidence_gap" in case:
            raise CasebookValidationError(
                f"ANSWERABLE case {case_id} must not have evidence_gap"
            )
        unused = sorted(supporting_ids - used_supporting_ids)
        if unused:
            raise CasebookValidationError(
                f"case {case_id} has unused supporting evidence: {', '.join(unused)}"
            )
    else:
        if supporting or points:
            raise CasebookValidationError(
                f"UNANSWERABLE case {case_id} must not have supporting evidence "
                "or points"
            )
        if "reference_answer" in case:
            raise CasebookValidationError(
                f"UNANSWERABLE case {case_id} must not have reference_answer"
            )
        _string(case.get("evidence_gap"), f"case {case_id}.evidence_gap")

    return case_id, len(supporting), len(near_miss)


def validate_casebook(
    casebook_path: Path, manifest_path: Path, corpus_root: Path
) -> CasebookValidationSummary:
    """Validate V1 structure and every Manifest/evidence byte identity."""

    casebook = _load_json(casebook_path, "Casebook")
    manifest = _load_json(manifest_path, "Corpus Manifest")
    manifest_id, source_by_key = _validate_manifest(
        manifest,
        corpus_root=corpus_root,
    )

    _require_exact_keys(
        casebook,
        required={
            "casebook_id",
            "version",
            "status",
            "case_count",
            "corpus_manifest_id",
            "corpus_manifest_sha256",
            "span_unit",
            "cases",
        },
        optional=set(),
        label="Casebook",
    )
    casebook_id = _string(casebook["casebook_id"], "casebook_id")
    if casebook["version"] != 1:
        raise CasebookValidationError("Casebook version must be 1")
    if casebook["status"] != "FROZEN_HUMAN_REVIEWED":
        raise CasebookValidationError("Casebook status must be FROZEN_HUMAN_REVIEWED")
    if casebook["span_unit"] != "UTF-8 byte offsets, half-open [start_byte,end_byte)":
        raise CasebookValidationError("Casebook has an unsupported span_unit")
    if casebook["corpus_manifest_id"] != manifest_id:
        raise CasebookValidationError("Casebook corpus_manifest_id mismatch")
    actual_manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if casebook["corpus_manifest_sha256"] != actual_manifest_sha:
        raise CasebookValidationError("Casebook Corpus Manifest SHA-256 mismatch")

    cases = _array(casebook["cases"], "cases")
    case_count = _integer(casebook["case_count"], "case_count")
    if case_count != len(cases) or case_count != _EXPECTED_CASE_COUNT:
        raise CasebookValidationError(
            f"Casebook V1 must contain exactly {_EXPECTED_CASE_COUNT} cases"
        )
    seen_case_ids: set[str] = set()
    supporting_count = 0
    near_miss_count = 0
    for index, raw_case in enumerate(cases):
        case_id, case_supporting_count, case_near_miss_count = _validate_case(
            raw_case,
            index=index,
            source_by_key=source_by_key,
        )
        if case_id in seen_case_ids:
            raise CasebookValidationError(f"duplicate case_id {case_id!r}")
        seen_case_ids.add(case_id)
        supporting_count += case_supporting_count
        near_miss_count += case_near_miss_count

    return CasebookValidationSummary(
        casebook_id=casebook_id,
        case_count=case_count,
        document_count=len(source_by_key),
        supporting_evidence_count=supporting_count,
        near_miss_evidence_count=near_miss_count,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("casebook", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("corpus_root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = validate_casebook(args.casebook, args.manifest, args.corpus_root)
    print(
        json.dumps(
            {
                "status": "PASS",
                "casebook_id": summary.casebook_id,
                "case_count": summary.case_count,
                "document_count": summary.document_count,
                "supporting_evidence_count": summary.supporting_evidence_count,
                "near_miss_evidence_count": summary.near_miss_evidence_count,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
