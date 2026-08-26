"""Focused structural and source-integrity checks for Casebook V1."""

import json
from pathlib import Path

import pytest

from langley.knowledge.casebook import CasebookValidationError, validate_casebook

_CASEBOOK = Path("eval/slice6/casebook/knowledge_casebook_v1.json")
_MANIFEST = Path("eval/slice6/casebook/corpus_manifest_v1.json")
_CORPUS_ROOT = Path("tests/fixtures/knowledge/retrieval")
_REMOVED_MATERIALIZER = Path("eval/slice6/materialize_casebook_v1.py")
_FORMAL_RUNNER = Path("eval/slice6/run_formal_casebook_baseline.py")


def _casebook_payload() -> dict[str, object]:
    value = json.loads(_CASEBOOK.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_frozen_knowledge_casebook_v1_matches_all_canonical_source_bytes() -> None:
    summary = validate_casebook(_CASEBOOK, _MANIFEST, _CORPUS_ROOT)

    assert summary.casebook_id == "knowledge-casebook-v1"
    assert summary.case_count == 25
    assert summary.document_count == 40
    assert summary.supporting_evidence_count == 28
    assert summary.near_miss_evidence_count == 4


def test_casebook_json_is_the_only_human_golden_authority() -> None:
    payload = _casebook_payload()
    cases = payload["cases"]
    assert isinstance(cases, list)
    unanswerable = [
        case
        for case in cases
        if isinstance(case, dict) and case.get("answerability") == "UNANSWERABLE"
    ]

    assert not _REMOVED_MATERIALIZER.exists()
    assert "materialize_casebook_v1" not in _FORMAL_RUNNER.read_text(encoding="utf-8")
    assert len(unanswerable) == 4
    assert {case["case_provenance"]["query_origin"] for case in unanswerable} == {
        "HUMAN_AUTHORED"
    }


def test_xiaolin_cn_003_uses_the_recalculated_minimal_p1_p2_source_slice() -> None:
    payload = _casebook_payload()
    cases = payload["cases"]
    assert isinstance(cases, list)
    case = next(
        case
        for case in cases
        if isinstance(case, dict) and case.get("case_id") == "xiaolin-cn-003"
    )
    evidence = case["supporting_evidence"][0]

    assert evidence == {
        "evidence_id": "E1",
        "document_key": "xiaolin_cn_data_link",
        "start_byte": 28488,
        "end_byte": 28720,
        "evidence_text": (
            "- **最小帧长**：总线传播时延 * 数据传输速率 * 2\n"
            "\t- 争用期内可发送的数据长度\n"
            "\t- 确保共享总线以太网上的每一个站点在发送完一个完整的帧之前，"
            "能够检测出是否产生了碰撞\n"
        ),
    }
    source = (_CORPUS_ROOT / "documents/xiaolin_cn_data_link.md").read_bytes()
    start = evidence["start_byte"]
    end = evidence["end_byte"]
    assert source[start:end].decode("utf-8") == evidence["evidence_text"]
    assert (
        source[end:]
        .decode("utf-8")
        .startswith("\t- 如果在争用期检测到碰撞就立即中止发送")
    )


def test_casebook_rejects_answer_points_that_reference_non_supporting_evidence(
    tmp_path: Path,
) -> None:
    payload = _casebook_payload()
    cases = payload["cases"]
    assert isinstance(cases, list)
    case = cases[0]
    assert isinstance(case, dict)
    points = case["required_answer_points"]
    assert isinstance(points, list)
    point = points[0]
    assert isinstance(point, dict)
    point["evidence_ids"] = ["NM1"]
    changed = tmp_path / "changed-casebook.json"
    changed.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(CasebookValidationError, match="non-supporting evidence"):
        validate_casebook(changed, _MANIFEST, _CORPUS_ROOT)


def test_casebook_rejects_evidence_text_not_extracted_from_the_declared_span(
    tmp_path: Path,
) -> None:
    payload = _casebook_payload()
    cases = payload["cases"]
    assert isinstance(cases, list)
    case = cases[0]
    assert isinstance(case, dict)
    evidence = case["supporting_evidence"]
    assert isinstance(evidence, list)
    item = evidence[0]
    assert isinstance(item, dict)
    item["evidence_text"] = f"{item['evidence_text']}tampered"
    changed = tmp_path / "changed-casebook.json"
    changed.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(CasebookValidationError, match="does not match source bytes"):
        validate_casebook(changed, _MANIFEST, _CORPUS_ROOT)
