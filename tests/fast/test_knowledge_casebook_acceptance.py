"""Focused accepted Casebook baseline contracts."""

import json
from pathlib import Path

import pytest

from langley.knowledge.casebook_acceptance import (
    render_accepted_baseline_markdown,
    write_accepted_baseline,
)


def _baseline(*, final_status: str = "SUCCEEDED") -> dict[str, object]:
    semantic = "PASS" if final_status == "SUCCEEDED" else "NOT_EVALUABLE"
    return {
        "schema_version": 1,
        "baseline_id": "knowledge-casebook-v1-b0",
        "status": "ACCEPTED_AS_BASELINE",
        "accepted_at": "2026-08-26T10:54:45Z",
        "source_observation": {
            "run_id": "run-1",
            "sha256": "a" * 64,
            "status": "OBSERVATION_COMPLETE_PENDING_HUMAN_REVIEW",
        },
        "human_review": {"overall": {"pass": 1, "fail": 0}},
        "deterministic_metrics": {"evaluated_case_count": 1},
        "failure_attribution": {"none": 1},
        "cases": [
            {
                "case_id": "case-1",
                "final_status": final_status,
                "human_adjudication": {
                    "answer_point": semantic,
                    "faithfulness": semantic,
                    "abstention": (
                        "NOT_APPLICABLE"
                        if final_status == "SUCCEEDED"
                        else "NOT_EVALUABLE"
                    ),
                    "overall": "PASS" if final_status == "SUCCEEDED" else "FAIL",
                    "failure_attribution": "clean positive control",
                },
            }
        ],
        "acceptance_notes": ["B0 remains an honest weak baseline."],
    }


def test_json_is_authority_and_markdown_renders_the_same_object(tmp_path: Path) -> None:
    baseline = _baseline()
    output = tmp_path / "accepted"

    write_accepted_baseline(baseline, output)

    loaded = json.loads((output / "baseline.json").read_text(encoding="utf-8"))
    assert loaded == baseline
    assert (output / "baseline.md").read_text(
        encoding="utf-8"
    ) == render_accepted_baseline_markdown(loaded)
    assert "ACCEPTED_AS_BASELINE" in (output / "baseline.md").read_text(
        encoding="utf-8"
    )


def test_failed_run_semantics_must_be_not_evaluable() -> None:
    baseline = _baseline(final_status="FAILED")
    baseline["cases"][0]["human_adjudication"]["faithfulness"] = "FAIL"  # type: ignore[index]

    with pytest.raises(ValueError, match="must be NOT_EVALUABLE"):
        render_accepted_baseline_markdown(baseline)
