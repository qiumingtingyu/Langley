"""Accepted Casebook baseline persistence and presentation contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ACCEPTED_BASELINE_STATUS = "ACCEPTED_AS_BASELINE"
_ADJUDICATION_VALUES = {"PASS", "FAIL", "NOT_APPLICABLE", "NOT_EVALUABLE"}


def validate_accepted_baseline(baseline: dict[str, Any]) -> None:
    """Validate the small set of invariants that make an accepted baseline usable."""

    if baseline.get("status") != ACCEPTED_BASELINE_STATUS:
        raise ValueError("accepted baseline must have ACCEPTED_AS_BASELINE status")
    source = baseline.get("source_observation")
    if (
        not isinstance(source, dict)
        or not source.get("run_id")
        or not source.get("sha256")
    ):
        raise ValueError("accepted baseline must identify its source Observation")
    cases = baseline.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("accepted baseline must contain adjudicated cases")
    for case in cases:
        adjudication = case.get("human_adjudication")
        if not isinstance(adjudication, dict):
            raise ValueError(f"case {case.get('case_id')} has no Human adjudication")
        for field in ("answer_point", "faithfulness", "abstention"):
            value = adjudication.get(field)
            if value not in _ADJUDICATION_VALUES:
                raise ValueError(
                    f"case {case.get('case_id')} has invalid {field}: {value}"
                )
        if case.get("final_status") != "SUCCEEDED" and any(
            adjudication.get(field) != "NOT_EVALUABLE"
            for field in ("answer_point", "faithfulness", "abstention")
        ):
            raise ValueError(f"failed case {case.get('case_id')} must be NOT_EVALUABLE")


def _markdown_cell(value: object) -> str:
    if value is None:
        return "—"
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_accepted_baseline_markdown(baseline: dict[str, Any]) -> str:
    """Render only the accepted JSON object; never recompute its judgments."""

    validate_accepted_baseline(baseline)
    source = baseline["source_observation"]
    human = baseline["human_review"]
    lines = [
        f"# Accepted Baseline {baseline['baseline_id']}",
        "",
        f"- Status: `{baseline['status']}`",
        f"- Accepted at: `{baseline['accepted_at']}`",
        f"- Observation run ID: `{source['run_id']}`",
        f"- Observation SHA-256: `{source['sha256']}`",
        f"- Observation status: `{source['status']}`",
        "- Provider rerun: `NO`",
        "- Original Observation mutation: `NO`",
        "",
        "## Human Review summary",
        "",
        "```json",
        json.dumps(human, ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "Failed Runs without a canonical final answer are `NOT_EVALUABLE`; they are "
        "not reclassified as hallucinations.",
        "",
        "## Deterministic metrics and failure attribution",
        "",
        "```json",
        json.dumps(
            {
                "metrics": baseline["deterministic_metrics"],
                "failure_attribution": baseline["failure_attribution"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        "```",
        "",
        "## Case adjudications",
        "",
        (
            "| Case | Final | Run error | Answer Point | Faithfulness | "
            "Abstention | Overall | Attribution |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for case in baseline["cases"]:
        adjudication = case["human_adjudication"]
        lines.append(
            "| "
            + " | ".join(
                _markdown_cell(value)
                for value in (
                    case["case_id"],
                    case["final_status"],
                    case.get("run_error_code"),
                    adjudication["answer_point"],
                    adjudication["faithfulness"],
                    adjudication["abstention"],
                    adjudication["overall"],
                    adjudication["failure_attribution"],
                )
            )
            + " |"
        )
    lines.extend(
        (
            "",
            "## Acceptance notes",
            "",
            *[f"- {note}" for note in baseline["acceptance_notes"]],
        )
    )
    return "\n".join(lines).rstrip() + "\n"


def write_accepted_baseline(baseline: dict[str, Any], output_directory: Path) -> None:
    """Atomically persist exactly one accepted JSON authority and its rendering."""

    validate_accepted_baseline(baseline)
    if output_directory.exists():
        raise FileExistsError(f"accepted baseline already exists: {output_directory}")
    temporary = output_directory.with_name(f".{output_directory.name}.part")
    if temporary.exists():
        raise FileExistsError(f"baseline temporary output already exists: {temporary}")
    temporary.mkdir(parents=True)
    (temporary / "baseline.json").write_text(
        json.dumps(baseline, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (temporary / "baseline.md").write_text(
        render_accepted_baseline_markdown(baseline),
        encoding="utf-8",
        newline="\n",
    )
    if sorted(path.name for path in temporary.iterdir()) != [
        "baseline.json",
        "baseline.md",
    ]:
        raise RuntimeError("accepted baseline directory has unexpected entries")
    temporary.replace(output_directory)
