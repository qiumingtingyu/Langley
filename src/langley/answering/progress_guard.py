"""Bounded, deterministic warnings over completed Tool batches; no execution IO."""

import hashlib
import json
from dataclasses import dataclass

from langley.answering.contracts import ToolCall, ToolResult, ToolResultKind


def _canonical(raw: str) -> str:
    try:
        return json.dumps(json.loads(raw), sort_keys=True, ensure_ascii=False)
    except (ValueError, RecursionError):
        return raw


def _digest(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


@dataclass(frozen=True)
class Observation:
    name: str
    action: str
    result: str
    failed: bool


@dataclass(frozen=True)
class ProgressState:
    recent: tuple[Observation, ...] = ()
    # A warning covers its entire unchanged repeating cycle, including rotation.
    warned_cycle: tuple[Observation, ...] = ()


@dataclass(frozen=True)
class ProgressWarning:
    rule: str
    cycle: tuple[Observation, ...]

    def feedback(self) -> str:
        evidence = "; ".join(
            f"{o.name}: arguments SHA256={o.action}, result SHA256={o.result}, "
            f"failure={o.failed}"
            for o in self.cycle
        )
        return (
            f"[Progress Guard: {self.rule}] Completed calls repeated with identical "
            f"arguments and observations ({evidence}). These fingerprints identify "
            "the calls and results already present immediately above; they are "
            "not proof of task failure or success. Summarize existing evidence, "
            "change strategy, or honestly report a blocker. Do not repeat an "
            "unchanged call without a reason. This is a reminder, not a tool veto."
        )


def observe_batch(
    state: ProgressState,
    calls: tuple[ToolCall, ...],
    results: tuple[ToolResult, ...],
) -> tuple[ProgressState, tuple[ProgressWarning, ...]]:
    """Match exact IDs/names before using any member; incomplete batches reset.

    Call declaration order is the deterministic order for a complete multi-call
    batch, independent of result arrival order. This does not infer wall time.
    All argument/result fields are retained; transport call IDs live separately.
    """
    ids = [c.call_id for c in calls]
    by_id = {r.call_id: r for r in results}
    if (
        not calls
        or len(set(ids)) != len(calls)
        or len(by_id) != len(results)
        or len(calls) != len(results)
        or set(ids) != set(by_id)
        or any(by_id[c.call_id].name != c.name for c in calls)
    ):
        return ProgressState(), ()
    recent, warned = state.recent, state.warned_cycle
    warnings = []
    for call in calls:
        result = by_id[call.call_id]
        failed = result.kind is not ToolResultKind.SUCCESS
        if call.name == "run_command":
            try:
                command = json.loads(result.content)
            except (ValueError, RecursionError):
                command = None
            if isinstance(command, dict):
                code = command.get("exit_code")
                failed |= command.get("timed_out") is True or (
                    type(code) is int and code != 0
                )
        observation = Observation(
            call.name,
            _digest(_canonical(call.raw_arguments)),
            _digest(result.kind.value + ":" + _canonical(result.content)),
            failed,
        )
        if warned:
            expected = warned[0]
            if observation == expected:
                warned = warned[1:] + warned[:1]
            else:
                warned = ()
        recent = (recent + (observation,))[-6:]
        rule = ""
        cycle: tuple[Observation, ...] = ()
        if len(recent) >= 3 and observation.failed and len(set(recent[-3:])) == 1:
            rule, cycle = "ACTION_ERROR_3", (observation,)
        elif len(recent) >= 4 and len(set(recent[-4:])) == 1:
            rule, cycle = "ACTION_OBSERVATION_4", (observation,)
        elif (
            len(recent) == 6
            and recent[:2] == recent[2:4] == recent[4:]
            and (recent[0].name, recent[0].action) != (recent[1].name, recent[1].action)
        ):
            rule, cycle = "ALTERNATING_AB_3", recent[:2]
        if rule and not warned:
            warnings.append(ProgressWarning(rule, cycle))
            warned = cycle
    return ProgressState(recent, warned), tuple(warnings)
