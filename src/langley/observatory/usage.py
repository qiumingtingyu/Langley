"""On-demand token accounting from existing rounds and RUN compaction events."""

from collections.abc import Iterable

from pydantic import BaseModel

from langley.observatory.store import ObservatoryStore, TraceRunSummary


class UsageTotals(BaseModel):
    """Known token components; complete is independent of trace_complete."""

    call_count: int
    known_input_tokens: int
    known_output_tokens: int
    known_total_tokens: int
    complete: bool


class RunUsage(BaseModel):
    generation: UsageTotals
    run_compaction: UsageTotals
    all_run: UsageTotals


def _tokens(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _totals(
    calls: Iterable[tuple[int | None, int | None]], *, coverage: bool = True
) -> UsageTotals:
    values = tuple(calls)
    input_ = sum(i for i, _ in values if i is not None)
    output = sum(o for _, o in values if o is not None)
    return UsageTotals(
        call_count=len(values),
        known_input_tokens=input_,
        known_output_tokens=output,
        known_total_tokens=input_ + output,
        complete=coverage and all(i is not None and o is not None for i, o in values),
    )


def _merge(
    calls: dict[tuple[object, ...], tuple[int | None, int | None]],
    key: tuple[object, ...],
    usage: tuple[int | None, int | None],
) -> bool:
    """Merge duplicate observations of one call, flag contradictory valid values."""
    previous = calls.get(key)
    if previous is None:
        calls[key] = usage
        return True
    consistent = all(a is None or b is None or a == b for a, b in zip(previous, usage))
    calls[key] = (
        previous[0] if previous[0] is not None else usage[0],
        previous[1] if previous[1] is not None else usage[1],
    )
    return consistent


def usage_for_run(store: ObservatoryStore, summary: TraceRunSummary) -> RunUsage:
    """Never mix Conversation compaction, cache tokens, or estimates into totals.

    RUN attempts have a monotonic attempt_count; revisions are not call IDs
    (failed compactions can share a revision). Older events fall back to their
    generation round or v1 sequence, then their source locator. Refreshes are
    idempotent: these are projections of observations, not accumulating counters.
    """
    generation_calls: dict[tuple[object, ...], tuple[int | None, int | None]] = {}
    generation_coverage = True
    for round_ in store.list_rounds(summary.run_id):
        generation_coverage &= _merge(
            generation_calls,
            (round_.round,),
            (_tokens(round_.input_tokens), _tokens(round_.output_tokens)),
        )
    compaction_calls: dict[tuple[object, ...], tuple[int | None, int | None]] = {}
    compaction_coverage = True
    for event in store.list_events(summary.run_id):
        if event.kind != "context.compact":
            continue
        try:
            raw = store.get_raw_event(event.event_id).event
        except (OSError, KeyError, RuntimeError, ValueError):
            # An unreadable context event cannot safely be classified or omitted
            # while claiming the all-run total is complete.
            compaction_coverage = False
            continue
        if raw.get("scope") != "RUN":
            continue
        usage = (
            _tokens(raw.get("provider_input_tokens")),
            _tokens(raw.get("provider_output_tokens")),
        )
        if raw.get("outcome") in ("NO_ROOM", "INPUT_TOO_LARGE") and usage == (
            None,
            None,
        ):
            continue  # Existing outcomes explicitly mean no summarizer invocation.
        attempt = _tokens(raw.get("attempt_count"))
        parent = _tokens(raw.get("parent_generation_round"))
        if attempt is not None and attempt > 0:
            key: tuple[object, ...] = ("attempt", attempt)
        elif parent is not None and parent > 0:
            key = ("generation_round", parent)
        elif event.schema_version == 1 and event.observed_seq is not None:
            key = ("seq", event.observed_seq)
        else:
            key = ("locator", event.source_path, event.ingest_order)
        compaction_coverage &= _merge(compaction_calls, key, usage)
    generation = _totals(generation_calls.values(), coverage=generation_coverage)
    compaction = _totals(compaction_calls.values(), coverage=compaction_coverage)
    all_run = _totals(
        (*generation_calls.values(), *compaction_calls.values()),
        coverage=summary.trace_complete and generation_coverage and compaction_coverage,
    )
    return RunUsage(generation=generation, run_compaction=compaction, all_run=all_run)
