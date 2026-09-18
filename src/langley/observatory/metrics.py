"""On-demand, low-cardinality metrics derived from Observatory projections."""

import math
from collections.abc import Mapping

from pydantic import BaseModel

from langley.observatory.store import (
    TraceEventRecord,
    TraceRoundRecord,
    TraceRunSummary,
)

PERCENTILE_METHOD = "nearest_rank:ceil(p*N),one_based"


class DistributionMetrics(BaseModel):
    sample_count: int
    average_ms: float | None
    p50_ms: float | None
    p95_ms: float | None


class TokenMetrics(BaseModel):
    sample_count: int
    total: int
    average_per_run: float | None


class ProviderTokenMetrics(BaseModel):
    input: TokenMetrics
    output: TokenMetrics
    total: TokenMetrics


class RunLevelMetrics(BaseModel):
    indexed_authorized_run_count: int
    terminal_observed_run_count: int
    business_success_count: int
    business_failure_count: int
    business_cancelled_count: int
    business_terminal_count: int
    business_success_rate: float | None
    trace_complete_count: int
    trace_complete_rate: float | None
    latency: DistributionMetrics
    provider_tokens: ProviderTokenMetrics
    rounds_per_run_average: float | None
    tools_per_run_average: float | None


class LLMTokenObservationMetrics(BaseModel):
    observed_total: int
    observed_round_count: int
    total_round_count: int
    coverage_rate: float | None


class LLMLevelMetrics(BaseModel):
    event_count: int
    latency: DistributionMetrics
    ttft: DistributionMetrics
    input_tokens: LLMTokenObservationMetrics
    output_tokens: LLMTokenObservationMetrics
    total_tokens: LLMTokenObservationMetrics


class ToolLevelMetrics(BaseModel):
    scope: str = "selected_runs"
    event_count: int
    latency: DistributionMetrics


class RuntimeMetrics(BaseModel):
    percentile_method: str = PERCENTILE_METHOD
    run: RunLevelMetrics
    llm: LLMLevelMetrics
    tool: ToolLevelMetrics


def derive_runtime_metrics(
    runs: tuple[TraceRunSummary, ...],
    rounds: tuple[TraceRoundRecord, ...],
    events: tuple[TraceEventRecord, ...],
    business_status_by_run_id: Mapping[int, str],
) -> RuntimeMetrics:
    """Derive one deterministic metrics snapshot from authorized diagnostics."""

    run_count = len(runs)
    diagnostic_terminal_count = sum(
        run.observed_outcome in {"SUCCEEDED", "FAILED"} for run in runs
    )
    business_success_count = sum(
        business_status_by_run_id.get(run.run_id) == "SUCCEEDED" for run in runs
    )
    business_failure_count = sum(
        business_status_by_run_id.get(run.run_id) == "FAILED" for run in runs
    )
    business_cancelled_count = sum(
        business_status_by_run_id.get(run.run_id) == "CANCELLED" for run in runs
    )
    business_terminal_count = (
        business_success_count + business_failure_count + business_cancelled_count
    )
    complete_count = sum(run.trace_complete for run in runs)
    started_run_ids = {event.run_id for event in events if event.kind == "run.start"}
    terminal_run_ids = {
        event.run_id for event in events if event.kind in {"run.success", "run.failure"}
    }
    tool_events = tuple(event for event in events if event.kind == "tool")
    return RuntimeMetrics(
        run=RunLevelMetrics(
            indexed_authorized_run_count=run_count,
            terminal_observed_run_count=diagnostic_terminal_count,
            business_success_count=business_success_count,
            business_failure_count=business_failure_count,
            business_cancelled_count=business_cancelled_count,
            business_terminal_count=business_terminal_count,
            business_success_rate=_rate(
                business_success_count,
                business_success_count + business_failure_count,
            ),
            trace_complete_count=complete_count,
            trace_complete_rate=_rate(complete_count, run_count),
            latency=_distribution(
                tuple(
                    run.observed_duration_ms
                    for run in runs
                    if run.run_id in started_run_ids
                    and run.run_id in terminal_run_ids
                    and run.observed_duration_ms is not None
                )
            ),
            provider_tokens=ProviderTokenMetrics(
                input=_token_metrics(
                    tuple(
                        run.provider_input_tokens
                        for run in runs
                        if run.provider_input_tokens is not None
                    )
                ),
                output=_token_metrics(
                    tuple(
                        run.provider_output_tokens
                        for run in runs
                        if run.provider_output_tokens is not None
                    )
                ),
                total=_token_metrics(
                    tuple(
                        run.provider_total_tokens
                        for run in runs
                        if run.provider_total_tokens is not None
                    )
                ),
            ),
            rounds_per_run_average=_average_count(
                sum(run.round_count for run in runs), run_count
            ),
            tools_per_run_average=_average_count(
                sum(run.tool_count for run in runs), run_count
            ),
        ),
        llm=LLMLevelMetrics(
            event_count=len(rounds),
            latency=_distribution(
                tuple(
                    round_.duration_ms
                    for round_ in rounds
                    if round_.duration_ms is not None
                )
            ),
            ttft=_distribution(
                tuple(round_.ttft_ms for round_ in rounds if round_.ttft_ms is not None)
            ),
            input_tokens=_llm_token_observation_metrics(
                tuple(round_.input_tokens for round_ in rounds),
            ),
            output_tokens=_llm_token_observation_metrics(
                tuple(round_.output_tokens for round_ in rounds),
            ),
            total_tokens=_llm_token_observation_metrics(
                tuple(round_.total_tokens for round_ in rounds),
            ),
        ),
        tool=ToolLevelMetrics(
            event_count=len(tool_events),
            latency=_distribution(
                tuple(
                    event.duration_ms
                    for event in tool_events
                    if event.duration_ms is not None
                )
            ),
        ),
    )


def _distribution(values: tuple[float, ...]) -> DistributionMetrics:
    return DistributionMetrics(
        sample_count=len(values),
        average_ms=_average(values),
        p50_ms=_nearest_rank(values, 0.50),
        p95_ms=_nearest_rank(values, 0.95),
    )


def _nearest_rank(values: tuple[float, ...], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def _average(values: tuple[float, ...]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def _token_metrics(values: tuple[int, ...]) -> TokenMetrics:
    total = sum(values)
    return TokenMetrics(
        sample_count=len(values),
        total=total,
        average_per_run=round(total / len(values), 3) if values else None,
    )


def _llm_token_observation_metrics(
    values: tuple[int | None, ...],
) -> LLMTokenObservationMetrics:
    observed = tuple(value for value in values if value is not None)
    return LLMTokenObservationMetrics(
        observed_total=sum(observed),
        observed_round_count=len(observed),
        total_round_count=len(values),
        coverage_rate=_rate(len(observed), len(values)),
    )


def _average_count(total: int, count: int) -> float | None:
    return round(total / count, 3) if count else None


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None
