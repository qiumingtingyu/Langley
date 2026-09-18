"""Development-only, owner-scoped Run Observatory HTTP reads."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NoReturn, TypeVar

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from langley.api.dependencies import get_current_user_id, get_session
from langley.observatory.metrics import RuntimeMetrics, derive_runtime_metrics
from langley.observatory.service import (
    ObservatoryRuntime,
    ObservatoryUnavailableError,
)
from langley.observatory.store import (
    ContextComponentRecord,
    ObservatoryStore,
    RawTraceEvent,
    TraceEventRecord,
    TraceRoundRecord,
    TraceRunSummary,
)
from langley.runs import RunNotFoundError, get_owned_run, list_owned_runs

router = APIRouter(prefix="/api/dev/observatory", tags=["developer-observatory"])
_Result = TypeVar("_Result")

_SAFE_EVENT_FIELDS: dict[str, tuple[str, ...]] = {
    "llm": ("finish_reason", "error_code", "tool_call_count", "provider_model"),
    "tool": ("tool_name", "result_kind", "error_code"),
    "knowledge.search": ("origin", "top_k", "hit_count", "success", "error_code"),
    "citation.validate": (
        "namespace",
        "available_evidence_count",
        "citation_count",
        "abstained",
        "success",
        "error_code",
        "abstention_control_token_leaked",
    ),
    "response.rejected": ("invalid_response_subtype", "finish_reason"),
    "context.compact": (
        "estimated_before",
        "estimated_after",
        "newly_compacted_turn_count",
        "recent_raw_turn_count",
        "compactor_model",
        "provider_model",
        "provider_input_tokens",
        "provider_output_tokens",
        "success",
        "outcome",
    ),
    "run.success": ("stop_reason",),
    "run.failure": ("error_code", "invalid_response_subtype"),
}


@dataclass(frozen=True)
class _BusinessRun:
    run_id: int
    status: str


class DiagnosticsSummaryResponse(BaseModel):
    available: bool
    observed_outcome: str | None
    trace_complete: bool | None
    first_observed_timestamp: str | None = None
    last_observed_timestamp: str | None = None
    provider: str | None = None
    configured_model: str | None = None
    capture_mode: str | None = None
    round_count: int | None = None
    tool_count: int | None = None
    provider_input_tokens: int | None = None
    provider_output_tokens: int | None = None
    provider_total_tokens: int | None = None
    observed_duration_ms: float | None = None


class RunObservatoryResponse(BaseModel):
    run_id: int
    business_status: str
    diagnostics: DiagnosticsSummaryResponse


class RunListResponse(BaseModel):
    runs: list[RunObservatoryResponse]


class TimelineEventResponse(BaseModel):
    event_id: int
    schema_version: int | None
    observed_seq: int | None
    kind: str
    round: int | None
    timestamp: str | None
    duration_ms: float | None
    tool_call_id: str | None
    parent_tool_call_id: str | None
    tool_ordinal: int | None
    metadata: dict[str, object]


class TimelineResponse(BaseModel):
    run_id: int
    events: list[TimelineEventResponse]


class RoundResponse(BaseModel):
    run_id: int
    round: int
    provider_input_tokens: int | None
    provider_output_tokens: int | None
    provider_total_tokens: int | None
    duration_ms: float | None
    ttft_ms: float | None
    finish_reason: str | None
    provider_model: str | None
    estimate_kind: str | None
    estimated_semantic_context_tokens: int | None


class ContextComponentResponse(BaseModel):
    key: str
    kind: str
    source: str | None
    chars: int
    utf8_bytes: int
    estimated_tokens: int
    content_sha256: str | None


class ContextResponse(BaseModel):
    run_id: int
    round: int
    estimate_kind: str | None
    estimated_semantic_context_tokens: int | None
    components: list[ContextComponentResponse]


class RawEventResponse(BaseModel):
    run_id: int
    event_id: int
    capture_mode: str | None
    event: dict[str, Any]


class MetricsFiltersResponse(BaseModel):
    provider: str | None
    configured_model: str | None
    provider_model: str | None
    observed_outcome: str | None
    capture_mode: str | None


class MetricsResponse(BaseModel):
    filters: MetricsFiltersResponse
    metrics: RuntimeMetrics


def get_observatory_runtime(request: Request) -> ObservatoryRuntime:
    runtime = getattr(request.app.state, "observatory_runtime", None)
    if runtime is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "OBSERVATORY_UNAVAILABLE"},
        )
    return runtime


@router.get("/runs", response_model=RunListResponse)
async def list_observatory_runs(
    session: AsyncSession = Depends(get_session),
    current_user_id: int = Depends(get_current_user_id),
    runtime: ObservatoryRuntime = Depends(get_observatory_runtime),
) -> RunListResponse:
    business_runs = await _owned_business_runs(session, current_user_id)
    business_by_id = {run.run_id: run for run in business_runs}
    summaries = await _runtime_read(runtime, lambda store: store.list_runs())
    return RunListResponse(
        runs=[
            _run_response(business_by_id[summary.run_id], summary)
            for summary in summaries
            if summary.run_id in business_by_id
        ]
    )


@router.get("/runs/{run_id}", response_model=RunObservatoryResponse)
async def get_observatory_run(
    run_id: int,
    session: AsyncSession = Depends(get_session),
    current_user_id: int = Depends(get_current_user_id),
    runtime: ObservatoryRuntime = Depends(get_observatory_runtime),
) -> RunObservatoryResponse:
    business_run = await _owned_business_run(session, current_user_id, run_id)
    summary = await _runtime_read(runtime, lambda store: store.get_run(run_id))
    return _run_response(business_run, summary)


@router.get("/runs/{run_id}/timeline", response_model=TimelineResponse)
async def get_observatory_timeline(
    run_id: int,
    session: AsyncSession = Depends(get_session),
    current_user_id: int = Depends(get_current_user_id),
    runtime: ObservatoryRuntime = Depends(get_observatory_runtime),
) -> TimelineResponse:
    await _owned_business_run(session, current_user_id, run_id)
    result = await _runtime_read(runtime, lambda store: _timeline(store, run_id))
    if result is None:
        _raise_diagnostics_not_found("OBSERVATORY_TRACE_NOT_FOUND")
    return TimelineResponse(run_id=run_id, events=list(result))


@router.get("/runs/{run_id}/rounds/{round_}", response_model=RoundResponse)
async def get_observatory_round(
    run_id: int,
    round_: int,
    session: AsyncSession = Depends(get_session),
    current_user_id: int = Depends(get_current_user_id),
    runtime: ObservatoryRuntime = Depends(get_observatory_runtime),
) -> RoundResponse:
    await _owned_business_run(session, current_user_id, run_id)
    trace_exists, round_record = await _runtime_read(
        runtime, lambda store: _round_lookup(store, run_id, round_)
    )
    if not trace_exists:
        _raise_diagnostics_not_found("OBSERVATORY_TRACE_NOT_FOUND")
    if round_record is None:
        _raise_diagnostics_not_found("OBSERVATORY_ROUND_NOT_FOUND")
    return _round_response(round_record)


@router.get("/runs/{run_id}/rounds/{round_}/context", response_model=ContextResponse)
async def get_observatory_context(
    run_id: int,
    round_: int,
    session: AsyncSession = Depends(get_session),
    current_user_id: int = Depends(get_current_user_id),
    runtime: ObservatoryRuntime = Depends(get_observatory_runtime),
) -> ContextResponse:
    await _owned_business_run(session, current_user_id, run_id)
    trace_exists, result = await _runtime_read(
        runtime, lambda store: _context_lookup(store, run_id, round_)
    )
    if not trace_exists:
        _raise_diagnostics_not_found("OBSERVATORY_TRACE_NOT_FOUND")
    if result is None:
        _raise_diagnostics_not_found("OBSERVATORY_ROUND_NOT_FOUND")
    round_record, components = result
    return ContextResponse(
        run_id=run_id,
        round=round_,
        estimate_kind=round_record.context_estimate_kind,
        estimated_semantic_context_tokens=round_record.estimated_context_tokens,
        components=[_component_response(component) for component in components],
    )


@router.get("/runs/{run_id}/events/{event_id}/raw", response_model=RawEventResponse)
async def get_observatory_raw_event(
    run_id: int,
    event_id: int,
    session: AsyncSession = Depends(get_session),
    current_user_id: int = Depends(get_current_user_id),
    runtime: ObservatoryRuntime = Depends(get_observatory_runtime),
) -> RawEventResponse:
    await _owned_business_run(session, current_user_id, run_id)
    trace_exists, result = await _runtime_read(
        runtime, lambda store: _raw_event_lookup(store, run_id, event_id)
    )
    if not trace_exists:
        _raise_diagnostics_not_found("OBSERVATORY_TRACE_NOT_FOUND")
    if result is None:
        _raise_diagnostics_not_found("OBSERVATORY_EVENT_NOT_FOUND")
    if isinstance(result, Exception):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "OBSERVATORY_RAW_EVENT_UNAVAILABLE"},
        )
    capture_mode = result.event.get("capture_mode")
    return RawEventResponse(
        run_id=run_id,
        event_id=event_id,
        capture_mode=capture_mode if isinstance(capture_mode, str) else None,
        event=result.event,
    )


@router.get("/metrics", response_model=MetricsResponse)
async def get_observatory_metrics(
    provider: str | None = Query(default=None),
    configured_model: str | None = Query(default=None),
    provider_model: str | None = Query(default=None),
    observed_outcome: str | None = Query(default=None),
    capture_mode: str | None = Query(default=None),
    session: AsyncSession = Depends(get_session),
    current_user_id: int = Depends(get_current_user_id),
    runtime: ObservatoryRuntime = Depends(get_observatory_runtime),
) -> MetricsResponse:
    business_runs = await _owned_business_runs(session, current_user_id)
    business_status_by_run_id = {run.run_id: run.status for run in business_runs}
    owned_ids = set(business_status_by_run_id)
    runs, rounds, events = await _runtime_read(
        runtime,
        lambda store: _metrics_inputs(
            store,
            owned_ids,
            provider=provider,
            configured_model=configured_model,
            provider_model=provider_model,
            observed_outcome=observed_outcome,
            capture_mode=capture_mode,
        ),
    )
    return MetricsResponse(
        filters=MetricsFiltersResponse(
            provider=provider,
            configured_model=configured_model,
            provider_model=provider_model,
            observed_outcome=observed_outcome,
            capture_mode=capture_mode,
        ),
        metrics=derive_runtime_metrics(
            runs,
            rounds,
            events,
            business_status_by_run_id,
        ),
    )


async def _runtime_read(
    runtime: ObservatoryRuntime, operation: Callable[[ObservatoryStore], _Result]
) -> _Result:
    try:
        return await runtime.read(operation)
    except ObservatoryUnavailableError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "OBSERVATORY_UNAVAILABLE"},
        ) from error


async def _owned_business_run(
    session: AsyncSession, user_id: int, run_id: int
) -> _BusinessRun:
    try:
        result = await get_owned_run(session, user_id=user_id, run_id=run_id)
        snapshot = _BusinessRun(run_id=result.run.id, status=result.run.status)
    except RunNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "RUN_NOT_FOUND"},
        ) from error
    finally:
        await session.rollback()
    return snapshot


async def _owned_business_runs(
    session: AsyncSession, user_id: int
) -> tuple[_BusinessRun, ...]:
    try:
        runs = await list_owned_runs(session, user_id=user_id)
        return tuple(_BusinessRun(run_id=run.id, status=run.status) for run in runs)
    finally:
        await session.rollback()


def _run_response(
    business_run: _BusinessRun, summary: TraceRunSummary | None
) -> RunObservatoryResponse:
    if summary is None:
        diagnostics = DiagnosticsSummaryResponse(
            available=False,
            observed_outcome=None,
            trace_complete=None,
        )
    else:
        diagnostics = DiagnosticsSummaryResponse(
            available=True,
            observed_outcome=summary.observed_outcome,
            trace_complete=summary.trace_complete,
            first_observed_timestamp=summary.first_observed_timestamp,
            last_observed_timestamp=summary.last_observed_timestamp,
            provider=summary.provider,
            configured_model=summary.configured_model,
            capture_mode=summary.capture_mode,
            round_count=summary.round_count,
            tool_count=summary.tool_count,
            provider_input_tokens=summary.provider_input_tokens,
            provider_output_tokens=summary.provider_output_tokens,
            provider_total_tokens=summary.provider_total_tokens,
            observed_duration_ms=summary.observed_duration_ms,
        )
    return RunObservatoryResponse(
        run_id=business_run.run_id,
        business_status=business_run.status,
        diagnostics=diagnostics,
    )


def _timeline(
    store: ObservatoryStore, run_id: int
) -> tuple[TimelineEventResponse, ...] | None:
    if store.get_run(run_id) is None:
        return None
    responses: list[TimelineEventResponse] = []
    for event in store.list_events(run_id):
        try:
            raw = store.get_raw_event(event.event_id).event
        except (OSError, RuntimeError):
            raw = {}
        responses.append(
            TimelineEventResponse(
                event_id=event.event_id,
                schema_version=event.schema_version,
                observed_seq=event.observed_seq,
                kind=event.kind,
                round=event.round,
                timestamp=event.timestamp,
                duration_ms=event.duration_ms,
                tool_call_id=event.tool_call_id,
                parent_tool_call_id=event.parent_tool_call_id,
                tool_ordinal=event.tool_ordinal,
                metadata=_safe_event_metadata(event.kind, raw),
            )
        )
    return tuple(responses)


def _safe_event_metadata(kind: str, event: dict[str, Any]) -> dict[str, object]:
    return {
        field: event[field]
        for field in _SAFE_EVENT_FIELDS.get(kind, ())
        if field in event and isinstance(event[field], (str, int, float, bool))
    }


def _round_lookup(
    store: ObservatoryStore, run_id: int, round_: int
) -> tuple[bool, TraceRoundRecord | None]:
    if store.get_run(run_id) is None:
        return False, None
    return True, store.get_round(run_id, round_)


def _context_lookup(
    store: ObservatoryStore, run_id: int, round_: int
) -> tuple[
    bool,
    tuple[TraceRoundRecord, tuple[ContextComponentRecord, ...]] | None,
]:
    trace_exists, round_record = _round_lookup(store, run_id, round_)
    if round_record is None:
        return trace_exists, None
    return True, (round_record, store.list_context_components(run_id, round_))


def _raw_event_lookup(
    store: ObservatoryStore, run_id: int, event_id: int
) -> tuple[bool, RawTraceEvent | Exception | None]:
    if store.get_run(run_id) is None:
        return False, None
    event_ids = {event.event_id for event in store.list_events(run_id)}
    if event_id not in event_ids:
        return True, None
    try:
        return True, store.get_raw_event(event_id)
    except (OSError, RuntimeError) as error:
        return True, error


def _round_response(round_record: TraceRoundRecord) -> RoundResponse:
    return RoundResponse(
        run_id=round_record.run_id,
        round=round_record.round,
        provider_input_tokens=round_record.input_tokens,
        provider_output_tokens=round_record.output_tokens,
        provider_total_tokens=round_record.total_tokens,
        duration_ms=round_record.duration_ms,
        ttft_ms=round_record.ttft_ms,
        finish_reason=round_record.finish_reason,
        provider_model=round_record.provider_model,
        estimate_kind=round_record.context_estimate_kind,
        estimated_semantic_context_tokens=round_record.estimated_context_tokens,
    )


def _component_response(component: ContextComponentRecord) -> ContextComponentResponse:
    return ContextComponentResponse(
        key=component.key,
        kind=component.kind,
        source=component.source,
        chars=component.char_count,
        utf8_bytes=component.byte_count,
        estimated_tokens=component.estimated_tokens,
        content_sha256=component.content_sha256,
    )


def _metrics_inputs(
    store: ObservatoryStore,
    owned_ids: set[int],
    *,
    provider: str | None,
    configured_model: str | None,
    provider_model: str | None,
    observed_outcome: str | None,
    capture_mode: str | None,
) -> tuple[
    tuple[TraceRunSummary, ...],
    tuple[TraceRoundRecord, ...],
    tuple[TraceEventRecord, ...],
]:
    runs = tuple(
        run
        for run in store.list_runs()
        if run.run_id in owned_ids
        and (provider is None or run.provider == provider)
        and (configured_model is None or run.configured_model == configured_model)
        and (
            provider_model is None
            or any(
                round_.provider_model == provider_model
                for round_ in store.list_rounds(run.run_id)
            )
        )
        and (observed_outcome is None or run.observed_outcome == observed_outcome)
        and (capture_mode is None or run.capture_mode == capture_mode)
    )
    run_ids = {run.run_id for run in runs}
    rounds = tuple(
        round_
        for run_id in run_ids
        for round_ in store.list_rounds(run_id)
        if provider_model is None or round_.provider_model == provider_model
    )
    events = tuple(event for run_id in run_ids for event in store.list_events(run_id))
    return runs, rounds, events


def _raise_diagnostics_not_found(code: str) -> NoReturn:
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"code": code},
    )
