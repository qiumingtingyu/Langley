"""Fast contracts for Task 2B metrics, routing, and refresh coordination."""

import asyncio
import threading
import time

import pytest
from fastapi.testclient import TestClient

import langley.main as main_module
from langley.main import create_app
from langley.observatory.metrics import PERCENTILE_METHOD, derive_runtime_metrics
from langley.observatory.service import (
    ObservatoryRuntime,
    ObservatoryUnavailableError,
)
from langley.observatory.store import (
    ObservatoryStore,
    TraceEventRecord,
    TraceRoundRecord,
    TraceRunSummary,
)
from langley.settings import Settings


def _run(
    run_id: int,
    outcome: str | None,
    *,
    complete: bool,
    duration_ms: float | None,
    input_tokens: int | None,
    output_tokens: int | None,
    total_tokens: int | None,
    round_count: int,
    tool_count: int,
) -> TraceRunSummary:
    return TraceRunSummary(
        run_id=run_id,
        first_observed_timestamp=None,
        last_observed_timestamp=None,
        provider="fake",
        configured_model="model",
        capture_mode="METADATA_ONLY",
        observed_outcome=outcome,
        trace_complete=complete,
        round_count=round_count,
        tool_count=tool_count,
        provider_input_tokens=input_tokens,
        provider_output_tokens=output_tokens,
        provider_total_tokens=total_tokens,
        observed_duration_ms=duration_ms,
        raw_trace_size_bytes=1,
        source_count=1,
    )


def _round(
    run_id: int,
    round_: int,
    *,
    duration_ms: float | None,
    ttft_ms: float | None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    total_tokens: int | None = None,
) -> TraceRoundRecord:
    return TraceRoundRecord(
        run_id=run_id,
        round=round_,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        duration_ms=duration_ms,
        ttft_ms=ttft_ms,
        finish_reason="STOP",
        provider_model="model",
        estimated_context_tokens=None,
        context_estimate_kind=None,
    )


def _trace_event(
    event_id: int,
    run_id: int,
    kind: str,
    duration_ms: float | None = None,
) -> TraceEventRecord:
    return TraceEventRecord(
        event_id=event_id,
        run_id=run_id,
        source_path="trace.jsonl",
        ingest_order=event_id,
        schema_version=1,
        observed_seq=event_id,
        kind=kind,
        round=1 if kind == "tool" else None,
        timestamp=None,
        tool_call_id=f"call-{event_id}" if kind == "tool" else None,
        parent_tool_call_id=None,
        tool_ordinal=event_id if kind == "tool" else None,
        duration_ms=duration_ms,
        raw_byte_offset=0,
        raw_byte_length=1,
    )


def _tool_event(event_id: int, run_id: int, duration_ms: float | None):
    return _trace_event(event_id, run_id, "tool", duration_ms)


def test_metrics_rates_percentiles_null_ttft_and_aggregates_are_deterministic():
    runs = (
        _run(
            1,
            "SUCCEEDED",
            complete=True,
            duration_ms=10,
            input_tokens=10,
            output_tokens=4,
            total_tokens=14,
            round_count=1,
            tool_count=1,
        ),
        _run(
            2,
            "FAILED",
            complete=True,
            duration_ms=20,
            input_tokens=20,
            output_tokens=6,
            total_tokens=26,
            round_count=2,
            tool_count=2,
        ),
        _run(
            3,
            None,
            complete=False,
            duration_ms=30,
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
            round_count=0,
            tool_count=0,
        ),
    )
    rounds = (
        _round(
            1,
            1,
            duration_ms=100,
            ttft_ms=None,
            input_tokens=10,
            output_tokens=4,
            total_tokens=14,
        ),
        _round(
            2,
            1,
            duration_ms=200,
            ttft_ms=40,
            input_tokens=20,
            output_tokens=6,
            total_tokens=26,
        ),
        _round(2, 2, duration_ms=300, ttft_ms=10),
    )
    metrics = derive_runtime_metrics(
        runs,
        rounds,
        (
            _trace_event(1, 1, "run.start"),
            _tool_event(2, 1, 5),
            _trace_event(3, 1, "run.success"),
            _trace_event(4, 2, "run.start"),
            _tool_event(5, 2, None),
            _trace_event(6, 2, "run.failure"),
        ),
        {1: "SUCCEEDED", 2: "FAILED", 3: "CANCELLED"},
    )

    assert metrics.percentile_method == PERCENTILE_METHOD
    assert metrics.run.business_success_count == 1
    assert metrics.run.business_failure_count == 1
    assert metrics.run.business_cancelled_count == 1
    assert metrics.run.business_terminal_count == 3
    assert metrics.run.business_success_rate == 0.5
    assert metrics.run.trace_complete_rate == 0.666667
    assert metrics.run.latency.model_dump() == {
        "sample_count": 2,
        "average_ms": 15.0,
        "p50_ms": 10.0,
        "p95_ms": 20.0,
    }
    assert metrics.run.provider_tokens.input.model_dump() == {
        "sample_count": 2,
        "total": 30,
        "average_per_run": 15.0,
    }
    assert metrics.run.rounds_per_run_average == 1.0
    assert metrics.run.tools_per_run_average == 1.0
    assert metrics.llm.latency.p50_ms == 200
    assert metrics.llm.latency.p95_ms == 300
    assert metrics.llm.ttft.sample_count == 2
    assert metrics.llm.ttft.average_ms == 25
    assert metrics.llm.input_tokens.model_dump() == {
        "observed_total": 30,
        "observed_round_count": 2,
        "total_round_count": 3,
        "coverage_rate": 0.666667,
    }
    assert metrics.llm.output_tokens.model_dump() == {
        "observed_total": 10,
        "observed_round_count": 2,
        "total_round_count": 3,
        "coverage_rate": 0.666667,
    }
    assert metrics.llm.total_tokens.model_dump() == {
        "observed_total": 40,
        "observed_round_count": 2,
        "total_round_count": 3,
        "coverage_rate": 0.666667,
    }
    assert metrics.tool.event_count == 2
    assert metrics.tool.scope == "selected_runs"
    assert metrics.tool.latency.sample_count == 1
    assert metrics.tool.latency.p50_ms == 5


def test_empty_and_single_sample_metrics_are_explicit():
    empty = derive_runtime_metrics((), (), (), {})
    assert empty.run.business_success_rate is None
    assert empty.run.trace_complete_rate is None
    assert empty.run.latency.p50_ms is None
    assert empty.llm.ttft.sample_count == 0
    assert empty.llm.ttft.p95_ms is None
    assert empty.llm.input_tokens.model_dump() == {
        "observed_total": 0,
        "observed_round_count": 0,
        "total_round_count": 0,
        "coverage_rate": None,
    }
    assert empty.llm.output_tokens == empty.llm.input_tokens
    assert empty.llm.total_tokens == empty.llm.input_tokens

    single = derive_runtime_metrics(
        (
            _run(
                1,
                "SUCCEEDED",
                complete=True,
                duration_ms=12.5,
                input_tokens=None,
                output_tokens=None,
                total_tokens=None,
                round_count=0,
                tool_count=0,
            ),
        ),
        (),
        (
            _trace_event(1, 1, "run.start"),
            _trace_event(2, 1, "run.success"),
        ),
        {1: "SUCCEEDED"},
    )
    assert single.run.latency.p50_ms == 12.5
    assert single.run.latency.p95_ms == 12.5


def test_llm_token_observations_report_independent_coverage_and_count_zero():
    rounds = (
        _round(
            1,
            1,
            duration_ms=None,
            ttft_ms=None,
            input_tokens=10,
            output_tokens=None,
            total_tokens=0,
        ),
        _round(
            1,
            2,
            duration_ms=None,
            ttft_ms=None,
            input_tokens=20,
            output_tokens=5,
            total_tokens=None,
        ),
    )

    metrics = derive_runtime_metrics((), rounds, (), {})

    assert metrics.llm.input_tokens.model_dump() == {
        "observed_total": 30,
        "observed_round_count": 2,
        "total_round_count": 2,
        "coverage_rate": 1.0,
    }
    assert metrics.llm.output_tokens.model_dump() == {
        "observed_total": 5,
        "observed_round_count": 1,
        "total_round_count": 2,
        "coverage_rate": 0.5,
    }
    assert metrics.llm.total_tokens.model_dump() == {
        "observed_total": 0,
        "observed_round_count": 1,
        "total_round_count": 2,
        "coverage_rate": 0.5,
    }


def test_run_latency_requires_observed_start_and_terminal_but_not_complete_trace():
    runs = (
        _run(
            1,
            "SUCCEEDED",
            complete=True,
            duration_ms=10,
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
            round_count=0,
            tool_count=0,
        ),
        _run(
            2,
            "SUCCEEDED",
            complete=True,
            duration_ms=20,
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
            round_count=0,
            tool_count=0,
        ),
        _run(
            3,
            "FAILED",
            complete=False,
            duration_ms=30,
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
            round_count=0,
            tool_count=0,
        ),
        _run(
            4,
            None,
            complete=False,
            duration_ms=40,
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
            round_count=0,
            tool_count=0,
        ),
    )
    events = (
        _trace_event(1, 1, "run.start"),
        _trace_event(2, 1, "run.success"),
        _trace_event(3, 2, "run.success"),
        _trace_event(4, 3, "run.start"),
        _trace_event(6, 3, "run.failure"),
        _trace_event(7, 4, "run.start"),
    )

    metrics = derive_runtime_metrics(
        runs,
        (),
        events,
        {1: "SUCCEEDED", 2: "SUCCEEDED", 3: "FAILED", 4: "RUNNING"},
    )

    assert metrics.run.latency.model_dump() == {
        "sample_count": 2,
        "average_ms": 20.0,
        "p50_ms": 10.0,
        "p95_ms": 30.0,
    }


def test_run_token_metrics_exclude_unknown_aggregates_and_preserve_zero():
    runs = (
        _run(
            1,
            "SUCCEEDED",
            complete=True,
            duration_ms=None,
            input_tokens=None,
            output_tokens=12,
            total_tokens=None,
            round_count=2,
            tool_count=0,
        ),
        _run(
            2,
            "SUCCEEDED",
            complete=True,
            duration_ms=None,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            round_count=1,
            tool_count=0,
        ),
    )

    metrics = derive_runtime_metrics(
        runs,
        (),
        (),
        {1: "SUCCEEDED", 2: "SUCCEEDED"},
    )

    assert metrics.run.provider_tokens.input.model_dump() == {
        "sample_count": 1,
        "total": 0,
        "average_per_run": 0.0,
    }
    assert metrics.run.provider_tokens.output.model_dump() == {
        "sample_count": 2,
        "total": 12,
        "average_per_run": 6.0,
    }
    assert metrics.run.provider_tokens.total.model_dump() == {
        "sample_count": 1,
        "total": 0,
        "average_per_run": 0.0,
    }


def test_cancelled_business_run_is_separate_from_failure_and_success_rate():
    runs = (
        _run(
            1,
            "SUCCEEDED",
            complete=True,
            duration_ms=10,
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
            round_count=0,
            tool_count=0,
        ),
        _run(
            2,
            "FAILED",
            complete=False,
            duration_ms=5,
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
            round_count=0,
            tool_count=0,
        ),
    )

    metrics = derive_runtime_metrics(
        runs,
        (),
        (),
        {1: "SUCCEEDED", 2: "CANCELLED"},
    )

    assert metrics.run.business_success_count == 1
    assert metrics.run.business_failure_count == 0
    assert metrics.run.business_cancelled_count == 1
    assert metrics.run.business_terminal_count == 2
    assert metrics.run.business_success_rate == 1.0
    assert metrics.run.trace_complete_rate == 0.5


@pytest.mark.anyio
async def test_application_runtime_serializes_concurrent_refreshes(
    tmp_path, monkeypatch
):
    store = ObservatoryStore(tmp_path / "observatory.sqlite", tmp_path / "traces")
    runtime = ObservatoryRuntime(store)
    guard = threading.Lock()
    active = 0
    maximum = 0
    calls = 0

    def refresh():
        nonlocal active, maximum, calls
        with guard:
            active += 1
            calls += 1
            maximum = max(maximum, active)
        time.sleep(0.03)
        with guard:
            active -= 1

    monkeypatch.setattr(store, "refresh", refresh)
    await asyncio.gather(
        runtime.read(lambda _: "first"),
        runtime.read(lambda _: "second"),
    )

    assert calls == 2
    assert maximum == 1


def test_routes_are_absent_outside_development_and_present_in_development(tmp_path):
    test_app = create_app(Settings(environment="test"))
    assert not any(
        path.startswith("/api/dev/observatory") for path in test_app.openapi()["paths"]
    )
    with TestClient(test_app) as client:
        assert client.get("/api/dev/observatory/runs").status_code == 404

    development_app = create_app(
        Settings(
            environment="development",
            observatory_database_path=tmp_path / "observatory.sqlite",
            local_run_diagnostics_root=tmp_path / "traces",
        )
    )
    paths = set(development_app.openapi()["paths"])
    assert "/api/dev/observatory/runs" in paths
    assert "/api/dev/observatory/metrics" in paths
    assert Settings().observatory_database_path.as_posix() == (
        ".runtime/observatory.sqlite"
    )


def test_observatory_initialization_failure_is_fail_open(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        del args, kwargs
        raise OSError("injected Observatory failure")

    monkeypatch.setattr(main_module, "ObservatoryStore", fail)
    app = create_app(
        Settings(
            environment="development",
            observatory_database_path=tmp_path / "unavailable.sqlite",
        )
    )

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert app.state.observatory_runtime.store is None
    with pytest.raises(ObservatoryUnavailableError):
        asyncio.run(app.state.observatory_runtime.read(lambda _: None))
