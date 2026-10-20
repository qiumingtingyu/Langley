"""RUN-only usage accounting through real local trace/store and API seams."""

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import langley.api.observatory as api
from langley.answering.local_tracing import LocalJsonTracer
from langley.api.dependencies import get_current_user_id, get_session
from langley.observatory.service import ObservatoryRuntime
from langley.observatory.store import ObservatoryStore
from langley.observatory.usage import usage_for_run


def _store(tmp_path, summaries=(), *, generation=(42, 8), terminal=True):
    root = tmp_path / "traces"
    trace = LocalJsonTracer(root, include_content=False).start(1, "fake", "fake", False)
    trace.event(
        "llm",
        round=1,
        input_tokens=generation[0],
        output_tokens=generation[1],
        total_tokens=sum(generation) if None not in generation else None,
    )
    for fields in summaries:
        trace.context_compact(**fields)
    if terminal:
        trace.success("not recorded")
    store = ObservatoryStore(tmp_path / "store.sqlite", root)
    store.refresh()
    return store


def _compact(input_=100, output=20, **extra):
    return dict(
        scope="RUN",
        provider_input_tokens=input_,
        provider_output_tokens=output,
        outcome="COMPACTED",
        **extra,
    )


def _usage(store):
    return usage_for_run(store, store.get_run(1))


@pytest.mark.parametrize(
    "summaries,expected,complete,count",
    [
        ([_compact()], (142, 28, 170), True, 1),
        ([_compact(None, None)], (42, 8, 50), False, 1),
        ([_compact(100, None)], (142, 8, 150), False, 1),
        ([_compact(0, 0)], (42, 8, 50), True, 1),
        (
            [
                dict(
                    scope="RUN",
                    outcome="SUMMARY_FAILED",
                    success=False,
                    provider_input_tokens=100,
                    provider_output_tokens=20,
                )
            ],
            (142, 28, 170),
            True,
            1,
        ),
        (
            [
                dict(
                    scope="CONVERSATION",
                    provider_input_tokens=100,
                    provider_output_tokens=20,
                )
            ],
            (42, 8, 50),
            True,
            0,
        ),
        (
            [dict(provider_input_tokens=100, provider_output_tokens=20)],
            (42, 8, 50),
            True,
            0,
        ),
        (
            [
                dict(scope="RUN", outcome="NO_ROOM"),
                dict(scope="RUN", outcome="INPUT_TOO_LARGE"),
            ],
            (42, 8, 50),
            True,
            0,
        ),
        ([], (42, 8, 50), True, 0),
    ],
)
def test_accounting_scope_failures_missing_usage_and_legacy(
    tmp_path, summaries, expected, complete, count
):
    store = _store(tmp_path, summaries)
    usage = _usage(store)
    assert (
        usage.all_run.known_input_tokens,
        usage.all_run.known_output_tokens,
        usage.all_run.known_total_tokens,
    ) == expected
    assert usage.all_run.complete is complete
    assert usage.run_compaction.call_count == count
    assert usage.generation.known_total_tokens == 50
    assert store.get_run(1).provider_total_tokens == 50  # Legacy generation-only field.
    assert store.get_run(1).trace_complete is True  # Not the usage completeness flag.
    store.refresh()
    assert _usage(store) == usage  # Idempotent refresh/read, no additive counters.


def test_each_actual_attempt_once_even_failed_revisions_and_duplicate_observations(
    tmp_path,
):
    first = _compact(100, 20, attempt_count=1, revision=0)
    second = _compact(60, 10, attempt_count=2, revision=0)
    usage = _usage(_store(tmp_path, [first, second, second]))
    assert usage.run_compaction.call_count == 2
    assert usage.run_compaction.known_total_tokens == 190
    assert usage.all_run.known_total_tokens == 240
    assert usage.all_run.complete is True


def test_contradictory_duplicate_never_claims_complete(tmp_path):
    usage = _usage(
        _store(
            tmp_path, [_compact(attempt_count=1), _compact(200, 20, attempt_count=1)]
        )
    )
    assert usage.run_compaction.call_count == 1
    assert usage.all_run.complete is False


@pytest.mark.parametrize("invalid", [True, -1, 10.5, "100"])
def test_invalid_token_counts_are_missing_not_silently_zero(tmp_path, invalid):
    usage = _usage(_store(tmp_path, [_compact(invalid, 20)]))
    assert usage.run_compaction.known_input_tokens == 0
    assert usage.run_compaction.complete is False
    assert usage.all_run.known_output_tokens == 28
    assert usage.all_run.complete is False


def test_generation_missing_and_trace_structural_coverage_are_separate(tmp_path):
    usage = _usage(_store(tmp_path, [_compact()], generation=(None, 8)))
    assert usage.generation.complete is False
    assert usage.run_compaction.complete is True
    assert usage.all_run.complete is False
    assert usage.all_run.known_total_tokens == 128
    other = _usage(_store(tmp_path / "partial", [_compact()], terminal=False))
    assert other.generation.complete and other.run_compaction.complete
    assert other.all_run.complete is False


def test_unreadable_context_event_cannot_be_hidden_as_complete(tmp_path):
    store = _store(tmp_path, [_compact()])
    (store.traces_root / "run-1.jsonl").unlink()
    assert _usage(store).all_run.complete is False


def test_api_run_and_list_expose_breakdown_preserving_generation_fields(
    tmp_path, monkeypatch
):
    store = _store(tmp_path, [_compact()])

    async def owned(*args):
        return api._BusinessRun(1, "SUCCEEDED")

    async def owned_list(*args):
        return (api._BusinessRun(1, "SUCCEEDED"),)

    monkeypatch.setattr(api, "_owned_business_run", owned)
    monkeypatch.setattr(api, "_owned_business_runs", owned_list)
    app = FastAPI()
    app.include_router(api.router)
    app.dependency_overrides[get_session] = lambda: object()
    app.dependency_overrides[get_current_user_id] = lambda: 1
    app.dependency_overrides[api.get_observatory_runtime] = lambda: ObservatoryRuntime(
        store
    )
    with TestClient(app) as client:
        detail = client.get("/api/dev/observatory/runs/1").json()
        listed = client.get("/api/dev/observatory/runs").json()["runs"][0]
    assert detail == listed
    diagnostics = detail["diagnostics"]
    assert diagnostics["provider_total_tokens"] == 50
    assert diagnostics["usage"]["all_run"] == {
        "call_count": 2,
        "known_input_tokens": 142,
        "known_output_tokens": 28,
        "known_total_tokens": 170,
        "complete": True,
    }
    assert "not recorded" not in json.dumps(detail)
