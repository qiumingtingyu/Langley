"""Real-MySQL ownership tests for the development-only Observatory API."""

import asyncio
import json
from argparse import Namespace
from datetime import timedelta
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from langley.bootstrap import bootstrap_local_user
from langley.business_time import utc_now
from langley.infrastructure.database import (
    create_database_engine,
    create_session_factory,
    dispose_database_engine,
)
from langley.infrastructure.models import Conversation, Message, Run
from langley.main import create_app
from langley.settings import Settings


@pytest.fixture
def migrated_database(test_database_url: str, reset_database) -> str:
    reset_database()
    config = Config("alembic.ini")
    config.cmd_opts = Namespace(x=["use_test_database=true"])
    command.upgrade(config, "head")
    return test_database_url


def _settings(database_url: str, root: Path, user_id: int = 1) -> Settings:
    return Settings(
        environment="development",
        database_url=database_url,
        local_user_id=user_id,
        local_run_diagnostics_root=root / "traces",
        observatory_database_path=root / "observatory.sqlite",
    )


def _bootstrap(database_url: str, user_id: int) -> None:
    settings = Settings(
        environment="test", database_url=database_url, local_user_id=user_id
    )
    asyncio.run(bootstrap_local_user(settings))


async def _seed_run(
    session: AsyncSession,
    *,
    user_id: int,
    status: str,
    label: str,
) -> int:
    now = utc_now()
    conversation = Conversation(
        user_id=user_id,
        title=label,
        created_at=now,
        updated_at=now,
        last_message_at=now,
        deleted_at=None,
    )
    session.add(conversation)
    await session.flush()
    user_message = Message(
        conversation_id=conversation.id,
        sequence_no=1,
        role="USER",
        content=f"question-{label}",
        run_id=None,
        regenerated_from_message_id=None,
        created_at=now,
    )
    session.add(user_message)
    await session.flush()
    terminal = status in {"SUCCEEDED", "FAILED", "CANCELLED"}
    run = Run(
        conversation_id=conversation.id,
        input_message_id=user_message.id,
        knowledge_base_id=None,
        grounding_policy="AUTO",
        client_request_id=f"observatory-{label}",
        attempt_no=1,
        status=status,
        started_at=now if status in {"RUNNING", "SUCCEEDED", "FAILED"} else None,
        finished_at=now + timedelta(seconds=1) if terminal else None,
        error_code="ANSWER_EXECUTION_FAILED" if status == "FAILED" else None,
        created_at=now,
        updated_at=now,
    )
    session.add(run)
    await session.flush()
    if status == "SUCCEEDED":
        session.add(
            Message(
                conversation_id=conversation.id,
                sequence_no=2,
                role="ASSISTANT",
                content=f"answer-{label}",
                run_id=run.id,
                regenerated_from_message_id=None,
                created_at=now,
            )
        )
    return run.id


async def _seed_runs(database_url: str) -> dict[str, int]:
    engine = create_database_engine(database_url)
    session_factory: async_sessionmaker[AsyncSession] = create_session_factory(engine)
    try:
        async with session_factory() as session:
            async with session.begin():
                return {
                    "observed_success": await _seed_run(
                        session,
                        user_id=1,
                        status="SUCCEEDED",
                        label="observed-success",
                    ),
                    "observed_failure": await _seed_run(
                        session, user_id=1, status="FAILED", label="observed-failure"
                    ),
                    "partial": await _seed_run(
                        session, user_id=1, status="FAILED", label="partial"
                    ),
                    "cancelled": await _seed_run(
                        session, user_id=1, status="CANCELLED", label="cancelled"
                    ),
                    "missing_trace": await _seed_run(
                        session, user_id=1, status="SUCCEEDED", label="missing-trace"
                    ),
                    "unowned": await _seed_run(
                        session, user_id=2, status="FAILED", label="unowned"
                    ),
                }
    finally:
        await dispose_database_engine(engine)


def _event(run_id: int, seq: int, kind: str, **fields: object) -> dict[str, object]:
    return {
        "schema_version": 1,
        "timestamp": f"2026-09-17T00:00:{seq:02d}+00:00",
        "run_id": run_id,
        "seq": seq,
        "kind": kind,
        "capture_mode": "FULL_CONTENT",
        **fields,
    }


def _write_trace(path: Path, events: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
            for event in events
        ),
        encoding="utf-8",
    )


def _seed_traces(root: Path, run_ids: dict[str, int]) -> int:
    success = run_ids["observed_success"]
    _write_trace(
        root / f"run-{success}.jsonl",
        [
            _event(success, 1, "run.start", provider="fake", configured_model="m1"),
            _event(
                success,
                2,
                "llm",
                round=1,
                duration_ms=100.0,
                ttft_ms=None,
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
                finish_reason="STOP",
                provider_model="provider-m1",
                context_frame={
                    "estimate_kind": "CONSERVATIVE_MULTILINGUAL_ESTIMATE_V1",
                    "estimated_tokens": 7,
                    "components": [
                        {
                            "key": "system",
                            "kind": "system",
                            "source": "system_input",
                            "char_count": 3,
                            "byte_count": 7,
                            "estimated_tokens": 3,
                            "content_sha256": "a" * 64,
                        }
                    ],
                },
                request={"system_input": "TIMELINE_SECRET_REQUEST"},
                assistant_content="TIMELINE_SECRET_ASSISTANT",
            ),
            _event(
                success,
                3,
                "llm",
                round=2,
                duration_ms=50.0,
                ttft_ms=5.0,
                input_tokens=7,
                output_tokens=3,
                total_tokens=10,
                finish_reason="STOP",
                provider_model="provider-m2",
            ),
            _event(
                success,
                4,
                "tool",
                round=2,
                ordinal=1,
                call_id="call-secret",
                tool_name="search_knowledge",
                duration_ms=30.0,
                result_kind="SUCCESS",
                error_code=None,
                raw_arguments="TIMELINE_SECRET_ARGUMENTS",
                result_content="TIMELINE_SECRET_RESULT",
            ),
            _event(success, 5, "run.success", stop_reason="FINAL_ANSWER"),
        ],
    )
    failed = run_ids["observed_failure"]
    _write_trace(
        root / f"run-{failed}.jsonl",
        [
            _event(failed, 1, "run.start", provider="fake", configured_model="m1"),
            _event(failed, 2, "run.failure", error_code="LLM_PROVIDER_FAILED"),
        ],
    )
    partial = run_ids["partial"]
    metadata_event = _event(
        partial,
        2,
        "llm",
        round=1,
        duration_ms=200.0,
        ttft_ms=20.0,
        input_tokens=20,
        output_tokens=10,
        total_tokens=30,
        finish_reason="STOP",
        provider_model="provider-m1",
        context_frame={
            "estimate_kind": "CONSERVATIVE_MULTILINGUAL_ESTIMATE_V1",
            "estimated_tokens": 4,
            "components": [],
        },
    )
    metadata_event["capture_mode"] = "METADATA_ONLY"
    start = _event(partial, 1, "run.start", provider="fake", configured_model="m1")
    start["capture_mode"] = "METADATA_ONLY"
    _write_trace(root / f"run-{partial}.jsonl", [start, metadata_event])

    cancelled = run_ids["cancelled"]
    _write_trace(
        root / f"run-{cancelled}.jsonl",
        [
            _event(cancelled, 1, "run.start", provider="fake", configured_model="m1"),
            _event(cancelled, 2, "run.failure", error_code="CANCELLED"),
        ],
    )

    unowned = run_ids["unowned"]
    _write_trace(
        root / f"run-{unowned}.jsonl",
        [_event(unowned, 1, "run.start"), _event(unowned, 2, "run.success")],
    )
    orphan = 9_999_999
    _write_trace(
        root / f"run-{orphan}.jsonl",
        [_event(orphan, 1, "run.start"), _event(orphan, 2, "run.success")],
    )
    return orphan


def test_developer_observatory_api_is_owner_scoped_safe_and_derived(
    migrated_database: str, tmp_path: Path, monkeypatch
) -> None:
    _bootstrap(migrated_database, 1)
    _bootstrap(migrated_database, 2)
    run_ids = asyncio.run(_seed_runs(migrated_database))
    settings = _settings(migrated_database, tmp_path)
    orphan_id = _seed_traces(settings.local_run_diagnostics_root, run_ids)
    app = create_app(settings)

    with TestClient(app) as client:
        listed = client.get("/api/dev/observatory/runs")
        assert listed.status_code == 200
        listed_ids = {item["run_id"] for item in listed.json()["runs"]}
        assert listed_ids == {
            run_ids["observed_success"],
            run_ids["observed_failure"],
            run_ids["partial"],
            run_ids["cancelled"],
        }
        assert run_ids["unowned"] not in listed_ids
        assert orphan_id not in listed_ids

        detail = client.get(f"/api/dev/observatory/runs/{run_ids['observed_success']}")
        assert detail.status_code == 200
        assert detail.json()["business_status"] == "SUCCEEDED"
        assert detail.json()["diagnostics"]["observed_outcome"] == "SUCCEEDED"
        assert detail.json()["diagnostics"]["trace_complete"] is True

        cancelled_detail = client.get(
            f"/api/dev/observatory/runs/{run_ids['cancelled']}"
        )
        assert cancelled_detail.status_code == 200
        assert cancelled_detail.json()["business_status"] == "CANCELLED"
        assert cancelled_detail.json()["diagnostics"]["observed_outcome"] == "FAILED"

        missing = client.get(f"/api/dev/observatory/runs/{run_ids['missing_trace']}")
        assert missing.status_code == 200
        assert missing.json()["diagnostics"] == {
            "available": False,
            "observed_outcome": None,
            "trace_complete": None,
            "first_observed_timestamp": None,
            "last_observed_timestamp": None,
            "provider": None,
            "configured_model": None,
            "capture_mode": None,
            "round_count": None,
            "tool_count": None,
            "provider_input_tokens": None,
            "provider_output_tokens": None,
            "provider_total_tokens": None,
            "observed_duration_ms": None,
        }
        missing_timeline = client.get(
            f"/api/dev/observatory/runs/{run_ids['missing_trace']}/timeline"
        )
        assert missing_timeline.status_code == 404
        assert missing_timeline.json() == {
            "detail": {"code": "OBSERVATORY_TRACE_NOT_FOUND"}
        }

        unowned = client.get(f"/api/dev/observatory/runs/{run_ids['unowned']}")
        assert unowned.status_code == 404
        assert unowned.json() == {"detail": {"code": "RUN_NOT_FOUND"}}

        timeline = client.get(
            f"/api/dev/observatory/runs/{run_ids['observed_success']}/timeline"
        )
        assert timeline.status_code == 200
        timeline_text = timeline.text
        assert "search_knowledge" in timeline_text
        assert "SUCCESS" in timeline_text
        assert "TIMELINE_SECRET" not in timeline_text
        llm_event_id = next(
            event["event_id"]
            for event in timeline.json()["events"]
            if event["kind"] == "llm"
        )

        raw = client.get(
            f"/api/dev/observatory/runs/{run_ids['observed_success']}"
            f"/events/{llm_event_id}/raw"
        )
        assert raw.status_code == 200
        assert raw.json()["capture_mode"] == "FULL_CONTENT"
        assert raw.json()["event"]["request"]["system_input"] == (
            "TIMELINE_SECRET_REQUEST"
        )

        wrong_run = client.get(
            f"/api/dev/observatory/runs/{run_ids['partial']}/events/{llm_event_id}/raw"
        )
        assert wrong_run.status_code == 404
        assert wrong_run.json() == {"detail": {"code": "OBSERVATORY_EVENT_NOT_FOUND"}}
        guessed_unowned = client.get(
            f"/api/dev/observatory/runs/{run_ids['unowned']}/events/{llm_event_id}/raw"
        )
        assert guessed_unowned.status_code == 404
        assert guessed_unowned.json() == {"detail": {"code": "RUN_NOT_FOUND"}}

        partial_timeline = client.get(
            f"/api/dev/observatory/runs/{run_ids['partial']}/timeline"
        ).json()
        metadata_event_id = next(
            event["event_id"]
            for event in partial_timeline["events"]
            if event["kind"] == "llm"
        )
        metadata_raw = client.get(
            f"/api/dev/observatory/runs/{run_ids['partial']}"
            f"/events/{metadata_event_id}/raw"
        )
        assert metadata_raw.status_code == 200
        assert metadata_raw.json()["capture_mode"] == "METADATA_ONLY"
        assert "request" not in metadata_raw.json()["event"]
        assert "assistant_content" not in metadata_raw.json()["event"]

        round_response = client.get(
            f"/api/dev/observatory/runs/{run_ids['observed_success']}/rounds/1"
        )
        assert round_response.status_code == 200
        assert round_response.json()["ttft_ms"] is None
        assert round_response.json()["provider_input_tokens"] == 10
        missing_round = client.get(
            f"/api/dev/observatory/runs/{run_ids['observed_success']}/rounds/99"
        )
        assert missing_round.status_code == 404
        assert missing_round.json() == {
            "detail": {"code": "OBSERVATORY_ROUND_NOT_FOUND"}
        }

        context = client.get(
            f"/api/dev/observatory/runs/{run_ids['observed_success']}/rounds/1/context"
        )
        assert context.status_code == 200
        context_body = context.json()
        assert context_body["estimate_kind"] == (
            "CONSERVATIVE_MULTILINGUAL_ESTIMATE_V1"
        )
        assert context_body["estimated_semantic_context_tokens"] == 7
        assert context_body["components"][0]["chars"] == 3
        assert context_body["components"][0]["utf8_bytes"] == 7
        assert "exact" not in context.text.lower()

        metrics = client.get("/api/dev/observatory/metrics")
        assert metrics.status_code == 200
        run_metrics = metrics.json()["metrics"]["run"]
        assert run_metrics["indexed_authorized_run_count"] == 4
        assert run_metrics["terminal_observed_run_count"] == 3
        assert run_metrics["business_success_count"] == 1
        assert run_metrics["business_failure_count"] == 2
        assert run_metrics["business_cancelled_count"] == 1
        assert run_metrics["business_terminal_count"] == 4
        assert run_metrics["business_success_rate"] == 0.333333
        assert run_metrics["trace_complete_count"] == 3
        assert run_metrics["trace_complete_rate"] == 0.75
        assert run_metrics["latency"]["sample_count"] == 3
        assert run_metrics["provider_tokens"]["input"]["total"] == 37
        llm_metrics = metrics.json()["metrics"]["llm"]
        assert llm_metrics["ttft"]["sample_count"] == 2
        assert llm_metrics["ttft"]["average_ms"] == 12.5
        assert llm_metrics["input_tokens"] == {
            "observed_total": 37,
            "observed_round_count": 3,
            "total_round_count": 3,
            "coverage_rate": 1.0,
        }
        assert llm_metrics["output_tokens"] == {
            "observed_total": 18,
            "observed_round_count": 3,
            "total_round_count": 3,
            "coverage_rate": 1.0,
        }
        assert llm_metrics["total_tokens"] == {
            "observed_total": 55,
            "observed_round_count": 3,
            "total_round_count": 3,
            "coverage_rate": 1.0,
        }
        tool_metrics = metrics.json()["metrics"]["tool"]
        assert tool_metrics["scope"] == "selected_runs"
        assert tool_metrics["event_count"] == 1

        filtered = client.get(
            "/api/dev/observatory/metrics", params={"capture_mode": "METADATA_ONLY"}
        )
        assert filtered.json()["metrics"]["run"]["indexed_authorized_run_count"] == 1
        provider_model_filtered = client.get(
            "/api/dev/observatory/metrics",
            params={"provider_model": "provider-m1"},
        )
        assert (
            provider_model_filtered.json()["metrics"]["run"][
                "indexed_authorized_run_count"
            ]
            == 2
        )
        provider_model_llm = provider_model_filtered.json()["metrics"]["llm"]
        assert provider_model_llm["event_count"] == 2
        assert provider_model_llm["input_tokens"] == {
            "observed_total": 30,
            "observed_round_count": 2,
            "total_round_count": 2,
            "coverage_rate": 1.0,
        }
        assert provider_model_llm["output_tokens"] == {
            "observed_total": 15,
            "observed_round_count": 2,
            "total_round_count": 2,
            "coverage_rate": 1.0,
        }
        assert provider_model_llm["total_tokens"] == {
            "observed_total": 45,
            "observed_round_count": 2,
            "total_round_count": 2,
            "coverage_rate": 1.0,
        }
        provider_model_tool = provider_model_filtered.json()["metrics"]["tool"]
        assert provider_model_tool["scope"] == "selected_runs"
        assert provider_model_tool["event_count"] == 1

        business = client.get(f"/api/runs/{run_ids['observed_success']}")
        assert business.status_code == 200
        assert business.json()["run"]["status"] == "SUCCEEDED"

        store = app.state.observatory_runtime.store
        assert store is not None

        def unavailable_raw(event_id: int):
            del event_id
            raise RuntimeError("injected missing raw bytes")

        monkeypatch.setattr(store, "get_raw_event", unavailable_raw)
        unavailable = client.get(
            f"/api/dev/observatory/runs/{run_ids['observed_success']}"
            f"/events/{llm_event_id}/raw"
        )
        assert unavailable.status_code == 409
        assert unavailable.json() == {
            "detail": {"code": "OBSERVATORY_RAW_EVENT_UNAVAILABLE"}
        }
