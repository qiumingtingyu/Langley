"""Real MySQL acceptance coverage for Personal Memory HTTP resources."""

import asyncio
from argparse import Namespace
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from threading import Event, Thread

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient

from langley.answering.contracts import LLMFinishReason, LLMResponseCompleted
from langley.answering.errors import RunErrorCode, WorkflowFailure
from langley.answering.fake_provider import FakeProvider, ScriptedProviderRound
from langley.api.responses import as_utc
from langley.bootstrap import bootstrap_local_user
from langley.business_time import utc_now
from langley.infrastructure.database import (
    create_database_engine,
    create_session_factory,
    dispose_database_engine,
)
from langley.infrastructure.models import Conversation, Memory, Message, User
from langley.main import create_app
from langley.memory.policy import estimate_load_all_memory_contribution
from langley.settings import Settings

from ._support import _completion, _insert_memory, _scalar


@pytest.fixture
def migrated_database(test_database_url: str, reset_database) -> str:
    reset_database()
    config = Config("alembic.ini")
    config.cmd_opts = Namespace(x=["use_test_database=true"])
    command.upgrade(config, "head")
    return test_database_url


def _settings(
    database_url: str,
    *,
    configured: bool = True,
    estimated_token_budget: int = 10_000,
    budget_only: bool = False,
) -> Settings:
    return Settings(
        environment="test",
        database_url=database_url,
        local_user_id=1,
        qwen_api_key=None,
        qwen_base_url=None,
        memory_policy_model="test-memory-policy" if configured else None,
        memory_policy_estimated_token_budget=(
            estimated_token_budget if configured or budget_only else None
        ),
    )


def _bootstrap(
    database_url: str,
    *,
    configured: bool = True,
    estimated_token_budget: int = 10_000,
    budget_only: bool = False,
) -> None:
    assert asyncio.run(
        bootstrap_local_user(
            _settings(
                database_url,
                configured=configured,
                estimated_token_budget=estimated_token_budget,
                budget_only=budget_only,
            )
        )
    )


def _seed_pending_evidence(
    database_url: str,
    conversations: list[tuple[list[str], bool]],
) -> list[tuple[int, datetime]]:
    async def seed() -> list[tuple[int, datetime]]:
        engine = create_database_engine(database_url)
        session_factory = create_session_factory(engine)
        try:
            async with session_factory() as session, session.begin():
                base = utc_now().replace(microsecond=0)
                facts: list[tuple[int, datetime]] = []
                offset = 0
                for contents, deleted in conversations:
                    conversation = Conversation(
                        user_id=1,
                        title=None,
                        created_at=base,
                        updated_at=base,
                        last_message_at=base,
                        deleted_at=base if deleted else None,
                    )
                    session.add(conversation)
                    await session.flush()
                    for sequence_no, content in enumerate(contents, start=1):
                        created_at = base + timedelta(seconds=offset)
                        offset += 1
                        message = Message(
                            conversation_id=conversation.id,
                            sequence_no=sequence_no,
                            role="USER",
                            content=content,
                            run_id=None,
                            regenerated_from_message_id=None,
                            created_at=created_at,
                        )
                        session.add(message)
                        await session.flush()
                        facts.append((message.id, created_at))
                return facts
        finally:
            await dispose_database_engine(engine)

    return asyncio.run(seed())


def test_memory_status_and_empty_sync_do_not_require_policy(
    migrated_database: str,
) -> None:
    _bootstrap(migrated_database, configured=False)
    app = create_app(_settings(migrated_database, configured=False))

    with TestClient(app) as client:
        assert client.get("/api/memory-status").json() == {
            "auto_memory_enabled": True,
            "policy_status": "NOT_CONFIGURED",
            "pending_evidence_count": 0,
            "oldest_pending_message_id": None,
            "oldest_pending_created_at": None,
        }
        response = client.post("/api/memory-sync")
        assert response.status_code == 200
        assert response.json() == {
            "processed_count": 0,
            "remaining_count": 0,
            "complete": True,
            "stop_reason": "COMPLETE",
            "oldest_pending_message_id": None,
            "oldest_pending_created_at": None,
        }


def test_empty_backlog_direct_crud_uses_budget_without_policy_provider(
    migrated_database: str,
) -> None:
    budget = estimate_load_all_memory_contribution(["abc"])
    _bootstrap(
        migrated_database,
        configured=False,
        estimated_token_budget=budget,
        budget_only=True,
    )
    delete_id = _insert_memory(migrated_database, content="delete me")
    app = create_app(
        _settings(
            migrated_database,
            configured=False,
            estimated_token_budget=budget,
            budget_only=True,
        )
    )

    with TestClient(app) as client:
        assert client.delete(f"/api/memories/{delete_id}").status_code == 204

        created = client.post("/api/memories", json={"content": "abc"})
        assert created.status_code == 201
        memory_id = created.json()["id"]

        corrected = client.put(f"/api/memories/{memory_id}", json={"content": "xyz"})
        assert corrected.status_code == 200
        assert corrected.json()["content"] == "xyz"


def test_empty_backlog_growing_write_still_requires_capacity_budget(
    migrated_database: str,
) -> None:
    _bootstrap(migrated_database, configured=False)
    app = create_app(_settings(migrated_database, configured=False))

    with TestClient(app) as client:
        response = client.post("/api/memories", json={"content": "direct"})
        assert response.status_code == 503
        assert response.json() == {"detail": {"code": "MEMORY_CAPACITY_UNAVAILABLE"}}


def test_memory_status_and_explicit_sync_progress_are_user_global(
    migrated_database: str,
) -> None:
    _bootstrap(migrated_database)
    facts = _seed_pending_evidence(
        migrated_database,
        [(["one", "two"], True), (["three", "four", "five"], False)],
    )
    provider = FakeProvider(
        [
            ScriptedProviderRound(
                events=(
                    _completion(
                        {"mutations": [], "user_requested_memory_action": False}
                    ),
                )
            )
            for _ in range(5)
        ]
    )
    app = create_app(_settings(migrated_database), memory_provider=provider)

    with TestClient(app) as client:
        assert client.get("/api/memory-status").json() == {
            "auto_memory_enabled": True,
            "policy_status": "READY",
            "pending_evidence_count": 5,
            "oldest_pending_message_id": facts[0][0],
            "oldest_pending_created_at": as_utc(facts[0][1]),
        }

        first = client.post("/api/memory-sync")
        assert first.status_code == 200
        assert first.json() == {
            "processed_count": 4,
            "remaining_count": 1,
            "complete": False,
            "stop_reason": "LIMIT_REACHED",
            "oldest_pending_message_id": facts[4][0],
            "oldest_pending_created_at": as_utc(facts[4][1]),
        }
        assert (
            _scalar(
                migrated_database,
                "SELECT COUNT(*) FROM messages WHERE memory_processed_at IS NOT NULL",
            )
            == 4
        )

        second = client.post("/api/memory-sync")
        assert second.status_code == 200
        assert second.json() == {
            "processed_count": 1,
            "remaining_count": 0,
            "complete": True,
            "stop_reason": "COMPLETE",
            "oldest_pending_message_id": None,
            "oldest_pending_created_at": None,
        }
        assert (
            _scalar(
                migrated_database,
                "SELECT COUNT(*) FROM messages WHERE memory_processed_at IS NOT NULL",
            )
            == 5
        )

        no_backlog = client.post("/api/memory-sync")
        assert no_backlog.status_code == 200
        assert no_backlog.json()["processed_count"] == 0
        assert no_backlog.json()["stop_reason"] == "COMPLETE"
        assert len(provider.requests) == 5


def test_direct_add_reports_durable_bounded_progress_before_retry(
    migrated_database: str,
) -> None:
    _bootstrap(migrated_database)
    _seed_pending_evidence(
        migrated_database,
        [([f"pending {index}" for index in range(5)], False)],
    )
    provider = FakeProvider(
        [
            ScriptedProviderRound(
                events=(
                    _completion(
                        {"mutations": [], "user_requested_memory_action": False}
                    ),
                )
            )
            for _ in range(5)
        ]
    )
    app = create_app(_settings(migrated_database), memory_provider=provider)

    with TestClient(app) as client:
        incomplete = client.post("/api/memories", json={"content": "direct"})
        assert incomplete.status_code == 409
        assert incomplete.json() == {
            "detail": {
                "code": "MEMORY_SYNC_INCOMPLETE",
                "stop_reason": "LIMIT_REACHED",
                "processed_count": 4,
                "remaining_count": 1,
            }
        }
        assert client.get("/api/memories").json() == []

        continued = client.post("/api/memory-sync")
        assert continued.status_code == 200
        assert continued.json()["stop_reason"] == "COMPLETE"
        assert continued.json()["processed_count"] == 1

        retried = client.post("/api/memories", json={"content": "direct"})
        assert retried.status_code == 201
        assert len(provider.requests) == 5


def test_off_to_on_reports_bounded_progress_and_remains_off_until_complete(
    migrated_database: str,
) -> None:
    _bootstrap(migrated_database)
    _seed_pending_evidence(
        migrated_database,
        [([f"pending {index}" for index in range(5)], False)],
    )
    provider = FakeProvider(
        [
            ScriptedProviderRound(
                events=(
                    _completion(
                        {"mutations": [], "user_requested_memory_action": False}
                    ),
                )
            )
            for _ in range(5)
        ]
    )
    app = create_app(_settings(migrated_database), memory_provider=provider)

    with TestClient(app) as client:
        disabled = client.patch(
            "/api/memory-settings", json={"auto_memory_enabled": False}
        )
        assert disabled.status_code == 200
        assert provider.requests == []

        incomplete = client.patch(
            "/api/memory-settings", json={"auto_memory_enabled": True}
        )
        assert incomplete.status_code == 409
        assert incomplete.json() == {
            "detail": {
                "code": "MEMORY_SYNC_INCOMPLETE",
                "stop_reason": "LIMIT_REACHED",
                "processed_count": 4,
                "remaining_count": 1,
            }
        }
        assert client.get("/api/memory-settings").json() == {
            "auto_memory_enabled": False
        }

        completed = client.patch(
            "/api/memory-settings", json={"auto_memory_enabled": True}
        )
        assert completed.status_code == 200
        assert completed.json() == {"auto_memory_enabled": True}
        assert len(provider.requests) == 5


def test_memory_sync_unavailable_provider_configuration_preserves_pending(
    migrated_database: str,
) -> None:
    _bootstrap(migrated_database)
    facts = _seed_pending_evidence(migrated_database, [(["pending"], False)])
    app = create_app(_settings(migrated_database))

    with TestClient(app) as client:
        assert (
            client.get("/api/memory-status").json()["policy_status"]
            == "PROVIDER_CONFIGURATION_UNAVAILABLE"
        )
        response = client.post("/api/memory-sync")
        assert response.status_code == 503
        assert response.json() == {
            "detail": {
                "code": "MEMORY_SYNC_UNAVAILABLE",
                "stop_reason": "POLICY_UNAVAILABLE",
                "processed_count": 0,
                "remaining_count": 1,
            }
        }
        status_response = client.get("/api/memory-status").json()
        assert status_response["pending_evidence_count"] == 1
        assert status_response["oldest_pending_message_id"] == facts[0][0]


def test_memory_sync_provider_failure_preserves_pending(
    migrated_database: str,
) -> None:
    _bootstrap(migrated_database)
    _seed_pending_evidence(migrated_database, [(["pending"], False)])
    provider = FakeProvider(
        [
            ScriptedProviderRound(
                events=(), failure=WorkflowFailure(RunErrorCode.LLM_PROVIDER_FAILED)
            )
        ]
    )
    app = create_app(_settings(migrated_database), memory_provider=provider)

    with TestClient(app) as client:
        response = client.post("/api/memory-sync")
        assert response.status_code == 503
        assert response.json() == {
            "detail": {
                "code": "MEMORY_SYNC_UNAVAILABLE",
                "stop_reason": "PROVIDER_FAILURE",
                "processed_count": 0,
                "remaining_count": 1,
            }
        }
        assert client.get("/api/memory-status").json()["pending_evidence_count"] == 1


def test_memory_sync_context_infeasible_is_distinct_and_preserves_pending(
    migrated_database: str,
) -> None:
    budget = estimate_load_all_memory_contribution(["existing"]) - 1
    _bootstrap(migrated_database, estimated_token_budget=budget)
    _seed_pending_evidence(migrated_database, [(["pending"], False)])
    _insert_memory(migrated_database, content="existing")
    provider = FakeProvider([])
    app = create_app(
        _settings(migrated_database, estimated_token_budget=budget),
        memory_provider=provider,
    )

    with TestClient(app) as client:
        response = client.post("/api/memory-sync")
        assert response.status_code == 409
        assert response.json() == {
            "detail": {
                "code": "MEMORY_SYNC_BLOCKED",
                "stop_reason": "CONTEXT_INFEASIBLE",
                "processed_count": 0,
                "remaining_count": 1,
            }
        }
        assert client.get("/api/memory-status").json()["pending_evidence_count"] == 1
        assert provider.requests == []


def test_direct_barrier_provider_failure_uses_stable_progress_mapping(
    migrated_database: str,
) -> None:
    _bootstrap(migrated_database)
    _seed_pending_evidence(migrated_database, [(["pending"], False)])
    provider = FakeProvider(
        [
            ScriptedProviderRound(
                events=(), failure=WorkflowFailure(RunErrorCode.LLM_PROVIDER_FAILED)
            )
        ]
    )
    app = create_app(_settings(migrated_database), memory_provider=provider)

    with TestClient(app) as client:
        response = client.post("/api/memories", json={"content": "direct"})
        assert response.status_code == 503
        assert response.json() == {
            "detail": {
                "code": "MEMORY_SYNC_UNAVAILABLE",
                "stop_reason": "PROVIDER_FAILURE",
                "processed_count": 0,
                "remaining_count": 1,
            }
        }


def test_direct_barrier_context_infeasible_uses_blocked_mapping(
    migrated_database: str,
) -> None:
    budget = estimate_load_all_memory_contribution(["existing"]) - 1
    _bootstrap(migrated_database, estimated_token_budget=budget)
    _seed_pending_evidence(migrated_database, [(["pending"], False)])
    memory_id = _insert_memory(migrated_database, content="existing")
    provider = FakeProvider([])
    app = create_app(
        _settings(migrated_database, estimated_token_budget=budget),
        memory_provider=provider,
    )

    with TestClient(app) as client:
        response = client.delete(f"/api/memories/{memory_id}")
        assert response.status_code == 409
        assert response.json() == {
            "detail": {
                "code": "MEMORY_SYNC_BLOCKED",
                "stop_reason": "CONTEXT_INFEASIBLE",
                "processed_count": 0,
                "remaining_count": 1,
            }
        }
        assert client.get(f"/api/memories/{memory_id}/source").status_code == 200
        assert provider.requests == []


def test_direct_barrier_timeout_uses_stable_progress_mapping(
    migrated_database: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _bootstrap(migrated_database)
    _seed_pending_evidence(migrated_database, [(["pending"], False)])
    provider = FakeProvider(
        [
            ScriptedProviderRound(
                events=(
                    _completion(
                        {"mutations": [], "user_requested_memory_action": False}
                    ),
                )
            )
        ]
    )
    monkeypatch.setattr("langley.memory.processing.MANUAL_SYNC_TIMEOUT_SECONDS", 0)
    app = create_app(_settings(migrated_database), memory_provider=provider)

    with TestClient(app) as client:
        response = client.post("/api/memories", json={"content": "direct"})
        assert response.status_code == 503
        assert response.json() == {
            "detail": {
                "code": "MEMORY_SYNC_UNAVAILABLE",
                "stop_reason": "TIMEOUT",
                "processed_count": 0,
                "remaining_count": 1,
            }
        }


def test_memory_api_crud_validation_and_settings(migrated_database: str) -> None:
    _bootstrap(migrated_database)
    app = create_app(_settings(migrated_database), memory_provider=FakeProvider([]))
    future = (datetime.now(UTC) + timedelta(days=2)).isoformat()
    with TestClient(app) as client:
        created = client.post(
            "/api/memories", json={"content": "prefers tea", "valid_until": future}
        )
        assert created.status_code == 201
        memory_id = created.json()["id"]
        assert created.json()["valid_until"].endswith("Z")

        second = client.post("/api/memories", json={"content": "uses vim"})
        assert second.status_code == 201
        listed = client.get("/api/memories")
        assert [item["content"] for item in listed.json()] == [
            "uses vim",
            "prefers tea",
        ]

        corrected = client.put(
            f"/api/memories/{memory_id}", json={"content": "prefers oolong"}
        )
        assert corrected.status_code == 200
        assert corrected.json()["source_message_id"] is None
        assert client.delete(f"/api/memories/{memory_id}").status_code == 204
        assert client.delete(f"/api/memories/{memory_id}").status_code == 204

        for payload in (
            {"content": ""},
            {"content": "x" * 1001},
            {"content": "valid", "valid_until": "not-a-date"},
            {"content": "valid", "valid_until": "2030-01-01T00:00:00"},
            {"content": "valid", "valid_until": "2000-01-01T00:00:00Z"},
        ):
            response = client.post("/api/memories", json=payload)
            assert response.status_code == 422
            assert response.json() == {"detail": {"code": "VALIDATION_ERROR"}}

        assert client.get("/api/memory-settings").json() == {
            "auto_memory_enabled": True
        }
        assert client.patch(
            "/api/memory-settings", json={"auto_memory_enabled": False}
        ).json() == {"auto_memory_enabled": False}
        assert client.patch(
            "/api/memory-settings", json={"auto_memory_enabled": True}
        ).json() == {"auto_memory_enabled": True}


def test_memory_api_enforces_direct_write_capacity(migrated_database: str) -> None:
    estimated_token_budget = estimate_load_all_memory_contribution(["abc"])
    _bootstrap(migrated_database, estimated_token_budget=estimated_token_budget)
    app = create_app(
        _settings(migrated_database, estimated_token_budget=estimated_token_budget),
        memory_provider=FakeProvider([]),
    )

    with TestClient(app) as client:
        exact_fit = client.post("/api/memories", json={"content": "abc"})
        assert exact_fit.status_code == 201
        memory_id = exact_fit.json()["id"]

        over_capacity_add = client.post("/api/memories", json={"content": "x"})
        assert over_capacity_add.status_code == 409
        assert over_capacity_add.json() == {
            "detail": {"code": "MEMORY_CAPACITY_REACHED"}
        }

        within_capacity_correction = client.put(
            f"/api/memories/{memory_id}", json={"content": "a"}
        )
        assert within_capacity_correction.status_code == 200
        assert within_capacity_correction.json()["content"] == "a"

        over_capacity_correction = client.put(
            f"/api/memories/{memory_id}", json={"content": "xxxx"}
        )
        assert over_capacity_correction.status_code == 409
        assert over_capacity_correction.json() == {
            "detail": {"code": "MEMORY_CAPACITY_REACHED"}
        }
        assert client.get("/api/memories").json()[0]["content"] == "a"

        assert client.delete(f"/api/memories/{memory_id}").status_code == 204
        assert client.get("/api/memories").json() == []


def test_memory_api_source_ownership_and_unavailable_barrier(
    migrated_database: str,
) -> None:
    async def seed() -> tuple[int, int, int]:
        engine = create_database_engine(migrated_database)
        session_factory = create_session_factory(engine)
        try:
            async with session_factory() as session, session.begin():
                now = utc_now()
                session.add_all(
                    [User(id=1, created_at=now), User(id=2, created_at=now)]
                )
                await session.flush()
                conversation = Conversation(
                    user_id=1,
                    title="Archived source",
                    created_at=now,
                    updated_at=now,
                    last_message_at=now,
                    deleted_at=now,
                )
                session.add(conversation)
                await session.flush()
                messages = [
                    Message(
                        conversation_id=conversation.id,
                        sequence_no=index,
                        role="USER",
                        content=content,
                        run_id=None,
                        regenerated_from_message_id=None,
                        created_at=now,
                    )
                    for index, content in enumerate(
                        ("before", "evidence", "after"), start=1
                    )
                ]
                session.add_all(messages)
                await session.flush()
                memory = Memory(
                    user_id=1,
                    content="from source",
                    source_message_id=messages[1].id,
                    valid_until=None,
                    created_at=now,
                    updated_at=now,
                )
                other = Memory(
                    user_id=2,
                    content="other",
                    source_message_id=None,
                    valid_until=None,
                    created_at=now,
                    updated_at=now,
                )
                session.add_all([memory, other])
                await session.flush()
                return memory.id, other.id, conversation.id
        finally:
            await dispose_database_engine(engine)

    memory_id, other_id, _ = asyncio.run(seed())
    app = create_app(_settings(migrated_database, configured=False))
    with TestClient(app) as client:
        source = client.get(f"/api/memories/{memory_id}/source")
        assert source.status_code == 200
        assert source.json()["kind"] == "conversation"
        assert source.json()["conversation_deleted"] is True
        assert [item["content"] for item in source.json()["context_messages"]] == [
            "before",
            "evidence",
            "after",
        ]
        assert client.get(f"/api/memories/{other_id}/source").status_code == 404
        unavailable = client.post("/api/memories", json={"content": "direct"})
        assert unavailable.status_code == 503
        assert unavailable.json() == {
            "detail": {
                "code": "MEMORY_SYNC_UNAVAILABLE",
                "stop_reason": "POLICY_UNAVAILABLE",
                "processed_count": 0,
                "remaining_count": 3,
            }
        }
        assert (
            client.put(
                f"/api/memories/{memory_id}", json={"content": "corrected"}
            ).json()
            == unavailable.json()
        )
        assert client.delete(f"/api/memories/{memory_id}").json() == unavailable.json()
        assert (
            client.patch(
                "/api/memory-settings", json={"auto_memory_enabled": False}
            ).status_code
            == 200
        )
        failed_enable = client.patch(
            "/api/memory-settings", json={"auto_memory_enabled": True}
        )
        assert failed_enable.status_code == 503
        assert client.get("/api/memory-settings").json() == {
            "auto_memory_enabled": False
        }


def test_delete_releases_preread_session_before_slow_memory_barrier(
    migrated_database: str,
) -> None:
    active_sessions = 0

    class GatedProvider:
        def __init__(self) -> None:
            self.started = Event()
            self.release = Event()

        async def stream(self, _request):
            assert active_sessions == 0
            self.started.set()
            await asyncio.to_thread(self.release.wait)
            yield LLMResponseCompleted(
                assistant_content="[]",
                tool_calls=(),
                finish_reason=LLMFinishReason.STOP,
                usage=None,
            )

    async def seed() -> int:
        engine = create_database_engine(migrated_database)
        session_factory = create_session_factory(engine)
        try:
            async with session_factory() as session, session.begin():
                now = utc_now()
                user = User(id=1, created_at=now)
                session.add(user)
                await session.flush()
                conversation = Conversation(
                    user_id=1,
                    title=None,
                    created_at=now,
                    updated_at=now,
                    last_message_at=now,
                    deleted_at=None,
                )
                session.add(conversation)
                await session.flush()
                session.add(
                    Message(
                        conversation_id=conversation.id,
                        sequence_no=1,
                        role="USER",
                        content="slow policy evidence",
                        run_id=None,
                        regenerated_from_message_id=None,
                        created_at=now,
                    )
                )
                memory = Memory(
                    user_id=1,
                    content="delete me",
                    source_message_id=None,
                    valid_until=None,
                    created_at=now,
                    updated_at=now,
                )
                session.add(memory)
                await session.flush()
                return memory.id
        finally:
            await dispose_database_engine(engine)

    memory_id = asyncio.run(seed())
    provider = GatedProvider()
    app = create_app(_settings(migrated_database), memory_provider=provider)
    original_factory = app.state.session_factory

    @asynccontextmanager
    async def observing_factory():
        nonlocal active_sessions
        async with original_factory() as session:
            active_sessions += 1
            try:
                yield session
            finally:
                active_sessions -= 1

    app.state.session_factory = observing_factory
    response = None

    def delete() -> None:
        nonlocal response
        with TestClient(app) as client:
            response = client.delete(f"/api/memories/{memory_id}")

    thread = Thread(target=delete)
    thread.start()
    assert provider.started.wait(timeout=5)
    provider.release.set()
    thread.join(timeout=5)
    assert response is not None
    assert response.status_code == 204
