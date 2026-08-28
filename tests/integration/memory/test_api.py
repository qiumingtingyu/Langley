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
from langley.answering.fake_provider import FakeProvider
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
) -> Settings:
    return Settings(
        environment="test",
        database_url=database_url,
        local_user_id=1,
        memory_policy_model="test-memory-policy" if configured else None,
        memory_policy_estimated_token_budget=(
            estimated_token_budget if configured else None
        ),
    )


def _bootstrap(database_url: str, *, estimated_token_budget: int = 10_000) -> None:
    assert asyncio.run(
        bootstrap_local_user(
            _settings(database_url, estimated_token_budget=estimated_token_budget)
        )
    )


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
        assert unavailable.json() == {"detail": {"code": "MEMORY_SYNC_UNAVAILABLE"}}
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
