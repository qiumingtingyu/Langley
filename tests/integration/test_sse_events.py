"""Real MySQL integration tests for Slice 3 transient Run SSE observation."""

import asyncio
import json
from argparse import Namespace
from collections.abc import AsyncIterator, Callable, Coroutine
from typing import Any

import pytest
from agent_workflow import workflow_for
from alembic import command
from alembic.config import Config
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

import langley.api.runs as runs_api
from langley.answer_execution import AnswerExecutionManager
from langley.answering.contracts import (
    AssistantContentDelta,
    LLMFinishReason,
    LLMResponseCompleted,
)
from langley.answering.fake_provider import FakeProvider, ScriptedProviderRound
from langley.api.runs import get_run_events
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
    """Provide an empty real MySQL database migrated to current head."""

    reset_database()
    config = Config("alembic.ini")
    config.cmd_opts = Namespace(x=["use_test_database=true"])
    command.upgrade(config, "head")
    return test_database_url


def _settings(database_url: str) -> Settings:
    return Settings(environment="test", database_url=database_url, local_user_id=1)


def _bootstrap(database_url: str) -> None:
    assert asyncio.run(bootstrap_local_user(_settings(database_url)))


class GatedSseFake:
    """Yield deterministic chunks only when the test opens explicit gates."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.allow_first = asyncio.Event()
        self.first_sent = asyncio.Event()
        self.release = asyncio.Event()

        self.provider = FakeProvider(
            [
                ScriptedProviderRound(
                    events=(
                        AssistantContentDelta(content="AB"),
                        AssistantContentDelta(content="CD"),
                        AssistantContentDelta(content="EF"),
                        LLMResponseCompleted(
                            assistant_content="ABCDEF",
                            tool_calls=(),
                            finish_reason=LLMFinishReason.STOP,
                            usage=None,
                        ),
                    ),
                    started=self.started,
                    blocked_until=self.allow_first,
                    event_reached=self.first_sent,
                    event_reached_after_count=1,
                    additional_blocked_after_events=((1, self.release),),
                )
            ]
        )

    @property
    def calls(self) -> int:
        return len(self.provider.requests)


class CapturingLogger:
    """Capture structured runtime-availability diagnostics."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def warning(self, event: str, **fields: object) -> None:
        self.events.append((event, fields))


async def _exercise(
    database_url: str,
    operation: Callable[[async_sessionmaker[AsyncSession]], Coroutine[Any, Any, None]],
) -> None:
    engine = create_database_engine(database_url)
    try:
        await operation(create_session_factory(engine))
    finally:
        await dispose_database_engine(engine)


async def _run(session_factory: async_sessionmaker[AsyncSession], run_id: int) -> Run:
    async with session_factory() as session:
        run = await session.get(Run, run_id)
        assert run is not None
        return run


async def _assistant_count(
    session_factory: async_sessionmaker[AsyncSession], run_id: int
) -> int:
    async with session_factory() as session:
        return len(
            list(
                (
                    await session.scalars(
                        select(Message).where(Message.run_id == run_id)
                    )
                ).all()
            )
        )


async def _new_run(client: AsyncClient, request_id: str) -> int:
    conversation = await client.post("/api/conversations", json={})
    assert conversation.status_code == 201
    accepted = await client.post(
        f"/api/conversations/{conversation.json()['id']}/messages",
        json={"content": "question", "client_request_id": request_id},
    )
    assert accepted.status_code == 202
    return accepted.json()["run"]["id"]


async def _open_events(
    session_factory: async_sessionmaker[AsyncSession],
    manager: AnswerExecutionManager,
    run_id: int,
):
    async with session_factory() as session:
        return await get_run_events(
            run_id,
            session=session,
            execution_manager=manager,
            current_user_id=1,
        )


async def _next_event(
    body: AsyncIterator[bytes | str],
) -> tuple[str, dict[str, Any]]:
    frame = await anext(body)
    text = frame.decode() if isinstance(frame, bytes) else frame
    assert text.endswith("\n\n")
    event_line, data_line = text[:-2].split("\n")
    assert event_line.startswith("event: ")
    assert data_line.startswith("data: ")
    return event_line.removeprefix("event: "), json.loads(
        data_line.removeprefix("data: ")
    )


async def _seed_terminal_run(
    session_factory: async_sessionmaker[AsyncSession], *, status: str
) -> Run:
    now = utc_now()
    async with session_factory() as session:
        async with session.begin():
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
            user_message = Message(
                conversation_id=conversation.id,
                sequence_no=1,
                role="USER",
                content="question",
                run_id=None,
                regenerated_from_message_id=None,
                created_at=now,
            )
            session.add(user_message)
            await session.flush()
            run = Run(
                conversation_id=conversation.id,
                input_message_id=user_message.id,
                client_request_id=f"terminal-{status}-{conversation.id}",
                attempt_no=1,
                status=status,
                started_at=now if status in {"RUNNING", "SUCCEEDED"} else None,
                finished_at=(
                    now if status in {"SUCCEEDED", "FAILED", "CANCELLED"} else None
                ),
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
                        content="persisted assistant",
                        run_id=run.id,
                        regenerated_from_message_id=None,
                        created_at=now,
                    )
                )
    return run


def test_pending_sse_subscription_streams_framed_events_after_commit(
    migrated_database: str,
) -> None:
    _bootstrap(migrated_database)

    async def operation(session_factory: async_sessionmaker[AsyncSession]) -> None:
        fake = GatedSseFake()
        start_execution = asyncio.Event()

        def delayed_scheduler(coroutine):
            async def delayed_start() -> None:
                try:
                    await start_execution.wait()
                    await coroutine
                except BaseException:
                    coroutine.close()
                    raise

            return asyncio.create_task(delayed_start())

        app = create_app(_settings(migrated_database))
        app.state.execution_manager = AnswerExecutionManager(
            session_factory,
            lambda: workflow_for(fake.provider),
            task_scheduler=delayed_scheduler,
        )
        async with app.router.lifespan_context(app):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                run_id = await _new_run(client, "pending-sse")
                assert (await _run(session_factory, run_id)).status == "PENDING"
                response = await _open_events(
                    session_factory, app.state.execution_manager, run_id
                )
                assert response.media_type == "text/event-stream"
                assert response.headers["cache-control"] == "no-cache"
                body = response.body_iterator

                start_execution.set()
                assert await _next_event(body) == ("run.started", {"run_id": run_id})
                assert (await _run(session_factory, run_id)).status == "RUNNING"
                fake.allow_first.set()
                assert await _next_event(body) == (
                    "message.delta",
                    {"run_id": run_id, "delta": "AB"},
                )
                await fake.first_sent.wait()
                fake.release.set()
                assert await _next_event(body) == (
                    "message.delta",
                    {"run_id": run_id, "delta": "CD"},
                )
                assert await _next_event(body) == (
                    "message.delta",
                    {"run_id": run_id, "delta": "EF"},
                )
                assert await _next_event(body) == ("run.succeeded", {"run_id": run_id})
                with pytest.raises(StopAsyncIteration):
                    await anext(body)
                assert (await _run(session_factory, run_id)).status == "SUCCEEDED"
                assert await _assistant_count(session_factory, run_id) == 1

    asyncio.run(_exercise(migrated_database, operation))


def test_late_subscribe_receives_full_prefix_then_gap_free_future_deltas(
    migrated_database: str,
) -> None:
    _bootstrap(migrated_database)

    async def operation(session_factory: async_sessionmaker[AsyncSession]) -> None:
        fake = GatedSseFake()
        app = create_app(_settings(migrated_database), provider=fake.provider)
        async with app.router.lifespan_context(app):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                run_id = await _new_run(client, "late-prefix")
                await fake.started.wait()
                fake.allow_first.set()
                await fake.first_sent.wait()

                response = await _open_events(
                    session_factory, app.state.execution_manager, run_id
                )
                body = response.body_iterator
                assert await _next_event(body) == (
                    "message.delta",
                    {"run_id": run_id, "delta": "AB"},
                )
                fake.release.set()
                assert await _next_event(body) == (
                    "message.delta",
                    {"run_id": run_id, "delta": "CD"},
                )
                assert await _next_event(body) == (
                    "message.delta",
                    {"run_id": run_id, "delta": "EF"},
                )
                assert await _next_event(body) == ("run.succeeded", {"run_id": run_id})

    asyncio.run(_exercise(migrated_database, operation))


def test_terminal_before_subscribe_emits_authoritative_terminal_and_closes(
    migrated_database: str,
) -> None:
    _bootstrap(migrated_database)

    async def operation(session_factory: async_sessionmaker[AsyncSession]) -> None:
        terminal_runs = {
            status: await _seed_terminal_run(session_factory, status=status)
            for status in ("SUCCEEDED", "FAILED", "CANCELLED")
        }
        app = create_app(_settings(migrated_database))
        async with app.router.lifespan_context(app):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                for status, run in terminal_runs.items():
                    response = await client.get(f"/api/runs/{run.id}/events")
                    assert response.status_code == 200
                    assert response.headers["content-type"].startswith(
                        "text/event-stream"
                    )
                    if status == "FAILED":
                        expected = (
                            f'event: run.failed\ndata: {{"run_id":{run.id},'
                            '"error_code":"ANSWER_EXECUTION_FAILED"}\n\n'
                        )
                    else:
                        expected = (
                            f"event: run.{status.lower()}\ndata: "
                            f'{{"run_id":{run.id}}}\n\n'
                        )
                    assert response.text == expected

    asyncio.run(_exercise(migrated_database, operation))


def test_active_run_without_local_runtime_records_availability_warning(
    migrated_database: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An active authoritative Run with no local owner closes SSE diagnostically."""

    _bootstrap(migrated_database)
    logger = CapturingLogger()
    monkeypatch.setattr(runs_api.structlog, "get_logger", lambda _: logger)

    async def operation(session_factory: async_sessionmaker[AsyncSession]) -> None:
        app = create_app(_settings(migrated_database))
        async with app.router.lifespan_context(app):
            run = await _seed_terminal_run(session_factory, status="PENDING")
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.get(f"/api/runs/{run.id}/events")

        assert response.status_code == 200
        assert response.text == ""
        missing_runtime_events = [
            event
            for event in logger.events
            if event[0] == "answer.run.active_answer_missing"
        ]
        assert missing_runtime_events == [
            (
                "answer.run.active_answer_missing",
                {
                    "run_id": run.id,
                    "conversation_id": run.conversation_id,
                    "run_status": "PENDING",
                },
            )
        ]

    asyncio.run(_exercise(migrated_database, operation))


def test_subscribe_terminal_handshake_rereads_authority_without_hanging(
    migrated_database: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bootstrap(migrated_database)

    async def operation(session_factory: async_sessionmaker[AsyncSession]) -> None:
        fake = GatedSseFake()
        app = create_app(_settings(migrated_database), provider=fake.provider)
        first_read = asyncio.Event()
        release_attach = asyncio.Event()
        original_get_owned_run = runs_api.get_owned_run
        calls = 0

        async def gated_get_owned_run(*args, **kwargs):
            nonlocal calls
            result = await original_get_owned_run(*args, **kwargs)
            calls += 1
            if calls == 1:
                first_read.set()
                await release_attach.wait()
            return result

        monkeypatch.setattr(runs_api, "get_owned_run", gated_get_owned_run)
        async with app.router.lifespan_context(app):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                run_id = await _new_run(client, "handshake-race")
                await fake.started.wait()
                async with session_factory() as session:
                    response_task = asyncio.create_task(
                        get_run_events(
                            run_id,
                            session=session,
                            execution_manager=app.state.execution_manager,
                            current_user_id=1,
                        )
                    )
                    await first_read.wait()
                    cancelled = await client.post(f"/api/runs/{run_id}/cancel")
                    assert cancelled.status_code == 200
                    release_attach.set()
                    response = await response_task
                body = response.body_iterator
                assert await _next_event(body) == ("run.cancelled", {"run_id": run_id})
                with pytest.raises(StopAsyncIteration):
                    await anext(body)

    asyncio.run(_exercise(migrated_database, operation))


def test_disconnect_and_multiple_subscribers_do_not_stop_generation(
    migrated_database: str,
) -> None:
    _bootstrap(migrated_database)

    async def operation(session_factory: async_sessionmaker[AsyncSession]) -> None:
        fake = GatedSseFake()
        app = create_app(_settings(migrated_database), provider=fake.provider)
        async with app.router.lifespan_context(app):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                run_id = await _new_run(client, "two-subscribers")
                await fake.started.wait()
                first = await _open_events(
                    session_factory, app.state.execution_manager, run_id
                )
                second = await _open_events(
                    session_factory, app.state.execution_manager, run_id
                )
                first_body = first.body_iterator
                second_body = second.body_iterator
                fake.allow_first.set()

                for body in (first_body, second_body):
                    assert await _next_event(body) == (
                        "message.delta",
                        {"run_id": run_id, "delta": "AB"},
                    )
                await first_body.aclose()
                fake.release.set()
                assert await _next_event(second_body) == (
                    "message.delta",
                    {"run_id": run_id, "delta": "CD"},
                )
                assert await _next_event(second_body) == (
                    "message.delta",
                    {"run_id": run_id, "delta": "EF"},
                )
                assert await _next_event(second_body) == (
                    "run.succeeded",
                    {"run_id": run_id},
                )
                assert (await _run(session_factory, run_id)).status == "SUCCEEDED"

    asyncio.run(_exercise(migrated_database, operation))


def test_zero_subscribers_do_not_block_generation(migrated_database: str) -> None:
    _bootstrap(migrated_database)

    async def operation(session_factory: async_sessionmaker[AsyncSession]) -> None:
        fake = GatedSseFake()
        app = create_app(_settings(migrated_database), provider=fake.provider)
        async with app.router.lifespan_context(app):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                run_id = await _new_run(client, "zero-subscribers")
                answer = app.state.execution_manager._active_answers.get(run_id)
                assert answer is not None and answer.task is not None
                await fake.started.wait()
                fake.allow_first.set()
                await fake.first_sent.wait()
                fake.release.set()
                await answer.task
                assert (await _run(session_factory, run_id)).status == "SUCCEEDED"
                assert await _assistant_count(session_factory, run_id) == 1

    asyncio.run(_exercise(migrated_database, operation))
