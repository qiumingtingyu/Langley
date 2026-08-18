"""Real MySQL integration tests for single-worker interruption lifecycle repair."""

import asyncio
from argparse import Namespace
from collections.abc import Callable, Coroutine
from datetime import timedelta
from typing import Any

import pytest
from agent_workflow import workflow_for
from alembic import command
from alembic.config import Config
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from uvicorn import Config as UvicornConfig
from uvicorn import Server

import langley.answer_lifecycle as answer_lifecycle
from langley.answer_execution import (
    AnswerExecutionManager,
    _commit_success,
)
from langley.answer_lifecycle import interrupt_active_runs
from langley.answering.contracts import (
    AssistantContentDelta,
    LLMFinishReason,
    LLMResponseCompleted,
)
from langley.answering.fake_provider import FakeProvider, ScriptedProviderRound
from langley.bootstrap import bootstrap_local_user
from langley.business_time import utc_now
from langley.conversation_commands import admit_new_question
from langley.conversations import create_conversation
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


class GatedFake:
    """Keep generation active until the test decides to release or interrupt it."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.provider = FakeProvider(
            [
                ScriptedProviderRound(
                    events=(
                        AssistantContentDelta(content="answer"),
                        LLMResponseCompleted(
                            assistant_content="answer",
                            tool_calls=(),
                            finish_reason=LLMFinishReason.STOP,
                            usage=None,
                        ),
                    ),
                    started=self.started,
                    blocked_until=self.release,
                )
            ]
        )


class PartiallyGatedFake:
    """Expose one SSE delta, then keep the execution active."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.provider = FakeProvider(
            [
                ScriptedProviderRound(
                    events=(
                        AssistantContentDelta(content="partial"),
                        AssistantContentDelta(content="answer"),
                        LLMResponseCompleted(
                            assistant_content="partialanswer",
                            tool_calls=(),
                            finish_reason=LLMFinishReason.STOP,
                            usage=None,
                        ),
                    ),
                    started=self.started,
                    blocked_until=self.release,
                    blocked_after_event_count=1,
                )
            ]
        )


class CapturingLogger:
    """Capture only structured lifecycle diagnostics in one focused test."""

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


async def _seed_run(
    session_factory: async_sessionmaker[AsyncSession], *, status: str
) -> tuple[Run, int, object]:
    now = utc_now()
    started_at = (
        now - timedelta(seconds=1) if status in {"RUNNING", "SUCCEEDED"} else None
    )
    finished_at = now if status in {"SUCCEEDED", "FAILED", "CANCELLED"} else None
    error_code = "ANSWER_EXECUTION_FAILED" if status == "FAILED" else None
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
                client_request_id=f"seed-{status}-{conversation.id}",
                attempt_no=1,
                status=status,
                started_at=started_at,
                finished_at=finished_at,
                error_code=error_code,
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
                        content="persisted",
                        run_id=run.id,
                        regenerated_from_message_id=None,
                        created_at=now,
                    )
                )
    return run, conversation.id, now


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


async def _conversation_last_message_at(
    session_factory: async_sessionmaker[AsyncSession], conversation_id: int
):
    async with session_factory() as session:
        conversation = await session.get(Conversation, conversation_id)
        assert conversation is not None
        return conversation.last_message_at


async def _wait_for_server_start(server: Server) -> None:
    """Wait for the real Uvicorn listener without fixed startup sleeps."""

    async with asyncio.timeout(5):
        while not server.started:
            await asyncio.sleep(0.01)


def test_uvicorn_graceful_shutdown_with_open_sse_repairs_active_run(
    migrated_database: str,
) -> None:
    """Exercise real Uvicorn disconnect-before-lifespan shutdown ordering."""

    _bootstrap(migrated_database)

    async def operation(session_factory: async_sessionmaker[AsyncSession]) -> None:
        fake = PartiallyGatedFake()
        app = create_app(_settings(migrated_database), provider=fake.provider)
        config = UvicornConfig(
            app,
            host="127.0.0.1",
            port=0,
            log_config=None,
            access_log=False,
        )
        assert config.timeout_graceful_shutdown is None
        server = Server(config)
        server_task = asyncio.create_task(server.serve())
        response = None
        run_id = None
        try:
            await _wait_for_server_start(server)
            port = server.servers[0].sockets[0].getsockname()[1]
            async with AsyncClient(base_url=f"http://127.0.0.1:{port}") as client:
                conversation = await client.post("/api/conversations", json={})
                assert conversation.status_code == 201
                accepted = await client.post(
                    f"/api/conversations/{conversation.json()['id']}/messages",
                    json={"content": "question", "client_request_id": "open-sse"},
                )
                assert accepted.status_code == 202
                run_id = accepted.json()["run"]["id"]
                await fake.started.wait()

                response = await client.send(
                    client.build_request("GET", f"/api/runs/{run_id}/events"),
                    stream=True,
                )
                assert response.status_code == 200
                assert "message.delta" in await anext(response.aiter_text())
                assert (await _run(session_factory, run_id)).status == "RUNNING"

                server.should_exit = True
                await asyncio.wait_for(server_task, timeout=2)
        finally:
            if response is not None:
                await response.aclose()
            if not server_task.done():
                server.force_exit = True
                await asyncio.wait_for(server_task, timeout=5)

        assert run_id is not None
        final = await _run(session_factory, run_id)
        assert final.status == "FAILED"
        assert final.error_code == "PROCESS_INTERRUPTED"
        assert await _assistant_count(session_factory, run_id) == 0
        assert not fake.release.is_set()

    asyncio.run(_exercise(migrated_database, operation))


def test_startup_repairs_residual_runs_and_leaves_terminal_facts_untouched(
    migrated_database: str,
) -> None:
    _bootstrap(migrated_database)

    async def operation(session_factory: async_sessionmaker[AsyncSession]) -> None:
        pending, pending_conversation_id, pending_last = await _seed_run(
            session_factory, status="PENDING"
        )
        running, running_conversation_id, running_last = await _seed_run(
            session_factory, status="RUNNING"
        )
        succeeded, _, _ = await _seed_run(session_factory, status="SUCCEEDED")
        cancelled, _, _ = await _seed_run(session_factory, status="CANCELLED")
        app = create_app(_settings(migrated_database))
        async with app.router.lifespan_context(app):
            repaired_pending = await _run(session_factory, pending.id)
            repaired_running = await _run(session_factory, running.id)
            assert repaired_pending.status == "FAILED"
            assert repaired_pending.started_at is None
            assert repaired_pending.error_code == "PROCESS_INTERRUPTED"
            assert repaired_pending.finished_at is not None
            assert repaired_running.status == "FAILED"
            assert repaired_running.started_at == running.started_at
            assert repaired_running.error_code == "PROCESS_INTERRUPTED"
            assert repaired_running.finished_at is not None
            assert (await _run(session_factory, succeeded.id)).status == "SUCCEEDED"
            assert (await _run(session_factory, cancelled.id)).status == "CANCELLED"
            assert await _assistant_count(session_factory, pending.id) == 0
            assert await _assistant_count(session_factory, running.id) == 0
            assert (
                await _conversation_last_message_at(
                    session_factory, pending_conversation_id
                )
            ) == pending_last
            assert (
                await _conversation_last_message_at(
                    session_factory, running_conversation_id
                )
            ) == running_last

    asyncio.run(_exercise(migrated_database, operation))


def test_startup_repair_completes_before_normal_retry_admission(
    migrated_database: str,
) -> None:
    _bootstrap(migrated_database)

    async def operation(session_factory: async_sessionmaker[AsyncSession]) -> None:
        residual, conversation_id, _ = await _seed_run(
            session_factory, status="PENDING"
        )
        app = create_app(_settings(migrated_database))
        async with app.router.lifespan_context(app):
            assert (
                await _run(session_factory, residual.id)
            ).error_code == "PROCESS_INTERRUPTED"
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                retry = await client.post(
                    f"/api/conversations/{conversation_id}/retry",
                    json={"client_request_id": "retry-after-interruption"},
                )
                assert retry.status_code == 202
                assert (
                    retry.json()["run"]["input_message_id"] == residual.input_message_id
                )
                assert retry.json()["run"]["attempt_no"] == 2
                answer = app.state.execution_manager._active_answers.get(
                    retry.json()["run"]["id"]
                )
                assert answer is not None and answer.task is not None
                await answer.task

    asyncio.run(_exercise(migrated_database, operation))


def test_shutdown_sweep_uses_mysql_not_only_local_runtime_registry(
    migrated_database: str,
) -> None:
    _bootstrap(migrated_database)

    async def operation(session_factory: async_sessionmaker[AsyncSession]) -> None:
        app = create_app(_settings(migrated_database))
        lifecycle = app.router.lifespan_context(app)
        await lifecycle.__aenter__()
        try:
            residual, _, _ = await _seed_run(session_factory, status="RUNNING")
            assert app.state.execution_manager._active_answers.get(residual.id) is None
        finally:
            await lifecycle.__aexit__(None, None, None)
        repaired = await _run(session_factory, residual.id)
        assert repaired.status == "FAILED"
        assert repaired.error_code == "PROCESS_INTERRUPTED"
        assert await _assistant_count(session_factory, residual.id) == 0

    asyncio.run(_exercise(migrated_database, operation))


def test_shutdown_interruption_and_success_race_preserve_first_terminal_commit(
    migrated_database: str,
) -> None:
    _bootstrap(migrated_database)

    async def operation(session_factory: async_sessionmaker[AsyncSession]) -> None:
        run, _, _ = await _seed_run(session_factory, status="RUNNING")
        gate = asyncio.Event()

        async def interrupt() -> list[int]:
            await gate.wait()
            return await interrupt_active_runs(session_factory)

        async def succeed() -> object:
            await gate.wait()
            return await _commit_success(
                session_factory,
                conversation_id=run.conversation_id,
                run_id=run.id,
                content="race answer",
            )

        interruption_task = asyncio.create_task(interrupt())
        success_task = asyncio.create_task(succeed())
        gate.set()
        await asyncio.gather(interruption_task, success_task, return_exceptions=True)
        final = await _run(session_factory, run.id)
        if final.status == "SUCCEEDED":
            assert await _assistant_count(session_factory, run.id) == 1
        else:
            assert final.status == "FAILED"
            assert final.error_code == "PROCESS_INTERRUPTED"
            assert await _assistant_count(session_factory, run.id) == 0

    asyncio.run(_exercise(migrated_database, operation))


def test_interruption_log_includes_run_and_conversation_identity(
    migrated_database: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The interruption diagnostic remains correlated to its durable Run fact."""

    _bootstrap(migrated_database)
    logger = CapturingLogger()
    monkeypatch.setattr(answer_lifecycle.structlog, "get_logger", lambda _: logger)

    async def operation(session_factory: async_sessionmaker[AsyncSession]) -> None:
        run, conversation_id, _ = await _seed_run(session_factory, status="RUNNING")
        assert await interrupt_active_runs(session_factory) == [run.id]
        assert logger.events == [
            (
                "answer.run.interrupted",
                {
                    "run_id": run.id,
                    "conversation_id": conversation_id,
                    "error_code": "PROCESS_INTERRUPTED",
                },
            )
        ]

    asyncio.run(_exercise(migrated_database, operation))


def test_interruption_then_local_task_cancel_does_not_overwrite_failed_run(
    migrated_database: str,
) -> None:
    _bootstrap(migrated_database)

    async def operation(session_factory: async_sessionmaker[AsyncSession]) -> None:
        fake = GatedFake()
        manager = AnswerExecutionManager(
            session_factory, lambda: workflow_for(fake.provider)
        )
        async with session_factory() as session:
            conversation = await create_conversation(session, user_id=1, title=None)
        async with session_factory() as session:
            admission = await admit_new_question(
                session,
                user_id=1,
                conversation_id=conversation.id,
                content="question",
                client_request_id="interruption-task",
            )
        await manager.schedule(admission)
        answer = manager._active_answers.get(admission.run.id)
        assert answer is not None and answer.task is not None
        await fake.started.wait()
        repaired_ids = await interrupt_active_runs(session_factory)
        assert repaired_ids == [admission.run.id]
        await manager.stop_interrupted_runs(repaired_ids)
        with pytest.raises(asyncio.CancelledError):
            await answer.task
        final = await _run(session_factory, admission.run.id)
        assert final.status == "FAILED"
        assert final.error_code == "PROCESS_INTERRUPTED"
        assert await _assistant_count(session_factory, admission.run.id) == 0

    asyncio.run(_exercise(migrated_database, operation))
