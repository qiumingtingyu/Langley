"""Real MySQL integration tests for authoritative Run recovery and cancellation."""

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

import langley.answer_execution as answer_execution
from langley import runs as run_service
from langley.answer_execution import (
    AnswerExecutionManager,
    _commit_success,
    _start_running,
    mark_run_failed_if_running,
)
from langley.answering.contracts import (
    AssistantContentDelta,
    LLMFinishReason,
    LLMResponseCompleted,
)
from langley.answering.errors import RunErrorCode, WorkflowFailure
from langley.answering.fake_provider import FakeProvider, ScriptedProviderRound
from langley.bootstrap import bootstrap_local_user
from langley.business_time import utc_now
from langley.infrastructure.database import (
    create_database_engine,
    create_session_factory,
    dispose_database_engine,
)
from langley.infrastructure.models import Conversation, Message, Run, User
from langley.main import create_app
from langley.runs import cancel_owned_run
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
    """Yield one partial chunk, then wait deterministically before completion."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.after_partial = asyncio.Event()
        self.release = asyncio.Event()
        self.provider = FakeProvider(
            [
                ScriptedProviderRound(
                    events=(
                        AssistantContentDelta(content="AB"),
                        AssistantContentDelta(content="CD"),
                        LLMResponseCompleted(
                            assistant_content="ABCD",
                            tool_calls=(),
                            finish_reason=LLMFinishReason.STOP,
                            usage=None,
                        ),
                    ),
                    started=self.started,
                    blocked_until=self.release,
                    blocked_after_event_count=1,
                    event_reached=self.after_partial,
                    event_reached_after_count=1,
                )
            ]
        )

    @property
    def calls(self) -> int:
        return len(self.provider.requests)


class CancelStopSpy:
    """Record whether an API cancel tries to stop a local runtime."""

    def __init__(self) -> None:
        self.cancelled_run_ids: list[int] = []

    async def stop_cancelled_run(self, run_id: int, *, user_id: int) -> None:
        del user_id
        self.cancelled_run_ids.append(run_id)

    async def stop_interrupted_runs(self, run_ids: list[int]) -> None:
        del run_ids


class CancelStopFailureSpy(CancelStopSpy):
    """Simulate local cleanup failure after the authoritative Cancel commit."""

    async def stop_cancelled_run(self, run_id: int, *, user_id: int) -> None:
        del user_id
        self.cancelled_run_ids.append(run_id)
        raise RuntimeError("injected local stop failure")


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


async def _seed_run(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    status: str,
    user_id: int = 1,
    assistant_content: str = "persisted assistant",
) -> Run:
    now = utc_now()
    async with session_factory() as session:
        async with session.begin():
            conversation = Conversation(
                user_id=user_id,
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
            active = status == "RUNNING"
            terminal = status in {"SUCCEEDED", "FAILED", "CANCELLED"}
            run = Run(
                conversation_id=conversation.id,
                input_message_id=user_message.id,
                client_request_id=f"seed-{status}-{conversation.id}",
                attempt_no=1,
                status=status,
                started_at=now if status in {"RUNNING", "SUCCEEDED"} else None,
                finished_at=now if terminal else None,
                error_code="ANSWER_EXECUTION_FAILED" if status == "FAILED" else None,
                created_at=now,
                updated_at=now,
            )
            assert not (active and terminal)
            session.add(run)
            await session.flush()
            if status == "SUCCEEDED":
                session.add(
                    Message(
                        conversation_id=conversation.id,
                        sequence_no=2,
                        role="ASSISTANT",
                        content=assistant_content,
                        run_id=run.id,
                        regenerated_from_message_id=None,
                        created_at=now,
                    )
                )
    return run


async def _new_run_via_api(
    client: AsyncClient, client_request_id: str
) -> tuple[int, int]:
    conversation = await client.post("/api/conversations", json={})
    assert conversation.status_code == 201
    conversation_id = conversation.json()["id"]
    accepted = await client.post(
        f"/api/conversations/{conversation_id}/messages",
        json={"content": "question", "client_request_id": client_request_id},
    )
    assert accepted.status_code == 202
    return conversation_id, accepted.json()["run"]["id"]


def test_run_query_reads_all_states_and_enforces_ownership(
    migrated_database: str,
) -> None:
    _bootstrap(migrated_database)

    async def operation(session_factory: async_sessionmaker[AsyncSession]) -> None:
        async with session_factory() as session:
            async with session.begin():
                session.add(User(id=2, created_at=utc_now()))
        app = create_app(_settings(migrated_database))
        async with app.router.lifespan_context(app):
            states = {
                state: await _seed_run(session_factory, status=state)
                for state in (
                    "PENDING",
                    "RUNNING",
                    "FAILED",
                    "CANCELLED",
                    "SUCCEEDED",
                )
            }
            other_owner = await _seed_run(session_factory, status="PENDING", user_id=2)
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                for state, run in states.items():
                    response = await client.get(f"/api/runs/{run.id}")
                    assert response.status_code == 200
                    payload = response.json()
                    assert payload["run"]["status"] == state
                    if state == "SUCCEEDED":
                        assert (
                            payload["assistant_message"]["content"]
                            == "persisted assistant"
                        )
                        assert payload["assistant_message"]["run_id"] == run.id
                    else:
                        assert payload["assistant_message"] is None

                for run_id in (other_owner.id, 987654):
                    response = await client.get(f"/api/runs/{run_id}")
                    assert response.status_code == 404
                    assert response.json()["detail"] == {"code": "RUN_NOT_FOUND"}
                for run in (*states.values(), other_owner):
                    assert (
                        app.state.execution_manager._active_answers.get(run.id) is None
                    )

    asyncio.run(_exercise(migrated_database, operation))


def test_pending_cancel_commits_before_fake_and_is_idempotent(
    migrated_database: str,
) -> None:
    _bootstrap(migrated_database)

    async def operation(session_factory: async_sessionmaker[AsyncSession]) -> None:
        fake = GatedFake()
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
                _, run_id = await _new_run_via_api(client, "pending-cancel")
                answer = app.state.execution_manager._active_answers.get(run_id)
                assert answer is not None and answer.task is not None

                cancelled = await client.post(f"/api/runs/{run_id}/cancel")
                assert cancelled.status_code == 200
                first_finished_at = cancelled.json()["finished_at"]
                assert cancelled.json()["status"] == "CANCELLED"
                assert cancelled.json()["error_code"] is None
                with pytest.raises(asyncio.CancelledError):
                    await answer.task
                assert fake.calls == 0
                assert await _assistant_count(session_factory, run_id) == 0
                persisted = await _run(session_factory, run_id)
                assert persisted.finished_at is not None
                assert (
                    cancelled.json()["finished_at"].removesuffix("Z")
                    == persisted.finished_at.isoformat()
                )

                repeated = await client.post(f"/api/runs/{run_id}/cancel")
                assert repeated.status_code == 200
                assert repeated.json()["finished_at"] == first_finished_at
                assert (await _run(session_factory, run_id)).status == "CANCELLED"

    asyncio.run(_exercise(migrated_database, operation))


def test_cancel_returns_committed_snapshot_when_local_stop_fails(
    migrated_database: str,
) -> None:
    """Local best-effort cleanup cannot turn a committed Cancel into HTTP 500."""

    _bootstrap(migrated_database)

    async def operation(session_factory: async_sessionmaker[AsyncSession]) -> None:
        app = create_app(_settings(migrated_database))
        stop_spy = CancelStopFailureSpy()
        app.state.execution_manager = stop_spy
        async with app.router.lifespan_context(app):
            run = await _seed_run(session_factory, status="PENDING")
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.post(f"/api/runs/{run.id}/cancel")

        assert response.status_code == 200
        assert response.json()["status"] == "CANCELLED"
        assert stop_spy.cancelled_run_ids == [run.id]
        assert (await _run(session_factory, run.id)).status == "CANCELLED"

    asyncio.run(_exercise(migrated_database, operation))


def test_running_partial_cancel_never_persists_assistant(
    migrated_database: str,
) -> None:
    _bootstrap(migrated_database)

    async def operation(session_factory: async_sessionmaker[AsyncSession]) -> None:
        fake = GatedFake()
        app = create_app(_settings(migrated_database), provider=fake.provider)
        async with app.router.lifespan_context(app):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                _, run_id = await _new_run_via_api(client, "running-cancel")
                answer = app.state.execution_manager._active_answers.get(run_id)
                assert answer is not None and answer.task is not None
                await fake.after_partial.wait()
                assert answer.partial_text == "AB"

                cancelled = await client.post(f"/api/runs/{run_id}/cancel")
                assert cancelled.status_code == 200
                assert cancelled.json()["status"] == "CANCELLED"
                with pytest.raises(asyncio.CancelledError):
                    await answer.task
                assert (await _run(session_factory, run_id)).status == "CANCELLED"
                assert await _assistant_count(session_factory, run_id) == 0

    asyncio.run(_exercise(migrated_database, operation))


def test_cancel_conflicts_after_succeeded_or_failed_run(migrated_database: str) -> None:
    _bootstrap(migrated_database)

    async def operation(session_factory: async_sessionmaker[AsyncSession]) -> None:
        succeeded = await _seed_run(session_factory, status="SUCCEEDED")
        failed = await _seed_run(session_factory, status="FAILED")
        app = create_app(_settings(migrated_database))
        async with app.router.lifespan_context(app):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                for run_id in (succeeded.id, failed.id):
                    response = await client.post(f"/api/runs/{run_id}/cancel")
                    assert response.status_code == 409
                    assert response.json()["detail"] == {"code": "RUN_NOT_CANCELLABLE"}

    asyncio.run(_exercise(migrated_database, operation))


def test_cancel_success_and_failure_races_preserve_first_terminal_commit(
    migrated_database: str,
) -> None:
    _bootstrap(migrated_database)

    async def operation(session_factory: async_sessionmaker[AsyncSession]) -> None:
        async def race_success() -> None:
            run = await _seed_run(session_factory, status="PENDING")
            await _start_running(
                session_factory, conversation_id=run.conversation_id, run_id=run.id
            )
            gate = asyncio.Event()

            async def cancel() -> object:
                async with session_factory() as session:
                    await gate.wait()
                    return await cancel_owned_run(session, user_id=1, run_id=run.id)

            async def succeed() -> object:
                await gate.wait()
                return await _commit_success(
                    session_factory,
                    conversation_id=run.conversation_id,
                    run_id=run.id,
                    content="race answer",
                )

            cancel_task = asyncio.create_task(cancel())
            success_task = asyncio.create_task(succeed())
            gate.set()
            await asyncio.gather(cancel_task, success_task, return_exceptions=True)
            final = await _run(session_factory, run.id)
            if final.status == "CANCELLED":
                assert await _assistant_count(session_factory, run.id) == 0
            else:
                assert final.status == "SUCCEEDED"
                assert await _assistant_count(session_factory, run.id) == 1

        async def race_failure() -> None:
            run = await _seed_run(session_factory, status="PENDING")
            await _start_running(
                session_factory, conversation_id=run.conversation_id, run_id=run.id
            )
            gate = asyncio.Event()

            async def cancel() -> object:
                async with session_factory() as session:
                    await gate.wait()
                    return await cancel_owned_run(session, user_id=1, run_id=run.id)

            async def fail() -> object:
                await gate.wait()
                return await mark_run_failed_if_running(
                    session_factory,
                    conversation_id=run.conversation_id,
                    run_id=run.id,
                    error_code="ANSWER_EXECUTION_FAILED",
                )

            cancel_task = asyncio.create_task(cancel())
            failure_task = asyncio.create_task(fail())
            gate.set()
            await asyncio.gather(cancel_task, failure_task, return_exceptions=True)
            final = await _run(session_factory, run.id)
            assert final.status in {"CANCELLED", "FAILED"}
            assert await _assistant_count(session_factory, run.id) == 0

        await race_success()
        await race_failure()

    asyncio.run(_exercise(migrated_database, operation))


@pytest.mark.parametrize("winner", ("SUCCEEDED", "FAILED"))
def test_cancel_loser_returns_authoritative_winner_without_local_stop(
    migrated_database: str,
    monkeypatch: pytest.MonkeyPatch,
    winner: str,
) -> None:
    """Force a terminal commit between Cancel's read and conditional UPDATE."""

    _bootstrap(migrated_database)

    async def operation(session_factory: async_sessionmaker[AsyncSession]) -> None:
        cancel_read = asyncio.Event()
        allow_cancel_update = asyncio.Event()
        original_get_owned_run = run_service._get_owned_run

        async def pause_cancel_after_read(
            session: AsyncSession, *, user_id: int, run_id: int
        ) -> Run:
            run = await original_get_owned_run(session, user_id=user_id, run_id=run_id)
            cancel_read.set()
            await allow_cancel_update.wait()
            return run

        monkeypatch.setattr(run_service, "_get_owned_run", pause_cancel_after_read)
        stop_spy = CancelStopSpy()
        app = create_app(_settings(migrated_database))
        app.state.execution_manager = stop_spy
        async with app.router.lifespan_context(app):
            run = await _seed_run(session_factory, status="PENDING")
            await _start_running(
                session_factory, conversation_id=run.conversation_id, run_id=run.id
            )
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                cancel_response = asyncio.create_task(
                    client.post(f"/api/runs/{run.id}/cancel")
                )
                await cancel_read.wait()
                if winner == "SUCCEEDED":
                    await _commit_success(
                        session_factory,
                        conversation_id=run.conversation_id,
                        run_id=run.id,
                        content="race answer",
                    )
                else:
                    await mark_run_failed_if_running(
                        session_factory,
                        conversation_id=run.conversation_id,
                        run_id=run.id,
                        error_code="ANSWER_EXECUTION_FAILED",
                    )
                allow_cancel_update.set()
                response = await cancel_response

            assert response.status_code == 409
            assert response.json()["detail"] == {"code": "RUN_NOT_CANCELLABLE"}
            assert stop_spy.cancelled_run_ids == []
            persisted = await _run(session_factory, run.id)
            assert persisted.status == winner
            assert await _assistant_count(session_factory, run.id) == (
                1 if winner == "SUCCEEDED" else 0
            )

    asyncio.run(_exercise(migrated_database, operation))


def test_concurrent_cancels_return_the_winning_persisted_snapshot(
    migrated_database: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both callers must return the one CANCELLED fact committed by MySQL."""

    _bootstrap(migrated_database)

    async def operation(session_factory: async_sessionmaker[AsyncSession]) -> None:
        initial_reads = 0
        both_reads_complete = asyncio.Event()
        allow_cancel_updates = asyncio.Event()
        original_get_owned_run = run_service._get_owned_run
        first_finished_at = utc_now()
        second_finished_at = first_finished_at + timedelta(microseconds=1)
        finished_at_values = iter((first_finished_at, second_finished_at))

        async def pause_both_cancels_after_read(
            session: AsyncSession, *, user_id: int, run_id: int
        ) -> Run:
            nonlocal initial_reads
            run = await original_get_owned_run(session, user_id=user_id, run_id=run_id)
            initial_reads += 1
            if initial_reads == 2:
                both_reads_complete.set()
            await allow_cancel_updates.wait()
            return run

        monkeypatch.setattr(
            run_service, "_get_owned_run", pause_both_cancels_after_read
        )
        monkeypatch.setattr(run_service, "utc_now", lambda: next(finished_at_values))

        run = await _seed_run(session_factory, status="PENDING")
        await _start_running(
            session_factory, conversation_id=run.conversation_id, run_id=run.id
        )

        async def cancel() -> Run:
            async with session_factory() as session:
                return await cancel_owned_run(session, user_id=1, run_id=run.id)

        first_cancel = asyncio.create_task(cancel())
        second_cancel = asyncio.create_task(cancel())
        await both_reads_complete.wait()
        allow_cancel_updates.set()
        first_result, second_result = await asyncio.gather(first_cancel, second_cancel)

        persisted = await _run(session_factory, run.id)
        assert persisted.status == "CANCELLED"
        assert persisted.finished_at is not None
        assert first_result.status == "CANCELLED"
        assert second_result.status == "CANCELLED"
        assert first_result.finished_at == persisted.finished_at
        assert second_result.finished_at == persisted.finished_at

    asyncio.run(_exercise(migrated_database, operation))


def test_explicit_cancelled_error_and_cross_conversation_stop_are_isolated(
    migrated_database: str,
) -> None:
    _bootstrap(migrated_database)

    async def operation(session_factory: async_sessionmaker[AsyncSession]) -> None:
        fake_a = GatedFake()
        fake_b = GatedFake()
        fakes = iter((fake_a, fake_b))
        app = create_app(_settings(migrated_database))
        app.state.execution_manager = AnswerExecutionManager(
            session_factory, lambda: workflow_for(next(fakes).provider)
        )
        async with app.router.lifespan_context(app):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                _, run_a_id = await _new_run_via_api(client, "isolate-a")
                _, run_b_id = await _new_run_via_api(client, "isolate-b")
                answer_a = app.state.execution_manager._active_answers.get(run_a_id)
                answer_b = app.state.execution_manager._active_answers.get(run_b_id)
                assert answer_a is not None and answer_b is not None
                await fake_a.after_partial.wait()
                await fake_b.after_partial.wait()

                cancelled = await client.post(f"/api/runs/{run_a_id}/cancel")
                assert cancelled.status_code == 200
                assert answer_a.task is not None
                with pytest.raises(asyncio.CancelledError):
                    await answer_a.task
                assert (await _run(session_factory, run_a_id)).status == "CANCELLED"
                assert (await _run(session_factory, run_b_id)).status == "RUNNING"

                fake_b.release.set()
                assert answer_b.task is not None
                await answer_b.task
                assert (await _run(session_factory, run_b_id)).status == "SUCCEEDED"
                assert await _assistant_count(session_factory, run_a_id) == 0
                assert await _assistant_count(session_factory, run_b_id) == 1

    asyncio.run(_exercise(migrated_database, operation))


def test_workflow_failure_losing_cancel_race_returns_without_task_error(
    migrated_database: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A handled WorkflowFailure accepts CANCELLED as the terminal winner."""

    _bootstrap(migrated_database)

    async def operation(session_factory: async_sessionmaker[AsyncSession]) -> None:
        started = asyncio.Event()
        release_failure = asyncio.Event()
        failure_ready_to_commit = asyncio.Event()
        allow_failure_commit = asyncio.Event()
        provider = FakeProvider(
            [
                ScriptedProviderRound(
                    events=(),
                    failure=WorkflowFailure(RunErrorCode.LLM_PROVIDER_FAILED),
                    started=started,
                    blocked_until=release_failure,
                )
            ]
        )
        original_mark_failed = answer_execution.mark_run_failed_if_running

        async def pause_failure_commit(*args, **kwargs) -> bool:
            failure_ready_to_commit.set()
            await allow_failure_commit.wait()
            return await original_mark_failed(*args, **kwargs)

        monkeypatch.setattr(
            answer_execution, "mark_run_failed_if_running", pause_failure_commit
        )
        app = create_app(_settings(migrated_database), provider=provider)
        async with app.router.lifespan_context(app):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                _, run_id = await _new_run_via_api(client, "failure-cancel-race")
                answer = app.state.execution_manager._active_answers.get(run_id)
                assert answer is not None and answer.task is not None
                await started.wait()
                release_failure.set()
                await failure_ready_to_commit.wait()

                async with session_factory() as session:
                    cancelled = await cancel_owned_run(
                        session, user_id=1, run_id=run_id
                    )
                assert cancelled.status == "CANCELLED"

                allow_failure_commit.set()
                await answer.task
                assert (await _run(session_factory, run_id)).status == "CANCELLED"
                assert await _assistant_count(session_factory, run_id) == 0

    asyncio.run(_exercise(migrated_database, operation))
