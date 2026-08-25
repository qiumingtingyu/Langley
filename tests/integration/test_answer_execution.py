"""Real MySQL integration tests for the production execution shell."""

import asyncio
import json
from argparse import Namespace

import pytest
from agent_workflow import workflow_for
from alembic import command
from alembic.config import Config
from sqlalchemy import select, text
from sqlalchemy.exc import DBAPIError

from langley.answer_execution import (
    AnswerExecutionManager,
    _commit_success,
    _start_running,
)
from langley.answering.contracts import (
    AssistantRuntimeMessage,
    LLMFinishReason,
    LLMResponseCompleted,
    ToolCall,
    UserRuntimeMessage,
)
from langley.answering.errors import RunErrorCode, WorkflowFailure
from langley.answering.fake_provider import FakeProvider, ScriptedProviderRound
from langley.answering.knowledge_qa import AnswerCompletion
from langley.bootstrap import bootstrap_local_user
from langley.conversation_commands import (
    AdmissionDisposition,
    NewQuestionAdmission,
    RegenerateAdmission,
    RetryAdmission,
    admit_new_question,
    admit_regenerate,
    admit_retry,
)
from langley.conversations import create_conversation
from langley.infrastructure.database import (
    create_database_engine,
    create_session_factory,
    dispose_database_engine,
)
from langley.infrastructure.models import Message, Run
from langley.main import _memory_lifecycle_callbacks, _memory_provider_for
from langley.memory.policy import MemoryPolicy
from langley.runs import cancel_owned_run
from langley.settings import Settings

Admission = NewQuestionAdmission | RetryAdmission | RegenerateAdmission


@pytest.fixture
def migrated_database(test_database_url: str, reset_database) -> str:
    """Provide an empty real MySQL database migrated to the current head."""

    reset_database()
    config = Config("alembic.ini")
    config.cmd_opts = Namespace(x=["use_test_database=true"])
    command.upgrade(config, "head")
    return test_database_url


def _settings(database_url: str) -> Settings:
    return Settings(environment="test", database_url=database_url, local_user_id=1)


def _bootstrap(database_url: str) -> None:
    assert asyncio.run(bootstrap_local_user(_settings(database_url)))


def _create_conversation(database_url: str) -> int:
    async def create() -> int:
        engine = create_database_engine(database_url)
        session_factory = create_session_factory(engine)
        try:
            async with session_factory() as session:
                conversation = await create_conversation(session, user_id=1, title=None)
                return conversation.id
        finally:
            await dispose_database_engine(engine)

    return asyncio.run(create())


def _admit_new(
    database_url: str,
    conversation_id: int,
    client_request_id: str,
    content: str,
) -> NewQuestionAdmission:
    async def admit() -> NewQuestionAdmission:
        engine = create_database_engine(database_url)
        session_factory = create_session_factory(engine)
        try:
            async with session_factory() as session:
                return await admit_new_question(
                    session,
                    user_id=1,
                    conversation_id=conversation_id,
                    client_request_id=client_request_id,
                    content=content,
                )
        finally:
            await dispose_database_engine(engine)

    return asyncio.run(admit())


def _admit_retry(
    database_url: str, conversation_id: int, client_request_id: str
) -> RetryAdmission:
    async def admit() -> RetryAdmission:
        engine = create_database_engine(database_url)
        session_factory = create_session_factory(engine)
        try:
            async with session_factory() as session:
                return await admit_retry(
                    session,
                    user_id=1,
                    conversation_id=conversation_id,
                    client_request_id=client_request_id,
                )
        finally:
            await dispose_database_engine(engine)

    return asyncio.run(admit())


def _admit_regenerate(
    database_url: str, conversation_id: int, client_request_id: str
) -> RegenerateAdmission:
    async def admit() -> RegenerateAdmission:
        engine = create_database_engine(database_url)
        session_factory = create_session_factory(engine)
        try:
            async with session_factory() as session:
                return await admit_regenerate(
                    session,
                    user_id=1,
                    conversation_id=conversation_id,
                    client_request_id=client_request_id,
                )
        finally:
            await dispose_database_engine(engine)

    return asyncio.run(admit())


def _completion(content: str) -> LLMResponseCompleted:
    return LLMResponseCompleted(
        assistant_content=content,
        tool_calls=(),
        finish_reason=LLMFinishReason.STOP,
        usage=None,
    )


def _execute(
    database_url: str, admission: Admission, provider: FakeProvider
) -> tuple[Run, Message | None]:
    async def execute() -> tuple[Run, Message | None]:
        engine = create_database_engine(database_url)
        session_factory = create_session_factory(engine)
        manager = AnswerExecutionManager(
            session_factory, lambda: workflow_for(provider)
        )
        try:
            await manager.schedule(admission)
            if admission.disposition is not AdmissionDisposition.REPLAY:
                answer = manager._active_answers.get(admission.run.id)
                assert answer is not None and answer.task is not None
                await answer.task
            async with session_factory() as session:
                run = await session.get(Run, admission.run.id)
                assert run is not None
                assistant = await session.scalar(
                    select(Message)
                    .where(
                        Message.run_id == admission.run.id, Message.role == "ASSISTANT"
                    )
                    .limit(1)
                )
                return run, assistant
        finally:
            await dispose_database_engine(engine)

    return asyncio.run(execute())


def _scalar(database_url: str, statement: str) -> object:
    async def execute() -> object:
        engine = create_database_engine(database_url)
        try:
            async with engine.connect() as connection:
                return (await connection.execute(text(statement))).scalar_one()
        finally:
            await dispose_database_engine(engine)

    return asyncio.run(execute())


def _run(database_url: str, statement: str) -> None:
    async def execute() -> None:
        engine = create_database_engine(database_url)
        try:
            async with engine.begin() as connection:
                await connection.execute(text(statement))
        finally:
            await dispose_database_engine(engine)

    asyncio.run(execute())


def test_scheduler_failure_publishes_only_after_durable_failed_transition(
    migrated_database: str,
) -> None:
    _bootstrap(migrated_database)
    conversation_id = _create_conversation(migrated_database)
    admission = _admit_new(migrated_database, conversation_id, "scheduler-failure", "q")

    async def exercise() -> None:
        engine = create_database_engine(migrated_database)
        session_factory = create_session_factory(engine)
        scheduling_attempts = 0
        drain_started = asyncio.Event()

        def schedule(coroutine):
            nonlocal scheduling_attempts
            scheduling_attempts += 1
            if scheduling_attempts == 1:
                raise RuntimeError("no answer task")
            return asyncio.create_task(coroutine)

        async def capture_boundary(user_id: int) -> int | None:
            assert user_id == 1
            return admission.user_message.id

        async def drain(user_id: int, boundary: int) -> None:
            assert user_id == 1
            assert boundary == admission.user_message.id
            drain_started.set()

        manager = AnswerExecutionManager(
            session_factory,
            lambda: workflow_for(FakeProvider([])),
            task_scheduler=schedule,
            memory_boundary_capture=capture_boundary,
            memory_background_drain=drain,
        )
        observed: list[str] = []
        original_close = manager._close_after_terminal

        def observe_terminal(answer, terminal):
            observed.append(terminal[0])
            original_close(answer, terminal)

        manager._close_after_terminal = observe_terminal
        try:
            await manager.schedule(admission)
            async with session_factory() as session:
                run = await session.get(Run, admission.run.id)
                assert run is not None and run.status == "FAILED"
            assert observed == ["run.failed"]
            await asyncio.wait_for(drain_started.wait(), timeout=2)
            assert scheduling_attempts == 2
        finally:
            await dispose_database_engine(engine)

    asyncio.run(exercise())


def test_scheduler_failure_loser_emits_no_terminal_event_or_memory_wake(
    migrated_database: str,
) -> None:
    """A losing scheduler failure has no presentation authority."""

    _bootstrap(migrated_database)
    conversation_id = _create_conversation(migrated_database)
    admission = _admit_new(migrated_database, conversation_id, "scheduler-loser", "q")
    _run(
        migrated_database,
        "UPDATE runs SET status = 'CANCELLED', "
        "finished_at = NOW(6), updated_at = NOW(6) "
        f"WHERE id = {admission.run.id}",
    )

    async def exercise() -> None:
        engine = create_database_engine(migrated_database)
        session_factory = create_session_factory(engine)
        wakes: list[int] = []
        manager = AnswerExecutionManager(
            session_factory,
            lambda: workflow_for(FakeProvider([])),
            task_scheduler=lambda coroutine: (_ for _ in ()).throw(
                RuntimeError("no task")
            ),
            memory_boundary_capture=lambda user_id: _record_wake(wakes, user_id),
            memory_background_drain=lambda user_id, boundary: _unexpected_drain(
                user_id, boundary
            ),
        )
        observed: list[str] = []
        original_close = manager._close_after_terminal

        def observe_terminal(answer, terminal):
            observed.append(terminal[0])
            original_close(answer, terminal)

        manager._close_after_terminal = observe_terminal
        try:
            await manager.schedule(admission)
            async with session_factory() as session:
                run = await session.get(Run, admission.run.id)
                assert run is not None and run.status == "CANCELLED"
            assert observed == []
            assert wakes == []
        finally:
            await dispose_database_engine(engine)

    asyncio.run(exercise())


async def _record_wake(wakes: list[int], user_id: int) -> int | None:
    wakes.append(user_id)
    return None


async def _unexpected_drain(user_id: int, boundary: int) -> None:
    raise AssertionError(f"unexpected memory drain for {user_id=} {boundary=}")


def test_successful_execution_commits_assistant_and_succeeded_run(
    migrated_database: str,
) -> None:
    """The production Workflow result commits exactly one visible assistant fact."""

    _bootstrap(migrated_database)
    conversation_id = _create_conversation(migrated_database)
    admission = _admit_new(migrated_database, conversation_id, "new-key", "question")
    provider = FakeProvider([ScriptedProviderRound(events=(_completion("answer"),))])

    run, assistant = _execute(migrated_database, admission, provider)

    assert run.status == "SUCCEEDED"
    assert run.started_at is not None
    assert run.finished_at is not None
    assert run.error_code is None
    assert assistant is not None
    assert assistant.sequence_no == 2
    assert assistant.content == "answer"
    assert provider.requests[0].transcript == (UserRuntimeMessage(content="question"),)
    assert _scalar(migrated_database, "SELECT COUNT(*) FROM messages") == 2


def test_provider_failure_persists_failed_run_without_assistant_or_conversation_update(
    migrated_database: str,
) -> None:
    """A stable Provider failure leaves no partial assistant message in MySQL."""

    _bootstrap(migrated_database)
    conversation_id = _create_conversation(migrated_database)
    admission = _admit_new(migrated_database, conversation_id, "new-key", "question")
    last_message_at = _scalar(
        migrated_database,
        f"SELECT last_message_at FROM conversations WHERE id = {conversation_id}",
    )
    provider = FakeProvider(
        [
            ScriptedProviderRound(
                events=(),
                failure=WorkflowFailure(RunErrorCode.LLM_PROVIDER_FAILED),
            )
        ]
    )

    run, assistant = _execute(migrated_database, admission, provider)

    assert run.status == "FAILED"
    assert run.started_at is not None
    assert run.finished_at is not None
    assert run.error_code == "LLM_PROVIDER_FAILED"
    assert assistant is None
    assert len(provider.requests) == 1
    assert _scalar(migrated_database, "SELECT COUNT(*) FROM messages") == 1
    assert (
        _scalar(
            migrated_database,
            f"SELECT last_message_at FROM conversations WHERE id = {conversation_id}",
        )
        == last_message_at
    )


def test_tool_execution_persists_only_the_final_visible_assistant_message(
    migrated_database: str,
) -> None:
    """Tool observations remain transient while the final answer commits once."""

    _bootstrap(migrated_database)
    conversation_id = _create_conversation(migrated_database)
    admission = _admit_new(migrated_database, conversation_id, "tool-key", "time?")
    provider = FakeProvider(
        [
            ScriptedProviderRound(
                events=(
                    LLMResponseCompleted(
                        assistant_content="I will check the time first.",
                        tool_calls=(
                            ToolCall(
                                call_id="time-1",
                                name="get_current_time",
                                raw_arguments='{"timezone":"UTC"}',
                            ),
                        ),
                        finish_reason=LLMFinishReason.TOOL_CALLS,
                        usage=None,
                    ),
                )
            ),
            ScriptedProviderRound(events=(_completion("It is 09:00 UTC."),)),
        ]
    )

    run, assistant = _execute(migrated_database, admission, provider)

    assert run.status == "SUCCEEDED"
    assert assistant is not None
    assert assistant.content == "It is 09:00 UTC."
    assert len(provider.requests) == 2
    assert (
        _scalar(
            migrated_database,
            f"SELECT COUNT(*) FROM messages WHERE run_id = {admission.run.id}",
        )
        == 1
    )


def test_replay_does_not_claim_or_execute_the_persisted_pending_run(
    migrated_database: str,
) -> None:
    """A same-key replay never starts a second production Workflow invocation."""

    _bootstrap(migrated_database)
    conversation_id = _create_conversation(migrated_database)
    initiating = _admit_new(migrated_database, conversation_id, "same-key", "question")
    replay = _admit_new(migrated_database, conversation_id, "same-key", "question")
    provider = FakeProvider([])

    run, assistant = _execute(migrated_database, replay, provider)

    assert initiating.run.id == run.id
    assert run.status == "PENDING"
    assert assistant is None
    assert provider.requests == []


def test_late_execution_cannot_overwrite_a_terminal_run(
    migrated_database: str,
) -> None:
    """A duplicate local schedule has no authority over a completed Run."""

    _bootstrap(migrated_database)
    conversation_id = _create_conversation(migrated_database)
    admission = _admit_new(migrated_database, conversation_id, "new-key", "question")
    _execute(
        migrated_database,
        admission,
        FakeProvider([ScriptedProviderRound(events=(_completion("answer"),))]),
    )
    late_provider = FakeProvider([])

    run, assistant = _execute(migrated_database, admission, late_provider)

    assert run.status == "SUCCEEDED"
    assert assistant is not None
    assert late_provider.requests == []
    assert _scalar(migrated_database, "SELECT COUNT(*) FROM messages") == 2


def test_retry_context_excludes_failed_attempts_and_keeps_reused_user_current(
    migrated_database: str,
) -> None:
    """Retry builds the real ContextBuilder transcript from authoritative facts."""

    _bootstrap(migrated_database)
    conversation_id = _create_conversation(migrated_database)
    first = _admit_new(migrated_database, conversation_id, "first-key", "first")
    _execute(
        migrated_database,
        first,
        FakeProvider([ScriptedProviderRound(events=(_completion("first answer"),))]),
    )
    second = _admit_new(migrated_database, conversation_id, "second-key", "second")
    _execute(
        migrated_database,
        second,
        FakeProvider(
            [
                ScriptedProviderRound(
                    events=(),
                    failure=WorkflowFailure(RunErrorCode.LLM_PROVIDER_FAILED),
                )
            ]
        ),
    )
    retry = _admit_retry(migrated_database, conversation_id, "retry-key")
    provider = FakeProvider(
        [ScriptedProviderRound(events=(_completion("retry answer"),))]
    )

    run, _ = _execute(migrated_database, retry, provider)

    assert run.status == "SUCCEEDED"
    assert provider.requests[0].transcript == (
        UserRuntimeMessage(content="first"),
        AssistantRuntimeMessage(content="first answer", tool_calls=()),
        UserRuntimeMessage(content="second"),
    )


def test_regenerate_context_keeps_original_turn_before_copied_current_user(
    migrated_database: str,
) -> None:
    """Regenerate exercises the production ContextBuilder's linear fact shape."""

    _bootstrap(migrated_database)
    conversation_id = _create_conversation(migrated_database)
    first = _admit_new(migrated_database, conversation_id, "first-key", "question")
    _execute(
        migrated_database,
        first,
        FakeProvider([ScriptedProviderRound(events=(_completion("first answer"),))]),
    )
    regenerate = _admit_regenerate(migrated_database, conversation_id, "regenerate-key")
    provider = FakeProvider(
        [ScriptedProviderRound(events=(_completion("second answer"),))]
    )

    run, _ = _execute(migrated_database, regenerate, provider)

    assert run.status == "SUCCEEDED"
    assert provider.requests[0].transcript == (
        UserRuntimeMessage(content="question"),
        AssistantRuntimeMessage(content="first answer", tool_calls=()),
        UserRuntimeMessage(content="question"),
    )


def test_success_transaction_rolls_back_run_transition_when_assistant_insert_fails(
    migrated_database: str,
) -> None:
    """A MySQL write failure after conditional success leaves the Run as RUNNING."""

    _bootstrap(migrated_database)
    conversation_id = _create_conversation(migrated_database)
    admission = _admit_new(migrated_database, conversation_id, "new-key", "question")
    _run(
        migrated_database,
        "UPDATE messages SET sequence_no = 9223372036854775807 "
        f"WHERE id = {admission.user_message.id}",
    )

    async def commit() -> None:
        engine = create_database_engine(migrated_database)
        session_factory = create_session_factory(engine)
        try:
            await _start_running(
                session_factory,
                conversation_id=admission.run.conversation_id,
                run_id=admission.run.id,
            )
            await _commit_success(
                session_factory,
                conversation_id=admission.run.conversation_id,
                run_id=admission.run.id,
                content="answer",
            )
        finally:
            await dispose_database_engine(engine)

    with pytest.raises(DBAPIError):
        asyncio.run(commit())

    assert (
        _scalar(
            migrated_database,
            f"SELECT status FROM runs WHERE id = {admission.run.id}",
        )
        == "RUNNING"
    )
    assert _scalar(migrated_database, "SELECT COUNT(*) FROM messages") == 1


def _memory_no_change() -> LLMResponseCompleted:
    return _completion(
        json.dumps({"mutations": [], "user_requested_memory_action": False})
    )


def test_memory_catch_up_precedes_workflow_and_terminal_drain_is_finite(
    migrated_database: str,
) -> None:
    """T5 uses T4 before and after Answer without widening its boundary."""

    _bootstrap(migrated_database)
    conversation_id = _create_conversation(migrated_database)
    first = _admit_new(migrated_database, conversation_id, "first", "first fact")
    _execute(
        migrated_database,
        first,
        FakeProvider([ScriptedProviderRound(events=(_completion("first answer"),))]),
    )
    second = _admit_new(migrated_database, conversation_id, "second", "second fact")

    async def exercise() -> None:
        engine = create_database_engine(migrated_database)
        session_factory = create_session_factory(engine)
        policy = MemoryPolicy(
            provider=FakeProvider(
                [
                    ScriptedProviderRound(events=(_memory_no_change(),)),
                    ScriptedProviderRound(events=(_memory_no_change(),)),
                ]
            ),
            memory_policy_estimated_token_budget=10_000,
        )
        settings = Settings(
            environment="test",
            database_url=migrated_database,
            local_user_id=1,
            memory_policy_model="test-memory-policy",
            memory_policy_estimated_token_budget=10_000,
        )
        catch_up, capture_boundary, background_drain = _memory_lifecycle_callbacks(
            settings, session_factory, policy
        )
        catch_up_completed = asyncio.Event()
        background_completed = asyncio.Event()

        async def observed_catch_up(command: Admission) -> None:
            await catch_up(command)
            catch_up_completed.set()

        async def observed_background(user_id: int, boundary: int) -> None:
            assert boundary == second.user_message.id
            await background_drain(user_id, boundary)
            background_completed.set()

        class CheckingWorkflow:
            async def execute(self, *args, **kwargs) -> AnswerCompletion:
                del args, kwargs
                assert catch_up_completed.is_set()
                return AnswerCompletion(
                    content="second answer", citations=(), abstained=False
                )

        manager = AnswerExecutionManager(
            session_factory,
            lambda: CheckingWorkflow(),
            memory_catch_up=observed_catch_up,
            memory_boundary_capture=capture_boundary,
            memory_background_drain=observed_background,
        )
        try:
            await manager.schedule(second)
            answer = manager._active_answers.get(second.run.id)
            assert answer is not None and answer.task is not None
            await answer.task
            await asyncio.wait_for(background_completed.wait(), timeout=2)
            async with session_factory() as session:
                first_message = await session.get(Message, first.user_message.id)
                second_message = await session.get(Message, second.user_message.id)
                assert first_message is not None
                assert second_message is not None
                assert first_message.memory_processed_at is not None
                assert second_message.memory_processed_at is not None
        finally:
            await dispose_database_engine(engine)

    asyncio.run(exercise())


def test_unexpected_memory_catch_up_failure_fails_open_to_answer(
    migrated_database: str,
) -> None:
    """The outer optional-subsystem seam preserves the Answer terminal outcome."""

    _bootstrap(migrated_database)
    conversation_id = _create_conversation(migrated_database)
    admission = _admit_new(migrated_database, conversation_id, "new-key", "question")

    async def exercise() -> None:
        engine = create_database_engine(migrated_database)
        session_factory = create_session_factory(engine)

        async def broken_catch_up(command: Admission) -> None:
            del command
            raise RuntimeError("programmer failure")

        manager = AnswerExecutionManager(
            session_factory,
            lambda: workflow_for(
                FakeProvider([ScriptedProviderRound(events=(_completion("answer"),))])
            ),
            memory_catch_up=broken_catch_up,
        )
        try:
            await manager.schedule(admission)
            answer = manager._active_answers.get(admission.run.id)
            assert answer is not None and answer.task is not None
            await answer.task
            async with session_factory() as session:
                run = await session.get(Run, admission.run.id)
                assert run is not None
                assert run.status == "SUCCEEDED"
        finally:
            await dispose_database_engine(engine)

    asyncio.run(exercise())


def test_background_memory_failure_cannot_change_a_failed_run(
    migrated_database: str,
) -> None:
    """A detached post-terminal failure never rewrites durable Run state."""

    _bootstrap(migrated_database)
    conversation_id = _create_conversation(migrated_database)
    admission = _admit_new(migrated_database, conversation_id, "new-key", "question")

    async def exercise() -> None:
        engine = create_database_engine(migrated_database)
        session_factory = create_session_factory(engine)
        drain_started = asyncio.Event()

        async def capture_boundary(user_id: int) -> int | None:
            assert user_id == 1
            return admission.user_message.id

        async def broken_drain(user_id: int, boundary: int) -> None:
            assert user_id == 1
            assert boundary == admission.user_message.id
            drain_started.set()
            raise RuntimeError("background failure")

        manager = AnswerExecutionManager(
            session_factory,
            lambda: workflow_for(
                FakeProvider(
                    [
                        ScriptedProviderRound(
                            events=(),
                            failure=WorkflowFailure(RunErrorCode.LLM_PROVIDER_FAILED),
                        )
                    ]
                )
            ),
            memory_boundary_capture=capture_boundary,
            memory_background_drain=broken_drain,
        )
        try:
            await manager.schedule(admission)
            answer = manager._active_answers.get(admission.run.id)
            assert answer is not None and answer.task is not None
            await answer.task
            await asyncio.wait_for(drain_started.wait(), timeout=2)
            await asyncio.sleep(0)
            async with session_factory() as session:
                run = await session.get(Run, admission.run.id)
                assert run is not None
                assert run.status == "FAILED"
                assert run.error_code == "LLM_PROVIDER_FAILED"
        finally:
            await dispose_database_engine(engine)

    asyncio.run(exercise())


def test_cancelled_run_schedules_one_post_commit_memory_wake(
    migrated_database: str,
) -> None:
    """Only the durable CANCELLED winner schedules its detached finite drain."""

    _bootstrap(migrated_database)
    conversation_id = _create_conversation(migrated_database)
    admission = _admit_new(migrated_database, conversation_id, "new-key", "question")

    async def exercise() -> None:
        engine = create_database_engine(migrated_database)
        session_factory = create_session_factory(engine)
        provider_started = asyncio.Event()
        release_provider = asyncio.Event()
        drain_started = asyncio.Event()
        boundaries: list[int] = []

        async def capture_boundary(user_id: int) -> int | None:
            assert user_id == 1
            return admission.user_message.id

        async def drain(user_id: int, boundary: int) -> None:
            assert user_id == 1
            boundaries.append(boundary)
            drain_started.set()

        manager = AnswerExecutionManager(
            session_factory,
            lambda: workflow_for(
                FakeProvider(
                    [
                        ScriptedProviderRound(
                            events=(),
                            started=provider_started,
                            blocked_until=release_provider,
                        )
                    ]
                )
            ),
            memory_boundary_capture=capture_boundary,
            memory_background_drain=drain,
        )
        try:
            await manager.schedule(admission)
            answer = manager._active_answers.get(admission.run.id)
            assert answer is not None and answer.task is not None
            await asyncio.wait_for(provider_started.wait(), timeout=2)
            async with session_factory() as session:
                cancelled = await cancel_owned_run(
                    session, user_id=1, run_id=admission.run.id
                )
            assert cancelled.status == "CANCELLED"
            await manager.stop_cancelled_run(admission.run.id, user_id=1)
            await asyncio.wait_for(drain_started.wait(), timeout=2)
            with pytest.raises(asyncio.CancelledError):
                await answer.task
            assert boundaries == [admission.user_message.id]
            async with session_factory() as session:
                run = await session.get(Run, admission.run.id)
                assert run is not None
                assert run.status == "CANCELLED"
        finally:
            release_provider.set()
            await dispose_database_engine(engine)

    asyncio.run(exercise())


def test_memory_model_none_constructs_no_policy_provider(
    migrated_database: str,
) -> None:
    """An injected Memory fake stays untouched without explicit model configuration."""

    settings = _settings(migrated_database)
    memory_provider = FakeProvider([])

    assert _memory_provider_for(settings, memory_provider) is None
    configured_settings = Settings(
        environment="test",
        database_url=migrated_database,
        local_user_id=1,
        memory_policy_model="explicit-memory-model",
        memory_policy_estimated_token_budget=10_000,
    )

    assert _memory_provider_for(configured_settings, memory_provider) is memory_provider
