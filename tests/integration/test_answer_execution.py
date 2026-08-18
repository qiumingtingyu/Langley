"""Real MySQL integration tests for the production execution shell."""

import asyncio
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
                        assistant_content="",
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
