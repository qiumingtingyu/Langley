"""Real MySQL transaction tests for new-question command admission."""

import asyncio
from argparse import Namespace

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from langley.business_time import utc_now
from langley.conversation_commands import (
    ActiveRunExistsError,
    AdmissionDisposition,
    ClientRequestIdReusedError,
    ConversationNotFoundError,
    NewQuestionAdmission,
    admit_new_question,
)
from langley.infrastructure.database import (
    create_database_engine,
    create_session_factory,
    dispose_database_engine,
)
from langley.infrastructure.models import Conversation, User


@pytest.fixture
def migrated_database(test_database_url: str, reset_database) -> str:
    """Provide an empty real MySQL database migrated to the current head."""

    reset_database()
    config = Config("alembic.ini")
    config.cmd_opts = Namespace(x=["use_test_database=true"])
    command.upgrade(config, "head")
    return test_database_url


def _create_conversation(database_url: str, user_id: int = 1) -> int:
    async def create() -> int:
        engine = create_database_engine(database_url)
        session_factory = create_session_factory(engine)
        try:
            async with session_factory() as session:
                async with session.begin():
                    user = User(id=user_id, created_at=utc_now())
                    conversation = Conversation(
                        user_id=user_id,
                        title=None,
                        created_at=utc_now(),
                        updated_at=utc_now(),
                        last_message_at=None,
                        deleted_at=None,
                    )
                    session.add(user)
                    await session.flush()
                    session.add(conversation)
                    await session.flush()
                    return conversation.id
        finally:
            await dispose_database_engine(engine)

    return asyncio.run(create())


def _admit(
    database_url: str,
    conversation_id: int,
    client_request_id: str,
    content: str = "question",
    user_id: int = 1,
) -> NewQuestionAdmission:
    async def admit() -> NewQuestionAdmission:
        engine = create_database_engine(database_url)
        session_factory = create_session_factory(engine)
        try:
            async with session_factory() as session:
                return await admit_new_question(
                    session,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    content=content,
                    client_request_id=client_request_id,
                )
        finally:
            await dispose_database_engine(engine)

    return asyncio.run(admit())


def _scalar(database_url: str, statement: str) -> object:
    async def execute() -> object:
        engine = create_database_engine(database_url)
        try:
            async with engine.connect() as connection:
                return (await connection.execute(text(statement))).scalar_one()
        finally:
            await dispose_database_engine(engine)

    return asyncio.run(execute())


def test_new_question_atomically_creates_user_and_pending_run(
    migrated_database: str,
) -> None:
    """A newly accepted command commits its USER and PENDING Run together."""

    conversation_id = _create_conversation(migrated_database)
    result = _admit(migrated_database, conversation_id, "new-key", "hello")

    assert result.disposition is AdmissionDisposition.NEWLY_ACCEPTED
    assert result.user_message.sequence_no == 1
    assert result.user_message.role == "USER"
    assert result.run.input_message_id == result.user_message.id
    assert result.run.attempt_no == 1
    assert result.run.status == "PENDING"
    assert _scalar(migrated_database, "SELECT COUNT(*) FROM messages") == 1
    assert _scalar(migrated_database, "SELECT COUNT(*) FROM runs") == 1


def test_same_key_pending_replay_returns_facts_without_new_admission(
    migrated_database: str,
) -> None:
    """A PENDING same-key replay is explicitly distinct from the initiating path."""

    conversation_id = _create_conversation(migrated_database)
    initiating = _admit(migrated_database, conversation_id, "same-key")
    replay = _admit(migrated_database, conversation_id, "same-key")

    assert initiating.disposition is AdmissionDisposition.NEWLY_ACCEPTED
    assert replay.disposition is AdmissionDisposition.REPLAY
    assert replay.user_message.id == initiating.user_message.id
    assert replay.run.id == initiating.run.id
    assert replay.run.status == "PENDING"
    assert _scalar(migrated_database, "SELECT COUNT(*) FROM messages") == 1
    assert _scalar(migrated_database, "SELECT COUNT(*) FROM runs") == 1


def test_same_key_with_different_content_is_rejected_before_active_run_check(
    migrated_database: str,
) -> None:
    """A reused key cannot change the semantic new-question command."""

    conversation_id = _create_conversation(migrated_database)
    _admit(migrated_database, conversation_id, "same-key", "first")

    with pytest.raises(ClientRequestIdReusedError):
        _admit(migrated_database, conversation_id, "same-key", "different")


def test_same_key_blank_content_is_reused_before_blank_content_validation(
    migrated_database: str,
) -> None:
    """Replay semantic validation precedes normal new-question content validation."""

    conversation_id = _create_conversation(migrated_database)
    _admit(migrated_database, conversation_id, "same-key", "first")

    with pytest.raises(ClientRequestIdReusedError):
        _admit(migrated_database, conversation_id, "same-key", "   ")


def test_new_key_blank_content_is_rejected_after_lock_and_active_run_checks(
    migrated_database: str,
) -> None:
    """A new blank question remains invalid without creating command facts."""

    conversation_id = _create_conversation(migrated_database)

    with pytest.raises(ValueError, match="content must not be blank"):
        _admit(migrated_database, conversation_id, "blank-key", "   ")

    assert _scalar(migrated_database, "SELECT COUNT(*) FROM messages") == 0
    assert _scalar(migrated_database, "SELECT COUNT(*) FROM runs") == 0


def test_different_key_is_rejected_while_a_conversation_has_an_active_run(
    migrated_database: str,
) -> None:
    """A Conversation accepts at most one PENDING or RUNNING Run at a time."""

    conversation_id = _create_conversation(migrated_database)
    _admit(migrated_database, conversation_id, "first-key")

    with pytest.raises(ActiveRunExistsError):
        _admit(migrated_database, conversation_id, "second-key")


def test_run_insert_failure_rolls_back_the_preceding_user_message(
    migrated_database: str,
) -> None:
    """A real MySQL write failure after USER flush leaves no partial command fact."""

    conversation_id = _create_conversation(migrated_database)

    with pytest.raises(DBAPIError):
        _admit(migrated_database, conversation_id, "x" * 65)

    assert _scalar(migrated_database, "SELECT COUNT(*) FROM messages") == 0
    assert _scalar(migrated_database, "SELECT COUNT(*) FROM runs") == 0


def test_new_question_uses_ownership_scoped_conversation_lock(
    migrated_database: str,
) -> None:
    """Another configured user cannot admit a command against this Conversation."""

    conversation_id = _create_conversation(migrated_database)

    with pytest.raises(ConversationNotFoundError):
        _admit(migrated_database, conversation_id, "other-user", user_id=2)


def test_concurrent_different_keys_create_at_most_one_active_run(
    migrated_database: str,
) -> None:
    """Independent MySQL sessions serialize competing new commands by Conversation."""

    conversation_id = _create_conversation(migrated_database)
    start_gate = asyncio.Barrier(2)

    async def admit_concurrently(client_request_id: str) -> object:
        engine = create_database_engine(migrated_database)
        session_factory = create_session_factory(engine)
        try:
            async with session_factory() as session:
                await start_gate.wait()
                return await admit_new_question(
                    session,
                    user_id=1,
                    conversation_id=conversation_id,
                    content="question",
                    client_request_id=client_request_id,
                )
        except ActiveRunExistsError as error:
            return error
        finally:
            await dispose_database_engine(engine)

    async def run_concurrent_admissions() -> tuple[object, object]:
        return await asyncio.gather(
            admit_concurrently("concurrent-a"),
            admit_concurrently("concurrent-b"),
        )

    first, second = asyncio.run(run_concurrent_admissions())

    assert (
        sum(
            isinstance(result, NewQuestionAdmission)
            and result.disposition is AdmissionDisposition.NEWLY_ACCEPTED
            for result in (first, second)
        )
        == 1
    )
    assert (
        sum(isinstance(result, ActiveRunExistsError) for result in (first, second)) == 1
    )
    assert (
        _scalar(
            migrated_database,
            "SELECT COUNT(*) FROM runs WHERE status IN ('PENDING', 'RUNNING')",
        )
        == 1
    )


def test_concurrent_same_key_returns_one_new_admission_and_one_replay(
    migrated_database: str,
) -> None:
    """Only the fact-creating path receives execution continuation eligibility."""

    conversation_id = _create_conversation(migrated_database)
    start_gate = asyncio.Barrier(2)

    async def admit_concurrently() -> NewQuestionAdmission:
        engine = create_database_engine(migrated_database)
        session_factory = create_session_factory(engine)
        try:
            async with session_factory() as session:
                await start_gate.wait()
                return await admit_new_question(
                    session,
                    user_id=1,
                    conversation_id=conversation_id,
                    content="question",
                    client_request_id="same-concurrent-key",
                )
        finally:
            await dispose_database_engine(engine)

    async def run_concurrent_admissions() -> tuple[
        NewQuestionAdmission, NewQuestionAdmission
    ]:
        return await asyncio.gather(admit_concurrently(), admit_concurrently())

    first, second = asyncio.run(run_concurrent_admissions())

    assert {first.disposition, second.disposition} == {
        AdmissionDisposition.NEWLY_ACCEPTED,
        AdmissionDisposition.REPLAY,
    }
    assert first.run.id == second.run.id
    assert first.run.status == second.run.status == "PENDING"
    assert _scalar(migrated_database, "SELECT COUNT(*) FROM messages") == 1
    assert _scalar(migrated_database, "SELECT COUNT(*) FROM runs") == 1
