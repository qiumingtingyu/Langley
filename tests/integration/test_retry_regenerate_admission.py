"""Real MySQL transaction tests for Retry and Regenerate command admission."""

import asyncio
from argparse import Namespace

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import select, text

from langley.business_time import utc_now
from langley.conversation_commands import (
    ActiveRunExistsError,
    AdmissionDisposition,
    ClientRequestIdReusedError,
    RegenerateAdmission,
    RegenerateNotAllowedError,
    RetryAdmission,
    RetryNotAllowedError,
    admit_regenerate,
    admit_retry,
)
from langley.infrastructure.database import (
    create_database_engine,
    create_session_factory,
    dispose_database_engine,
)
from langley.infrastructure.models import Conversation, Message, Run, User


@pytest.fixture
def migrated_database(test_database_url: str, reset_database) -> str:
    """Provide an empty real MySQL database migrated to the current head."""

    reset_database()
    config = Config("alembic.ini")
    config.cmd_opts = Namespace(x=["use_test_database=true"])
    command.upgrade(config, "head")
    return test_database_url


def _seed_turn(database_url: str, status: str) -> tuple[int, int, int]:
    async def seed() -> tuple[int, int, int]:
        engine = create_database_engine(database_url)
        session_factory = create_session_factory(engine)
        try:
            async with session_factory() as session:
                async with session.begin():
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
                        client_request_id="initial-key",
                        attempt_no=1,
                        status=status,
                        started_at=now if status != "PENDING" else None,
                        finished_at=now
                        if status in {"SUCCEEDED", "FAILED", "CANCELLED"}
                        else None,
                        error_code="ANSWER_EXECUTION_FAILED"
                        if status == "FAILED"
                        else None,
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
                                content="answer",
                                run_id=run.id,
                                regenerated_from_message_id=None,
                                created_at=now,
                            )
                        )
                    return conversation.id, user_message.id, run.id
        finally:
            await dispose_database_engine(engine)

    return asyncio.run(seed())


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


def _scalar(database_url: str, statement: str) -> object:
    async def execute() -> object:
        engine = create_database_engine(database_url)
        try:
            async with engine.connect() as connection:
                return (await connection.execute(text(statement))).scalar_one()
        finally:
            await dispose_database_engine(engine)

    return asyncio.run(execute())


def _complete_pending_regenerated_turn(
    database_url: str, conversation_id: int, run_id: int
) -> None:
    async def complete() -> None:
        engine = create_database_engine(database_url)
        session_factory = create_session_factory(engine)
        try:
            async with session_factory() as session:
                async with session.begin():
                    now = utc_now()
                    run = await session.get(Run, run_id)
                    if run is None:
                        raise AssertionError("expected pending regenerated run")
                    run.status = "SUCCEEDED"
                    run.started_at = now
                    run.finished_at = now
                    run.updated_at = now
                    last_sequence_no = await session.scalar(
                        select(Message.sequence_no)
                        .where(Message.conversation_id == conversation_id)
                        .order_by(Message.sequence_no.desc())
                        .limit(1)
                    )
                    session.add(
                        Message(
                            conversation_id=conversation_id,
                            sequence_no=(last_sequence_no or 0) + 1,
                            role="ASSISTANT",
                            content="regenerated answer",
                            run_id=run.id,
                            regenerated_from_message_id=None,
                            created_at=now,
                        )
                    )
                    conversation = await session.get(Conversation, conversation_id)
                    if conversation is None:
                        raise AssertionError("expected conversation")
                    conversation.last_message_at = now
                    conversation.updated_at = now
        finally:
            await dispose_database_engine(engine)

    asyncio.run(complete())


def _fail_pending_run(database_url: str, run_id: int) -> None:
    async def fail() -> None:
        engine = create_database_engine(database_url)
        session_factory = create_session_factory(engine)
        try:
            async with session_factory() as session:
                async with session.begin():
                    run = await session.get(Run, run_id)
                    if run is None:
                        raise AssertionError("expected pending run")
                    now = utc_now()
                    run.status = "FAILED"
                    run.started_at = now
                    run.finished_at = now
                    run.error_code = "ANSWER_EXECUTION_FAILED"
                    run.updated_at = now
        finally:
            await dispose_database_engine(engine)

    asyncio.run(fail())


def test_retry_reuses_failed_latest_user_and_does_not_update_conversation_time(
    migrated_database: str,
) -> None:
    """Retry appends only attempt 2 and keeps the existing USER as the input."""

    conversation_id, user_message_id, _ = _seed_turn(migrated_database, "FAILED")
    previous_last_message_at = _scalar(
        migrated_database,
        f"SELECT last_message_at FROM conversations WHERE id = {conversation_id}",
    )

    result = _admit_retry(migrated_database, conversation_id, "retry-key")

    assert result.disposition is AdmissionDisposition.NEWLY_ACCEPTED
    assert result.user_message.id == user_message_id
    assert result.run.input_message_id == user_message_id
    assert result.run.attempt_no == 2
    assert result.run.status == "PENDING"
    assert result.memory_catchup_through_message_id == user_message_id
    assert _scalar(migrated_database, "SELECT COUNT(*) FROM messages") == 1
    assert (
        _scalar(
            migrated_database,
            f"SELECT last_message_at FROM conversations WHERE id = {conversation_id}",
        )
        == previous_last_message_at
    )


def test_retry_same_key_replays_existing_pending_attempt(
    migrated_database: str,
) -> None:
    """Retry replay remains distinct from a newly accepted execution path."""

    conversation_id, user_message_id, _ = _seed_turn(migrated_database, "FAILED")
    initiating = _admit_retry(migrated_database, conversation_id, "retry-key")
    replay = _admit_retry(migrated_database, conversation_id, "retry-key")

    assert initiating.disposition is AdmissionDisposition.NEWLY_ACCEPTED
    assert replay.disposition is AdmissionDisposition.REPLAY
    assert replay.user_message.id == user_message_id
    assert replay.run.id == initiating.run.id
    assert replay.memory_catchup_through_message_id is None
    assert _scalar(migrated_database, "SELECT COUNT(*) FROM runs") == 2


def test_retry_rejects_latest_user_that_already_succeeded(
    migrated_database: str,
) -> None:
    """A USER with a successful answer cannot create another Retry attempt."""

    conversation_id, _, _ = _seed_turn(migrated_database, "SUCCEEDED")

    with pytest.raises(RetryNotAllowedError):
        _admit_retry(migrated_database, conversation_id, "retry-key")


def test_retry_accepts_cancelled_latest_user(
    migrated_database: str,
) -> None:
    """A cancelled latest USER is eligible for a new complete Retry attempt."""

    conversation_id, user_message_id, _ = _seed_turn(migrated_database, "CANCELLED")

    result = _admit_retry(migrated_database, conversation_id, "retry-key")

    assert result.disposition is AdmissionDisposition.NEWLY_ACCEPTED
    assert result.user_message.id == user_message_id
    assert result.run.attempt_no == 2
    assert result.run.status == "PENDING"


def test_retry_after_a_second_failed_attempt_creates_attempt_three(
    migrated_database: str,
) -> None:
    """Repeated failed complete attempts reuse the USER and advance attempt numbers."""

    conversation_id, user_message_id, _ = _seed_turn(migrated_database, "FAILED")
    second_attempt = _admit_retry(migrated_database, conversation_id, "retry-two")
    _fail_pending_run(migrated_database, second_attempt.run.id)

    third_attempt = _admit_retry(migrated_database, conversation_id, "retry-three")

    assert third_attempt.user_message.id == user_message_id
    assert third_attempt.run.input_message_id == user_message_id
    assert third_attempt.run.attempt_no == 3
    assert _scalar(migrated_database, "SELECT COUNT(*) FROM messages") == 1


def test_concurrent_retry_creates_one_new_active_attempt(
    migrated_database: str,
) -> None:
    """A start gate forces two real MySQL Retry sessions to compete for the lock."""

    conversation_id, _, _ = _seed_turn(migrated_database, "FAILED")
    start_gate = asyncio.Barrier(2)

    async def admit_concurrently(client_request_id: str) -> object:
        engine = create_database_engine(migrated_database)
        session_factory = create_session_factory(engine)
        try:
            async with session_factory() as session:
                await start_gate.wait()
                return await admit_retry(
                    session,
                    user_id=1,
                    conversation_id=conversation_id,
                    client_request_id=client_request_id,
                )
        except ActiveRunExistsError as error:
            return error
        finally:
            await dispose_database_engine(engine)

    async def run_concurrent_admissions() -> tuple[object, object]:
        return await asyncio.gather(
            admit_concurrently("retry-a"), admit_concurrently("retry-b")
        )

    first, second = asyncio.run(run_concurrent_admissions())

    assert (
        sum(
            isinstance(result, RetryAdmission)
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


def test_regenerate_appends_a_copied_user_and_pending_attempt(
    migrated_database: str,
) -> None:
    """Regenerate preserves the successful turn and appends a new linear turn anchor."""

    conversation_id, original_user_id, _ = _seed_turn(migrated_database, "SUCCEEDED")

    result = _admit_regenerate(migrated_database, conversation_id, "regenerate-key")

    assert result.disposition is AdmissionDisposition.NEWLY_ACCEPTED
    assert result.user_message.sequence_no == 3
    assert result.user_message.content == "question"
    assert result.user_message.regenerated_from_message_id == original_user_id
    assert result.run.input_message_id == result.user_message.id
    assert result.run.attempt_no == 1
    assert result.run.status == "PENDING"
    assert result.memory_catchup_through_message_id == original_user_id
    assert _scalar(migrated_database, "SELECT COUNT(*) FROM messages") == 3
    assert _scalar(migrated_database, "SELECT COUNT(*) FROM runs") == 2


def test_regenerate_same_key_replays_existing_pending_copy(
    migrated_database: str,
) -> None:
    """Regenerate replay cannot create a second copied USER while its Run is active."""

    conversation_id, _, _ = _seed_turn(migrated_database, "SUCCEEDED")
    initiating = _admit_regenerate(migrated_database, conversation_id, "regenerate-key")
    replay = _admit_regenerate(migrated_database, conversation_id, "regenerate-key")

    assert initiating.disposition is AdmissionDisposition.NEWLY_ACCEPTED
    assert replay.disposition is AdmissionDisposition.REPLAY
    assert replay.user_message.id == initiating.user_message.id
    assert replay.run.id == initiating.run.id
    assert replay.memory_catchup_through_message_id is None
    assert _scalar(migrated_database, "SELECT COUNT(*) FROM messages") == 3
    assert _scalar(migrated_database, "SELECT COUNT(*) FROM runs") == 2


def test_repeated_regenerate_keeps_provenance_on_the_original_user(
    migrated_database: str,
) -> None:
    """Copied USER Messages point to the original USER instead of forming a chain."""

    conversation_id, original_user_id, _ = _seed_turn(migrated_database, "SUCCEEDED")
    first = _admit_regenerate(migrated_database, conversation_id, "regenerate-one")
    _complete_pending_regenerated_turn(migrated_database, conversation_id, first.run.id)
    second = _admit_regenerate(migrated_database, conversation_id, "regenerate-two")

    assert first.user_message.regenerated_from_message_id == original_user_id
    assert second.user_message.regenerated_from_message_id == original_user_id
    assert second.user_message.sequence_no == 5


def test_retry_reuses_a_failed_regenerated_user_without_another_copy(
    migrated_database: str,
) -> None:
    """Retry after regenerated failure targets the copied latest USER directly."""

    conversation_id, _, _ = _seed_turn(migrated_database, "SUCCEEDED")
    regenerated = _admit_regenerate(
        migrated_database, conversation_id, "regenerate-key"
    )
    _fail_pending_run(migrated_database, regenerated.run.id)

    retried = _admit_retry(migrated_database, conversation_id, "retry-key")

    assert retried.user_message.id == regenerated.user_message.id
    assert retried.run.input_message_id == regenerated.user_message.id
    assert retried.run.attempt_no == 2
    assert _scalar(migrated_database, "SELECT COUNT(*) FROM messages") == 3


def test_regenerate_rejects_latest_user_without_a_successful_answer(
    migrated_database: str,
) -> None:
    """A failed latest USER cannot be the source of a Regenerate command."""

    conversation_id, _, _ = _seed_turn(migrated_database, "FAILED")

    with pytest.raises(RegenerateNotAllowedError):
        _admit_regenerate(migrated_database, conversation_id, "regenerate-key")


def test_retry_or_regenerate_same_key_mismatch_is_rejected(
    migrated_database: str,
) -> None:
    """An existing Retry key cannot be reused as a Regenerate command identity."""

    conversation_id, _, _ = _seed_turn(migrated_database, "FAILED")
    _admit_retry(migrated_database, conversation_id, "shared-key")

    with pytest.raises(ClientRequestIdReusedError):
        _admit_regenerate(migrated_database, conversation_id, "shared-key")
