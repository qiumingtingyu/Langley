"""Real MySQL coverage for Slice 5 ordered Memory processing."""

import asyncio
import json
from argparse import Namespace
from datetime import timedelta

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text

from langley.answering.contracts import LLMFinishReason, LLMResponseCompleted
from langley.answering.errors import RunErrorCode, WorkflowFailure
from langley.answering.fake_provider import FakeProvider, ScriptedProviderRound
from langley.business_time import utc_now
from langley.infrastructure.database import (
    create_database_engine,
    create_session_factory,
    dispose_database_engine,
)
from langley.infrastructure.models import Conversation, Memory, Message, User
from langley.memory_policy import (
    MemoryPolicy,
    MemoryPolicyInput,
    MemoryPolicyInvalidOutputError,
    MemoryPolicyResult,
)
from langley.memory_processing import (
    MemorySynchronizationUnavailableError,
    add_memory_direct,
    correct_memory_direct,
    forget_memory_direct,
    process_memory_through,
    set_auto_memory_enabled,
)


@pytest.fixture
def migrated_database(test_database_url: str, reset_database) -> str:
    reset_database()
    config = Config("alembic.ini")
    config.cmd_opts = Namespace(x=["use_test_database=true"])
    command.upgrade(config, "head")
    return test_database_url


def _completion(payload: dict[str, object] | str) -> LLMResponseCompleted:
    return LLMResponseCompleted(
        assistant_content=payload if isinstance(payload, str) else json.dumps(payload),
        tool_calls=(),
        finish_reason=LLMFinishReason.STOP,
        usage=None,
    )


def _policy(rounds: list[ScriptedProviderRound]) -> MemoryPolicy:
    return MemoryPolicy(
        provider=FakeProvider(rounds), memory_policy_estimated_token_budget=10_000
    )


def _seed_evidence(database_url: str, contents: list[str]) -> tuple[int, list[int]]:
    async def seed() -> tuple[int, list[int]]:
        engine = create_database_engine(database_url)
        session_factory = create_session_factory(engine)
        try:
            async with session_factory() as session, session.begin():
                now = utc_now()
                user = User(id=1, created_at=now)
                conversation = Conversation(
                    user_id=1,
                    title=None,
                    created_at=now,
                    updated_at=now,
                    last_message_at=now,
                    deleted_at=None,
                )
                session.add(user)
                await session.flush()
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
                    for index, content in enumerate(contents, start=1)
                ]
                session.add_all(messages)
                await session.flush()
                return conversation.id, [message.id for message in messages]
        finally:
            await dispose_database_engine(engine)

    return asyncio.run(seed())


def _process(
    database_url: str, *, through_message_id: int, policy: MemoryPolicy, limit: int = 4
):
    async def process():
        engine = create_database_engine(database_url)
        session_factory = create_session_factory(engine)
        try:
            return await process_memory_through(
                session_factory,
                user_id=1,
                through_message_id=through_message_id,
                policy=policy,
                local_timezone="Asia/Shanghai",
                lane=asyncio.Lock(),
                limit=limit,
            )
        finally:
            await dispose_database_engine(engine)

    return asyncio.run(process())


def _scalar(database_url: str, statement: str) -> object:
    async def execute() -> object:
        engine = create_database_engine(database_url)
        try:
            async with engine.connect() as connection:
                return (await connection.execute(text(statement))).scalar_one()
        finally:
            await dispose_database_engine(engine)

    return asyncio.run(execute())


def _run(coroutine):
    return asyncio.run(coroutine)


def _result(payload: dict[str, object]) -> MemoryPolicyResult:
    return MemoryPolicyResult.model_validate(payload)


class _GatePolicy(MemoryPolicy):
    """Small deterministic policy gate for real-MySQL interleaving tests."""

    def __init__(
        self, first: MemoryPolicyResult | Exception, later: MemoryPolicyResult
    ):
        self._first = first
        self._later = later
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.inputs: list[MemoryPolicyInput] = []

    async def decide(self, policy_input: MemoryPolicyInput) -> MemoryPolicyResult:
        self.inputs.append(policy_input)
        if len(self.inputs) == 1:
            self.started.set()
            await self.release.wait()
            if isinstance(self._first, Exception):
                raise self._first
            return self._first
        return self._later


def _insert_memory(database_url: str, *, content: str = "existing") -> int:
    async def insert() -> int:
        engine = create_database_engine(database_url)
        session_factory = create_session_factory(engine)
        try:
            async with session_factory() as session, session.begin():
                memory = Memory(
                    user_id=1,
                    content=content,
                    source_message_id=None,
                    valid_until=None,
                    created_at=utc_now(),
                    updated_at=utc_now(),
                )
                session.add(memory)
                await session.flush()
                return memory.id
        finally:
            await dispose_database_engine(engine)

    return _run(insert())


def test_invalid_output_closes_only_its_evidence_then_continues_oldest_first(
    migrated_database: str,
) -> None:
    """Two invalid outputs conservatively close one item; later evidence proceeds."""

    _, message_ids = _seed_evidence(migrated_database, ["first", "second"])
    policy = _policy(
        [
            ScriptedProviderRound(events=(_completion("not-json"),)),
            ScriptedProviderRound(events=(_completion("still-not-json"),)),
            ScriptedProviderRound(
                events=(
                    _completion(
                        {
                            "mutations": [
                                {"operation": "NEW", "content": "second fact"}
                            ],
                            "user_requested_memory_action": False,
                        }
                    ),
                )
            ),
        ]
    )

    result = _process(
        migrated_database, through_message_id=message_ids[-1], policy=policy
    )

    assert result.processed_count == 2
    assert result.complete is True
    assert (
        _scalar(
            migrated_database,
            "SELECT COUNT(*) FROM messages WHERE id IN "
            f"({message_ids[0]}, {message_ids[1]}) "
            "AND memory_processed_at IS NOT NULL",
        )
        == 2
    )
    assert _scalar(migrated_database, "SELECT COUNT(*) FROM memories") == 1


def test_transient_failure_leaves_oldest_marker_null_and_stops_prefix(
    migrated_database: str,
) -> None:
    """A provider failure neither closes nor skips the oldest pending evidence."""

    _, message_ids = _seed_evidence(migrated_database, ["first", "second"])
    policy = _policy(
        [
            ScriptedProviderRound(
                events=(), failure=WorkflowFailure(RunErrorCode.LLM_PROVIDER_FAILED)
            )
        ]
    )

    result = _process(
        migrated_database, through_message_id=message_ids[-1], policy=policy
    )

    assert result.processed_count == 0
    assert result.complete is False
    assert (
        _scalar(
            migrated_database,
            "SELECT COUNT(*) FROM messages WHERE id IN "
            f"({message_ids[0]}, {message_ids[1]}) "
            "AND memory_processed_at IS NOT NULL",
        )
        == 0
    )
    assert _scalar(migrated_database, "SELECT COUNT(*) FROM memories") == 0


def test_policy_mutation_and_marker_commit_atomically(migrated_database: str) -> None:
    """A successful NEW writes its source provenance and marker in one transaction."""

    _, message_ids = _seed_evidence(migrated_database, ["remember this"])
    policy = _policy(
        [
            ScriptedProviderRound(
                events=(
                    _completion(
                        {
                            "mutations": [
                                {"operation": "NEW", "content": "durable fact"}
                            ],
                            "user_requested_memory_action": True,
                        }
                    ),
                )
            )
        ]
    )

    result = _process(
        migrated_database, through_message_id=message_ids[0], policy=policy
    )

    assert result == type(result)(processed_count=1, complete=True)
    assert (
        _scalar(
            migrated_database,
            "SELECT source_message_id FROM memories WHERE content = 'durable fact'",
        )
        == message_ids[0]
    )
    assert (
        _scalar(
            migrated_database,
            "SELECT memory_processed_at IS NOT NULL FROM messages "
            f"WHERE id = {message_ids[0]}",
        )
        == 1
    )


def test_boundary_does_not_consume_newer_canonical_evidence(
    migrated_database: str,
) -> None:
    """A finite captured high-water leaves later admissions for a future wake."""

    _, message_ids = _seed_evidence(migrated_database, ["first", "second"])
    policy = _policy(
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

    result = _process(
        migrated_database, through_message_id=message_ids[0], policy=policy
    )

    assert result.complete is True
    assert (
        _scalar(
            migrated_database,
            "SELECT memory_processed_at IS NOT NULL FROM messages "
            f"WHERE id = {message_ids[0]}",
        )
        == 1
    )
    assert (
        _scalar(
            migrated_database,
            "SELECT memory_processed_at IS NULL FROM messages "
            f"WHERE id = {message_ids[1]}",
        )
        == 1
    )


def test_manual_add_refuses_to_bypass_an_unavailable_ordered_barrier(
    migrated_database: str,
) -> None:
    """A failed earlier policy attempt prevents the direct mutation from committing."""

    _, message_ids = _seed_evidence(migrated_database, ["earlier evidence"])
    policy = _policy(
        [
            ScriptedProviderRound(
                events=(), failure=WorkflowFailure(RunErrorCode.LLM_PROVIDER_FAILED)
            )
        ]
    )

    async def add() -> None:
        engine = create_database_engine(migrated_database)
        session_factory = create_session_factory(engine)
        try:
            with pytest.raises(MemorySynchronizationUnavailableError):
                await add_memory_direct(
                    session_factory,
                    user_id=1,
                    content="manual fact",
                    valid_until=None,
                    policy=policy,
                    local_timezone="Asia/Shanghai",
                    lane=asyncio.Lock(),
                )
        finally:
            await dispose_database_engine(engine)

    _run(add())

    assert _scalar(migrated_database, "SELECT COUNT(*) FROM memories") == 0
    assert (
        _scalar(
            migrated_database,
            "SELECT memory_processed_at IS NULL FROM messages "
            f"WHERE id = {message_ids[0]}",
        )
        == 1
    )


def test_manual_barrier_propagates_an_unexpected_policy_error(
    migrated_database: str,
) -> None:
    """Programmer faults are never relabelled as retriable synchronization failures."""

    _seed_evidence(migrated_database, ["earlier evidence"])
    policy = _policy([ScriptedProviderRound(events=(), failure=RuntimeError("bug"))])

    async def add() -> None:
        engine = create_database_engine(migrated_database)
        session_factory = create_session_factory(engine)
        try:
            with pytest.raises(RuntimeError, match="bug"):
                await add_memory_direct(
                    session_factory,
                    user_id=1,
                    content="manual fact",
                    valid_until=None,
                    policy=policy,
                    local_timezone="Asia/Shanghai",
                    lane=asyncio.Lock(),
                )
        finally:
            await dispose_database_engine(engine)

    _run(add())
    assert _scalar(migrated_database, "SELECT COUNT(*) FROM memories") == 0


def test_disabling_auto_memory_does_not_invoke_or_wait_for_policy(
    migrated_database: str,
) -> None:
    """ON-to-OFF is only a short User-row transaction, even with pending evidence."""

    _seed_evidence(migrated_database, ["pending evidence"])

    async def disable() -> None:
        engine = create_database_engine(migrated_database)
        session_factory = create_session_factory(engine)
        try:
            await set_auto_memory_enabled(
                session_factory,
                user_id=1,
                enabled=False,
                policy=None,
                local_timezone="Asia/Shanghai",
                lane=asyncio.Lock(),
            )
        finally:
            await dispose_database_engine(engine)

    _run(disable())

    assert _scalar(migrated_database, "SELECT auto_memory_enabled FROM users") == 0
    assert (
        _scalar(
            migrated_database,
            "SELECT COUNT(*) FROM messages WHERE memory_processed_at IS NULL",
        )
        == 1
    )


def test_change_and_marker_close_together(migrated_database: str) -> None:
    """CHANGE keeps the stable id and closes its source evidence atomically."""

    _, message_ids = _seed_evidence(migrated_database, ["correct preference"])
    memory_id = _insert_memory(migrated_database, content="old preference")
    policy = _policy(
        [
            ScriptedProviderRound(
                events=(
                    _completion(
                        {
                            "mutations": [
                                {
                                    "operation": "CHANGE",
                                    "target_memory_id": memory_id,
                                    "content": "new preference",
                                }
                            ],
                            "user_requested_memory_action": True,
                        }
                    ),
                )
            )
        ]
    )

    assert _process(
        migrated_database, through_message_id=message_ids[0], policy=policy
    ).complete
    assert (
        _scalar(
            migrated_database, f"SELECT content FROM memories WHERE id = {memory_id}"
        )
        == "new preference"
    )
    assert (
        _scalar(
            migrated_database,
            "SELECT memory_processed_at IS NOT NULL FROM messages "
            f"WHERE id = {message_ids[0]}",
        )
        == 1
    )


def test_forget_and_marker_close_together(migrated_database: str) -> None:
    """FORGET deletes its target and closes its source evidence in one commit."""

    _, message_ids = _seed_evidence(migrated_database, ["forget preference"])
    memory_id = _insert_memory(migrated_database)
    policy = _policy(
        [
            ScriptedProviderRound(
                events=(
                    _completion(
                        {
                            "mutations": [
                                {"operation": "FORGET", "target_memory_id": memory_id}
                            ],
                            "user_requested_memory_action": True,
                        }
                    ),
                )
            )
        ]
    )

    assert _process(
        migrated_database, through_message_id=message_ids[0], policy=policy
    ).complete
    assert (
        _scalar(
            migrated_database, f"SELECT COUNT(*) FROM memories WHERE id = {memory_id}"
        )
        == 0
    )
    assert (
        _scalar(
            migrated_database,
            "SELECT memory_processed_at IS NOT NULL FROM messages "
            f"WHERE id = {message_ids[0]}",
        )
        == 1
    )


def test_apply_revalidates_a_target_that_expired_after_detached_policy_read(
    migrated_database: str,
) -> None:
    """A stale CHANGE is not applied; the source is re-evaluated against fresh state."""

    _, message_ids = _seed_evidence(migrated_database, ["correct preference"])
    memory_id = _insert_memory(migrated_database, content="old preference")
    policy = _GatePolicy(
        _result(
            {
                "mutations": [
                    {
                        "operation": "CHANGE",
                        "target_memory_id": memory_id,
                        "content": "stale replacement",
                    }
                ],
                "user_requested_memory_action": True,
            }
        ),
        _result({"mutations": [], "user_requested_memory_action": False}),
    )

    async def process_with_expiry() -> None:
        engine = create_database_engine(migrated_database)
        session_factory = create_session_factory(engine)
        try:
            task = asyncio.create_task(
                process_memory_through(
                    session_factory,
                    user_id=1,
                    through_message_id=message_ids[0],
                    policy=policy,
                    local_timezone="Asia/Shanghai",
                    lane=asyncio.Lock(),
                    limit=4,
                )
            )
            await policy.started.wait()
            async with session_factory() as session, session.begin():
                memory = await session.get(Memory, memory_id)
                if memory is None:
                    raise AssertionError("expected memory")
                memory.valid_until = utc_now() - timedelta(seconds=1)
            policy.release.set()
            assert (await task).complete
        finally:
            await dispose_database_engine(engine)

    _run(process_with_expiry())

    assert (
        _scalar(
            migrated_database, f"SELECT content FROM memories WHERE id = {memory_id}"
        )
        == "old preference"
    )
    assert [item.auto_memory_enabled for item in policy.inputs] == [True, True]
    assert policy.inputs[1].current_memories == ()


def test_old_on_valid_result_is_reloaded_under_off_after_toggle(
    migrated_database: str,
) -> None:
    """The OFF transaction completes while Policy is gated, proving no held DB lock."""

    _, message_ids = _seed_evidence(migrated_database, ["implicit preference"])
    policy = _GatePolicy(
        _result(
            {
                "mutations": [{"operation": "NEW", "content": "old on fact"}],
                "user_requested_memory_action": False,
            }
        ),
        _result({"mutations": [], "user_requested_memory_action": False}),
    )

    async def process_then_disable() -> None:
        engine = create_database_engine(migrated_database)
        session_factory = create_session_factory(engine)
        try:
            task = asyncio.create_task(
                process_memory_through(
                    session_factory,
                    user_id=1,
                    through_message_id=message_ids[0],
                    policy=policy,
                    local_timezone="Asia/Shanghai",
                    lane=asyncio.Lock(),
                    limit=4,
                )
            )
            await policy.started.wait()
            await set_auto_memory_enabled(
                session_factory,
                user_id=1,
                enabled=False,
                policy=None,
                local_timezone="Asia/Shanghai",
                lane=asyncio.Lock(),
            )
            async with session_factory() as session:
                assert (
                    await session.scalar(
                        text(
                            "SELECT memory_processed_at IS NULL FROM messages "
                            f"WHERE id = {message_ids[0]}"
                        )
                    )
                    == 1
                )
            policy.release.set()
            assert (await task).complete
        finally:
            await dispose_database_engine(engine)

    _run(process_then_disable())

    assert _scalar(migrated_database, "SELECT COUNT(*) FROM memories") == 0
    assert [item.auto_memory_enabled for item in policy.inputs] == [True, False]


def test_old_on_invalid_result_is_reloaded_under_off_after_toggle(
    migrated_database: str,
) -> None:
    """Bounded invalid closure observes the same User-row mode linearization."""

    _, message_ids = _seed_evidence(migrated_database, ["implicit preference"])
    policy = _GatePolicy(
        MemoryPolicyInvalidOutputError("bounded invalid output"),
        _result({"mutations": [], "user_requested_memory_action": False}),
    )

    async def process_then_disable() -> None:
        engine = create_database_engine(migrated_database)
        session_factory = create_session_factory(engine)
        try:
            task = asyncio.create_task(
                process_memory_through(
                    session_factory,
                    user_id=1,
                    through_message_id=message_ids[0],
                    policy=policy,
                    local_timezone="Asia/Shanghai",
                    lane=asyncio.Lock(),
                    limit=4,
                )
            )
            await policy.started.wait()
            await set_auto_memory_enabled(
                session_factory,
                user_id=1,
                enabled=False,
                policy=None,
                local_timezone="Asia/Shanghai",
                lane=asyncio.Lock(),
            )
            async with session_factory() as session:
                assert (
                    await session.scalar(
                        text(
                            "SELECT memory_processed_at IS NULL FROM messages "
                            f"WHERE id = {message_ids[0]}"
                        )
                    )
                    == 1
                )
            policy.release.set()
            assert (await task).complete
        finally:
            await dispose_database_engine(engine)

    _run(process_then_disable())

    assert [item.auto_memory_enabled for item in policy.inputs] == [True, False]
    assert _scalar(migrated_database, "SELECT COUNT(*) FROM memories") == 0


def test_invalid_closure_before_off_remains_a_valid_linearization(
    migrated_database: str,
) -> None:
    """Once invalid closure commits under ON, a later OFF does not reopen evidence."""

    _, message_ids = _seed_evidence(migrated_database, ["implicit preference"])
    policy = _policy(
        [
            ScriptedProviderRound(events=(_completion("invalid"),)),
            ScriptedProviderRound(events=(_completion("still invalid"),)),
        ]
    )
    assert _process(
        migrated_database, through_message_id=message_ids[0], policy=policy
    ).complete

    async def disable() -> None:
        engine = create_database_engine(migrated_database)
        session_factory = create_session_factory(engine)
        try:
            await set_auto_memory_enabled(
                session_factory,
                user_id=1,
                enabled=False,
                policy=None,
                local_timezone="Asia/Shanghai",
                lane=asyncio.Lock(),
            )
        finally:
            await dispose_database_engine(engine)

    _run(disable())
    assert (
        _scalar(
            migrated_database,
            "SELECT memory_processed_at IS NOT NULL FROM messages "
            f"WHERE id = {message_ids[0]}",
        )
        == 1
    )


def test_manual_correct_and_forget_reread_current_target_after_barrier(
    migrated_database: str,
) -> None:
    """Manual direct mutations resolve their target after an empty completed barrier."""

    _seed_evidence(migrated_database, [])
    memory_id = _insert_memory(migrated_database, content="old fact")
    policy = _policy([])

    async def correct_then_forget() -> None:
        engine = create_database_engine(migrated_database)
        session_factory = create_session_factory(engine)
        try:
            corrected = await correct_memory_direct(
                session_factory,
                user_id=1,
                memory_id=memory_id,
                content="corrected fact",
                valid_until=None,
                policy=policy,
                local_timezone="Asia/Shanghai",
                lane=asyncio.Lock(),
            )
            assert corrected.id == memory_id
            await forget_memory_direct(
                session_factory,
                user_id=1,
                memory_id=memory_id,
                policy=policy,
                local_timezone="Asia/Shanghai",
                lane=asyncio.Lock(),
            )
        finally:
            await dispose_database_engine(engine)

    _run(correct_then_forget())
    assert (
        _scalar(
            migrated_database, f"SELECT COUNT(*) FROM memories WHERE id = {memory_id}"
        )
        == 0
    )


def test_off_to_on_processes_the_captured_prefix_before_enabling(
    migrated_database: str,
) -> None:
    """Enable waits for OFF semantics to close the pre-captured canonical evidence."""

    _, message_ids = _seed_evidence(migrated_database, ["implicit preference"])
    no_change = _result({"mutations": [], "user_requested_memory_action": False})
    policy = _GatePolicy(no_change, no_change)

    async def disable_then_enable() -> None:
        engine = create_database_engine(migrated_database)
        session_factory = create_session_factory(engine)
        try:
            await set_auto_memory_enabled(
                session_factory,
                user_id=1,
                enabled=False,
                policy=None,
                local_timezone="Asia/Shanghai",
                lane=asyncio.Lock(),
            )
            enable = asyncio.create_task(
                set_auto_memory_enabled(
                    session_factory,
                    user_id=1,
                    enabled=True,
                    policy=policy,
                    local_timezone="Asia/Shanghai",
                    lane=asyncio.Lock(),
                )
            )
            await policy.started.wait()
            policy.release.set()
            await enable
        finally:
            await dispose_database_engine(engine)

    _run(disable_then_enable())
    assert [item.auto_memory_enabled for item in policy.inputs] == [False]
    assert _scalar(migrated_database, "SELECT auto_memory_enabled FROM users") == 1
    assert (
        _scalar(
            migrated_database,
            "SELECT memory_processed_at IS NOT NULL FROM messages "
            f"WHERE id = {message_ids[0]}",
        )
        == 1
    )


def test_off_to_on_failure_keeps_auto_memory_disabled(migrated_database: str) -> None:
    """Enable fails closed when the ordered prefix cannot be processed."""

    _, _ = _seed_evidence(migrated_database, ["implicit preference"])
    policy = _policy(
        [
            ScriptedProviderRound(
                events=(), failure=WorkflowFailure(RunErrorCode.LLM_PROVIDER_FAILED)
            )
        ]
    )

    async def disable_then_fail_enable() -> None:
        engine = create_database_engine(migrated_database)
        session_factory = create_session_factory(engine)
        try:
            await set_auto_memory_enabled(
                session_factory,
                user_id=1,
                enabled=False,
                policy=None,
                local_timezone="Asia/Shanghai",
                lane=asyncio.Lock(),
            )
            with pytest.raises(MemorySynchronizationUnavailableError):
                await set_auto_memory_enabled(
                    session_factory,
                    user_id=1,
                    enabled=True,
                    policy=policy,
                    local_timezone="Asia/Shanghai",
                    lane=asyncio.Lock(),
                )
        finally:
            await dispose_database_engine(engine)

    _run(disable_then_fail_enable())
    assert _scalar(migrated_database, "SELECT auto_memory_enabled FROM users") == 0


def test_off_to_on_does_not_chase_evidence_after_its_captured_boundary(
    migrated_database: str,
) -> None:
    """A new canonical USER admitted during enable remains for a later wake."""

    conversation_id, message_ids = _seed_evidence(migrated_database, ["first"])
    no_change = _result({"mutations": [], "user_requested_memory_action": False})
    policy = _GatePolicy(no_change, no_change)

    async def enable_with_later_evidence() -> None:
        engine = create_database_engine(migrated_database)
        session_factory = create_session_factory(engine)
        try:
            await set_auto_memory_enabled(
                session_factory,
                user_id=1,
                enabled=False,
                policy=None,
                local_timezone="Asia/Shanghai",
                lane=asyncio.Lock(),
            )
            enable = asyncio.create_task(
                set_auto_memory_enabled(
                    session_factory,
                    user_id=1,
                    enabled=True,
                    policy=policy,
                    local_timezone="Asia/Shanghai",
                    lane=asyncio.Lock(),
                )
            )
            await policy.started.wait()
            async with session_factory() as session, session.begin():
                session.add(
                    Message(
                        conversation_id=conversation_id,
                        sequence_no=2,
                        role="USER",
                        content="later",
                        run_id=None,
                        regenerated_from_message_id=None,
                        created_at=utc_now(),
                    )
                )
            policy.release.set()
            await enable
        finally:
            await dispose_database_engine(engine)

    _run(enable_with_later_evidence())
    assert (
        _scalar(
            migrated_database,
            "SELECT memory_processed_at IS NOT NULL FROM messages "
            f"WHERE id = {message_ids[0]}",
        )
        == 1
    )
    assert (
        _scalar(
            migrated_database,
            "SELECT COUNT(*) FROM messages WHERE content = 'later' "
            "AND memory_processed_at IS NULL",
        )
        == 1
    )
