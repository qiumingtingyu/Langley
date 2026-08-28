"""Real MySQL coverage for Slice 5 ordered Memory processing."""

import asyncio

import pytest
from sqlalchemy import text

from langley.answering.errors import RunErrorCode, WorkflowFailure
from langley.answering.fake_provider import ScriptedProviderRound
from langley.business_time import utc_now
from langley.infrastructure.database import (
    create_database_engine,
    create_session_factory,
    dispose_database_engine,
)
from langley.infrastructure.models import Message
from langley.memory.events import MemoryOutcome
from langley.memory.policy import (
    MemoryPolicy,
    MemoryPolicyInput,
    MemoryPolicyInvalidOutputError,
    MemoryPolicyResult,
)
from langley.memory.processing import (
    MemorySynchronizationUnavailableError,
    process_memory_through,
    set_auto_memory_enabled,
)

from ._support import (
    _completion,
    _policy,
    _process,
    _result,
    _run,
    _scalar,
    _seed_evidence,
)


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

    @property
    def estimated_token_budget(self) -> int:
        return 10_000

    async def decide(self, policy_input: MemoryPolicyInput) -> MemoryPolicyResult:
        self.inputs.append(policy_input)
        if len(self.inputs) == 1:
            self.started.set()
            await self.release.wait()
            if isinstance(self._first, Exception):
                raise self._first
            return self._first
        return self._later


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


def test_repeated_off_contradiction_closes_one_source_then_continues(
    migrated_database: str,
) -> None:
    conversation_id, message_ids = _seed_evidence(
        migrated_database, ["implicit preference", "nothing else to save"]
    )

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
    contradictory_result = {
        "mutations": [{"operation": "NEW", "content": "implicit fact"}],
        "user_requested_memory_action": False,
    }
    policy = _policy(
        [
            ScriptedProviderRound(events=(_completion(contradictory_result),)),
            ScriptedProviderRound(events=(_completion(contradictory_result),)),
            ScriptedProviderRound(
                events=(
                    _completion(
                        {"mutations": [], "user_requested_memory_action": False}
                    ),
                )
            ),
        ]
    )
    outcomes: list[MemoryOutcome] = []

    result = _process(
        migrated_database,
        through_message_id=message_ids[-1],
        policy=policy,
        outcome_callback=outcomes.append,
    )

    assert result.processed_count == 2
    assert result.complete is True
    assert outcomes == [
        MemoryOutcome(
            user_id=1,
            conversation_id=conversation_id,
            source_message_id=message_ids[0],
            kind="not_saved",
        )
    ]
    assert _scalar(migrated_database, "SELECT COUNT(*) FROM memories") == 0
    assert (
        _scalar(
            migrated_database,
            "SELECT COUNT(*) FROM messages WHERE id IN "
            f"({message_ids[0]}, {message_ids[1]}) "
            "AND memory_processed_at IS NOT NULL",
        )
        == 2
    )


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
