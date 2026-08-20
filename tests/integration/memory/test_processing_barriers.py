"""Real MySQL coverage for Slice 5 ordered Memory processing."""

import asyncio

import pytest

from langley.answering.errors import RunErrorCode, WorkflowFailure
from langley.answering.fake_provider import ScriptedProviderRound
from langley.infrastructure.database import (
    create_database_engine,
    create_session_factory,
    dispose_database_engine,
)
from langley.memory.processing import (
    MemorySynchronizationUnavailableError,
    add_memory_direct,
    correct_memory_direct,
    forget_memory_direct,
)

from ._support import (
    _completion,
    _insert_memory,
    _policy,
    _run,
    _scalar,
    _seed_evidence,
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


def test_manual_barrier_stops_after_its_bounded_prefix_without_mass_closure(
    migrated_database: str,
) -> None:
    """Direct control cannot close an over-limit backlog to proceed."""

    _, message_ids = _seed_evidence(
        migrated_database, [f"backlog {index}" for index in range(5)]
    )
    policy = _policy(
        [
            ScriptedProviderRound(
                events=(
                    _completion(
                        {"mutations": [], "user_requested_memory_action": False}
                    ),
                )
            )
            for _ in range(4)
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
                    content="must not bypass backlog",
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
            "SELECT COUNT(*) FROM messages WHERE id IN "
            f"({', '.join(map(str, message_ids))}) AND memory_processed_at IS NULL",
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
