"""Real MySQL coverage for Slice 5 ordered Memory processing."""

import asyncio
from datetime import timedelta

import pytest
from sqlalchemy import event
from sqlalchemy.orm import Session

from langley.answering.fake_provider import FakeProvider, ScriptedProviderRound
from langley.business_time import utc_now
from langley.infrastructure.database import (
    create_database_engine,
    create_session_factory,
    dispose_database_engine,
)
from langley.infrastructure.models import Memory
from langley.memory.events import MemoryOutcome
from langley.memory.policy import (
    MemoryPolicy,
    MemoryPolicyInput,
    MemoryPolicyResult,
    estimate_load_all_memory_contribution,
)
from langley.memory.processing import (
    MemoryProcessingStopReason,
    process_memory_through,
)

from ._support import (
    _completion,
    _insert_memory,
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


def test_capacity_rejected_new_closes_source_and_later_evidence_progresses(
    migrated_database: str,
) -> None:
    conversation_id, message_ids = _seed_evidence(
        migrated_database, ["remember another fact", "nothing else to save"]
    )
    _insert_memory(migrated_database, content="a")
    outcomes: list[MemoryOutcome] = []
    policy = _policy(
        [
            ScriptedProviderRound(
                events=(
                    _completion(
                        {
                            "mutations": [{"operation": "NEW", "content": "b"}],
                            "user_requested_memory_action": True,
                        }
                    ),
                )
            ),
            ScriptedProviderRound(
                events=(
                    _completion(
                        {"mutations": [], "user_requested_memory_action": True}
                    ),
                )
            ),
        ],
        estimated_token_budget=estimate_load_all_memory_contribution(["a"]),
    )

    result = _process(
        migrated_database,
        through_message_id=message_ids[-1],
        policy=policy,
        limit=2,
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
        ),
        MemoryOutcome(
            user_id=1,
            conversation_id=conversation_id,
            source_message_id=message_ids[1],
            kind="no_change",
            user_requested_memory_action=True,
        ),
    ]
    assert _scalar(migrated_database, "SELECT COUNT(*) FROM memories") == 1
    assert _scalar(migrated_database, "SELECT content FROM memories") == "a"
    assert (
        _scalar(
            migrated_database,
            "SELECT COUNT(*) FROM messages WHERE memory_processed_at IS NOT NULL",
        )
        == 2
    )


def test_capacity_rejected_whole_result_applies_no_partial_mutation(
    migrated_database: str,
) -> None:
    conversation_id, message_ids = _seed_evidence(
        migrated_database, ["replace, forget, and add"]
    )
    first_id = _insert_memory(migrated_database, content="a")
    second_id = _insert_memory(migrated_database, content="b")
    outcomes: list[MemoryOutcome] = []
    policy = _policy(
        [
            ScriptedProviderRound(
                events=(
                    _completion(
                        {
                            "mutations": [
                                {
                                    "operation": "CHANGE",
                                    "target_memory_id": first_id,
                                    "content": "aa",
                                },
                                {
                                    "operation": "FORGET",
                                    "target_memory_id": second_id,
                                },
                                {"operation": "NEW", "content": "x" * 40},
                            ],
                            "user_requested_memory_action": True,
                        }
                    ),
                )
            )
        ],
        estimated_token_budget=(estimate_load_all_memory_contribution(["a", "b"]) + 4),
    )

    result = _process(
        migrated_database,
        through_message_id=message_ids[0],
        policy=policy,
        limit=1,
        outcome_callback=outcomes.append,
    )

    assert result.processed_count == 1
    assert result.complete is True
    assert outcomes == [
        MemoryOutcome(
            user_id=1,
            conversation_id=conversation_id,
            source_message_id=message_ids[0],
            kind="not_saved",
        )
    ]
    assert _scalar(migrated_database, "SELECT COUNT(*) FROM memories") == 2
    assert (
        _scalar(
            migrated_database,
            f"SELECT content FROM memories WHERE id = {first_id}",
        )
        == "a"
    )
    assert (
        _scalar(
            migrated_database,
            f"SELECT content FROM memories WHERE id = {second_id}",
        )
        == "b"
    )
    assert (
        _scalar(
            migrated_database,
            "SELECT memory_processed_at IS NOT NULL FROM messages "
            f"WHERE id = {message_ids[0]}",
        )
        == 1
    )


def test_exact_fit_automatic_post_state_succeeds(migrated_database: str) -> None:
    _, message_ids = _seed_evidence(migrated_database, ["remember exact fit"])
    policy = _policy(
        [
            ScriptedProviderRound(
                events=(
                    _completion(
                        {
                            "mutations": [{"operation": "NEW", "content": "abc"}],
                            "user_requested_memory_action": True,
                        }
                    ),
                )
            )
        ],
        estimated_token_budget=estimate_load_all_memory_contribution(["abc"]),
    )

    result = _process(
        migrated_database, through_message_id=message_ids[0], policy=policy, limit=1
    )

    assert result.processed_count == 1
    assert result.complete is True
    assert _scalar(migrated_database, "SELECT content FROM memories") == "abc"
    assert (
        _scalar(
            migrated_database,
            "SELECT memory_processed_at IS NOT NULL FROM messages "
            f"WHERE id = {message_ids[0]}",
        )
        == 1
    )


def test_preexisting_overflow_remains_pending_before_provider_invocation(
    migrated_database: str,
) -> None:
    conversation_id, message_ids = _seed_evidence(
        migrated_database, ["pending behind legacy overflow"]
    )
    _insert_memory(migrated_database, content="ab")
    provider = FakeProvider(
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
    policy = MemoryPolicy(
        provider=provider,
        memory_policy_estimated_token_budget=(
            estimate_load_all_memory_contribution(["ab"]) - 1
        ),
    )
    outcomes: list[MemoryOutcome] = []

    result = _process(
        migrated_database,
        through_message_id=message_ids[0],
        policy=policy,
        limit=1,
        outcome_callback=outcomes.append,
    )

    assert result.processed_count == 0
    assert result.complete is False
    assert provider.requests == []
    assert outcomes == [
        MemoryOutcome(
            user_id=1,
            conversation_id=conversation_id,
            source_message_id=message_ids[0],
            kind="retry_pending",
        )
    ]
    assert (
        _scalar(
            migrated_database,
            "SELECT memory_processed_at IS NULL FROM messages "
            f"WHERE id = {message_ids[0]}",
        )
        == 1
    )


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

    assert result == type(result)(
        processed_count=1,
        complete=True,
        stop_reason=MemoryProcessingStopReason.COMPLETE,
    )
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


def test_forget_does_not_resurrect_from_already_closed_old_evidence(
    migrated_database: str,
) -> None:
    """A physical FORGET cannot be undone by reprocessing its old provenance."""

    _, message_ids = _seed_evidence(migrated_database, ["remember tea", "forget tea"])
    first = _process(
        migrated_database,
        through_message_id=message_ids[0],
        policy=_policy(
            [
                ScriptedProviderRound(
                    events=(
                        _completion(
                            {
                                "mutations": [
                                    {"operation": "NEW", "content": "prefers tea"}
                                ],
                                "user_requested_memory_action": True,
                            }
                        ),
                    )
                )
            ]
        ),
    )
    assert first.complete
    memory_id = int(_scalar(migrated_database, "SELECT id FROM memories"))

    forgotten = _process(
        migrated_database,
        through_message_id=message_ids[1],
        policy=_policy(
            [
                ScriptedProviderRound(
                    events=(
                        _completion(
                            {
                                "mutations": [
                                    {
                                        "operation": "FORGET",
                                        "target_memory_id": memory_id,
                                    }
                                ],
                                "user_requested_memory_action": True,
                            }
                        ),
                    )
                )
            ]
        ),
    )
    assert forgotten.complete
    assert _scalar(migrated_database, "SELECT COUNT(*) FROM memories") == 0

    replay_old_boundary = _process(
        migrated_database,
        through_message_id=message_ids[0],
        policy=_policy([]),
    )
    assert replay_old_boundary == type(replay_old_boundary)(
        processed_count=0,
        complete=True,
        stop_reason=MemoryProcessingStopReason.COMPLETE,
    )
    assert _scalar(migrated_database, "SELECT COUNT(*) FROM memories") == 0


def test_apply_transaction_rollback_leaves_effect_and_marker_undurable(
    migrated_database: str,
) -> None:
    """An apply-transaction failure rolls back both NEW and its source marker."""

    _, message_ids = _seed_evidence(migrated_database, ["remember durable fact"])

    def fail_dirty_commit(session) -> None:
        if session.new or session.dirty or session.deleted:
            raise RuntimeError("forced apply rollback")

    event.listen(
        Session,
        "before_commit",
        fail_dirty_commit,
    )
    try:
        with pytest.raises(RuntimeError, match="forced apply rollback"):
            _process(
                migrated_database,
                through_message_id=message_ids[0],
                policy=_policy(
                    [
                        ScriptedProviderRound(
                            events=(
                                _completion(
                                    {
                                        "mutations": [
                                            {
                                                "operation": "NEW",
                                                "content": "durable fact",
                                            }
                                        ],
                                        "user_requested_memory_action": True,
                                    }
                                ),
                            )
                        )
                    ]
                ),
            )
    finally:
        event.remove(
            Session,
            "before_commit",
            fail_dirty_commit,
        )

    assert _scalar(migrated_database, "SELECT COUNT(*) FROM memories") == 0
    assert (
        _scalar(
            migrated_database,
            "SELECT memory_processed_at IS NULL FROM messages "
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
