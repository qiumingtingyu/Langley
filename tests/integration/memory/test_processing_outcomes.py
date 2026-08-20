"""Real MySQL coverage for Slice 5 ordered Memory processing."""

import asyncio

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
    process_memory_through,
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


class _RecordingPolicy(MemoryPolicy):
    """Immediate deterministic policy used where an interleaving gate is unnecessary."""

    def __init__(self, decisions: list[MemoryPolicyResult | Exception]):
        self._decisions = decisions
        self.inputs: list[MemoryPolicyInput] = []

    async def decide(self, policy_input: MemoryPolicyInput) -> MemoryPolicyResult:
        self.inputs.append(policy_input)
        decision = self._decisions.pop(0)
        if isinstance(decision, Exception):
            raise decision
        return decision


def test_real_processing_outcomes_match_durable_dispositions_and_publish_fail_open(
    migrated_database: str,
) -> None:
    """Post-commit outcomes retain source correlation and cannot undo durable Memory."""

    conversation_id, message_ids = _seed_evidence(
        migrated_database,
        ["remember", "explicit no change", "retry later", "bad output"],
    )
    outcomes = []

    def broken_publish(outcome) -> None:
        outcomes.append(outcome)
        raise RuntimeError("subscriber gone")

    policy = _policy(
        [
            ScriptedProviderRound(
                events=(
                    _completion(
                        {
                            "mutations": [
                                {"operation": "NEW", "content": "saved by outcome"}
                            ],
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
            ScriptedProviderRound(
                events=(), failure=WorkflowFailure(RunErrorCode.LLM_PROVIDER_FAILED)
            ),
        ]
    )

    async def process() -> object:
        engine = create_database_engine(migrated_database)
        session_factory = create_session_factory(engine)
        try:
            return await process_memory_through(
                session_factory,
                user_id=1,
                through_message_id=message_ids[-1],
                policy=policy,
                local_timezone="Asia/Shanghai",
                lane=asyncio.Lock(),
                limit=4,
                outcome_callback=broken_publish,
            )
        finally:
            await dispose_database_engine(engine)

    result = _run(process())
    assert result.complete is False
    assert [(outcome.kind, outcome.source_message_id) for outcome in outcomes] == [
        ("updated", message_ids[0]),
        ("no_change", message_ids[1]),
        ("retry_pending", message_ids[2]),
    ]
    assert all(outcome.conversation_id == conversation_id for outcome in outcomes)
    assert outcomes[0].created_count == 1
    assert outcomes[0].user_requested_memory_action is True
    assert outcomes[1].user_requested_memory_action is True
    assert _scalar(migrated_database, "SELECT COUNT(*) FROM memories") == 1
    memory_id = int(_scalar(migrated_database, "SELECT id FROM memories"))
    assert (
        _scalar(
            migrated_database,
            "SELECT COUNT(*) FROM messages "
            "WHERE memory_processed_at IS NOT NULL "
            f"AND id IN ({message_ids[0]}, {message_ids[1]})",
        )
        == 2
    )

    invalid_outcomes: list[MemoryOutcome] = []
    invalid = _process(
        migrated_database,
        through_message_id=message_ids[-1],
        policy=_RecordingPolicy(
            [MemoryPolicyInvalidOutputError("bounded invalid output")]
        ),
        limit=1,
        outcome_callback=invalid_outcomes.append,
    )
    assert invalid.complete is False
    assert invalid_outcomes == [
        MemoryOutcome(
            user_id=1,
            conversation_id=conversation_id,
            source_message_id=message_ids[2],
            kind="not_saved",
        )
    ]

    changed_outcomes: list[MemoryOutcome] = []
    changed = _process(
        migrated_database,
        through_message_id=message_ids[3],
        policy=_RecordingPolicy(
            [
                _result(
                    {
                        "mutations": [
                            {
                                "operation": "CHANGE",
                                "target_memory_id": memory_id,
                                "content": "changed by outcome",
                            }
                        ],
                        "user_requested_memory_action": True,
                    }
                )
            ]
        ),
        limit=1,
        outcome_callback=changed_outcomes.append,
    )
    assert changed.complete is True
    assert changed_outcomes == [
        MemoryOutcome(
            user_id=1,
            conversation_id=conversation_id,
            source_message_id=message_ids[3],
            kind="updated",
            user_requested_memory_action=True,
            changed_count=1,
        )
    ]
    assert (
        _scalar(
            migrated_database,
            f"SELECT content FROM memories WHERE id = {memory_id}",
        )
        == "changed by outcome"
    )

    async def add_forget_evidence() -> int:
        engine = create_database_engine(migrated_database)
        session_factory = create_session_factory(engine)
        try:
            async with session_factory() as session, session.begin():
                source = Message(
                    conversation_id=conversation_id,
                    sequence_no=5,
                    role="USER",
                    content="forget changed fact",
                    run_id=None,
                    regenerated_from_message_id=None,
                    created_at=utc_now(),
                )
                session.add(source)
                await session.flush()
                return source.id
        finally:
            await dispose_database_engine(engine)

    forget_message_id = _run(add_forget_evidence())
    forgotten_outcomes: list[MemoryOutcome] = []
    forgotten = _process(
        migrated_database,
        through_message_id=forget_message_id,
        policy=_RecordingPolicy(
            [
                _result(
                    {
                        "mutations": [
                            {"operation": "FORGET", "target_memory_id": memory_id}
                        ],
                        "user_requested_memory_action": True,
                    }
                )
            ]
        ),
        limit=1,
        outcome_callback=forgotten_outcomes.append,
    )
    assert forgotten.complete is True
    assert forgotten_outcomes == [
        MemoryOutcome(
            user_id=1,
            conversation_id=conversation_id,
            source_message_id=forget_message_id,
            kind="updated",
            user_requested_memory_action=True,
            forgotten_count=1,
        )
    ]
    assert (
        _scalar(
            migrated_database,
            f"SELECT COUNT(*) FROM memories WHERE id = {memory_id}",
        )
        == 0
    )
