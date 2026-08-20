"""Real MySQL coverage for Slice 5 ordered Memory processing."""

from langley.answering.errors import RunErrorCode, WorkflowFailure
from langley.answering.fake_provider import ScriptedProviderRound
from langley.business_time import utc_now
from langley.infrastructure.database import (
    create_database_engine,
    create_session_factory,
    dispose_database_engine,
)
from langley.infrastructure.models import Conversation, Message
from langley.memory.policy import (
    MemoryPolicy,
    MemoryPolicyInput,
    MemoryPolicyResult,
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


def test_processed_retry_evidence_is_not_processed_twice(
    migrated_database: str,
) -> None:
    """Retry's stable canonical Message identity permits exactly one disposition."""

    _, message_ids = _seed_evidence(migrated_database, ["same retried user evidence"])
    policy = _policy(
        [
            ScriptedProviderRound(
                events=(
                    _completion(
                        {
                            "mutations": [
                                {"operation": "NEW", "content": "one durable fact"}
                            ],
                            "user_requested_memory_action": False,
                        }
                    ),
                )
            )
        ]
    )

    assert _process(
        migrated_database, through_message_id=message_ids[0], policy=policy
    ).complete
    retry_drain = _process(
        migrated_database, through_message_id=message_ids[0], policy=policy
    )

    assert retry_drain == type(retry_drain)(processed_count=0, complete=True)
    assert _scalar(migrated_database, "SELECT COUNT(*) FROM memories") == 1


def test_regenerated_copy_is_never_selected_as_memory_evidence(
    migrated_database: str,
) -> None:
    """A copied USER Message stays excluded even when it falls below the boundary."""

    conversation_id, message_ids = _seed_evidence(migrated_database, ["canonical"])

    async def add_copy() -> int:
        engine = create_database_engine(migrated_database)
        session_factory = create_session_factory(engine)
        try:
            async with session_factory() as session, session.begin():
                copied = Message(
                    conversation_id=conversation_id,
                    sequence_no=2,
                    role="USER",
                    content="regenerated copy",
                    run_id=None,
                    regenerated_from_message_id=message_ids[0],
                    created_at=utc_now(),
                )
                session.add(copied)
                await session.flush()
                return copied.id
        finally:
            await dispose_database_engine(engine)

    copied_id = _run(add_copy())
    policy = _RecordingPolicy(
        [_result({"mutations": [], "user_requested_memory_action": False})]
    )
    assert _process(
        migrated_database, through_message_id=copied_id, policy=policy
    ).complete
    assert [item.evidence_message_id for item in policy.inputs] == [message_ids[0]]
    assert [item.evidence_content for item in policy.inputs] == ["canonical"]
    assert (
        _scalar(
            migrated_database,
            f"SELECT memory_processed_at IS NULL FROM messages WHERE id = {copied_id}",
        )
        == 1
    )


def test_cross_conversation_transient_failure_blocks_the_user_global_prefix(
    migrated_database: str,
) -> None:
    """An older Conversation A failure prevents semantic processing of later B."""

    _, first_ids = _seed_evidence(migrated_database, ["older A"])

    async def add_later_conversation() -> int:
        engine = create_database_engine(migrated_database)
        session_factory = create_session_factory(engine)
        try:
            async with session_factory() as session, session.begin():
                now = utc_now()
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
                message = Message(
                    conversation_id=conversation.id,
                    sequence_no=1,
                    role="USER",
                    content="later B",
                    run_id=None,
                    regenerated_from_message_id=None,
                    created_at=now,
                )
                session.add(message)
                await session.flush()
                return message.id
        finally:
            await dispose_database_engine(engine)

    later_id = _run(add_later_conversation())
    policy = _RecordingPolicy([WorkflowFailure(RunErrorCode.LLM_PROVIDER_FAILED)])
    result = _process(migrated_database, through_message_id=later_id, policy=policy)

    assert result == type(result)(processed_count=0, complete=False)
    assert [item.evidence_message_id for item in policy.inputs] == [first_ids[0]]
    assert (
        _scalar(
            migrated_database,
            "SELECT COUNT(*) FROM messages WHERE id IN "
            f"({first_ids[0]}, {later_id}) AND memory_processed_at IS NULL",
        )
        == 2
    )
