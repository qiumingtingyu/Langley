"""Real MySQL coverage for Slice 5 ordered Memory processing."""

import asyncio
import json
from argparse import Namespace
from datetime import timedelta

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import event, text
from sqlalchemy.orm import Session

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
from langley.memory_events import MemoryOutcome
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
    database_url: str,
    *,
    through_message_id: int,
    policy: MemoryPolicy,
    limit: int = 4,
    outcome_callback=None,
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
                outcome_callback=outcome_callback,
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
        processed_count=0, complete=True
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
