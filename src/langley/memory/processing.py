"""Ordered, transaction-safe persistence of Personal Context Memory decisions."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, cast

import structlog
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from langley.answering.errors import WorkflowFailure
from langley.business_time import utc_naive_to_local_reference, utc_now
from langley.infrastructure.models import Conversation, Memory, Message, User
from langley.memory.events import MemoryOutcome
from langley.memory.policy import (
    MemoryMutationDecision,
    MemoryPolicy,
    MemoryPolicyContextInfeasibleError,
    MemoryPolicyConversationMessage,
    MemoryPolicyInput,
    MemoryPolicyInvalidOutputError,
    MemoryPolicyItem,
    MemoryPolicyResult,
    MemoryPolicyUnavailableError,
)

BACKGROUND_BATCH_LIMIT = 4
PRE_ANSWER_CATCHUP_LIMIT = 2
PRE_ANSWER_CATCHUP_TIMEOUT_SECONDS = 5
MANUAL_SYNC_LIMIT = 4
MANUAL_SYNC_TIMEOUT_SECONDS = 20

logger = structlog.get_logger(__name__)


class MemorySynchronizationUnavailableError(RuntimeError):
    """The mandatory ordered Memory barrier could not complete safely."""


@dataclass(frozen=True)
class MemoryProcessingResult:
    """The finite result of processing a canonical USER prefix."""

    processed_count: int
    complete: bool
    outcomes: tuple[MemoryOutcome, ...] = ()


@dataclass(frozen=True)
class _DetachedEvidence:
    """Policy input plus the mode snapshot that must be revalidated on apply."""

    policy_input: MemoryPolicyInput
    auto_memory_enabled: bool
    conversation_id: int


SessionFactory = Callable[[], AsyncSession]


async def process_memory_through(
    session_factory: SessionFactory,
    *,
    user_id: int,
    through_message_id: int | None,
    policy: MemoryPolicy,
    local_timezone: str,
    lane: asyncio.Lock,
    limit: int = BACKGROUND_BATCH_LIMIT,
    outcome_callback: Callable[[MemoryOutcome], None] | None = None,
) -> MemoryProcessingResult:
    """Process the oldest-first finite canonical USER prefix under one local lane."""

    if through_message_id is None:
        return MemoryProcessingResult(processed_count=0, complete=True)
    if limit < 1:
        raise ValueError("limit must be positive")
    async with lane:
        result = await _process_memory_through_locked(
            session_factory,
            user_id=user_id,
            through_message_id=through_message_id,
            policy=policy,
            local_timezone=local_timezone,
            limit=limit,
        )
    if outcome_callback is None:
        return MemoryProcessingResult(
            processed_count=result.processed_count, complete=result.complete
        )
    for outcome in result.outcomes:
        try:
            outcome_callback(outcome)
        except Exception:
            logger.exception(
                "memory.outcome_publish_failed",
                outcome_kind=outcome.kind,
                conversation_id=outcome.conversation_id,
                source_message_id=outcome.source_message_id,
            )
    return result


async def capture_memory_high_water(
    session_factory: SessionFactory, *, user_id: int
) -> int | None:
    """Capture one User-serialized finite canonical USER boundary."""

    return await _capture_canonical_boundary(session_factory, user_id)


async def add_memory_direct(
    session_factory: SessionFactory,
    *,
    user_id: int,
    content: str,
    valid_until: datetime | None,
    policy: MemoryPolicy,
    local_timezone: str,
    lane: asyncio.Lock,
) -> Memory:
    """Synchronize earlier evidence, then add one direct current Memory fact."""

    if not content.strip():
        raise ValueError("content must not be blank")
    async with lane:
        await _require_manual_barrier(
            session_factory,
            user_id=user_id,
            policy=policy,
            local_timezone=local_timezone,
        )
        now = utc_now()
        _validate_direct_valid_until(valid_until, now)
        async with session_factory() as session, session.begin():
            await _lock_user(session, user_id)
            memory = Memory(
                user_id=user_id,
                content=content,
                source_message_id=None,
                valid_until=valid_until,
                created_at=now,
                updated_at=now,
            )
            session.add(memory)
            await session.flush()
            return memory


async def correct_memory_direct(
    session_factory: SessionFactory,
    *,
    user_id: int,
    memory_id: int,
    content: str,
    valid_until: datetime | None,
    policy: MemoryPolicy,
    local_timezone: str,
    lane: asyncio.Lock,
) -> Memory:
    """Synchronize earlier evidence, then replace one current Memory fact."""

    if not content.strip():
        raise ValueError("content must not be blank")
    async with lane:
        await _require_manual_barrier(
            session_factory,
            user_id=user_id,
            policy=policy,
            local_timezone=local_timezone,
        )
        now = utc_now()
        _validate_direct_valid_until(valid_until, now)
        async with session_factory() as session, session.begin():
            await _lock_user(session, user_id)
            memory = await _lock_current_memory(session, user_id, memory_id, now)
            if memory is None:
                raise KeyError("current memory not found")
            memory.content = content
            memory.valid_until = valid_until
            memory.source_message_id = None
            memory.updated_at = now
            return memory


async def forget_memory_direct(
    session_factory: SessionFactory,
    *,
    user_id: int,
    memory_id: int,
    policy: MemoryPolicy,
    local_timezone: str,
    lane: asyncio.Lock,
) -> None:
    """Synchronize earlier evidence, then physically delete one current Memory fact."""

    async with lane:
        await _require_manual_barrier(
            session_factory,
            user_id=user_id,
            policy=policy,
            local_timezone=local_timezone,
        )
        async with session_factory() as session, session.begin():
            await _lock_user(session, user_id)
            memory = await _lock_current_memory(session, user_id, memory_id, utc_now())
            if memory is None:
                raise KeyError("current memory not found")
            await session.delete(memory)


async def set_auto_memory_enabled(
    session_factory: SessionFactory,
    *,
    user_id: int,
    enabled: bool,
    policy: MemoryPolicy | None,
    local_timezone: str,
    lane: asyncio.Lock,
) -> None:
    """Toggle automatic Memory with the frozen ON/OFF synchronization semantics."""

    if not enabled:
        async with session_factory() as session, session.begin():
            user = await _lock_user(session, user_id)
            user.auto_memory_enabled = False
        return

    if policy is None:
        raise MemorySynchronizationUnavailableError

    async with lane:
        boundary = await _capture_canonical_boundary(session_factory, user_id)
        try:
            async with asyncio.timeout(MANUAL_SYNC_TIMEOUT_SECONDS):
                result = await _process_memory_through_locked(
                    session_factory,
                    user_id=user_id,
                    through_message_id=boundary,
                    policy=policy,
                    local_timezone=local_timezone,
                    limit=MANUAL_SYNC_LIMIT,
                )
        except TimeoutError as error:
            raise MemorySynchronizationUnavailableError from error
        if not result.complete:
            raise MemorySynchronizationUnavailableError
        async with session_factory() as session, session.begin():
            user = await _lock_user(session, user_id)
            user.auto_memory_enabled = True


async def _require_manual_barrier(
    session_factory: SessionFactory,
    *,
    user_id: int,
    policy: MemoryPolicy,
    local_timezone: str,
) -> None:
    boundary = await _capture_canonical_boundary(session_factory, user_id)
    try:
        async with asyncio.timeout(MANUAL_SYNC_TIMEOUT_SECONDS):
            result = await _process_memory_through_locked(
                session_factory,
                user_id=user_id,
                through_message_id=boundary,
                policy=policy,
                local_timezone=local_timezone,
                limit=MANUAL_SYNC_LIMIT,
            )
    except TimeoutError as error:
        raise MemorySynchronizationUnavailableError from error
    if not result.complete:
        raise MemorySynchronizationUnavailableError


async def _process_memory_through_locked(
    session_factory: SessionFactory,
    *,
    user_id: int,
    through_message_id: int | None,
    policy: MemoryPolicy,
    local_timezone: str,
    limit: int,
) -> MemoryProcessingResult:
    if through_message_id is None:
        return MemoryProcessingResult(processed_count=0, complete=True)
    processed_count = 0
    outcomes: list[MemoryOutcome] = []
    while processed_count < limit:
        evidence = await _load_oldest_evidence(
            session_factory,
            user_id=user_id,
            through_message_id=through_message_id,
            local_timezone=local_timezone,
        )
        if evidence is None:
            return MemoryProcessingResult(
                processed_count=processed_count, complete=True, outcomes=tuple(outcomes)
            )
        try:
            result = await policy.decide(evidence.policy_input)
        except MemoryPolicyInvalidOutputError:
            if (
                await _conservatively_close_invalid(
                    session_factory,
                    user_id=user_id,
                    message_id=evidence.policy_input.evidence_message_id,
                    expected_auto_memory_enabled=evidence.auto_memory_enabled,
                )
                == "closed"
            ):
                processed_count += 1
                outcomes.append(
                    MemoryOutcome(
                        user_id=user_id,
                        conversation_id=evidence.conversation_id,
                        source_message_id=evidence.policy_input.evidence_message_id,
                        kind="not_saved",
                    )
                )
            continue
        except (
            WorkflowFailure,
            MemoryPolicyUnavailableError,
            MemoryPolicyContextInfeasibleError,
        ):
            outcomes.append(
                MemoryOutcome(
                    user_id=user_id,
                    conversation_id=evidence.conversation_id,
                    source_message_id=evidence.policy_input.evidence_message_id,
                    kind="retry_pending",
                )
            )
            return MemoryProcessingResult(
                processed_count=processed_count,
                complete=False,
                outcomes=tuple(outcomes),
            )

        outcome = await _apply_policy_result(
            session_factory,
            user_id=user_id,
            evidence=evidence,
            result=result,
        )
        if outcome == "applied":
            processed_count += 1
            mutation_counts = _mutation_counts(result)
            if mutation_counts == (0, 0, 0):
                if result.user_requested_memory_action:
                    outcomes.append(
                        MemoryOutcome(
                            user_id=user_id,
                            conversation_id=evidence.conversation_id,
                            source_message_id=evidence.policy_input.evidence_message_id,
                            kind="no_change",
                            user_requested_memory_action=True,
                        )
                    )
            else:
                outcomes.append(
                    MemoryOutcome(
                        user_id=user_id,
                        conversation_id=evidence.conversation_id,
                        source_message_id=evidence.policy_input.evidence_message_id,
                        kind="updated",
                        user_requested_memory_action=result.user_requested_memory_action,
                        created_count=mutation_counts[0],
                        changed_count=mutation_counts[1],
                        forgotten_count=mutation_counts[2],
                    )
                )
        # A concurrent apply/marker/toggle made the detached snapshot stale.
        # Reload the same oldest evidence under current facts rather than closing it.

    remaining = await _has_unprocessed_evidence(
        session_factory, user_id=user_id, through_message_id=through_message_id
    )
    return MemoryProcessingResult(
        processed_count=processed_count,
        complete=not remaining,
        outcomes=tuple(outcomes),
    )


async def _load_oldest_evidence(
    session_factory: SessionFactory,
    *,
    user_id: int,
    through_message_id: int,
    local_timezone: str,
) -> _DetachedEvidence | None:
    """Build detached policy context in a short transaction; no provider runs here."""

    async with session_factory() as session, session.begin():
        source = await session.scalar(
            select(Message)
            .join(Conversation, Message.conversation_id == Conversation.id)
            .where(
                Conversation.user_id == user_id,
                Message.id <= through_message_id,
                Message.role == "USER",
                Message.regenerated_from_message_id.is_(None),
                Message.memory_processed_at.is_(None),
            )
            .order_by(Message.id.asc())
            .limit(1)
        )
        if source is None:
            return None
        user = await session.get(User, user_id)
        if user is None:
            return None
        previous = list(
            (
                await session.scalars(
                    select(Message)
                    .where(
                        Message.conversation_id == source.conversation_id,
                        Message.sequence_no < source.sequence_no,
                    )
                    .order_by(Message.sequence_no.desc())
                    .limit(4)
                )
            ).all()
        )
        previous.reverse()
        now = utc_now()
        current_memories = list(
            (
                await session.scalars(
                    select(Memory)
                    .where(
                        Memory.user_id == user_id,
                        or_(Memory.valid_until.is_(None), Memory.valid_until > now),
                    )
                    .order_by(Memory.updated_at.desc(), Memory.id.desc())
                )
            ).all()
        )
        return _DetachedEvidence(
            policy_input=MemoryPolicyInput(
                evidence_message_id=source.id,
                evidence_content=source.content,
                evidence_created_at=source.created_at,
                previous_messages=tuple(
                    MemoryPolicyConversationMessage(
                        role=_policy_role(message.role), content=message.content
                    )
                    for message in previous
                ),
                current_memories=tuple(
                    MemoryPolicyItem(
                        memory_id=memory.id,
                        content=memory.content,
                        valid_until=memory.valid_until,
                    )
                    for memory in current_memories
                ),
                auto_memory_enabled=user.auto_memory_enabled,
                local_temporal_reference=utc_naive_to_local_reference(
                    source.created_at, local_timezone
                ),
            ),
            auto_memory_enabled=user.auto_memory_enabled,
            conversation_id=source.conversation_id,
        )


async def _apply_policy_result(
    session_factory: SessionFactory,
    *,
    user_id: int,
    evidence: _DetachedEvidence,
    result: MemoryPolicyResult,
) -> str:
    """Atomically apply a still-current semantic decision and close its evidence."""

    async with session_factory() as session, session.begin():
        user = await _lock_user(session, user_id)
        if user.auto_memory_enabled != evidence.auto_memory_enabled:
            return "stale"
        source = await session.scalar(
            select(Message)
            .join(Conversation, Message.conversation_id == Conversation.id)
            .where(
                Message.id == evidence.policy_input.evidence_message_id,
                Conversation.user_id == user_id,
                Message.role == "USER",
                Message.regenerated_from_message_id.is_(None),
                Message.memory_processed_at.is_(None),
            )
            .with_for_update()
        )
        if source is None:
            return "stale"
        targets = await _lock_result_targets(
            session, user_id=user_id, result=result, now=utc_now()
        )
        if targets is None:
            return "stale"
        now = utc_now()
        for mutation in result.mutations:
            if mutation.operation == "NEW":
                session.add(
                    Memory(
                        user_id=user_id,
                        content=_required_content(mutation),
                        source_message_id=source.id,
                        valid_until=mutation.valid_until,
                        created_at=now,
                        updated_at=now,
                    )
                )
            elif mutation.operation == "CHANGE":
                target = targets[_required_target_id(mutation)]
                target.content = _required_content(mutation)
                target.valid_until = mutation.valid_until
                target.source_message_id = source.id
                target.updated_at = now
            else:
                await session.delete(targets[_required_target_id(mutation)])
        source.memory_processed_at = now
        return "applied"


async def _conservatively_close_invalid(
    session_factory: SessionFactory,
    *,
    user_id: int,
    message_id: int,
    expected_auto_memory_enabled: bool,
) -> str:
    """Close only one still-canonical evidence item after bounded invalid output."""

    async with session_factory() as session, session.begin():
        user = await _lock_user(session, user_id)
        if user.auto_memory_enabled != expected_auto_memory_enabled:
            return "stale"
        source = await session.scalar(
            select(Message)
            .join(Conversation, Message.conversation_id == Conversation.id)
            .where(
                Message.id == message_id,
                Conversation.user_id == user_id,
                Message.role == "USER",
                Message.regenerated_from_message_id.is_(None),
                Message.memory_processed_at.is_(None),
            )
            .with_for_update()
        )
        if source is None:
            return "stale"
        source.memory_processed_at = utc_now()
        return "closed"


async def _lock_result_targets(
    session: AsyncSession,
    *,
    user_id: int,
    result: MemoryPolicyResult,
    now: datetime,
) -> dict[int, Memory] | None:
    target_ids = sorted(
        mutation.target_memory_id
        for mutation in result.mutations
        if mutation.target_memory_id is not None
    )
    if not target_ids:
        return {}
    targets = list(
        (
            await session.scalars(
                select(Memory)
                .where(
                    Memory.id.in_(target_ids),
                    Memory.user_id == user_id,
                    or_(Memory.valid_until.is_(None), Memory.valid_until > now),
                )
                .order_by(Memory.id.asc())
                .with_for_update()
            )
        ).all()
    )
    if len(targets) != len(target_ids):
        return None
    return {target.id: target for target in targets}


async def _has_unprocessed_evidence(
    session_factory: SessionFactory, *, user_id: int, through_message_id: int
) -> bool:
    async with session_factory() as session:
        return bool(
            await session.scalar(
                select(Message.id)
                .join(Conversation, Message.conversation_id == Conversation.id)
                .where(
                    Conversation.user_id == user_id,
                    Message.id <= through_message_id,
                    Message.role == "USER",
                    Message.regenerated_from_message_id.is_(None),
                    Message.memory_processed_at.is_(None),
                )
                .order_by(Message.id.asc())
                .limit(1)
            )
        )


async def _capture_canonical_boundary(
    session_factory: SessionFactory, user_id: int
) -> int | None:
    async with session_factory() as session, session.begin():
        await _lock_user(session, user_id)
        return await session.scalar(
            select(func.max(Message.id))
            .join(Conversation, Message.conversation_id == Conversation.id)
            .where(
                Conversation.user_id == user_id,
                Message.role == "USER",
                Message.regenerated_from_message_id.is_(None),
            )
        )


async def _lock_user(session: AsyncSession, user_id: int) -> User:
    user = await session.scalar(
        select(User).where(User.id == user_id).with_for_update()
    )
    if user is None:
        raise KeyError("user not found")
    return user


async def _lock_current_memory(
    session: AsyncSession, user_id: int, memory_id: int, now: datetime
) -> Memory | None:
    return await session.scalar(
        select(Memory)
        .where(
            Memory.id == memory_id,
            Memory.user_id == user_id,
            or_(Memory.valid_until.is_(None), Memory.valid_until > now),
        )
        .with_for_update()
    )


def _required_content(mutation: MemoryMutationDecision) -> str:
    if mutation.content is None:
        raise RuntimeError("validated mutation content is missing")
    return mutation.content


def _required_target_id(mutation: MemoryMutationDecision) -> int:
    if mutation.target_memory_id is None:
        raise RuntimeError("validated mutation target is missing")
    return mutation.target_memory_id


def _policy_role(role: str) -> Literal["USER", "ASSISTANT"]:
    if role not in {"USER", "ASSISTANT"}:
        raise RuntimeError("persisted message role is invalid")
    return cast(Literal["USER", "ASSISTANT"], role)


def _mutation_counts(result: MemoryPolicyResult) -> tuple[int, int, int]:
    return (
        sum(mutation.operation == "NEW" for mutation in result.mutations),
        sum(mutation.operation == "CHANGE" for mutation in result.mutations),
        sum(mutation.operation == "FORGET" for mutation in result.mutations),
    )


def _validate_direct_valid_until(valid_until: datetime | None, now: datetime) -> None:
    if valid_until is not None and valid_until <= now:
        raise ValueError("valid_until must be in the future")
