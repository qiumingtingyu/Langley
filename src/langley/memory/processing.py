"""Ordered, transaction-safe persistence of Personal Context Memory decisions."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Literal, cast

import structlog
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement

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
    estimate_load_all_memory_contribution,
)

BACKGROUND_BATCH_LIMIT = 4
PRE_ANSWER_CATCHUP_LIMIT = 2
PRE_ANSWER_CATCHUP_TIMEOUT_SECONDS = 5
MANUAL_SYNC_LIMIT = 4
MANUAL_SYNC_TIMEOUT_SECONDS = 20

logger = structlog.get_logger(__name__)


class MemoryCapacityReachedError(RuntimeError):
    """A direct write would exceed the configured Load-All capacity."""


class MemoryCapacityUnavailableError(RuntimeError):
    """A growing direct write has no configured Load-All capacity contract."""


class MemoryProcessingStopReason(StrEnum):
    """Content-free reason one bounded processing invocation stopped."""

    COMPLETE = "COMPLETE"
    LIMIT_REACHED = "LIMIT_REACHED"
    POLICY_UNAVAILABLE = "POLICY_UNAVAILABLE"
    PROVIDER_FAILURE = "PROVIDER_FAILURE"
    CONTEXT_INFEASIBLE = "CONTEXT_INFEASIBLE"
    TIMEOUT = "TIMEOUT"


@dataclass(frozen=True)
class MemoryProcessingResult:
    """The finite result of processing a canonical USER prefix."""

    processed_count: int
    complete: bool
    stop_reason: MemoryProcessingStopReason
    outcomes: tuple[MemoryOutcome, ...] = ()


@dataclass(frozen=True)
class PendingMemoryEvidenceStatus:
    """Durable/rebuildable pending canonical USER evidence facts."""

    pending_evidence_count: int
    oldest_pending_message_id: int | None
    oldest_pending_created_at: datetime | None


@dataclass(frozen=True)
class MemorySyncResult:
    """One bounded explicit sync result plus its captured-prefix remainder."""

    processing: MemoryProcessingResult
    pending: PendingMemoryEvidenceStatus


class MemorySynchronizationError(RuntimeError):
    """A mandatory ordered barrier stopped before its captured prefix completed."""

    def __init__(self, sync: MemorySyncResult) -> None:
        super().__init__(sync.processing.stop_reason.value)
        self.sync = sync


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
        return MemoryProcessingResult(
            processed_count=0,
            complete=True,
            stop_reason=MemoryProcessingStopReason.COMPLETE,
        )
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
    return _publish_processing_outcomes(result, outcome_callback)


async def capture_memory_high_water(
    session_factory: SessionFactory, *, user_id: int
) -> int | None:
    """Capture one User-serialized finite canonical USER boundary."""

    return await _capture_canonical_boundary(session_factory, user_id)


async def get_pending_memory_evidence_status(
    session: AsyncSession,
    *,
    user_id: int,
    through_message_id: int | None = None,
) -> PendingMemoryEvidenceStatus:
    """Read count and oldest identity from the production canonical predicate."""

    conditions = list(_pending_evidence_conditions(user_id))
    if through_message_id is not None:
        conditions.append(Message.id <= through_message_id)
    row = (
        await session.execute(
            select(
                Message.id.label("oldest_pending_message_id"),
                Message.created_at.label("oldest_pending_created_at"),
                func.count(Message.id).over().label("pending_evidence_count"),
            )
            .join(Conversation, Message.conversation_id == Conversation.id)
            .where(*conditions)
            .order_by(Message.id.asc())
            .limit(1)
        )
    ).one_or_none()
    if row is None:
        return PendingMemoryEvidenceStatus(
            pending_evidence_count=0,
            oldest_pending_message_id=None,
            oldest_pending_created_at=None,
        )
    return PendingMemoryEvidenceStatus(
        pending_evidence_count=row.pending_evidence_count,
        oldest_pending_message_id=row.oldest_pending_message_id,
        oldest_pending_created_at=row.oldest_pending_created_at,
    )


async def synchronize_memory(
    session_factory: SessionFactory,
    *,
    user_id: int,
    policy: MemoryPolicy | None,
    local_timezone: str,
    lane: asyncio.Lock,
    outcome_callback: Callable[[MemoryOutcome], None] | None = None,
) -> MemorySyncResult:
    """Process one captured batch without adding durable operational state."""

    async with lane:
        sync = await _synchronize_memory_locked(
            session_factory,
            user_id=user_id,
            policy=policy,
            local_timezone=local_timezone,
        )
    return MemorySyncResult(
        processing=_publish_processing_outcomes(sync.processing, outcome_callback),
        pending=sync.pending,
    )


async def add_memory_direct(
    session_factory: SessionFactory,
    *,
    user_id: int,
    content: str,
    valid_until: datetime | None,
    policy: MemoryPolicy | None,
    estimated_token_budget: int | None,
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
            current_memories = await _lock_current_memories(session, user_id, now)
            _require_direct_post_state_capacity(
                before_contents=[memory.content for memory in current_memories],
                after_contents=[
                    *[memory.content for memory in current_memories],
                    content,
                ],
                estimated_token_budget=estimated_token_budget,
            )
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
    policy: MemoryPolicy | None,
    estimated_token_budget: int | None,
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
            current_memories = await _lock_current_memories(session, user_id, now)
            memory = next(
                (memory for memory in current_memories if memory.id == memory_id), None
            )
            if memory is None:
                raise KeyError("current memory not found")
            post_state_contents = [
                content if current.id == memory_id else current.content
                for current in current_memories
            ]
            _require_direct_post_state_capacity(
                before_contents=[memory.content for memory in current_memories],
                after_contents=post_state_contents,
                estimated_token_budget=estimated_token_budget,
            )
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
    policy: MemoryPolicy | None,
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

    async with lane:
        sync = await _synchronize_memory_locked(
            session_factory,
            user_id=user_id,
            policy=policy,
            local_timezone=local_timezone,
        )
        _require_complete_sync(sync)
        async with session_factory() as session, session.begin():
            user = await _lock_user(session, user_id)
            user.auto_memory_enabled = True


async def _require_manual_barrier(
    session_factory: SessionFactory,
    *,
    user_id: int,
    policy: MemoryPolicy | None,
    local_timezone: str,
) -> None:
    sync = await _synchronize_memory_locked(
        session_factory,
        user_id=user_id,
        policy=policy,
        local_timezone=local_timezone,
    )
    _require_complete_sync(sync)


async def _synchronize_memory_locked(
    session_factory: SessionFactory,
    *,
    user_id: int,
    policy: MemoryPolicy | None,
    local_timezone: str,
) -> MemorySyncResult:
    """Capture and process one finite prefix while the process-local lane is held."""

    boundary = await _capture_canonical_boundary(session_factory, user_id)
    before = await _pending_status_from_factory(
        session_factory,
        user_id=user_id,
        through_message_id=boundary,
    )
    if before.pending_evidence_count == 0:
        return MemorySyncResult(
            processing=MemoryProcessingResult(
                processed_count=0,
                complete=True,
                stop_reason=MemoryProcessingStopReason.COMPLETE,
            ),
            pending=before,
        )
    if policy is None:
        return MemorySyncResult(
            processing=MemoryProcessingResult(
                processed_count=0,
                complete=False,
                stop_reason=MemoryProcessingStopReason.POLICY_UNAVAILABLE,
            ),
            pending=before,
        )
    try:
        async with asyncio.timeout(MANUAL_SYNC_TIMEOUT_SECONDS):
            processing = await _process_memory_through_locked(
                session_factory,
                user_id=user_id,
                through_message_id=boundary,
                policy=policy,
                local_timezone=local_timezone,
                limit=MANUAL_SYNC_LIMIT,
            )
    except TimeoutError:
        remaining = await _pending_status_from_factory(
            session_factory,
            user_id=user_id,
            through_message_id=boundary,
        )
        processing = MemoryProcessingResult(
            processed_count=max(
                0,
                before.pending_evidence_count - remaining.pending_evidence_count,
            ),
            complete=False,
            stop_reason=MemoryProcessingStopReason.TIMEOUT,
        )
    else:
        remaining = await _pending_status_from_factory(
            session_factory,
            user_id=user_id,
            through_message_id=boundary,
        )
    return MemorySyncResult(processing=processing, pending=remaining)


def _require_complete_sync(sync: MemorySyncResult) -> None:
    if not sync.processing.complete:
        raise MemorySynchronizationError(sync)


def _require_direct_post_state_capacity(
    *,
    before_contents: list[str],
    after_contents: list[str],
    estimated_token_budget: int | None,
) -> None:
    before_estimate = estimate_load_all_memory_contribution(before_contents)
    after_estimate = estimate_load_all_memory_contribution(after_contents)
    if estimated_token_budget is None:
        if after_estimate > before_estimate:
            raise MemoryCapacityUnavailableError
        return
    if after_estimate > estimated_token_budget:
        raise MemoryCapacityReachedError


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
        return MemoryProcessingResult(
            processed_count=0,
            complete=True,
            stop_reason=MemoryProcessingStopReason.COMPLETE,
        )
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
                processed_count=processed_count,
                complete=True,
                stop_reason=MemoryProcessingStopReason.COMPLETE,
                outcomes=tuple(outcomes),
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
        except WorkflowFailure:
            return _failed_processing_result(
                user_id=user_id,
                evidence=evidence,
                processed_count=processed_count,
                outcomes=outcomes,
                stop_reason=MemoryProcessingStopReason.PROVIDER_FAILURE,
            )
        except MemoryPolicyUnavailableError:
            return _failed_processing_result(
                user_id=user_id,
                evidence=evidence,
                processed_count=processed_count,
                outcomes=outcomes,
                stop_reason=MemoryProcessingStopReason.POLICY_UNAVAILABLE,
            )
        except MemoryPolicyContextInfeasibleError:
            return _failed_processing_result(
                user_id=user_id,
                evidence=evidence,
                processed_count=processed_count,
                outcomes=outcomes,
                stop_reason=MemoryProcessingStopReason.CONTEXT_INFEASIBLE,
            )

        outcome = await _apply_policy_result(
            session_factory,
            user_id=user_id,
            evidence=evidence,
            result=result,
            estimated_token_budget=policy.estimated_token_budget,
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
        elif outcome == "capacity_rejected":
            processed_count += 1
            outcomes.append(
                MemoryOutcome(
                    user_id=user_id,
                    conversation_id=evidence.conversation_id,
                    source_message_id=evidence.policy_input.evidence_message_id,
                    kind="not_saved",
                )
            )
            logger.info(
                "memory.policy_result.capacity_rejected",
                user_id=user_id,
                conversation_id=evidence.conversation_id,
                source_message_id=evidence.policy_input.evidence_message_id,
                mutation_count=len(result.mutations),
            )
        # A concurrent apply/marker/toggle made the detached snapshot stale.
        # Reload the same oldest evidence under current facts rather than closing it.

    remaining = await _has_unprocessed_evidence(
        session_factory, user_id=user_id, through_message_id=through_message_id
    )
    return MemoryProcessingResult(
        processed_count=processed_count,
        complete=not remaining,
        stop_reason=(
            MemoryProcessingStopReason.LIMIT_REACHED
            if remaining
            else MemoryProcessingStopReason.COMPLETE
        ),
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
                *_pending_evidence_conditions(user_id),
                Message.id <= through_message_id,
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
    estimated_token_budget: int,
) -> Literal["applied", "capacity_rejected", "stale"]:
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
        now = utc_now()
        current_memories = await _lock_current_memories(session, user_id, now)
        targets = _result_targets(current_memories, result)
        if targets is None:
            return "stale"
        post_state_contents = _policy_post_state_contents(
            current_memories, result=result, now=now
        )
        if (
            estimate_load_all_memory_contribution(post_state_contents)
            > estimated_token_budget
        ):
            source.memory_processed_at = now
            return "capacity_rejected"
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


def _result_targets(
    current_memories: list[Memory], result: MemoryPolicyResult
) -> dict[int, Memory] | None:
    target_ids = {
        mutation.target_memory_id
        for mutation in result.mutations
        if mutation.target_memory_id is not None
    }
    current_by_id = {memory.id: memory for memory in current_memories}
    if not target_ids.issubset(current_by_id):
        return None
    return {target_id: current_by_id[target_id] for target_id in target_ids}


async def _lock_current_memories(
    session: AsyncSession,
    user_id: int,
    now: datetime,
) -> list[Memory]:
    return list(
        (
            await session.scalars(
                select(Memory)
                .where(
                    Memory.user_id == user_id,
                    or_(Memory.valid_until.is_(None), Memory.valid_until > now),
                )
                .order_by(Memory.id.asc())
                .with_for_update()
            )
        ).all()
    )


def _policy_post_state_contents(
    current_memories: list[Memory],
    *,
    result: MemoryPolicyResult,
    now: datetime,
) -> tuple[str, ...]:
    contents_by_id = {memory.id: memory.content for memory in current_memories}
    new_contents: list[str] = []
    for mutation in result.mutations:
        if mutation.operation == "NEW":
            if mutation.valid_until is None or mutation.valid_until > now:
                new_contents.append(_required_content(mutation))
            continue
        target_id = _required_target_id(mutation)
        if mutation.operation == "CHANGE":
            if mutation.valid_until is None or mutation.valid_until > now:
                contents_by_id[target_id] = _required_content(mutation)
            else:
                contents_by_id.pop(target_id)
        else:
            contents_by_id.pop(target_id)
    return (*contents_by_id.values(), *new_contents)


async def _has_unprocessed_evidence(
    session_factory: SessionFactory, *, user_id: int, through_message_id: int
) -> bool:
    async with session_factory() as session:
        return bool(
            await session.scalar(
                select(Message.id)
                .join(Conversation, Message.conversation_id == Conversation.id)
                .where(
                    *_pending_evidence_conditions(user_id),
                    Message.id <= through_message_id,
                )
                .order_by(Message.id.asc())
                .limit(1)
            )
        )


async def _pending_status_from_factory(
    session_factory: SessionFactory,
    *,
    user_id: int,
    through_message_id: int | None,
) -> PendingMemoryEvidenceStatus:
    async with session_factory() as session:
        return await get_pending_memory_evidence_status(
            session,
            user_id=user_id,
            through_message_id=through_message_id,
        )


def _pending_evidence_conditions(user_id: int) -> tuple[ColumnElement[bool], ...]:
    return (
        Conversation.user_id == user_id,
        Message.role == "USER",
        Message.regenerated_from_message_id.is_(None),
        Message.memory_processed_at.is_(None),
    )


def _failed_processing_result(
    *,
    user_id: int,
    evidence: _DetachedEvidence,
    processed_count: int,
    outcomes: list[MemoryOutcome],
    stop_reason: MemoryProcessingStopReason,
) -> MemoryProcessingResult:
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
        stop_reason=stop_reason,
        outcomes=tuple(outcomes),
    )


def _publish_processing_outcomes(
    result: MemoryProcessingResult,
    outcome_callback: Callable[[MemoryOutcome], None] | None,
) -> MemoryProcessingResult:
    if outcome_callback is None:
        return MemoryProcessingResult(
            processed_count=result.processed_count,
            complete=result.complete,
            stop_reason=result.stop_reason,
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
