"""Triggered structured conversation compaction plus lossless recent raw turns."""

import asyncio
from dataclasses import dataclass
from time import perf_counter

import structlog
from pydantic import ValidationError
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from langley.answering.context_builder import (
    AnswerContext,
    CompletedTurn,
    PersonalContextItem,
)
from langley.answering.conversation_context import (
    CONVERSATION_COMPACTOR_PROMPT_VERSION,
    ConversationCompactionInvalidOutputError,
    ConversationCompactionResult,
    ConversationCompactState,
    ConversationContextCompactor,
    ConversationContextMessage,
    estimate_message_tokens,
    render_conversation_compact_context,
)
from langley.answering.errors import WorkflowFailure
from langley.answering.tracing import current_context_compaction_trace_parent
from langley.business_time import utc_now
from langley.infrastructure.models import (
    Conversation,
    ConversationContextSnapshot,
    Memory,
    Message,
    Run,
)

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class _AuthoritativeContextFacts:
    messages: tuple[Message, ...]
    runs: tuple[Run, ...]
    memories: tuple[Memory, ...]
    snapshot: ConversationContextSnapshot | None


@dataclass(frozen=True)
class _ValidSnapshot:
    state: ConversationCompactState
    boundary_turn_index: int


class _NonBeneficialCompactionError(RuntimeError):
    """The candidate would not reduce visible older-history pressure."""


class ConversationContextBuilder:
    """Maintain a rebuildable older-history projection outside DB resources."""

    def __init__(
        self,
        *,
        working_context_budget_estimate: int,
        conversation_compaction_trigger_estimate: int = 12_000,
        recent_raw_target_estimate: int = 6_000,
        compact_state_target_estimate: int = 2_000,
        memory_estimated_token_budget: int = 8_192,
        compactor: ConversationContextCompactor | None = None,
    ) -> None:
        policy_values = {
            "working_context_budget_estimate": working_context_budget_estimate,
            "conversation_compaction_trigger_estimate": (
                conversation_compaction_trigger_estimate
            ),
            "recent_raw_target_estimate": recent_raw_target_estimate,
            "compact_state_target_estimate": compact_state_target_estimate,
            "memory_estimated_token_budget": memory_estimated_token_budget,
        }
        for name, value in policy_values.items():
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if conversation_compaction_trigger_estimate > working_context_budget_estimate:
            raise ValueError(
                "conversation_compaction_trigger_estimate must not exceed "
                "working_context_budget_estimate"
            )
        if recent_raw_target_estimate >= conversation_compaction_trigger_estimate:
            raise ValueError(
                "recent_raw_target_estimate must be below the compaction trigger"
            )
        if (
            recent_raw_target_estimate + compact_state_target_estimate
            > working_context_budget_estimate
        ):
            raise ValueError(
                "compact and recent targets must fit the working context budget"
            )
        self._working_context_budget_estimate = working_context_budget_estimate
        self._conversation_compaction_trigger_estimate = (
            conversation_compaction_trigger_estimate
        )
        self._recent_raw_target_estimate = recent_raw_target_estimate
        self._compact_state_target_estimate = compact_state_target_estimate
        self._memory_estimated_token_budget = memory_estimated_token_budget
        self._compactor = compactor

    async def build(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        conversation_id: int,
        current_user_message_id: int,
    ) -> AnswerContext:
        """Use short DB scopes around an optional detached compactor invocation."""

        facts = await self._read_facts(session_factory, conversation_id)
        messages_by_id = {message.id: message for message in facts.messages}
        current_user = messages_by_id.get(current_user_message_id)
        if current_user is None:
            raise ValueError("current user message is missing")
        completed_turns = self._completed_turns(
            messages=facts.messages,
            runs=facts.runs,
            current_user=current_user,
        )
        valid_snapshot = self._validated_snapshot(
            facts.snapshot,
            completed_turns=completed_turns,
            messages_by_id=messages_by_id,
        )
        previous_state = None if valid_snapshot is None else valid_snapshot.state
        unincorporated_turns = (
            completed_turns
            if valid_snapshot is None
            else completed_turns[valid_snapshot.boundary_turn_index + 1 :]
        )
        estimated_before = self._estimate_visible_conversation(
            previous_state, unincorporated_turns
        )
        snapshot_present = facts.snapshot is not None

        if estimated_before < self._conversation_compaction_trigger_estimate:
            self._log_compaction(
                triggered=False,
                previous_snapshot_present=snapshot_present,
                previous_snapshot_valid=valid_snapshot is not None,
                newly_compacted_turn_count=0,
                recent_raw_turn_count=len(unincorporated_turns),
                estimated_before=estimated_before,
                estimated_after=estimated_before,
                success=True,
                outcome="REUSED" if valid_snapshot is not None else "SHORT_HISTORY",
            )
            return self._answer_context(
                raw_turns=unincorporated_turns,
                compact_state=previous_state,
                current_user=current_user,
                memories=facts.memories,
                messages_by_id=messages_by_id,
            )

        aged_turns, recent_turns = self._partition_for_compaction(unincorporated_turns)
        if not aged_turns or self._compactor is None:
            fallback_turns = self._fallback_raw_turns(unincorporated_turns)
            self._log_compaction(
                triggered=False,
                previous_snapshot_present=snapshot_present,
                previous_snapshot_valid=valid_snapshot is not None,
                newly_compacted_turn_count=0,
                recent_raw_turn_count=len(fallback_turns),
                estimated_before=estimated_before,
                estimated_after=self._estimate_visible_conversation(
                    previous_state, fallback_turns
                ),
                success=self._compactor is not None,
                outcome=(
                    "NO_COMPLETE_TURN_TO_AGE"
                    if self._compactor is not None
                    else "COMPACTOR_UNAVAILABLE"
                ),
            )
            return self._answer_context(
                raw_turns=fallback_turns,
                compact_state=previous_state,
                current_user=current_user,
                memories=facts.memories,
                messages_by_id=messages_by_id,
            )

        started_at = perf_counter()
        completed_compaction: ConversationCompactionResult | None = None
        try:
            compaction = await self._compactor.compact(
                previous_state=previous_state,
                newly_aged_out_messages=self._messages_for_compactor(aged_turns),
            )
            completed_compaction = compaction
            estimated_after = self._estimate_visible_conversation(
                compaction.state, recent_turns
            )
            if (
                estimated_after >= estimated_before
                or estimated_after > self._working_context_budget_estimate
            ):
                raise _NonBeneficialCompactionError
            through_message_id = aged_turns[-1].assistant_message_id
            if through_message_id is None:
                raise RuntimeError("authoritative completed turn has no Assistant ID")
            await self._persist_snapshot(
                session_factory,
                conversation_id=conversation_id,
                through_message_id=through_message_id,
                compaction=compaction,
                observed_snapshot_through_message_id=(
                    None
                    if facts.snapshot is None
                    else facts.snapshot.through_message_id
                ),
                observed_snapshot_valid=valid_snapshot is not None,
            )
        except asyncio.CancelledError:
            usage = None if completed_compaction is None else completed_compaction.usage
            self._log_compaction(
                triggered=True,
                previous_snapshot_present=snapshot_present,
                previous_snapshot_valid=valid_snapshot is not None,
                newly_compacted_turn_count=len(aged_turns),
                recent_raw_turn_count=len(recent_turns),
                estimated_before=estimated_before,
                estimated_after=None,
                success=False,
                outcome="CANCELLED",
                duration_ms=self._duration_ms(started_at),
                compactor_model=self._compactor.model,
                provider_model=(
                    None
                    if completed_compaction is None
                    else completed_compaction.provider_model
                ),
                provider_input_tokens=None if usage is None else usage.input_tokens,
                provider_output_tokens=(None if usage is None else usage.output_tokens),
            )
            raise
        except Exception as error:
            fallback_turns = self._fallback_raw_turns(unincorporated_turns)
            usage = None if completed_compaction is None else completed_compaction.usage
            self._log_compaction(
                triggered=True,
                previous_snapshot_present=snapshot_present,
                previous_snapshot_valid=valid_snapshot is not None,
                newly_compacted_turn_count=len(aged_turns),
                recent_raw_turn_count=len(fallback_turns),
                estimated_before=estimated_before,
                estimated_after=self._estimate_visible_conversation(
                    previous_state, fallback_turns
                ),
                success=False,
                outcome=self._failure_outcome(error),
                duration_ms=self._duration_ms(started_at),
                compactor_model=self._compactor.model,
                provider_model=(
                    None
                    if completed_compaction is None
                    else completed_compaction.provider_model
                ),
                provider_input_tokens=None if usage is None else usage.input_tokens,
                provider_output_tokens=(None if usage is None else usage.output_tokens),
            )
            return self._answer_context(
                raw_turns=fallback_turns,
                compact_state=previous_state,
                current_user=current_user,
                memories=facts.memories,
                messages_by_id=messages_by_id,
            )

        usage = compaction.usage
        self._log_compaction(
            triggered=True,
            previous_snapshot_present=snapshot_present,
            previous_snapshot_valid=valid_snapshot is not None,
            newly_compacted_turn_count=len(aged_turns),
            recent_raw_turn_count=len(recent_turns),
            estimated_before=estimated_before,
            estimated_after=estimated_after,
            success=True,
            outcome="COMPACTED",
            duration_ms=self._duration_ms(started_at),
            compactor_model=self._compactor.model,
            provider_model=compaction.provider_model,
            provider_input_tokens=None if usage is None else usage.input_tokens,
            provider_output_tokens=None if usage is None else usage.output_tokens,
        )
        return self._answer_context(
            raw_turns=recent_turns,
            compact_state=compaction.state,
            current_user=current_user,
            memories=facts.memories,
            messages_by_id=messages_by_id,
        )

    async def _read_facts(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        conversation_id: int,
    ) -> _AuthoritativeContextFacts:
        async with session_factory() as session:
            async with session.begin():
                messages = tuple(
                    (
                        await session.scalars(
                            select(Message)
                            .where(Message.conversation_id == conversation_id)
                            .order_by(Message.sequence_no.asc())
                        )
                    ).all()
                )
                runs = tuple(
                    (
                        await session.scalars(
                            select(Run).where(Run.conversation_id == conversation_id)
                        )
                    ).all()
                )
                memories = tuple(
                    (
                        await session.scalars(
                            select(Memory)
                            .join(Conversation, Memory.user_id == Conversation.user_id)
                            .where(
                                Conversation.id == conversation_id,
                                or_(
                                    Memory.valid_until.is_(None),
                                    Memory.valid_until > utc_now(),
                                ),
                            )
                            .order_by(Memory.updated_at.desc(), Memory.id.desc())
                        )
                    ).all()
                )
                snapshot = await session.get(
                    ConversationContextSnapshot, conversation_id
                )
                return _AuthoritativeContextFacts(
                    messages=messages,
                    runs=runs,
                    memories=memories,
                    snapshot=snapshot,
                )

    async def _persist_snapshot(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        conversation_id: int,
        through_message_id: int,
        compaction: ConversationCompactionResult,
        observed_snapshot_through_message_id: int | None,
        observed_snapshot_valid: bool,
    ) -> None:
        now = utc_now()
        compactor = self._compactor
        if compactor is None:
            raise RuntimeError("snapshot persistence requires a configured compactor")
        async with session_factory() as session:
            async with session.begin():
                snapshot = await session.get(
                    ConversationContextSnapshot,
                    conversation_id,
                    with_for_update=True,
                )
                if (
                    snapshot is None
                    and observed_snapshot_through_message_id is not None
                ):
                    return
                if snapshot is not None and (
                    observed_snapshot_through_message_id is None
                    or snapshot.through_message_id
                    != observed_snapshot_through_message_id
                ):
                    return
                if (
                    snapshot is not None
                    and observed_snapshot_valid
                    and snapshot.through_message_id >= through_message_id
                ):
                    return
                if snapshot is None:
                    session.add(
                        ConversationContextSnapshot(
                            conversation_id=conversation_id,
                            through_message_id=through_message_id,
                            structured_state=compaction.state.model_dump(mode="json"),
                            compactor_model=compactor.model,
                            prompt_version=CONVERSATION_COMPACTOR_PROMPT_VERSION,
                            created_at=now,
                            updated_at=now,
                        )
                    )
                else:
                    snapshot.through_message_id = through_message_id
                    snapshot.structured_state = compaction.state.model_dump(mode="json")
                    snapshot.compactor_model = compactor.model
                    snapshot.prompt_version = CONVERSATION_COMPACTOR_PROMPT_VERSION
                    snapshot.updated_at = now

    @staticmethod
    def _completed_turns(
        *,
        messages: tuple[Message, ...],
        runs: tuple[Run, ...],
        current_user: Message,
    ) -> tuple[CompletedTurn, ...]:
        messages_by_id = {message.id: message for message in messages}
        successful_runs_by_input = {
            run.input_message_id: run for run in runs if run.status == "SUCCEEDED"
        }
        assistants_by_run = {
            message.run_id: message
            for message in messages
            if message.role == "ASSISTANT" and message.run_id is not None
        }
        complete_turns: list[tuple[int, CompletedTurn]] = []
        for input_message_id, run in successful_runs_by_input.items():
            user_message = messages_by_id.get(input_message_id)
            assistant_message = assistants_by_run.get(run.id)
            if (
                user_message is None
                or assistant_message is None
                or user_message.sequence_no >= current_user.sequence_no
            ):
                continue
            complete_turns.append(
                (
                    user_message.sequence_no,
                    CompletedTurn(
                        user_content=user_message.content,
                        assistant_content=assistant_message.content,
                        estimated_tokens=(
                            estimate_message_tokens(user_message.content)
                            + estimate_message_tokens(assistant_message.content)
                        ),
                        user_message_id=user_message.id,
                        assistant_message_id=assistant_message.id,
                    ),
                )
            )
        return tuple(
            turn for _, turn in sorted(complete_turns, key=lambda item: item[0])
        )

    def _validated_snapshot(
        self,
        snapshot: ConversationContextSnapshot | None,
        *,
        completed_turns: tuple[CompletedTurn, ...],
        messages_by_id: dict[int, Message],
    ) -> _ValidSnapshot | None:
        if snapshot is None:
            return None
        boundary_message = messages_by_id.get(snapshot.through_message_id)
        if boundary_message is None or boundary_message.role != "ASSISTANT":
            return None
        boundary_turn_index = next(
            (
                index
                for index, turn in enumerate(completed_turns)
                if turn.assistant_message_id == snapshot.through_message_id
            ),
            None,
        )
        if boundary_turn_index is None:
            return None
        try:
            state = ConversationCompactState.model_validate(snapshot.structured_state)
        except ValidationError:
            return None
        for source_message_id in state.source_message_ids:
            source = messages_by_id.get(source_message_id)
            if source is None or source.sequence_no > boundary_message.sequence_no:
                return None
        return _ValidSnapshot(state=state, boundary_turn_index=boundary_turn_index)

    def _partition_for_compaction(
        self, turns: tuple[CompletedTurn, ...]
    ) -> tuple[tuple[CompletedTurn, ...], tuple[CompletedTurn, ...]]:
        recent_newest_first: list[CompletedTurn] = []
        remaining = self._recent_raw_target_estimate
        for turn in reversed(turns):
            if recent_newest_first and turn.estimated_tokens > remaining:
                break
            recent_newest_first.append(turn)
            remaining -= turn.estimated_tokens
        recent_turns = tuple(reversed(recent_newest_first))
        return turns[: len(turns) - len(recent_turns)], recent_turns

    def _fallback_raw_turns(
        self, turns: tuple[CompletedTurn, ...]
    ) -> tuple[CompletedTurn, ...]:
        selected_newest_first: list[CompletedTurn] = []
        remaining = self._working_context_budget_estimate
        for turn in reversed(turns):
            if selected_newest_first and turn.estimated_tokens > remaining:
                break
            selected_newest_first.append(turn)
            remaining -= turn.estimated_tokens
        return tuple(reversed(selected_newest_first))

    def _answer_context(
        self,
        *,
        raw_turns: tuple[CompletedTurn, ...],
        compact_state: ConversationCompactState | None,
        current_user: Message,
        memories: tuple[Memory, ...],
        messages_by_id: dict[int, Message],
    ) -> AnswerContext:
        exposed_canonical_user_ids = {self._canonical_user_message_id(current_user)}
        exposed_canonical_user_ids.update(
            self._canonical_user_message_id(messages_by_id[turn.user_message_id])
            for turn in raw_turns
            if turn.user_message_id is not None
        )
        if compact_state is not None:
            exposed_canonical_user_ids.update(
                self._canonical_user_message_id(message)
                for source_id in compact_state.source_message_ids
                if (message := messages_by_id.get(source_id)) is not None
                and message.role == "USER"
            )
        personal_context_items = tuple(
            PersonalContextItem(memory_id=memory.id, content=memory.content)
            for memory in memories
            if memory.source_message_id not in exposed_canonical_user_ids
        )
        personal_context: tuple[PersonalContextItem, ...] | None = (
            personal_context_items
            if self._estimate_personal_context(personal_context_items)
            <= self._memory_estimated_token_budget
            else None
        )
        return AnswerContext(
            completed_turns=raw_turns,
            current_user_content=current_user.content,
            personal_context=personal_context,
            conversation_compact_context=(
                None
                if compact_state is None
                else render_conversation_compact_context(compact_state)
            ),
        )

    @staticmethod
    def _messages_for_compactor(
        turns: tuple[CompletedTurn, ...],
    ) -> tuple[ConversationContextMessage, ...]:
        messages: list[ConversationContextMessage] = []
        for turn in turns:
            if turn.user_message_id is None or turn.assistant_message_id is None:
                raise RuntimeError(
                    "compaction requires stable authoritative Message IDs"
                )
            messages.extend(
                (
                    ConversationContextMessage(
                        message_id=turn.user_message_id,
                        role="USER",
                        content=turn.user_content,
                    ),
                    ConversationContextMessage(
                        message_id=turn.assistant_message_id,
                        role="ASSISTANT",
                        content=turn.assistant_content,
                    ),
                )
            )
        return tuple(messages)

    @staticmethod
    def _canonical_user_message_id(message: Message) -> int:
        return message.regenerated_from_message_id or message.id

    @staticmethod
    def _estimate_personal_context(
        personal_context: tuple[PersonalContextItem, ...],
    ) -> int:
        return sum(len(item.content) + 32 for item in personal_context)

    @staticmethod
    def _estimate_visible_conversation(
        state: ConversationCompactState | None,
        turns: tuple[CompletedTurn, ...],
    ) -> int:
        compact_estimate = (
            0
            if state is None
            else estimate_message_tokens(render_conversation_compact_context(state))
        )
        return compact_estimate + sum(turn.estimated_tokens for turn in turns)

    @staticmethod
    def _duration_ms(started_at: float) -> float:
        return round((perf_counter() - started_at) * 1000, 3)

    @staticmethod
    def _failure_outcome(error: Exception) -> str:
        if isinstance(error, _NonBeneficialCompactionError):
            return "NON_BENEFICIAL_OUTPUT"
        if isinstance(error, ConversationCompactionInvalidOutputError):
            return "INVALID_OUTPUT"
        if isinstance(error, WorkflowFailure):
            return f"PROVIDER_{error.error_code.value}"
        return "MAINTENANCE_FAILED"

    def _log_compaction(
        self,
        *,
        triggered: bool,
        previous_snapshot_present: bool,
        previous_snapshot_valid: bool,
        newly_compacted_turn_count: int,
        recent_raw_turn_count: int,
        estimated_before: int,
        estimated_after: int | None,
        success: bool,
        outcome: str,
        duration_ms: float | None = None,
        compactor_model: str | None = None,
        provider_model: str | None = None,
        provider_input_tokens: int | None = None,
        provider_output_tokens: int | None = None,
    ) -> None:
        logger.info(
            "conversation_context.compaction",
            compaction_triggered=triggered,
            previous_snapshot_present=previous_snapshot_present,
            previous_snapshot_valid=previous_snapshot_valid,
            newly_compacted_message_count=newly_compacted_turn_count * 2,
            newly_compacted_turn_count=newly_compacted_turn_count,
            recent_raw_turn_count=recent_raw_turn_count,
            estimated_before=estimated_before,
            estimated_after=estimated_after,
            working_context_budget_estimate=self._working_context_budget_estimate,
            compaction_trigger_estimate=(
                self._conversation_compaction_trigger_estimate
            ),
            recent_raw_target_estimate=self._recent_raw_target_estimate,
            compact_state_target_estimate=self._compact_state_target_estimate,
            compactor_model=compactor_model,
            provider_model=provider_model,
            provider_input_tokens=provider_input_tokens,
            provider_output_tokens=provider_output_tokens,
            duration_ms=duration_ms,
            success=success,
            outcome=outcome,
        )
        if not triggered:
            return
        trace_parent = current_context_compaction_trace_parent()
        if trace_parent is None:
            return
        try:
            trace_parent.context_compact(
                estimated_before=estimated_before,
                estimated_after=estimated_after,
                newly_compacted_turn_count=newly_compacted_turn_count,
                recent_raw_turn_count=recent_raw_turn_count,
                compactor_model=compactor_model,
                provider_model=provider_model,
                provider_input_tokens=provider_input_tokens,
                provider_output_tokens=provider_output_tokens,
                duration_ms=duration_ms,
                success=success,
                outcome=outcome,
            )
        except Exception:
            logger.warning("conversation_context.compaction_trace_failed")
