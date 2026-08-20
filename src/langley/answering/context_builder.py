"""Detached authoritative context assembly for one Learning Assistant Run."""

from dataclasses import dataclass

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from langley.business_time import utc_now
from langley.infrastructure.models import Conversation, Memory, Message, Run


@dataclass(frozen=True)
class CompletedTurn:
    """One detached successful USER and ASSISTANT pair."""

    user_content: str
    assistant_content: str
    estimated_tokens: int


@dataclass(frozen=True)
class PersonalContextItem:
    """One detached current Memory fact available to answer generation."""

    memory_id: int
    content: str


@dataclass(frozen=True)
class AnswerContext:
    """Provider- and framework-neutral context for a current USER input."""

    completed_turns: tuple[CompletedTurn, ...]
    current_user_content: str
    personal_context: tuple[PersonalContextItem, ...] | None = ()


class AnswerContextBuilder:
    """Read authoritative facts briefly, then return only detached runtime data."""

    def __init__(
        self,
        *,
        history_estimated_token_budget: int,
        memory_estimated_token_budget: int = 8_192,
    ) -> None:
        if history_estimated_token_budget < 1:
            raise ValueError("history_estimated_token_budget must be positive")
        if memory_estimated_token_budget < 1:
            raise ValueError("memory_estimated_token_budget must be positive")
        self._history_estimated_token_budget = history_estimated_token_budget
        self._memory_estimated_token_budget = memory_estimated_token_budget

    async def build(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        conversation_id: int,
        current_user_message_id: int,
    ) -> AnswerContext:
        """Load facts in a short DB scope and release it before returning context."""

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
                            .join(
                                Conversation,
                                Memory.user_id == Conversation.user_id,
                            )
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
                return self._assemble(
                    messages=messages,
                    runs=runs,
                    memories=memories,
                    conversation_id=conversation_id,
                    current_user_message_id=current_user_message_id,
                )

    def _assemble(
        self,
        *,
        messages: tuple[Message, ...],
        runs: tuple[Run, ...],
        memories: tuple[Memory, ...] = (),
        conversation_id: int,
        current_user_message_id: int,
    ) -> AnswerContext:
        """Apply whole-turn selection to authoritative facts."""

        messages_by_id = {message.id: message for message in messages}
        current_user = messages_by_id.get(current_user_message_id)
        if current_user is None:
            raise ValueError("current user message is missing")

        successful_runs_by_input: dict[int, Run] = {}
        for run in runs:
            if run.status != "SUCCEEDED":
                continue
            successful_runs_by_input[run.input_message_id] = run

        assistants_by_run: dict[int, Message] = {}
        for message in messages:
            if message.role != "ASSISTANT":
                continue
            if message.run_id is not None:
                assistants_by_run[message.run_id] = message

        complete_turns: list[tuple[int, int, CompletedTurn]] = []
        for input_message_id, run in successful_runs_by_input.items():
            user_message = messages_by_id.get(input_message_id)
            assistant_message = assistants_by_run.get(run.id)
            if (
                user_message is not None
                and assistant_message is not None
                and user_message.sequence_no < current_user.sequence_no
            ):
                complete_turns.append(
                    (
                        user_message.sequence_no,
                        input_message_id,
                        CompletedTurn(
                            user_content=user_message.content,
                            assistant_content=assistant_message.content,
                            estimated_tokens=self._estimate_turn_tokens(
                                user_message.content, assistant_message.content
                            ),
                        ),
                    )
                )

        selected_newest_first: list[tuple[int, CompletedTurn]] = []
        remaining = self._history_estimated_token_budget
        for _, input_message_id, turn in sorted(
            complete_turns, key=lambda item: item[0], reverse=True
        ):
            if turn.estimated_tokens > remaining:
                break
            selected_newest_first.append(
                (
                    input_message_id,
                    turn,
                )
            )
            remaining -= turn.estimated_tokens

        exposed_canonical_user_ids = {
            self._canonical_user_message_id(current_user),
            *(
                self._canonical_user_message_id(messages_by_id[input_message_id])
                for input_message_id, _ in selected_newest_first
            ),
        }
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
            completed_turns=tuple(turn for _, turn in reversed(selected_newest_first)),
            current_user_content=current_user.content,
            personal_context=personal_context,
        )

    @staticmethod
    def _estimate_turn_tokens(user_content: str, assistant_content: str) -> int:
        """Use a deterministic character estimate until tokenization exists."""

        return len(user_content) + len(assistant_content)

    @staticmethod
    def _canonical_user_message_id(message: Message) -> int:
        """Map regenerated evidence to its original canonical USER identity."""

        return message.regenerated_from_message_id or message.id

    @staticmethod
    def _estimate_personal_context(
        personal_context: tuple[PersonalContextItem, ...],
    ) -> int:
        return sum(len(item.content) + 32 for item in personal_context)
