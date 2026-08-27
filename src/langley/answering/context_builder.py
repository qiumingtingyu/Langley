"""Detached answer-context DTOs and the context-builder seam."""

from dataclasses import dataclass
from typing import Protocol

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


@dataclass(frozen=True)
class CompletedTurn:
    """One detached successful USER and ASSISTANT pair."""

    user_content: str
    assistant_content: str
    estimated_tokens: int
    user_message_id: int | None = None
    assistant_message_id: int | None = None


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
    conversation_compact_context: str | None = None


class AnswerContextBuilder(Protocol):
    """Build detached answer context from authoritative conversation facts."""

    async def build(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        conversation_id: int,
        current_user_message_id: int,
    ) -> AnswerContext: ...
