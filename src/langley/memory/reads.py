"""Authoritative read helpers for current Personal Context Memory facts."""

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from langley.infrastructure.models import Conversation, Memory, Message


@dataclass(frozen=True)
class MemorySourceContext:
    """The owned Conversation facts surrounding one canonical USER source."""

    conversation: Conversation
    source_message: Message
    context_messages: tuple[Message, ...]


def _is_current_memory(user_id: int, now: datetime):
    return (
        Memory.user_id == user_id,
        or_(Memory.valid_until.is_(None), Memory.valid_until > now),
    )


async def list_current_memories(
    session: AsyncSession, *, user_id: int, now: datetime
) -> list[Memory]:
    """List one user's current/effective Memory in stable recent-first order."""
    statement = (
        select(Memory)
        .where(*_is_current_memory(user_id, now))
        .order_by(Memory.updated_at.desc(), Memory.id.desc())
    )
    return list((await session.scalars(statement)).all())


async def get_owned_current_memory(
    session: AsyncSession, *, user_id: int, memory_id: int, now: datetime
) -> Memory | None:
    """Return one owned current/effective Memory, if it exists."""
    statement = select(Memory).where(
        Memory.id == memory_id,
        *_is_current_memory(user_id, now),
    )
    return await session.scalar(statement)


async def get_memory_source_context(
    session: AsyncSession, *, user_id: int, memory: Memory
) -> MemorySourceContext | None:
    """Read an owned canonical USER provenance window, if the Memory has one."""
    if memory.user_id != user_id or memory.source_message_id is None:
        return None

    source_message = await session.scalar(
        select(Message)
        .join(Conversation, Message.conversation_id == Conversation.id)
        .where(
            Message.id == memory.source_message_id,
            Message.role == "USER",
            Message.regenerated_from_message_id.is_(None),
            Conversation.user_id == user_id,
        )
    )
    if source_message is None:
        return None

    conversation = await session.get(Conversation, source_message.conversation_id)
    if conversation is None:
        return None

    previous_message = await session.scalar(
        select(Message)
        .where(
            Message.conversation_id == source_message.conversation_id,
            Message.sequence_no < source_message.sequence_no,
        )
        .order_by(Message.sequence_no.desc())
        .limit(1)
    )
    next_message = await session.scalar(
        select(Message)
        .where(
            Message.conversation_id == source_message.conversation_id,
            Message.sequence_no > source_message.sequence_no,
        )
        .order_by(Message.sequence_no.asc())
        .limit(1)
    )
    context_messages = tuple(
        message
        for message in (previous_message, source_message, next_message)
        if message is not None
    )
    return MemorySourceContext(
        conversation=conversation,
        source_message=source_message,
        context_messages=context_messages,
    )
