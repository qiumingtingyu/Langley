"""Conversation persistence operations for the Slice 2 read and create APIs."""

from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from langley.business_time import utc_now
from langley.infrastructure.models import (
    Conversation,
    Message,
    MessageCitation,
    Run,
)


class ConversationHasActiveRunError(Exception):
    """Raised when an active Conversation cannot be logically deleted."""


@dataclass(frozen=True)
class MessageCitationRead:
    evidence_handle: int
    document_version_id: int
    evidence_text: str
    source_display_name: str
    heading_path: list[object]
    source_regions: list[object]


async def create_conversation(
    session: AsyncSession, user_id: int, title: str | None
) -> Conversation:
    """Create a Conversation owned by the resolved current user."""

    now = utc_now()
    conversation = Conversation(
        user_id=user_id,
        title=title,
        created_at=now,
        updated_at=now,
        last_message_at=None,
        deleted_at=None,
    )
    async with session.begin():
        session.add(conversation)
        await session.flush()
    return conversation


async def list_conversations(session: AsyncSession, user_id: int) -> list[Conversation]:
    """Return active Conversations for one owner in deterministic recent-first order."""

    statement = (
        select(Conversation)
        .where(
            Conversation.user_id == user_id,
            Conversation.deleted_at.is_(None),
        )
        .order_by(
            func.coalesce(Conversation.last_message_at, Conversation.created_at).desc(),
            Conversation.id.desc(),
        )
    )
    return list((await session.scalars(statement)).all())


async def rename_conversation(
    session: AsyncSession, *, user_id: int, conversation_id: int, title: str
) -> Conversation | None:
    """Rename one owned active Conversation without affecting answer facts."""

    normalized_title = title.strip()
    if not normalized_title:
        raise ValueError("title must not be blank")

    async with session.begin():
        conversation = await session.scalar(
            select(Conversation)
            .where(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id,
                Conversation.deleted_at.is_(None),
            )
            .with_for_update()
        )
        if conversation is None:
            return None
        conversation.title = normalized_title
        conversation.updated_at = utc_now()
        await session.flush()
    return conversation


async def delete_conversation(
    session: AsyncSession, *, user_id: int, conversation_id: int
) -> bool:
    """Logically delete an owned inactive Conversation in admission lock order."""

    async with session.begin():
        conversation = await session.scalar(
            select(Conversation)
            .where(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id,
                Conversation.deleted_at.is_(None),
            )
            .with_for_update()
        )
        if conversation is None:
            return False

        active_run_id = await session.scalar(
            select(Run.id)
            .where(
                Run.conversation_id == conversation_id,
                Run.status.in_(("PENDING", "RUNNING")),
            )
            .with_for_update()
            .limit(1)
        )
        if active_run_id is not None:
            raise ConversationHasActiveRunError

        now = utc_now()
        conversation.deleted_at = now
        conversation.updated_at = now
        await session.flush()
    return True


async def get_conversation_messages(
    session: AsyncSession, user_id: int, conversation_id: int
) -> (
    tuple[
        Conversation,
        list[Message],
        Run | None,
        dict[int, list[MessageCitationRead]],
    ]
    | None
):
    """Load owned message history and the latest Run for its latest USER Message."""

    conversation = await session.scalar(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id,
            Conversation.deleted_at.is_(None),
        )
    )
    if conversation is None:
        return None

    messages = list(
        (
            await session.scalars(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.sequence_no.asc())
            )
        ).all()
    )
    latest_user = next(
        (message for message in reversed(messages) if message.role == "USER"), None
    )
    if latest_user is None:
        return conversation, messages, None, await _message_citations(session, messages)

    latest_run = await session.scalar(
        select(Run)
        .where(Run.input_message_id == latest_user.id)
        .order_by(Run.attempt_no.desc())
        .limit(1)
    )
    citations = await _message_citations(session, messages)
    return conversation, messages, latest_run, citations


async def _message_citations(
    session: AsyncSession, messages: list[Message]
) -> dict[int, list[MessageCitationRead]]:
    """Bulk-load authoritative citation display facts for persisted messages."""

    message_ids = [message.id for message in messages if message.role == "ASSISTANT"]
    if not message_ids:
        return {}
    rows = (
        await session.execute(
            select(
                MessageCitation.message_id,
                MessageCitation.evidence_handle,
                MessageCitation.document_version_id,
                MessageCitation.evidence_text,
                MessageCitation.source_display_name_snapshot,
                MessageCitation.heading_path_snapshot,
                MessageCitation.source_regions_snapshot,
            )
            .where(MessageCitation.message_id.in_(message_ids))
            .order_by(
                MessageCitation.message_id.asc(), MessageCitation.evidence_handle.asc()
            )
        )
    ).all()
    citations: dict[int, list[MessageCitationRead]] = {}
    for row in rows:
        citations.setdefault(row[0], []).append(
            MessageCitationRead(
                evidence_handle=row[1],
                document_version_id=row[2],
                evidence_text=row[3],
                source_display_name=row[4],
                heading_path=list(row[5]),
                source_regions=list(row[6]),
            )
        )
    return citations
