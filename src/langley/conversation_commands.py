"""Transactional admission of new Conversation answer commands."""

from dataclasses import dataclass
from enum import StrEnum

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from langley.business_time import utc_now
from langley.infrastructure.models import Conversation, Message, Run


class AdmissionDisposition(StrEnum):
    """Compatibility label for existing callers; new code uses is_replay."""

    NEWLY_ACCEPTED = "NEWLY_ACCEPTED"
    REPLAY = "REPLAY"


@dataclass(frozen=True)
class AnswerCommandResult:
    """The persisted USER message and Run for one answer command."""

    user_message: Message
    run: Run
    is_replay: bool

    @property
    def disposition(self) -> AdmissionDisposition:
        return (
            AdmissionDisposition.REPLAY
            if self.is_replay
            else AdmissionDisposition.NEWLY_ACCEPTED
        )


NewQuestionAdmission = AnswerCommandResult
RetryAdmission = AnswerCommandResult
RegenerateAdmission = AnswerCommandResult


class ConversationNotFoundError(Exception):
    """Raised when an owned, active Conversation cannot be locked."""


class ActiveRunExistsError(Exception):
    """Raised when a different active Run already belongs to the Conversation."""


class ClientRequestIdReusedError(Exception):
    """Raised when an existing request key represents another semantic command."""


class RetryNotAllowedError(Exception):
    """Raised when the latest USER is not eligible for a new Retry attempt."""


class RegenerateNotAllowedError(Exception):
    """Raised when the latest USER is not eligible for Regenerate."""


async def admit_new_question(
    session: AsyncSession,
    *,
    user_id: int,
    conversation_id: int,
    content: str,
    client_request_id: str,
) -> AnswerCommandResult:
    """Atomically persist a new USER and PENDING Run or return a same-key replay."""

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
            raise ConversationNotFoundError

        existing_run = await session.scalar(
            select(Run)
            .where(
                Run.conversation_id == conversation_id,
                Run.client_request_id == client_request_id,
            )
            .with_for_update()
        )
        if existing_run is not None:
            existing_message = await session.get(Message, existing_run.input_message_id)
            if existing_message is None:
                raise RuntimeError("persisted run input message is missing")
            if not _is_new_question_replay(existing_run, existing_message, content):
                raise ClientRequestIdReusedError
            return AnswerCommandResult(
                user_message=existing_message,
                run=existing_run,
                is_replay=True,
            )

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
            raise ActiveRunExistsError

        if not content.strip():
            raise ValueError("content must not be blank")

        last_sequence_no = await session.scalar(
            select(Message.sequence_no)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.sequence_no.desc())
            .limit(1)
            .with_for_update()
        )
        now = utc_now()
        user_message = Message(
            conversation_id=conversation_id,
            sequence_no=(last_sequence_no or 0) + 1,
            role="USER",
            content=content,
            run_id=None,
            regenerated_from_message_id=None,
            created_at=now,
        )
        session.add(user_message)
        await session.flush()

        run = Run(
            conversation_id=conversation_id,
            input_message_id=user_message.id,
            client_request_id=client_request_id,
            attempt_no=1,
            status="PENDING",
            started_at=None,
            finished_at=None,
            error_code=None,
            created_at=now,
            updated_at=now,
        )
        session.add(run)
        conversation.last_message_at = now
        conversation.updated_at = now
        await session.flush()

    return AnswerCommandResult(
        user_message=user_message,
        run=run,
        is_replay=False,
    )


def _is_new_question_replay(run: Run, message: Message, content: str) -> bool:
    """Determine whether existing facts represent this endpoint's exact command."""

    return (
        run.attempt_no == 1
        and message.role == "USER"
        and message.regenerated_from_message_id is None
        and message.content == content
    )


async def admit_retry(
    session: AsyncSession,
    *,
    user_id: int,
    conversation_id: int,
    client_request_id: str,
) -> AnswerCommandResult:
    """Create a Retry attempt for the latest failed or cancelled USER Message."""

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
            raise ConversationNotFoundError

        existing_run = await session.scalar(
            select(Run)
            .where(
                Run.conversation_id == conversation_id,
                Run.client_request_id == client_request_id,
            )
            .with_for_update()
        )
        if existing_run is not None:
            existing_message = await session.get(Message, existing_run.input_message_id)
            if existing_message is None:
                raise RuntimeError("persisted run input message is missing")
            if existing_run.attempt_no <= 1:
                raise ClientRequestIdReusedError
            return AnswerCommandResult(
                user_message=existing_message,
                run=existing_run,
                is_replay=True,
            )

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
            raise ActiveRunExistsError

        latest_user = await session.scalar(
            select(Message)
            .where(
                Message.conversation_id == conversation_id,
                Message.role == "USER",
            )
            .order_by(Message.sequence_no.desc())
            .limit(1)
            .with_for_update()
        )
        if latest_user is None:
            raise RetryNotAllowedError

        succeeded_run_id = await session.scalar(
            select(Run.id)
            .where(
                Run.input_message_id == latest_user.id,
                Run.status == "SUCCEEDED",
            )
            .with_for_update()
            .limit(1)
        )
        prior_failed_or_cancelled_run_id = await session.scalar(
            select(Run.id)
            .where(
                Run.input_message_id == latest_user.id,
                Run.status.in_(("FAILED", "CANCELLED")),
            )
            .with_for_update()
            .limit(1)
        )
        if succeeded_run_id is not None or prior_failed_or_cancelled_run_id is None:
            raise RetryNotAllowedError

        last_attempt_no = await session.scalar(
            select(Run.attempt_no)
            .where(Run.input_message_id == latest_user.id)
            .order_by(Run.attempt_no.desc())
            .limit(1)
            .with_for_update()
        )
        if last_attempt_no is None:
            raise RuntimeError("latest user has no persisted attempts")

        now = utc_now()
        run = Run(
            conversation_id=conversation_id,
            input_message_id=latest_user.id,
            client_request_id=client_request_id,
            attempt_no=last_attempt_no + 1,
            status="PENDING",
            started_at=None,
            finished_at=None,
            error_code=None,
            created_at=now,
            updated_at=now,
        )
        session.add(run)
        await session.flush()

    return AnswerCommandResult(
        user_message=latest_user,
        run=run,
        is_replay=False,
    )


async def admit_regenerate(
    session: AsyncSession,
    *,
    user_id: int,
    conversation_id: int,
    client_request_id: str,
) -> AnswerCommandResult:
    """Append a copied latest successful USER and a new PENDING Run atomically."""

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
            raise ConversationNotFoundError

        existing_run = await session.scalar(
            select(Run)
            .where(
                Run.conversation_id == conversation_id,
                Run.client_request_id == client_request_id,
            )
            .with_for_update()
        )
        if existing_run is not None:
            existing_message = await session.get(Message, existing_run.input_message_id)
            if existing_message is None:
                raise RuntimeError("persisted run input message is missing")
            if (
                existing_run.attempt_no != 1
                or existing_message.regenerated_from_message_id is None
            ):
                raise ClientRequestIdReusedError
            return AnswerCommandResult(
                user_message=existing_message,
                run=existing_run,
                is_replay=True,
            )

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
            raise ActiveRunExistsError

        latest_user = await session.scalar(
            select(Message)
            .where(
                Message.conversation_id == conversation_id,
                Message.role == "USER",
            )
            .order_by(Message.sequence_no.desc())
            .limit(1)
            .with_for_update()
        )
        if latest_user is None:
            raise RegenerateNotAllowedError

        succeeded_run_id = await session.scalar(
            select(Run.id)
            .where(
                Run.input_message_id == latest_user.id,
                Run.status == "SUCCEEDED",
            )
            .with_for_update()
            .limit(1)
        )
        if succeeded_run_id is None:
            raise RegenerateNotAllowedError

        original_user_id = (
            latest_user.regenerated_from_message_id
            if latest_user.regenerated_from_message_id is not None
            else latest_user.id
        )
        last_sequence_no = await session.scalar(
            select(Message.sequence_no)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.sequence_no.desc())
            .limit(1)
            .with_for_update()
        )
        now = utc_now()
        copied_user = Message(
            conversation_id=conversation_id,
            sequence_no=(last_sequence_no or 0) + 1,
            role="USER",
            content=latest_user.content,
            run_id=None,
            regenerated_from_message_id=original_user_id,
            created_at=now,
        )
        session.add(copied_user)
        await session.flush()

        run = Run(
            conversation_id=conversation_id,
            input_message_id=copied_user.id,
            client_request_id=client_request_id,
            attempt_no=1,
            status="PENDING",
            started_at=None,
            finished_at=None,
            error_code=None,
            created_at=now,
            updated_at=now,
        )
        session.add(run)
        conversation.last_message_at = now
        conversation.updated_at = now
        await session.flush()

    return AnswerCommandResult(
        user_message=copied_user,
        run=run,
        is_replay=False,
    )
