"""Shared HTTP projections for persisted Conversation answer facts."""

from datetime import UTC, datetime

from pydantic import BaseModel

from langley.infrastructure.models import Message, Run


class MessageResponse(BaseModel):
    """A persisted, user-visible Message projection."""

    id: int
    sequence_no: int
    role: str
    content: str
    run_id: int | None
    regenerated_from_message_id: int | None
    created_at: str


class RunResponse(BaseModel):
    """The persisted execution state needed for recovery and commands."""

    id: int
    input_message_id: int
    attempt_no: int
    status: str
    started_at: str | None
    finished_at: str | None
    error_code: str | None


def as_utc(value: datetime) -> str:
    """Serialize a naive MySQL DATETIME with explicit UTC semantics."""

    return value.replace(tzinfo=UTC).isoformat().replace("+00:00", "Z")


def as_optional_utc(value: datetime | None) -> str | None:
    """Serialize an optional MySQL DATETIME with explicit UTC semantics."""

    return as_utc(value) if value is not None else None


def message_response(message: Message) -> MessageResponse:
    """Project one persisted Message without deriving transient content."""

    return MessageResponse(
        id=message.id,
        sequence_no=message.sequence_no,
        role=message.role,
        content=message.content,
        run_id=message.run_id,
        regenerated_from_message_id=message.regenerated_from_message_id,
        created_at=as_utc(message.created_at),
    )


def run_response(run: Run) -> RunResponse:
    """Project only authoritative Run fields."""

    return RunResponse(
        id=run.id,
        input_message_id=run.input_message_id,
        attempt_no=run.attempt_no,
        status=run.status,
        started_at=as_optional_utc(run.started_at),
        finished_at=as_optional_utc(run.finished_at),
        error_code=run.error_code,
    )
