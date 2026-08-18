"""SQLAlchemy mappings for Langley's authoritative business facts."""

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    ForeignKey,
    Index,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.mysql import DATETIME, LONGTEXT
from sqlalchemy.orm import Mapped, mapped_column

from langley.infrastructure.database import Base


class User(Base):
    """Minimal ownership anchor for Langley business resources."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DATETIME(fsp=6), nullable=False)


class Conversation(Base):
    """A user-owned, linearly ordered conversation."""

    __tablename__ = "conversations"
    __table_args__ = (Index("ix_conversations_user", "user_id"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("users.id", name="fk_conversations_user"),
        nullable=False,
    )
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DATETIME(fsp=6), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DATETIME(fsp=6), nullable=False)
    last_message_at: Mapped[datetime | None] = mapped_column(
        DATETIME(fsp=6), nullable=True
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DATETIME(fsp=6), nullable=True)


class Message(Base):
    """An immutable user-visible message in a conversation."""

    __tablename__ = "messages"
    __table_args__ = (
        UniqueConstraint(
            "conversation_id", "sequence_no", name="uq_messages_conversation_sequence"
        ),
        UniqueConstraint("run_id", name="uq_messages_run"),
        CheckConstraint("sequence_no > 0", name="ck_messages_sequence_positive"),
        CheckConstraint("role IN ('USER', 'ASSISTANT')", name="ck_messages_role_valid"),
        CheckConstraint(
            "(role = 'USER' AND run_id IS NULL) OR "
            "(role = 'ASSISTANT' AND run_id IS NOT NULL)",
            name="ck_messages_role_run",
        ),
        CheckConstraint(
            "regenerated_from_message_id IS NULL OR role = 'USER'",
            name="ck_messages_regenerated_from_role",
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("conversations.id", name="fk_messages_conversation"),
        nullable=False,
    )
    sequence_no: Mapped[int] = mapped_column(BigInteger, nullable=False)
    role: Mapped[str] = mapped_column(
        String(16, collation="utf8mb4_0900_bin"), nullable=False
    )
    content: Mapped[str] = mapped_column(LONGTEXT, nullable=False)
    run_id: Mapped[int | None] = mapped_column(
        BigInteger,
        ForeignKey("runs.id", name="fk_messages_run", use_alter=True),
        nullable=True,
    )
    regenerated_from_message_id: Mapped[int | None] = mapped_column(
        BigInteger,
        ForeignKey("messages.id", name="fk_messages_regenerated_from"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(DATETIME(fsp=6), nullable=False)


class Run(Base):
    """A persisted answer execution attempt for one user message."""

    __tablename__ = "runs"
    __table_args__ = (
        UniqueConstraint(
            "conversation_id",
            "client_request_id",
            name="uq_runs_conversation_client_request",
        ),
        UniqueConstraint(
            "input_message_id", "attempt_no", name="uq_runs_input_attempt"
        ),
        Index("ix_runs_conversation_status", "conversation_id", "status"),
        CheckConstraint("attempt_no > 0", name="ck_runs_attempt_positive"),
        CheckConstraint(
            "status IN ('PENDING', 'RUNNING', 'SUCCEEDED', 'FAILED', 'CANCELLED')",
            name="ck_runs_status_valid",
        ),
        CheckConstraint(
            "(status = 'PENDING' AND started_at IS NULL AND finished_at IS NULL "
            "AND error_code IS NULL) OR "
            "(status = 'RUNNING' AND started_at IS NOT NULL AND finished_at IS NULL "
            "AND error_code IS NULL) OR "
            "(status = 'SUCCEEDED' AND started_at IS NOT NULL "
            "AND finished_at IS NOT NULL AND error_code IS NULL) OR "
            "(status = 'FAILED' AND finished_at IS NOT NULL "
            "AND error_code IS NOT NULL) OR "
            "(status = 'CANCELLED' AND finished_at IS NOT NULL AND error_code IS NULL)",
            name="ck_runs_status_timestamps_error",
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("conversations.id", name="fk_runs_conversation"),
        nullable=False,
    )
    input_message_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("messages.id", name="fk_runs_input_message"),
        nullable=False,
    )
    client_request_id: Mapped[str] = mapped_column(
        String(64, collation="utf8mb4_0900_bin"), nullable=False
    )
    attempt_no: Mapped[int] = mapped_column(BigInteger, nullable=False)
    status: Mapped[str] = mapped_column(
        String(16, collation="utf8mb4_0900_bin"), nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(DATETIME(fsp=6), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DATETIME(fsp=6), nullable=True)
    error_code: Mapped[str | None] = mapped_column(
        String(64, collation="utf8mb4_0900_bin"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DATETIME(fsp=6), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DATETIME(fsp=6), nullable=False)
