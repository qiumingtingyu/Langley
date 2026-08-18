"""Create the Slice 2 conversation and answer fact schema."""

from typing import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

revision: str = "0002_conversation_answer_loop"
down_revision: str | None = "0001_initial_baseline"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None

_BINARY_COLLATION = "utf8mb4_0900_bin"


def upgrade() -> None:
    """Create authoritative conversation, message, and run facts."""

    op.create_table(
        "users",
        sa.Column("id", mysql.BIGINT(), autoincrement=True, nullable=False),
        sa.Column("created_at", mysql.DATETIME(fsp=6), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        mysql_engine="InnoDB",
        mysql_charset="utf8mb4",
    )
    op.create_table(
        "conversations",
        sa.Column("id", mysql.BIGINT(), autoincrement=True, nullable=False),
        sa.Column("user_id", mysql.BIGINT(), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=True),
        sa.Column("created_at", mysql.DATETIME(fsp=6), nullable=False),
        sa.Column("updated_at", mysql.DATETIME(fsp=6), nullable=False),
        sa.Column("last_message_at", mysql.DATETIME(fsp=6), nullable=True),
        sa.Column("deleted_at", mysql.DATETIME(fsp=6), nullable=True),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name="fk_conversations_user"
        ),
        sa.PrimaryKeyConstraint("id"),
        mysql_engine="InnoDB",
        mysql_charset="utf8mb4",
    )
    op.create_index("ix_conversations_user", "conversations", ["user_id"])
    op.create_table(
        "messages",
        sa.Column("id", mysql.BIGINT(), autoincrement=True, nullable=False),
        sa.Column("conversation_id", mysql.BIGINT(), nullable=False),
        sa.Column("sequence_no", mysql.BIGINT(), nullable=False),
        sa.Column(
            "role", sa.String(length=16, collation=_BINARY_COLLATION), nullable=False
        ),
        sa.Column("content", mysql.LONGTEXT(), nullable=False),
        sa.Column("run_id", mysql.BIGINT(), nullable=True),
        sa.Column("regenerated_from_message_id", mysql.BIGINT(), nullable=True),
        sa.Column("created_at", mysql.DATETIME(fsp=6), nullable=False),
        sa.CheckConstraint("sequence_no > 0", name="ck_messages_sequence_positive"),
        sa.CheckConstraint(
            "role IN ('USER', 'ASSISTANT')", name="ck_messages_role_valid"
        ),
        sa.CheckConstraint(
            "(role = 'USER' AND run_id IS NULL) OR "
            "(role = 'ASSISTANT' AND run_id IS NOT NULL)",
            name="ck_messages_role_run",
        ),
        sa.CheckConstraint(
            "regenerated_from_message_id IS NULL OR role = 'USER'",
            name="ck_messages_regenerated_from_role",
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"], ["conversations.id"], name="fk_messages_conversation"
        ),
        sa.ForeignKeyConstraint(
            ["regenerated_from_message_id"],
            ["messages.id"],
            name="fk_messages_regenerated_from",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "conversation_id", "sequence_no", name="uq_messages_conversation_sequence"
        ),
        sa.UniqueConstraint("run_id", name="uq_messages_run"),
        mysql_engine="InnoDB",
        mysql_charset="utf8mb4",
    )
    op.create_table(
        "runs",
        sa.Column("id", mysql.BIGINT(), autoincrement=True, nullable=False),
        sa.Column("conversation_id", mysql.BIGINT(), nullable=False),
        sa.Column("input_message_id", mysql.BIGINT(), nullable=False),
        sa.Column(
            "client_request_id",
            sa.String(length=64, collation=_BINARY_COLLATION),
            nullable=False,
        ),
        sa.Column("attempt_no", mysql.BIGINT(), nullable=False),
        sa.Column(
            "status", sa.String(length=16, collation=_BINARY_COLLATION), nullable=False
        ),
        sa.Column("started_at", mysql.DATETIME(fsp=6), nullable=True),
        sa.Column("finished_at", mysql.DATETIME(fsp=6), nullable=True),
        sa.Column(
            "error_code",
            sa.String(length=64, collation=_BINARY_COLLATION),
            nullable=True,
        ),
        sa.Column("created_at", mysql.DATETIME(fsp=6), nullable=False),
        sa.Column("updated_at", mysql.DATETIME(fsp=6), nullable=False),
        sa.CheckConstraint("attempt_no > 0", name="ck_runs_attempt_positive"),
        sa.CheckConstraint(
            "status IN ('PENDING', 'RUNNING', 'SUCCEEDED', 'FAILED', 'CANCELLED')",
            name="ck_runs_status_valid",
        ),
        sa.CheckConstraint(
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
        sa.ForeignKeyConstraint(
            ["conversation_id"], ["conversations.id"], name="fk_runs_conversation"
        ),
        sa.ForeignKeyConstraint(
            ["input_message_id"], ["messages.id"], name="fk_runs_input_message"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "conversation_id",
            "client_request_id",
            name="uq_runs_conversation_client_request",
        ),
        sa.UniqueConstraint(
            "input_message_id", "attempt_no", name="uq_runs_input_attempt"
        ),
        mysql_engine="InnoDB",
        mysql_charset="utf8mb4",
    )
    op.create_index(
        "ix_runs_conversation_status", "runs", ["conversation_id", "status"]
    )
    op.create_foreign_key("fk_messages_run", "messages", "runs", ["run_id"], ["id"])


def downgrade() -> None:
    """Drop the Slice 2 business schema in reverse dependency order."""

    op.drop_constraint("fk_messages_run", "messages", type_="foreignkey")
    op.drop_index("ix_runs_conversation_status", table_name="runs")
    op.drop_table("runs")
    op.drop_table("messages")
    op.drop_index("ix_conversations_user", table_name="conversations")
    op.drop_table("conversations")
    op.drop_table("users")
