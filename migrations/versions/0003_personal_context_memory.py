"""Add the Slice 5 personal context memory persistence foundation."""

from datetime import UTC, datetime
from typing import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

revision: str = "0003_personal_context_memory"
down_revision: str | None = "0002_conversation_answer_loop"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    """Create the minimal current-state Personal Memory schema."""
    op.add_column(
        "users",
        sa.Column(
            "auto_memory_enabled",
            sa.Boolean(),
            nullable=False,
            server_default=sa.true(),
        ),
    )
    op.add_column(
        "messages",
        sa.Column("memory_processed_at", mysql.DATETIME(fsp=6), nullable=True),
    )
    op.create_check_constraint(
        "ck_messages_memory_processed_canonical",
        "messages",
        "memory_processed_at IS NULL OR "
        "(role = 'USER' AND regenerated_from_message_id IS NULL)",
    )

    cutover_at = datetime.now(UTC).replace(tzinfo=None)
    op.get_bind().execute(
        sa.text(
            "UPDATE messages "
            "SET memory_processed_at = :cutover_at "
            "WHERE role = 'USER' "
            "AND regenerated_from_message_id IS NULL"
        ),
        {"cutover_at": cutover_at},
    )

    op.create_table(
        "memories",
        sa.Column("id", mysql.BIGINT(), autoincrement=True, nullable=False),
        sa.Column("user_id", mysql.BIGINT(), nullable=False),
        sa.Column("content", mysql.LONGTEXT(), nullable=False),
        sa.Column("source_message_id", mysql.BIGINT(), nullable=True),
        sa.Column("valid_until", mysql.DATETIME(fsp=6), nullable=True),
        sa.Column("created_at", mysql.DATETIME(fsp=6), nullable=False),
        sa.Column("updated_at", mysql.DATETIME(fsp=6), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name="fk_memories_user"),
        sa.ForeignKeyConstraint(
            ["source_message_id"],
            ["messages.id"],
            name="fk_memories_source_message",
        ),
        sa.PrimaryKeyConstraint("id"),
        mysql_engine="InnoDB",
        mysql_charset="utf8mb4",
    )
    op.create_index("ix_memories_user", "memories", ["user_id"])


def downgrade() -> None:
    """Remove the Slice 5 persistence foundation in reverse dependency order."""
    op.drop_table("memories")
    op.drop_constraint(
        "ck_messages_memory_processed_canonical",
        "messages",
        type_="check",
    )
    op.drop_column("messages", "memory_processed_at")
    op.drop_column("users", "auto_memory_enabled")
