"""Add the rebuildable current Conversation compact-state projection."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

revision = "0010_context_snapshot"
down_revision = "0009_grounding_policy"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "conversation_context_snapshots",
        sa.Column("conversation_id", sa.BigInteger(), nullable=False),
        sa.Column("through_message_id", sa.BigInteger(), nullable=False),
        sa.Column("structured_state", mysql.JSON(), nullable=False),
        sa.Column(
            "compactor_model",
            sa.String(length=255, collation="utf8mb4_0900_bin"),
            nullable=False,
        ),
        sa.Column(
            "prompt_version",
            sa.String(length=64, collation="utf8mb4_0900_bin"),
            nullable=False,
        ),
        sa.Column("created_at", mysql.DATETIME(fsp=6), nullable=False),
        sa.Column("updated_at", mysql.DATETIME(fsp=6), nullable=False),
        sa.CheckConstraint(
            "JSON_TYPE(structured_state) = 'OBJECT'",
            name="ck_conversation_context_snapshots_state_object",
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            name="fk_conversation_context_snapshots_conversation",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["through_message_id"],
            ["messages.id"],
            name="fk_conversation_context_snapshots_through_message",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("conversation_id"),
    )


def downgrade() -> None:
    op.drop_table("conversation_context_snapshots")
