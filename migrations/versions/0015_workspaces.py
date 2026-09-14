"""Add managed Workspace scope without changing existing Conversation binding."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.mysql import DATETIME

revision = "0015_workspaces"
down_revision = "0014_retrieval_cutover"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "workspaces",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column(
            "storage_key", sa.String(32, collation="utf8mb4_0900_bin"), nullable=False
        ),
        sa.Column("created_at", DATETIME(fsp=6), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name="fk_workspaces_user"),
        sa.UniqueConstraint("storage_key", name="uq_workspaces_storage_key"),
    )
    op.create_index("ix_workspaces_user", "workspaces", ["user_id"])
    op.add_column(
        "conversations", sa.Column("workspace_id", sa.BigInteger(), nullable=True)
    )
    op.create_foreign_key(
        "fk_conversations_workspace",
        "conversations",
        "workspaces",
        ["workspace_id"],
        ["id"],
    )
    op.create_index("ix_conversations_workspace", "conversations", ["workspace_id"])


def downgrade() -> None:
    if op.get_bind().scalar(sa.text("SELECT COUNT(*) FROM workspaces")):
        raise RuntimeError("Cannot discard existing Workspace binding")
    op.drop_constraint(
        "fk_conversations_workspace", "conversations", type_="foreignkey"
    )
    op.drop_index("ix_conversations_workspace", table_name="conversations")
    op.drop_column("conversations", "workspace_id")
    op.drop_table("workspaces")
