"""Add the explicit Run-level GroundingPolicy contract."""

import sqlalchemy as sa
from alembic import op

revision = "0009_grounding_policy"
down_revision = "0008_message_citations"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "runs",
        sa.Column(
            "grounding_policy",
            sa.String(length=16, collation="utf8mb4_0900_bin"),
            server_default=sa.text("'AUTO'"),
            nullable=True,
        ),
    )
    op.execute(
        "UPDATE runs SET grounding_policy = 'AUTO' WHERE grounding_policy IS NULL"
    )
    op.alter_column(
        "runs",
        "grounding_policy",
        existing_type=sa.String(length=16, collation="utf8mb4_0900_bin"),
        nullable=False,
        server_default=sa.text("'AUTO'"),
    )
    op.create_check_constraint(
        "ck_runs_grounding_policy_valid",
        "runs",
        "grounding_policy IN ('AUTO', 'REQUIRED')",
    )
    op.create_check_constraint(
        "ck_runs_required_grounding_has_knowledge_base",
        "runs",
        "grounding_policy != 'REQUIRED' OR knowledge_base_id IS NOT NULL",
    )


def downgrade() -> None:
    connection = op.get_bind()
    required_count = connection.scalar(
        sa.text("SELECT COUNT(*) FROM runs WHERE grounding_policy = 'REQUIRED'")
    )
    if required_count:
        raise RuntimeError(
            "cannot downgrade grounding policy while REQUIRED Runs exist"
        )
    op.drop_constraint(
        "ck_runs_required_grounding_has_knowledge_base", "runs", type_="check"
    )
    op.drop_constraint("ck_runs_grounding_policy_valid", "runs", type_="check")
    op.drop_column("runs", "grounding_policy")
