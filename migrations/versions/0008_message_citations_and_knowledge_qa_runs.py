"""Add durable Knowledge QA selector and assistant citation facts."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

revision = "0008_message_citations"
down_revision = "0007_chunk_config"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "runs", sa.Column("knowledge_base_id", sa.BigInteger(), nullable=True)
    )
    op.create_foreign_key(
        "fk_runs_knowledge_base",
        "runs",
        "knowledge_bases",
        ["knowledge_base_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.create_table(
        "message_citations",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("message_id", sa.BigInteger(), nullable=False),
        sa.Column("document_version_id", sa.BigInteger(), nullable=False),
        sa.Column("evidence_handle", sa.BigInteger(), nullable=False),
        sa.Column("evidence_text", mysql.LONGTEXT(), nullable=False),
        sa.Column("source_display_name_snapshot", sa.String(255), nullable=False),
        sa.Column("heading_path_snapshot", mysql.JSON(), nullable=False),
        sa.Column("source_regions_snapshot", mysql.JSON(), nullable=False),
        sa.CheckConstraint(
            "evidence_handle > 0", name="ck_message_citations_evidence_handle_positive"
        ),
        sa.CheckConstraint(
            "CHAR_LENGTH(TRIM(evidence_text)) > 0",
            name="ck_message_citations_evidence_text_nonblank",
        ),
        sa.CheckConstraint(
            "CHAR_LENGTH(TRIM(source_display_name_snapshot)) > 0",
            name="ck_message_citations_source_display_name_nonblank",
        ),
        sa.CheckConstraint(
            "JSON_TYPE(heading_path_snapshot) = 'ARRAY'",
            name="ck_message_citations_heading_path_snapshot_array",
        ),
        sa.CheckConstraint(
            "JSON_TYPE(source_regions_snapshot) = 'ARRAY' "
            "AND JSON_LENGTH(source_regions_snapshot) > 0",
            name="ck_message_citations_source_regions_snapshot_nonempty_array",
        ),
        sa.ForeignKeyConstraint(
            ["message_id"],
            ["messages.id"],
            name="fk_message_citations_message",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["document_version_id"],
            ["document_versions.id"],
            name="fk_message_citations_document_version",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "message_id",
            "evidence_handle",
            name="uq_message_citations_message_evidence_handle",
        ),
    )


def downgrade() -> None:
    op.drop_table("message_citations")
    op.drop_constraint("fk_runs_knowledge_base", "runs", type_="foreignkey")
    op.drop_column("runs", "knowledge_base_id")
