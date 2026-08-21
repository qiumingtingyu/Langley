"""Add current authoritative KnowledgeChunk persistence."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

revision = "0005_knowledge_chunks"
down_revision = "0004_knowledge_persistence"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "knowledge_chunks",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("document_version_id", sa.BigInteger(), nullable=False),
        sa.Column("ordinal", sa.BigInteger(), nullable=False),
        sa.Column("content", mysql.LONGTEXT(), nullable=False),
        sa.Column("heading_path", mysql.JSON(), nullable=False),
        sa.Column("source_regions", mysql.JSON(), nullable=False),
        sa.Column("created_at", mysql.DATETIME(fsp=6), nullable=False),
        sa.CheckConstraint("ordinal > 0", name="ck_knowledge_chunks_ordinal_positive"),
        sa.CheckConstraint(
            "CHAR_LENGTH(content) > 0", name="ck_knowledge_chunks_content_nonempty"
        ),
        sa.CheckConstraint(
            "JSON_TYPE(heading_path) = 'ARRAY'",
            name="ck_knowledge_chunks_heading_path_array",
        ),
        sa.CheckConstraint(
            "JSON_TYPE(source_regions) = 'ARRAY' AND JSON_LENGTH(source_regions) > 0",
            name="ck_knowledge_chunks_source_regions_nonempty_array",
        ),
        sa.ForeignKeyConstraint(
            ["document_version_id"],
            ["document_versions.id"],
            name="fk_knowledge_chunks_document_version",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "document_version_id", "ordinal", name="uq_knowledge_chunks_version_ordinal"
        ),
    )


def downgrade() -> None:
    op.drop_table("knowledge_chunks")
