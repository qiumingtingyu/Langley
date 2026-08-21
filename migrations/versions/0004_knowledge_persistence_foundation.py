"""Add the Slice 6 Knowledge persistence foundation."""

from typing import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

revision: str = "0004_knowledge_persistence"
down_revision: str | None = "0003_personal_context_memory"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    """Create authoritative KnowledgeBase, Document, and DocumentVersion facts."""
    op.create_table(
        "knowledge_bases",
        sa.Column("id", mysql.BIGINT(), autoincrement=True, nullable=False),
        sa.Column("user_id", mysql.BIGINT(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("created_at", mysql.DATETIME(fsp=6), nullable=False),
        sa.CheckConstraint(
            "CHAR_LENGTH(TRIM(name)) > 0", name="ck_knowledge_bases_name_nonblank"
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name="fk_knowledge_bases_user"
        ),
        sa.PrimaryKeyConstraint("id"),
        mysql_engine="InnoDB",
        mysql_charset="utf8mb4",
    )
    op.create_index("ix_knowledge_bases_user", "knowledge_bases", ["user_id"])

    op.create_table(
        "documents",
        sa.Column("id", mysql.BIGINT(), autoincrement=True, nullable=False),
        sa.Column("knowledge_base_id", mysql.BIGINT(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("created_at", mysql.DATETIME(fsp=6), nullable=False),
        sa.CheckConstraint(
            "CHAR_LENGTH(TRIM(name)) > 0", name="ck_documents_name_nonblank"
        ),
        sa.ForeignKeyConstraint(
            ["knowledge_base_id"],
            ["knowledge_bases.id"],
            name="fk_documents_knowledge_base",
        ),
        sa.PrimaryKeyConstraint("id"),
        mysql_engine="InnoDB",
        mysql_charset="utf8mb4",
    )
    op.create_index("ix_documents_knowledge_base", "documents", ["knowledge_base_id"])

    op.create_table(
        "document_versions",
        sa.Column("id", mysql.BIGINT(), autoincrement=True, nullable=False),
        sa.Column("document_id", mysql.BIGINT(), nullable=False),
        sa.Column("source_filename", sa.String(length=255), nullable=False),
        sa.Column(
            "source_media_type",
            sa.String(length=64, collation="utf8mb4_0900_bin"),
            nullable=False,
        ),
        sa.Column(
            "source_sha256",
            sa.String(length=64, collation="utf8mb4_0900_bin"),
            nullable=False,
        ),
        sa.Column("source_size_bytes", mysql.BIGINT(), nullable=False),
        sa.Column(
            "storage_key",
            sa.String(length=512, collation="utf8mb4_0900_bin"),
            nullable=False,
        ),
        sa.Column("created_at", mysql.DATETIME(fsp=6), nullable=False),
        sa.CheckConstraint(
            "CHAR_LENGTH(TRIM(source_filename)) > 0",
            name="ck_document_versions_source_filename_nonblank",
        ),
        sa.CheckConstraint(
            "CHAR_LENGTH(TRIM(source_media_type)) > 0",
            name="ck_document_versions_source_media_type_nonblank",
        ),
        sa.CheckConstraint(
            "CHAR_LENGTH(source_sha256) = 64",
            name="ck_document_versions_source_sha256_length",
        ),
        sa.CheckConstraint(
            "source_size_bytes > 0",
            name="ck_document_versions_source_size_bytes_positive",
        ),
        sa.CheckConstraint(
            "CHAR_LENGTH(TRIM(storage_key)) > 0",
            name="ck_document_versions_storage_key_nonblank",
        ),
        sa.ForeignKeyConstraint(
            ["document_id"], ["documents.id"], name="fk_document_versions_document"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("storage_key", name="uq_document_versions_storage_key"),
        mysql_engine="InnoDB",
        mysql_charset="utf8mb4",
    )
    op.create_index(
        "ix_document_versions_document", "document_versions", ["document_id"]
    )


def downgrade() -> None:
    """Remove the Knowledge persistence foundation in reverse dependency order."""
    op.drop_table("document_versions")
    op.drop_table("documents")
    op.drop_table("knowledge_bases")
