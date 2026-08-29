"""Add durable immutable-document processing attempts."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

revision = "0011_doc_processing"
down_revision = "0010_context_snapshot"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "document_processing_jobs",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("document_version_id", sa.BigInteger(), nullable=False),
        sa.Column("attempt_no", sa.BigInteger(), nullable=False),
        sa.Column(
            "status",
            sa.String(length=16, collation="utf8mb4_0900_bin"),
            nullable=False,
        ),
        sa.Column(
            "stage",
            sa.String(length=32, collation="utf8mb4_0900_bin"),
            nullable=True,
        ),
        sa.Column(
            "recipe_id",
            sa.String(length=64, collation="utf8mb4_0900_bin"),
            nullable=False,
        ),
        sa.Column(
            "error_code",
            sa.String(length=64, collation="utf8mb4_0900_bin"),
            nullable=True,
        ),
        sa.Column("error_message", sa.String(length=255), nullable=True),
        sa.Column("created_at", mysql.DATETIME(fsp=6), nullable=False),
        sa.Column("started_at", mysql.DATETIME(fsp=6), nullable=True),
        sa.Column("finished_at", mysql.DATETIME(fsp=6), nullable=True),
        sa.CheckConstraint("attempt_no > 0", name="ck_doc_processing_attempt_positive"),
        sa.CheckConstraint(
            "status IN ('PENDING', 'RUNNING', 'SUCCEEDED', 'FAILED', 'INTERRUPTED')",
            name="ck_doc_processing_status_valid",
        ),
        sa.CheckConstraint(
            "stage IS NULL OR stage IN ('VERIFYING_SOURCE', 'PARSING', 'CHUNKING', "
            "'VALIDATING', 'PUBLISHING')",
            name="ck_doc_processing_stage_valid",
        ),
        sa.CheckConstraint(
            "CHAR_LENGTH(TRIM(recipe_id)) > 0",
            name="ck_doc_processing_recipe_nonblank",
        ),
        sa.CheckConstraint(
            "error_code IS NULL OR error_code IN ("
            "'SOURCE_MISSING', 'SOURCE_INTEGRITY_MISMATCH', 'PDF_PROCESS_TIMEOUT', "
            "'PDF_PROCESS_RESOURCE_LIMIT', 'PDF_PARSE_FAILED', "
            "'PDF_CHUNKING_FAILED', 'PDF_OUTPUT_INVALID', "
            "'SOURCE_CHANGED_DURING_PROCESSING', 'PUBLICATION_FAILED', "
            "'PROCESS_INTERRUPTED')",
            name="ck_doc_processing_error_valid",
        ),
        sa.CheckConstraint(
            "(status = 'PENDING' AND stage IS NULL AND started_at IS NULL "
            "AND finished_at IS NULL AND error_code IS NULL "
            "AND error_message IS NULL) OR "
            "(status = 'RUNNING' AND stage IS NOT NULL AND started_at IS NOT NULL "
            "AND finished_at IS NULL AND error_code IS NULL "
            "AND error_message IS NULL) OR "
            "(status = 'SUCCEEDED' AND stage = 'PUBLISHING' "
            "AND started_at IS NOT NULL AND finished_at IS NOT NULL "
            "AND error_code IS NULL AND error_message IS NULL) OR "
            "(status = 'FAILED' AND stage IS NOT NULL AND started_at IS NOT NULL "
            "AND finished_at IS NOT NULL AND error_code IS NOT NULL "
            "AND error_code <> 'PROCESS_INTERRUPTED') OR "
            "(status = 'INTERRUPTED' AND finished_at IS NOT NULL "
            "AND error_code = 'PROCESS_INTERRUPTED' "
            "AND ((stage IS NULL AND started_at IS NULL) "
            "OR (stage IS NOT NULL AND started_at IS NOT NULL)))",
            name="ck_doc_processing_lifecycle_valid",
        ),
        sa.CheckConstraint(
            "(started_at IS NULL OR started_at >= created_at) AND "
            "(finished_at IS NULL OR finished_at >= created_at) AND "
            "(started_at IS NULL OR finished_at IS NULL "
            "OR finished_at >= started_at)",
            name="ck_doc_processing_timestamp_order",
        ),
        sa.ForeignKeyConstraint(
            ["document_version_id"],
            ["document_versions.id"],
            name="fk_doc_processing_document_version",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "document_version_id",
            "attempt_no",
            name="uq_doc_processing_version_attempt",
        ),
    )
    op.create_index(
        "ix_doc_processing_status_id",
        "document_processing_jobs",
        ["status", "id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_doc_processing_status_id", table_name="document_processing_jobs")
    op.drop_table("document_processing_jobs")
