"""Add durable manual Knowledge dense-index build state."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

revision = "0006_knowledge_index_builds"
down_revision = "0005_knowledge_chunks"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "knowledge_bases",
        sa.Column(
            "index_status",
            sa.String(length=16, collation="utf8mb4_0900_bin"),
            server_default="CHUNKED",
            nullable=False,
        ),
    )
    op.add_column(
        "knowledge_bases",
        sa.Column(
            "active_generation_id",
            sa.String(length=36, collation="utf8mb4_0900_bin"),
            nullable=True,
        ),
    )
    op.add_column(
        "knowledge_bases",
        sa.Column(
            "building_generation_id",
            sa.String(length=36, collation="utf8mb4_0900_bin"),
            nullable=True,
        ),
    )
    op.add_column(
        "knowledge_bases",
        sa.Column(
            "active_embedding_model",
            sa.String(length=255, collation="utf8mb4_0900_bin"),
        ),
    )
    op.add_column(
        "knowledge_bases",
        sa.Column(
            "active_embedding_revision",
            sa.String(length=64, collation="utf8mb4_0900_bin"),
        ),
    )
    op.add_column(
        "knowledge_bases", sa.Column("active_embedding_dimension", sa.BigInteger())
    )
    op.add_column(
        "knowledge_bases",
        sa.Column(
            "active_embedding_representation",
            sa.String(length=64, collation="utf8mb4_0900_bin"),
        ),
    )
    op.add_column(
        "knowledge_bases",
        sa.Column(
            "active_chunk_snapshot_sha256",
            sa.String(length=64, collation="utf8mb4_0900_bin"),
        ),
    )
    op.create_check_constraint(
        "ck_knowledge_bases_index_status_valid",
        "knowledge_bases",
        "index_status IN ('CHUNKED', 'INDEXING', 'READY', 'FAILED', 'STALE')",
    )
    op.create_table(
        "knowledge_index_jobs",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("knowledge_base_id", sa.BigInteger(), nullable=False),
        sa.Column(
            "generation_id",
            sa.String(length=36, collation="utf8mb4_0900_bin"),
            nullable=False,
        ),
        sa.Column(
            "status", sa.String(length=16, collation="utf8mb4_0900_bin"), nullable=False
        ),
        sa.Column(
            "stage", sa.String(length=32, collation="utf8mb4_0900_bin"), nullable=True
        ),
        sa.Column("processed_chunk_count", sa.BigInteger(), nullable=False),
        sa.Column("total_chunk_count", sa.BigInteger(), nullable=False),
        sa.Column(
            "embedding_model",
            sa.String(length=255, collation="utf8mb4_0900_bin"),
            nullable=False,
        ),
        sa.Column(
            "embedding_revision",
            sa.String(length=64, collation="utf8mb4_0900_bin"),
            nullable=False,
        ),
        sa.Column("embedding_dimension", sa.BigInteger(), nullable=False),
        sa.Column(
            "embedding_representation",
            sa.String(length=64, collation="utf8mb4_0900_bin"),
            nullable=False,
        ),
        sa.Column(
            "chunk_snapshot_sha256",
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
        sa.CheckConstraint(
            "status IN ('PENDING', 'RUNNING', 'SUCCEEDED', 'FAILED', 'INTERRUPTED')",
            name="ck_knowledge_index_jobs_status_valid",
        ),
        sa.CheckConstraint(
            "stage IS NULL OR stage IN ('SNAPSHOT', 'EMBEDDING', 'UPLOADING_INDEX', "
            "'VERIFYING', 'ACTIVATING')",
            name="ck_knowledge_index_jobs_stage_valid",
        ),
        sa.CheckConstraint(
            "processed_chunk_count >= 0 AND total_chunk_count >= 0 AND "
            "processed_chunk_count <= total_chunk_count",
            name="ck_knowledge_index_jobs_progress_valid",
        ),
        sa.ForeignKeyConstraint(
            ["knowledge_base_id"],
            ["knowledge_bases.id"],
            name="fk_knowledge_index_jobs_knowledge_base",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("generation_id", name="uq_knowledge_index_jobs_generation"),
    )
    op.create_index(
        "ix_knowledge_index_jobs_base_created",
        "knowledge_index_jobs",
        ["knowledge_base_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_knowledge_index_jobs_base_created", table_name="knowledge_index_jobs"
    )
    op.drop_table("knowledge_index_jobs")
    op.drop_constraint(
        "ck_knowledge_bases_index_status_valid", "knowledge_bases", type_="check"
    )
    for column in (
        "active_chunk_snapshot_sha256",
        "active_embedding_representation",
        "active_embedding_dimension",
        "active_embedding_revision",
        "active_embedding_model",
        "building_generation_id",
        "active_generation_id",
        "index_status",
    ):
        op.drop_column("knowledge_bases", column)
