"""Retire generation state and invalidate pre-cutover document projections."""

import sqlalchemy as sa
from alembic import op

revision = "0014_retrieval_cutover"
down_revision = "0013_document_index_contract"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("UPDATE document_versions SET indexed_chunk_revision = NULL")
    op.execute(
        "UPDATE knowledge_index_jobs SET status = 'INTERRUPTED', stage = NULL, "
        "error_code = 'INDEX_BUILD_INTERRUPTED', "
        "error_message = 'Index build interrupted by retrieval cutover.', "
        "finished_at = CURRENT_TIMESTAMP(6) "
        "WHERE status IN ('PENDING', 'RUNNING')"
    )
    op.execute(
        "UPDATE knowledge_bases SET index_status = 'CHUNKED', "
        "active_embedding_model = NULL, active_embedding_revision = NULL, "
        "active_embedding_dimension = NULL, "
        "active_embedding_representation = NULL"
    )
    op.drop_constraint(
        "uq_knowledge_index_jobs_generation",
        "knowledge_index_jobs",
        type_="unique",
    )
    op.drop_column("knowledge_index_jobs", "generation_id")
    op.drop_column("knowledge_bases", "active_chunk_snapshot_sha256")
    op.drop_column("knowledge_bases", "building_generation_id")
    op.drop_column("knowledge_bases", "active_generation_id")


def downgrade() -> None:
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
            "active_chunk_snapshot_sha256",
            sa.String(length=64, collation="utf8mb4_0900_bin"),
            nullable=True,
        ),
    )
    op.add_column(
        "knowledge_index_jobs",
        sa.Column(
            "generation_id",
            sa.String(length=36, collation="utf8mb4_0900_bin"),
            nullable=True,
        ),
    )
    op.execute("UPDATE knowledge_index_jobs SET generation_id = UUID()")
    op.alter_column(
        "knowledge_index_jobs",
        "generation_id",
        existing_type=sa.String(length=36, collation="utf8mb4_0900_bin"),
        nullable=False,
    )
    op.create_unique_constraint(
        "uq_knowledge_index_jobs_generation",
        "knowledge_index_jobs",
        ["generation_id"],
    )
