"""Persist the successful current chunk configuration for each document Version."""

import sqlalchemy as sa
from alembic import op

revision = "0007_chunk_config"
down_revision = "0006_knowledge_index_builds"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "document_versions",
        sa.Column("chunk_max_chars", sa.BigInteger(), nullable=True),
    )
    op.create_check_constraint(
        "ck_document_versions_chunk_max_chars_positive",
        "document_versions",
        "chunk_max_chars IS NULL OR chunk_max_chars > 0",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_document_versions_chunk_max_chars_positive",
        "document_versions",
        type_="check",
    )
    op.drop_column("document_versions", "chunk_max_chars")
