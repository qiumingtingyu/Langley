"""Add per-document chunk revision and durable index-attempt contracts."""

import json
from hashlib import sha256

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

revision = "0013_document_index_contract"
down_revision = "0012_pdf_worker_errors"
branch_labels = None
depends_on = None

_DOCUMENT_VERSIONS = "document_versions"
_DOCUMENT_INDEX_JOBS = "document_index_jobs"


def _json_array(value: object, field_name: str) -> list[object]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        raise RuntimeError(f"invalid {field_name} during chunk fingerprint backfill")
    return value


def _frozen_chunk_set_sha256(rows: list[dict[str, object]]) -> str:
    """Keep migration backfill independent from future application changes."""

    members: list[dict[str, object]] = []
    ordinals: set[int] = set()
    for row in rows:
        ordinal = row["ordinal"]
        content = row["content"]
        heading_path = _json_array(row["heading_path"], "heading_path")
        source_regions = _json_array(row["source_regions"], "source_regions")
        if (
            type(ordinal) is not int
            or ordinal <= 0
            or ordinal in ordinals
            or not isinstance(content, str)
            or not content
            or any(not isinstance(value, str) or not value for value in heading_path)
            or not source_regions
        ):
            raise RuntimeError("invalid chunk facts during fingerprint backfill")
        ordinals.add(ordinal)
        members.append(
            {
                "ordinal": ordinal,
                "content": content,
                "heading_path": heading_path,
                "source_regions": source_regions,
            }
        )
    serialized = json.dumps(
        sorted(members, key=lambda member: int(member["ordinal"])),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(serialized).hexdigest()


def upgrade() -> None:
    op.add_column(
        _DOCUMENT_VERSIONS,
        sa.Column("chunk_revision", sa.BigInteger(), nullable=True),
    )
    op.add_column(
        _DOCUMENT_VERSIONS,
        sa.Column(
            "chunk_set_sha256",
            sa.String(length=64, collation="utf8mb4_0900_bin"),
            nullable=True,
        ),
    )
    op.add_column(
        _DOCUMENT_VERSIONS,
        sa.Column("indexed_chunk_revision", sa.BigInteger(), nullable=True),
    )

    connection = op.get_bind()
    chunk_rows = connection.execute(
        sa.text(
            "SELECT document_version_id, ordinal, content, heading_path, "
            "source_regions FROM knowledge_chunks "
            "ORDER BY document_version_id, ordinal"
        )
    ).mappings()
    rows_by_version: dict[int, list[dict[str, object]]] = {}
    for row in chunk_rows:
        version_id = int(row["document_version_id"])
        rows_by_version.setdefault(version_id, []).append(dict(row))
    for version_id, rows in rows_by_version.items():
        connection.execute(
            sa.text(
                "UPDATE document_versions SET chunk_revision = 1, "
                "chunk_set_sha256 = :fingerprint, indexed_chunk_revision = NULL "
                "WHERE id = :version_id"
            ),
            {
                "fingerprint": _frozen_chunk_set_sha256(rows),
                "version_id": version_id,
            },
        )
    connection.execute(
        sa.text(
            "UPDATE document_versions SET chunk_revision = 0, "
            "chunk_set_sha256 = NULL, indexed_chunk_revision = NULL "
            "WHERE chunk_revision IS NULL"
        )
    )
    op.alter_column(
        _DOCUMENT_VERSIONS,
        "chunk_revision",
        existing_type=sa.BigInteger(),
        nullable=False,
        server_default=sa.text("0"),
    )
    op.create_check_constraint(
        "ck_document_versions_chunk_revision_nonnegative",
        _DOCUMENT_VERSIONS,
        "chunk_revision >= 0",
    )
    op.create_check_constraint(
        "ck_document_versions_chunk_set_state",
        _DOCUMENT_VERSIONS,
        "(chunk_revision = 0 AND chunk_set_sha256 IS NULL) OR "
        "(chunk_revision > 0 AND "
        "chunk_set_sha256 IS NOT NULL AND "
        "chunk_set_sha256 REGEXP '^[0-9a-f]{64}$')",
    )
    op.create_check_constraint(
        "ck_document_versions_indexed_revision_valid",
        _DOCUMENT_VERSIONS,
        "indexed_chunk_revision IS NULL OR "
        "(indexed_chunk_revision >= 1 "
        "AND indexed_chunk_revision <= chunk_revision)",
    )

    op.create_table(
        _DOCUMENT_INDEX_JOBS,
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("document_version_id", sa.BigInteger(), nullable=False),
        sa.Column("attempt_no", sa.BigInteger(), nullable=False),
        sa.Column("target_chunk_revision", sa.BigInteger(), nullable=False),
        sa.Column(
            "status",
            sa.String(length=16, collation="utf8mb4_0900_bin"),
            nullable=False,
        ),
        sa.Column(
            "stage",
            sa.String(length=16, collation="utf8mb4_0900_bin"),
            nullable=True,
        ),
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
            "error_code",
            sa.String(length=64, collation="utf8mb4_0900_bin"),
            nullable=True,
        ),
        sa.Column("error_message", sa.String(length=255), nullable=True),
        sa.Column("created_at", mysql.DATETIME(fsp=6), nullable=False),
        sa.Column("started_at", mysql.DATETIME(fsp=6), nullable=True),
        sa.Column("finished_at", mysql.DATETIME(fsp=6), nullable=True),
        sa.CheckConstraint("attempt_no > 0", name="ck_doc_index_attempt_positive"),
        sa.CheckConstraint(
            "target_chunk_revision > 0",
            name="ck_doc_index_target_revision_positive",
        ),
        sa.CheckConstraint(
            "status IN ('PENDING', 'RUNNING', 'SUCCEEDED', 'FAILED', 'INTERRUPTED')",
            name="ck_doc_index_status_valid",
        ),
        sa.CheckConstraint(
            "stage IS NULL OR stage IN ('EMBEDDING', 'PUBLISHING', 'VERIFYING')",
            name="ck_doc_index_stage_valid",
        ),
        sa.CheckConstraint(
            "CHAR_LENGTH(TRIM(embedding_model)) > 0 AND "
            "CHAR_LENGTH(TRIM(embedding_revision)) > 0 AND "
            "embedding_dimension > 0 AND "
            "CHAR_LENGTH(TRIM(embedding_representation)) > 0",
            name="ck_doc_index_embedding_config_valid",
        ),
        sa.CheckConstraint(
            "error_code IS NULL OR error_code IN ("
            "'SOURCE_CHUNKS_CHANGED', 'INDEX_CONFIGURATION_CHANGED', "
            "'EMBEDDING_FAILED', 'INVALID_EMBEDDING', "
            "'INDEX_PUBLICATION_FAILED', 'INDEX_VERIFICATION_FAILED', "
            "'INDEX_INTERRUPTED')",
            name="ck_doc_index_error_valid",
        ),
        sa.CheckConstraint(
            "(status = 'PENDING' AND stage IS NULL AND started_at IS NULL "
            "AND finished_at IS NULL AND error_code IS NULL "
            "AND error_message IS NULL) OR "
            "(status = 'RUNNING' AND stage IS NOT NULL AND started_at IS NOT NULL "
            "AND finished_at IS NULL AND error_code IS NULL "
            "AND error_message IS NULL) OR "
            "(status = 'SUCCEEDED' AND stage = 'VERIFYING' "
            "AND started_at IS NOT NULL AND finished_at IS NOT NULL "
            "AND error_code IS NULL AND error_message IS NULL) OR "
            "(status = 'FAILED' AND stage IS NOT NULL AND started_at IS NOT NULL "
            "AND finished_at IS NOT NULL AND error_code IS NOT NULL "
            "AND error_code <> 'INDEX_INTERRUPTED') OR "
            "(status = 'INTERRUPTED' AND finished_at IS NOT NULL "
            "AND error_code = 'INDEX_INTERRUPTED' "
            "AND ((stage IS NULL AND started_at IS NULL) "
            "OR (stage IS NOT NULL AND started_at IS NOT NULL)))",
            name="ck_doc_index_lifecycle_valid",
        ),
        sa.CheckConstraint(
            "(started_at IS NULL OR started_at >= created_at) AND "
            "(finished_at IS NULL OR finished_at >= created_at) AND "
            "(started_at IS NULL OR finished_at IS NULL "
            "OR finished_at >= started_at)",
            name="ck_doc_index_timestamp_order",
        ),
        sa.ForeignKeyConstraint(
            ["document_version_id"],
            ["document_versions.id"],
            name="fk_doc_index_document_version",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "document_version_id",
            "attempt_no",
            name="uq_doc_index_version_attempt",
        ),
    )
    op.create_index(
        "ix_doc_index_status_id",
        _DOCUMENT_INDEX_JOBS,
        ["status", "id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_doc_index_status_id", table_name=_DOCUMENT_INDEX_JOBS)
    op.drop_table(_DOCUMENT_INDEX_JOBS)
    op.drop_constraint(
        "ck_document_versions_indexed_revision_valid",
        _DOCUMENT_VERSIONS,
        type_="check",
    )
    op.drop_constraint(
        "ck_document_versions_chunk_set_state",
        _DOCUMENT_VERSIONS,
        type_="check",
    )
    op.drop_constraint(
        "ck_document_versions_chunk_revision_nonnegative",
        _DOCUMENT_VERSIONS,
        type_="check",
    )
    op.drop_column(_DOCUMENT_VERSIONS, "indexed_chunk_revision")
    op.drop_column(_DOCUMENT_VERSIONS, "chunk_set_sha256")
    op.drop_column(_DOCUMENT_VERSIONS, "chunk_revision")
