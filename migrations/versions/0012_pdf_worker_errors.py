"""Distinguish PDF worker launch and unexpected-exit failures."""

from alembic import op

revision = "0012_pdf_worker_errors"
down_revision = "0011_doc_processing"
branch_labels = None
depends_on = None

_TABLE = "document_processing_jobs"
_CONSTRAINT = "ck_doc_processing_error_valid"
_R01_ERROR_CHECK = (
    "error_code IS NULL OR error_code IN ("
    "'SOURCE_MISSING', 'SOURCE_INTEGRITY_MISMATCH', 'PDF_PROCESS_TIMEOUT', "
    "'PDF_PROCESS_RESOURCE_LIMIT', 'PDF_PARSE_FAILED', "
    "'PDF_CHUNKING_FAILED', 'PDF_OUTPUT_INVALID', "
    "'SOURCE_CHANGED_DURING_PROCESSING', 'PUBLICATION_FAILED', "
    "'PROCESS_INTERRUPTED')"
)
_R02_ERROR_CHECK = (
    "error_code IS NULL OR error_code IN ("
    "'SOURCE_MISSING', 'SOURCE_INTEGRITY_MISMATCH', 'PDF_PROCESS_TIMEOUT', "
    "'PDF_PROCESS_RESOURCE_LIMIT', 'PDF_PROCESS_LAUNCH_FAILED', "
    "'PDF_PROCESS_WORKER_EXITED', 'PDF_PARSE_FAILED', "
    "'PDF_CHUNKING_FAILED', 'PDF_OUTPUT_INVALID', "
    "'SOURCE_CHANGED_DURING_PROCESSING', 'PUBLICATION_FAILED', "
    "'PROCESS_INTERRUPTED')"
)


def upgrade() -> None:
    op.drop_constraint(_CONSTRAINT, _TABLE, type_="check")
    op.create_check_constraint(_CONSTRAINT, _TABLE, _R02_ERROR_CHECK)


def downgrade() -> None:
    op.drop_constraint(_CONSTRAINT, _TABLE, type_="check")
    op.create_check_constraint(_CONSTRAINT, _TABLE, _R01_ERROR_CHECK)
