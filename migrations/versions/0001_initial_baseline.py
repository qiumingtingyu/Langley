"""Initial empty baseline revision."""

from typing import Sequence

revision: str = "0001_initial_baseline"
down_revision: str | None = None
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    """Establish the migration revision chain without creating business tables."""


def downgrade() -> None:
    """Remove no schema objects because this baseline creates none."""
