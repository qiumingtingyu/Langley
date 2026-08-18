"""UTC clock helpers for MySQL DATETIME business facts."""

from datetime import UTC, datetime


def utc_now() -> datetime:
    """Return the current UTC time as a naive value for MySQL DATETIME(6)."""

    return datetime.now(UTC).replace(tzinfo=None)
