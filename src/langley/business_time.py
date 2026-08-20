"""UTC clock helpers for MySQL DATETIME business facts."""

from datetime import UTC, datetime
from zoneinfo import ZoneInfo


def utc_now() -> datetime:
    """Return the current UTC time as a naive value for MySQL DATETIME(6)."""

    return datetime.now(UTC).replace(tzinfo=None)


def normalize_aware_datetime_to_utc_naive(value: datetime) -> datetime:
    """Normalize an offset-aware instant for UTC-naive MySQL storage."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime must be offset-aware")

    return value.astimezone(UTC).replace(tzinfo=None)


def utc_naive_to_local_reference(value: datetime, timezone_name: str) -> datetime:
    """Convert a UTC-naive business value to an aware configured local reference."""
    if value.tzinfo is not None:
        raise ValueError("datetime must be UTC-naive")

    return value.replace(tzinfo=UTC).astimezone(ZoneInfo(timezone_name))
