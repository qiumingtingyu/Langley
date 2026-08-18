"""Tests for the business timestamp source."""

from datetime import UTC, datetime

from langley.business_time import utc_now


def test_utc_now_returns_a_naive_datetime_with_utc_semantics() -> None:
    """MySQL DATETIME values are naive, but generated from the UTC clock."""

    before = datetime.now(UTC).replace(tzinfo=None)
    value = utc_now()
    after = datetime.now(UTC).replace(tzinfo=None)

    assert value.tzinfo is None
    assert before <= value <= after
