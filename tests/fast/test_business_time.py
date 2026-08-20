"""Tests for the business timestamp source."""

from datetime import UTC, datetime, timedelta, timezone

import pytest

from langley.business_time import (
    normalize_aware_datetime_to_utc_naive,
    utc_naive_to_local_reference,
    utc_now,
)


def test_utc_now_returns_a_naive_datetime_with_utc_semantics() -> None:
    """MySQL DATETIME values are naive, but generated from the UTC clock."""

    before = datetime.now(UTC).replace(tzinfo=None)
    value = utc_now()
    after = datetime.now(UTC).replace(tzinfo=None)

    assert value.tzinfo is None
    assert before <= value <= after


def test_normalize_aware_datetime_to_utc_naive_preserves_the_instant() -> None:
    aware_value = datetime(2026, 8, 20, 10, 0, tzinfo=timezone(timedelta(hours=8)))

    assert normalize_aware_datetime_to_utc_naive(aware_value) == datetime(
        2026, 8, 20, 2, 0
    )


def test_normalize_aware_datetime_to_utc_naive_rejects_naive_values() -> None:
    with pytest.raises(ValueError, match="offset-aware"):
        normalize_aware_datetime_to_utc_naive(datetime(2026, 8, 20, 2, 0))


def test_utc_naive_to_local_reference_uses_the_explicit_timezone() -> None:
    local_reference = utc_naive_to_local_reference(
        datetime(2026, 8, 20, 2, 0), "Asia/Shanghai"
    )

    assert local_reference == datetime(
        2026, 8, 20, 10, 0, tzinfo=timezone(timedelta(hours=8))
    )


def test_utc_naive_to_local_reference_rejects_aware_storage_values() -> None:
    with pytest.raises(ValueError, match="UTC-naive"):
        utc_naive_to_local_reference(datetime(2026, 8, 20, 2, 0, tzinfo=UTC), "UTC")
