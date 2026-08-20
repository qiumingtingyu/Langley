"""Expose the shared Memory migration fixture to this pytest directory."""

import pytest

from ._support import migrated_database as _migrated_database


@pytest.fixture
def migrated_database(test_database_url: str, reset_database) -> str:
    return _migrated_database(test_database_url, reset_database)
