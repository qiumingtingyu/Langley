"""Fixtures for real MySQL integration tests."""

import asyncio

import pytest
from sqlalchemy import text
from sqlalchemy.engine import make_url

from langley.infrastructure.database import (
    TEST_DATABASE_NAME,
    create_database_engine,
    dispose_database_engine,
    require_test_database_url,
)
from langley.settings import Settings


@pytest.fixture(scope="session")
def test_database_url() -> str:
    """Require an explicitly configured dedicated integration database."""

    return require_test_database_url(Settings())


def reset_test_database(test_database_url: str) -> None:
    """Reset only the explicitly approved langley_test database."""

    test_url = make_url(test_database_url)
    if test_url.database != TEST_DATABASE_NAME:
        raise RuntimeError("Refusing to reset a database other than langley_test")

    async def reset() -> None:
        admin_url = test_url.set(database="")
        admin_engine = create_database_engine(
            admin_url.render_as_string(hide_password=False)
        )
        try:
            async with admin_engine.begin() as connection:
                await connection.execute(
                    text(f"DROP DATABASE IF EXISTS `{TEST_DATABASE_NAME}`")
                )
                await connection.execute(
                    text(f"CREATE DATABASE `{TEST_DATABASE_NAME}`")
                )
        finally:
            await dispose_database_engine(admin_engine)

    asyncio.run(reset())


@pytest.fixture
def reset_database(test_database_url: str):
    """Provide an explicit reset action for an approved test database."""

    def reset() -> None:
        reset_test_database(test_database_url)

    return reset
