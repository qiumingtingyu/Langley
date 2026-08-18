"""Async MySQL infrastructure primitives."""

from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from langley.settings import Settings

TEST_DATABASE_NAME = "langley_test"


class Base(DeclarativeBase):
    """Metadata base for future ORM models."""


def create_database_engine(database_url: str) -> AsyncEngine:
    """Create an async engine without opening a database connection."""

    return create_async_engine(database_url, pool_pre_ping=True)


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Create async sessions bound to an explicit engine."""

    return async_sessionmaker(engine, expire_on_commit=False)


async def dispose_database_engine(engine: AsyncEngine) -> None:
    """Release resources held by an explicitly created engine."""

    await engine.dispose()


def require_test_database_url(settings: Settings) -> str:
    """Return the dedicated test database URL or fail before destructive work."""

    test_database_url = settings.test_database_url
    if test_database_url is None:
        raise RuntimeError(
            "LANGLEY_TEST_DATABASE_URL must be configured for integration tests"
        )

    test_url = make_url(test_database_url)
    if test_url.database != TEST_DATABASE_NAME:
        raise RuntimeError(
            "LANGLEY_TEST_DATABASE_URL must target the dedicated langley_test database"
        )

    if settings.database_url is not None:
        development_url = make_url(settings.database_url)
        if development_url.database == TEST_DATABASE_NAME:
            raise RuntimeError(
                "LANGLEY_DATABASE_URL must not target the dedicated "
                "langley_test database"
            )

    return test_database_url
