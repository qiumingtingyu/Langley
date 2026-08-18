"""Fast tests for database infrastructure configuration."""

import asyncio

import pytest

from langley.infrastructure.database import (
    create_database_engine,
    create_session_factory,
    dispose_database_engine,
    require_test_database_url,
)
from langley.main import create_app
from langley.settings import Settings


def test_database_engine_and_session_factory_are_constructed_without_connecting() -> (
    None
):
    engine = create_database_engine(
        "mysql+asyncmy://root:password@127.0.0.1:3306/langley"
    )
    session_factory = create_session_factory(engine)

    assert engine.url.drivername == "mysql+asyncmy"
    assert session_factory.kw["bind"] is engine

    asyncio.run(dispose_database_engine(engine))


def test_test_database_url_never_falls_back_to_development_database() -> None:
    with pytest.raises(RuntimeError, match="LANGLEY_TEST_DATABASE_URL"):
        require_test_database_url(
            Settings(
                database_url="mysql+asyncmy://root:password@127.0.0.1:3306/langley",
                test_database_url=None,
            )
        )


def test_test_database_url_must_target_langley_test() -> None:
    with pytest.raises(RuntimeError, match="langley_test"):
        require_test_database_url(
            Settings(
                test_database_url="mysql+asyncmy://root:password@127.0.0.1:3306/other"
            )
        )


def test_development_database_must_not_target_langley_test() -> None:
    with pytest.raises(RuntimeError, match="LANGLEY_DATABASE_URL"):
        require_test_database_url(
            Settings(
                database_url="mysql+asyncmy://root:password@127.0.0.1:3306/langley_test",
                test_database_url="mysql+asyncmy://root:password@127.0.0.1:3306/langley_test",
            )
        )


def test_create_app_does_not_connect_to_configured_database() -> None:
    app = create_app(
        Settings(database_url="mysql+asyncmy://root:password@127.0.0.1:3306/langley")
    )

    assert app.state.settings.database_url is not None
