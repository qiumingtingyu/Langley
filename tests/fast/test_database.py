"""Fast tests for database infrastructure configuration."""

import asyncio

import pytest

from langley.answering.fake_provider import FakeProvider
from langley.infrastructure.database import (
    create_database_engine,
    create_session_factory,
    dispose_database_engine,
    require_test_database_url,
)
from langley.infrastructure.models import Memory, Message, User
from langley.main import create_app
from langley.memory.policy import MemoryPolicy, MemoryPolicyStatus
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


def test_create_app_constructs_no_memory_policy_without_model() -> None:
    app = create_app(
        Settings(
            database_url="mysql+asyncmy://root:password@127.0.0.1:3306/langley",
            memory_policy_estimated_token_budget=24_576,
        ),
        memory_provider=FakeProvider([]),
    )

    assert not hasattr(app.state, "memory_policy")
    assert app.state.memory_policy_status is MemoryPolicyStatus.NOT_CONFIGURED


def test_create_app_constructs_configured_memory_policy() -> None:
    app = create_app(
        Settings(
            database_url="mysql+asyncmy://root:password@127.0.0.1:3306/langley",
            memory_policy_model="qwen3.7-plus-2026-05-26",
            memory_policy_estimated_token_budget=24_576,
        ),
        memory_provider=FakeProvider([]),
    )

    assert isinstance(app.state.memory_policy, MemoryPolicy)
    assert app.state.memory_policy_status is MemoryPolicyStatus.READY


def test_create_app_reports_unavailable_memory_provider_configuration() -> None:
    app = create_app(
        Settings(
            database_url="mysql+asyncmy://root:password@127.0.0.1:3306/langley",
            memory_policy_model="qwen3.7-plus-2026-05-26",
            memory_policy_estimated_token_budget=24_576,
            qwen_api_key=None,
            qwen_base_url=None,
        )
    )

    assert isinstance(app.state.memory_policy, MemoryPolicy)
    assert (
        app.state.memory_policy_status
        is MemoryPolicyStatus.PROVIDER_CONFIGURATION_UNAVAILABLE
    )


def test_memory_orm_metadata_matches_the_persistence_contract() -> None:
    assert set(Memory.__table__.columns.keys()) == {
        "content",
        "created_at",
        "id",
        "source_message_id",
        "updated_at",
        "user_id",
        "valid_until",
    }
    assert {index.name for index in Memory.__table__.indexes} == {"ix_memories_user"}
    assert User.__table__.c.auto_memory_enabled.nullable is False
    assert Message.__table__.c.memory_processed_at.nullable is True
