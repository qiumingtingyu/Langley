"""Real MySQL integration tests for the Slice 2 identity and Conversation APIs."""

import asyncio
from argparse import Namespace

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import text

from langley.bootstrap import bootstrap_local_user
from langley.infrastructure.database import (
    create_database_engine,
    dispose_database_engine,
)
from langley.main import create_app
from langley.settings import Settings

_TIMESTAMP = "2026-08-11 00:00:00.123456"


@pytest.fixture
def migrated_database(test_database_url: str, reset_database) -> str:
    """Provide an empty real MySQL database migrated to the current head."""

    reset_database()
    config = Config("alembic.ini")
    config.cmd_opts = Namespace(x=["use_test_database=true"])
    command.upgrade(config, "head")
    return test_database_url


def _run(database_url: str, *statements: str) -> None:
    async def execute() -> None:
        engine = create_database_engine(database_url)
        try:
            async with engine.begin() as connection:
                for statement in statements:
                    await connection.execute(text(statement))
        finally:
            await dispose_database_engine(engine)

    asyncio.run(execute())


def _scalar(database_url: str, statement: str) -> object:
    async def execute() -> object:
        engine = create_database_engine(database_url)
        try:
            async with engine.connect() as connection:
                return (await connection.execute(text(statement))).scalar_one()
        finally:
            await dispose_database_engine(engine)

    return asyncio.run(execute())


def _settings(database_url: str, local_user_id: int | None = 1) -> Settings:
    return Settings(
        environment="test",
        database_url=database_url,
        local_user_id=local_user_id,
    )


def _bootstrap(database_url: str, user_id: int) -> bool:
    return asyncio.run(bootstrap_local_user(_settings(database_url, user_id)))


def test_explicit_local_user_bootstrap_is_idempotent(migrated_database: str) -> None:
    """The explicit bootstrap creates a configured user and never runs implicitly."""

    assert _bootstrap(migrated_database, 7)
    assert not _bootstrap(migrated_database, 7)
    assert _scalar(migrated_database, "SELECT COUNT(*) FROM users WHERE id = 7") == 1


def test_current_user_errors_are_configuration_and_bootstrap_safe(
    migrated_database: str,
) -> None:
    """API identity neither defaults to user 1 nor creates a missing user."""

    with TestClient(create_app(_settings(migrated_database, None))) as client:
        unconfigured = client.get("/api/conversations")
    with TestClient(create_app(_settings(migrated_database, 99))) as client:
        missing = client.get("/api/conversations")

    assert unconfigured.status_code == 500
    assert unconfigured.json() == {"detail": {"code": "LOCAL_USER_NOT_CONFIGURED"}}
    assert missing.status_code == 503
    assert missing.json() == {"detail": {"code": "LOCAL_USER_NOT_BOOTSTRAPPED"}}
    assert _scalar(migrated_database, "SELECT COUNT(*) FROM users") == 0


def test_create_and_list_conversations_are_owner_scoped_and_stably_ordered(
    migrated_database: str,
) -> None:
    """Current user identity controls creation, filtering, and recent-first ordering."""

    assert _bootstrap(migrated_database, 1)
    assert _bootstrap(migrated_database, 2)
    with TestClient(create_app(_settings(migrated_database, 1))) as client:
        first = client.post(
            "/api/conversations",
            json={"title": "First"},
        )
        unexpected_field = client.post(
            "/api/conversations",
            json={"title": "Rejected", "user_id": 2},
        )
        second = client.post("/api/conversations", json={"title": "Second"})

        assert first.status_code == 201
        assert unexpected_field.status_code == 422
        assert second.status_code == 201
        first_id = first.json()["id"]
        second_id = second.json()["id"]

        _run(
            migrated_database,
            "UPDATE conversations "
            f"SET last_message_at = '2026-08-11 00:00:00.000001' WHERE id = {first_id}",
            "UPDATE conversations "
            "SET last_message_at = '2026-08-11 00:00:01.000001' "
            f"WHERE id = {second_id}",
            "INSERT INTO conversations "
            "(id, user_id, title, created_at, updated_at, last_message_at, deleted_at) "
            f"VALUES (99, 2, 'Other user', '{_TIMESTAMP}', '{_TIMESTAMP}', NULL, NULL)",
        )
        listed = client.get("/api/conversations")

    assert listed.status_code == 200
    assert [conversation["id"] for conversation in listed.json()] == [
        second_id,
        first_id,
    ]
    assert (
        _scalar(
            migrated_database,
            f"SELECT user_id FROM conversations WHERE id = {first_id}",
        )
        == 1
    )
    assert _scalar(migrated_database, "SELECT COUNT(*) FROM conversations") == 3


def test_owned_message_query_restores_sequence_and_latest_run(
    migrated_database: str,
) -> None:
    """Message history uses sequence order and exposes only the latest user run."""

    assert _bootstrap(migrated_database, 1)
    assert _bootstrap(migrated_database, 2)
    statements = (
        "INSERT INTO conversations "
        "(id, user_id, title, created_at, updated_at, last_message_at, deleted_at) "
        "VALUES (10, 1, 'Owned', "
        f"'{_TIMESTAMP}', '{_TIMESTAMP}', '{_TIMESTAMP}', NULL)",
        "INSERT INTO conversations "
        "(id, user_id, title, created_at, updated_at, last_message_at, deleted_at) "
        f"VALUES (20, 2, 'Other', '{_TIMESTAMP}', '{_TIMESTAMP}', NULL, NULL)",
        "INSERT INTO messages "
        "(id, conversation_id, sequence_no, role, content, run_id, "
        "regenerated_from_message_id, created_at) "
        f"VALUES (101, 10, 1, 'USER', 'first', NULL, NULL, '{_TIMESTAMP}')",
        "INSERT INTO runs "
        "(id, conversation_id, input_message_id, client_request_id, attempt_no, "
        "status, "
        "started_at, finished_at, error_code, created_at, updated_at) "
        f"VALUES (201, 10, 101, 'first-request', 1, 'SUCCEEDED', '{_TIMESTAMP}', "
        f"'{_TIMESTAMP}', NULL, '{_TIMESTAMP}', '{_TIMESTAMP}')",
        "INSERT INTO messages "
        "(id, conversation_id, sequence_no, role, content, run_id, "
        "regenerated_from_message_id, created_at) "
        f"VALUES (102, 10, 2, 'ASSISTANT', 'first answer', 201, NULL, '{_TIMESTAMP}')",
        "INSERT INTO messages "
        "(id, conversation_id, sequence_no, role, content, run_id, "
        "regenerated_from_message_id, created_at) "
        f"VALUES (103, 10, 3, 'USER', 'latest', NULL, NULL, '{_TIMESTAMP}')",
        "INSERT INTO runs "
        "(id, conversation_id, input_message_id, client_request_id, attempt_no, "
        "status, "
        "started_at, finished_at, error_code, created_at, updated_at) "
        f"VALUES (202, 10, 103, 'latest-request', 1, 'PENDING', NULL, NULL, NULL, "
        f"'{_TIMESTAMP}', '{_TIMESTAMP}')",
    )
    with TestClient(create_app(_settings(migrated_database, 1))) as client:
        _run(migrated_database, *statements)
        owned = client.get("/api/conversations/10/messages")
        other_user = client.get("/api/conversations/20/messages")

    assert owned.status_code == 200
    payload = owned.json()
    assert [message["sequence_no"] for message in payload["messages"]] == [1, 2, 3]
    assert payload["latest_run"]["id"] == 202
    assert payload["latest_run"]["status"] == "PENDING"
    assert other_user.status_code == 404
    assert other_user.json() == {"detail": {"code": "CONVERSATION_NOT_FOUND"}}
