"""Real MySQL schema and constraint tests for Slice 2 facts."""

import asyncio
from argparse import Namespace

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from langley.infrastructure.database import (
    create_database_engine,
    dispose_database_engine,
)

_TIMESTAMP = "2026-08-11 00:00:00.123456"


@pytest.fixture
def migrated_database(test_database_url: str, reset_database) -> str:
    """Reset the approved test database and upgrade it to the Slice 2 head."""

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


def _seed_user_and_conversation(database_url: str) -> None:
    _run(
        database_url,
        f"INSERT INTO users (id, created_at) VALUES (1, '{_TIMESTAMP}')",
        "INSERT INTO conversations "
        "(id, user_id, created_at, updated_at, last_message_at, deleted_at) "
        f"VALUES (1, 1, '{_TIMESTAMP}', '{_TIMESTAMP}', NULL, NULL)",
    )


def _insert_user_message(database_url: str, message_id: int, sequence_no: int) -> None:
    _run(
        database_url,
        "INSERT INTO messages "
        "(id, conversation_id, sequence_no, role, content, run_id, "
        "regenerated_from_message_id, created_at) "
        f"VALUES ({message_id}, 1, {sequence_no}, 'USER', 'question', NULL, NULL, "
        f"'{_TIMESTAMP}')",
    )


def _insert_pending_run(
    database_url: str, run_id: int, message_id: int, client_request_id: str
) -> None:
    _run(
        database_url,
        "INSERT INTO runs "
        "(id, conversation_id, input_message_id, client_request_id, attempt_no, "
        "status, "
        "started_at, finished_at, error_code, created_at, updated_at) "
        f"VALUES ({run_id}, 1, {message_id}, '{client_request_id}', 1, 'PENDING', "
        f"NULL, NULL, NULL, '{_TIMESTAMP}', '{_TIMESTAMP}')",
    )


def test_business_migration_creates_schema_and_circular_foreign_keys(
    migrated_database: str,
) -> None:
    """The empty-database migration establishes every Slice 2 fact relation."""

    tables = _scalar(
        migrated_database,
        "SELECT COUNT(*) FROM information_schema.tables "
        "WHERE table_schema = DATABASE() "
        "AND table_name IN ('users', 'conversations', 'messages', 'runs')",
    )
    message_run_fk = _scalar(
        migrated_database,
        "SELECT COUNT(*) FROM information_schema.table_constraints "
        "WHERE table_schema = DATABASE() AND table_name = 'messages' "
        "AND constraint_name = 'fk_messages_run' "
        "AND constraint_type = 'FOREIGN KEY'",
    )
    run_input_fk = _scalar(
        migrated_database,
        "SELECT COUNT(*) FROM information_schema.table_constraints "
        "WHERE table_schema = DATABASE() AND table_name = 'runs' "
        "AND constraint_name = 'fk_runs_input_message' "
        "AND constraint_type = 'FOREIGN KEY'",
    )

    assert tables == 4
    assert message_run_fk == 1
    assert run_input_fk == 1


def test_machine_fields_are_binary_and_business_timestamps_have_no_db_default(
    migrated_database: str,
) -> None:
    """Exact machine equality and application-generated UTC time are schema facts."""

    binary_columns = _scalar(
        migrated_database,
        "SELECT COUNT(*) FROM information_schema.columns "
        "WHERE table_schema = DATABASE() "
        "AND ((table_name = 'messages' AND column_name = 'role') "
        "OR (table_name = 'runs' AND column_name IN "
        "('client_request_id', 'status', 'error_code'))) "
        "AND collation_name = 'utf8mb4_0900_bin'",
    )
    timestamp_defaults = _scalar(
        migrated_database,
        "SELECT COUNT(*) FROM information_schema.columns "
        "WHERE table_schema = DATABASE() AND data_type = 'datetime' "
        "AND column_default IS NOT NULL",
    )

    assert binary_columns == 4
    assert timestamp_defaults == 0


def test_message_constraints_preserve_linear_order_and_visible_roles(
    migrated_database: str,
) -> None:
    """MySQL rejects duplicate positions and invalid role/run combinations."""

    _seed_user_and_conversation(migrated_database)
    _insert_user_message(migrated_database, message_id=1, sequence_no=1)

    with pytest.raises(DBAPIError):
        _run(
            migrated_database,
            "INSERT INTO messages "
            "(conversation_id, sequence_no, role, content, run_id, "
            "regenerated_from_message_id, created_at) "
            f"VALUES (1, 1, 'USER', 'duplicate', NULL, NULL, '{_TIMESTAMP}')",
        )
    with pytest.raises(DBAPIError):
        _run(
            migrated_database,
            "INSERT INTO messages "
            "(conversation_id, sequence_no, role, content, run_id, "
            "regenerated_from_message_id, created_at) "
            f"VALUES (1, 2, 'user', 'lowercase', NULL, NULL, '{_TIMESTAMP}')",
        )
    with pytest.raises(DBAPIError):
        _run(
            migrated_database,
            "INSERT INTO messages "
            "(conversation_id, sequence_no, role, content, run_id, "
            "regenerated_from_message_id, created_at) "
            f"VALUES (1, 2, 'ASSISTANT', 'invalid', NULL, NULL, '{_TIMESTAMP}')",
        )

    _insert_user_message(migrated_database, message_id=2, sequence_no=2)
    _run(
        migrated_database,
        "INSERT INTO conversations "
        "(id, user_id, created_at, updated_at, last_message_at, deleted_at) "
        f"VALUES (2, 1, '{_TIMESTAMP}', '{_TIMESTAMP}', NULL, NULL)",
        "INSERT INTO messages "
        "(id, conversation_id, sequence_no, role, content, run_id, "
        "regenerated_from_message_id, created_at) "
        f"VALUES (3, 2, 1, 'USER', 'other conversation', NULL, NULL, "
        f"'{_TIMESTAMP}')",
    )
    _insert_pending_run(migrated_database, 1, 1, "message-test")
    _run(
        migrated_database,
        "INSERT INTO messages "
        "(conversation_id, sequence_no, role, content, run_id, "
        "regenerated_from_message_id, created_at) "
        f"VALUES (1, 3, 'ASSISTANT', 'answer', 1, NULL, '{_TIMESTAMP}')",
    )
    with pytest.raises(DBAPIError):
        _run(
            migrated_database,
            "INSERT INTO messages "
            "(conversation_id, sequence_no, role, content, run_id, "
            "regenerated_from_message_id, created_at) "
            f"VALUES (1, 4, 'ASSISTANT', 'duplicate answer', 1, NULL, "
            f"'{_TIMESTAMP}')",
        )


def test_run_constraints_enforce_exact_request_identity_and_lifecycle_shape(
    migrated_database: str,
) -> None:
    """Runs distinguish case-sensitive keys and reject illegal lifecycle values."""

    _seed_user_and_conversation(migrated_database)
    _insert_user_message(migrated_database, message_id=1, sequence_no=1)
    _insert_user_message(migrated_database, message_id=2, sequence_no=2)
    _insert_pending_run(migrated_database, 1, 1, "CaseKey")
    _insert_pending_run(migrated_database, 2, 2, "casekey")

    assert _scalar(migrated_database, "SELECT COUNT(*) FROM runs") == 2

    _insert_user_message(migrated_database, message_id=3, sequence_no=3)
    with pytest.raises(DBAPIError):
        _insert_pending_run(migrated_database, 3, 3, "CaseKey")
    with pytest.raises(DBAPIError):
        _run(
            migrated_database,
            "INSERT INTO runs "
            "(conversation_id, input_message_id, client_request_id, attempt_no, "
            "status, started_at, finished_at, error_code, created_at, updated_at) "
            f"VALUES (1, 1, 'second-attempt-key', 1, 'PENDING', NULL, NULL, "
            f"NULL, '{_TIMESTAMP}', '{_TIMESTAMP}')",
        )
    with pytest.raises(DBAPIError):
        _run(
            migrated_database,
            "INSERT INTO runs "
            "(conversation_id, input_message_id, client_request_id, attempt_no, "
            "status, "
            "started_at, finished_at, error_code, created_at, updated_at) "
            f"VALUES (1, 3, 'third', 1, 'running', '{_TIMESTAMP}', NULL, NULL, "
            f"'{_TIMESTAMP}', '{_TIMESTAMP}')",
        )
    with pytest.raises(DBAPIError):
        _run(
            migrated_database,
            "INSERT INTO runs "
            "(conversation_id, input_message_id, client_request_id, attempt_no, "
            "status, "
            "started_at, finished_at, error_code, created_at, updated_at) "
            f"VALUES (1, 3, 'third', 1, 'PENDING', '{_TIMESTAMP}', NULL, NULL, "
            f"'{_TIMESTAMP}', '{_TIMESTAMP}')",
        )
