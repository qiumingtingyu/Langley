"""Real MySQL persistence foundation tests."""

import asyncio
from argparse import Namespace
from datetime import datetime

import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from langley.infrastructure.database import (
    create_database_engine,
    dispose_database_engine,
)


def test_asyncmy_executes_real_select_one(
    test_database_url: str, reset_database
) -> None:
    """Verify the SQLAlchemy async engine reaches real MySQL through asyncmy."""

    reset_database()

    async def select_one() -> int:
        engine = create_database_engine(test_database_url)
        try:
            async with engine.connect() as connection:
                result = await connection.execute(text("SELECT 1"))
                return result.scalar_one()
        finally:
            await dispose_database_engine(engine)

    assert asyncio.run(select_one()) == 1


def test_migration_smoke_upgrades_empty_test_database(
    test_database_url: str, reset_database
) -> None:
    """Upgrade a reset test database and verify it reaches Alembic head."""

    reset_database()
    config = Config("alembic.ini")
    config.cmd_opts = Namespace(x=["use_test_database=true"])
    command.upgrade(config, "head")
    expected_revision = ScriptDirectory.from_config(config).get_current_head()

    async def current_revision() -> str | None:
        engine = create_database_engine(test_database_url)
        try:
            async with engine.connect() as connection:
                result = await connection.execute(
                    text("SELECT version_num FROM alembic_version")
                )
                return result.scalar_one_or_none()
        finally:
            await dispose_database_engine(engine)

    assert asyncio.run(current_revision()) == expected_revision


def test_memory_cutover_marks_only_pre_slice5_canonical_user_messages(
    test_database_url: str, reset_database
) -> None:
    """Apply 0003 over an actual 0002 schema containing both USER variants."""
    reset_database()
    config = Config("alembic.ini")
    config.cmd_opts = Namespace(x=["use_test_database=true"])
    command.upgrade(config, "0002_conversation_answer_loop")

    created_at = datetime(2026, 8, 20, 2, 0)

    async def seed_pre_slice5_messages() -> tuple[int, int]:
        engine = create_database_engine(test_database_url)
        try:
            async with engine.begin() as connection:
                user_result = await connection.execute(
                    text("INSERT INTO users (created_at) VALUES (:created_at)"),
                    {"created_at": created_at},
                )
                conversation_result = await connection.execute(
                    text(
                        "INSERT INTO conversations "
                        "(user_id, title, created_at, updated_at, last_message_at, "
                        "deleted_at) "
                        "VALUES (:user_id, NULL, :created_at, :created_at, NULL, NULL)"
                    ),
                    {"user_id": user_result.lastrowid, "created_at": created_at},
                )
                canonical_result = await connection.execute(
                    text(
                        "INSERT INTO messages "
                        "(conversation_id, sequence_no, role, content, run_id, "
                        "regenerated_from_message_id, created_at) "
                        "VALUES (:conversation_id, 1, 'USER', 'canonical', NULL, NULL, "
                        ":created_at)"
                    ),
                    {
                        "conversation_id": conversation_result.lastrowid,
                        "created_at": created_at,
                    },
                )
                copied_result = await connection.execute(
                    text(
                        "INSERT INTO messages "
                        "(conversation_id, sequence_no, role, content, run_id, "
                        "regenerated_from_message_id, created_at) "
                        "VALUES (:conversation_id, 2, 'USER', "
                        "'regenerated copy', NULL, :canonical_id, :created_at)"
                    ),
                    {
                        "conversation_id": conversation_result.lastrowid,
                        "canonical_id": canonical_result.lastrowid,
                        "created_at": created_at,
                    },
                )
                return canonical_result.lastrowid, copied_result.lastrowid
        finally:
            await dispose_database_engine(engine)

    canonical_id, copied_id = asyncio.run(seed_pre_slice5_messages())
    command.upgrade(config, "0003_personal_context_memory")

    async def mark_regenerated_copy() -> None:
        engine = create_database_engine(test_database_url)
        try:
            async with engine.begin() as connection:
                await connection.execute(
                    text(
                        "UPDATE messages SET memory_processed_at = :processed_at "
                        "WHERE id = :message_id"
                    ),
                    {
                        "processed_at": datetime(2026, 8, 20, 2, 1),
                        "message_id": copied_id,
                    },
                )
        finally:
            await dispose_database_engine(engine)

    with pytest.raises(
        OperationalError, match="ck_messages_memory_processed_canonical"
    ):
        asyncio.run(mark_regenerated_copy())

    async def read_cutover_result() -> tuple[
        object, object, object, set[str], int, int
    ]:
        engine = create_database_engine(test_database_url)
        try:
            async with engine.connect() as connection:
                canonical_processed_at = await connection.scalar(
                    text(
                        "SELECT memory_processed_at FROM messages "
                        "WHERE id = :message_id"
                    ),
                    {"message_id": canonical_id},
                )
                copied_processed_at = await connection.scalar(
                    text(
                        "SELECT memory_processed_at FROM messages "
                        "WHERE id = :message_id"
                    ),
                    {"message_id": copied_id},
                )
                auto_memory_enabled = await connection.scalar(
                    text("SELECT auto_memory_enabled FROM users LIMIT 1")
                )
                memory_columns = set(
                    (
                        await connection.scalars(
                            text(
                                "SELECT column_name FROM information_schema.columns "
                                "WHERE table_schema = DATABASE() "
                                "AND table_name = 'memories'"
                            )
                        )
                    ).all()
                )
                memory_index_count = await connection.scalar(
                    text(
                        "SELECT COUNT(*) FROM information_schema.statistics "
                        "WHERE table_schema = DATABASE() "
                        "AND table_name = 'memories' "
                        "AND index_name = 'ix_memories_user'"
                    )
                )
                memory_count = await connection.scalar(
                    text("SELECT COUNT(*) FROM memories")
                )
                return (
                    canonical_processed_at,
                    copied_processed_at,
                    auto_memory_enabled,
                    memory_columns,
                    memory_index_count,
                    memory_count,
                )
        finally:
            await dispose_database_engine(engine)

    (
        canonical_processed_at,
        copied_processed_at,
        auto_memory_enabled,
        memory_columns,
        memory_index_count,
        memory_count,
    ) = asyncio.run(read_cutover_result())

    assert canonical_processed_at is not None
    assert copied_processed_at is None
    assert auto_memory_enabled == 1
    assert memory_columns == {
        "content",
        "created_at",
        "id",
        "source_message_id",
        "updated_at",
        "user_id",
        "valid_until",
    }
    assert memory_index_count == 1
    assert memory_count == 0

    command.downgrade(config, "0002_conversation_answer_loop")
    command.upgrade(config, "head")

    async def current_revision() -> str | None:
        engine = create_database_engine(test_database_url)
        try:
            async with engine.connect() as connection:
                return await connection.scalar(
                    text("SELECT version_num FROM alembic_version")
                )
        finally:
            await dispose_database_engine(engine)

    assert asyncio.run(current_revision()) == "0004_knowledge_persistence"
