"""Real MySQL persistence foundation tests."""

import asyncio
from argparse import Namespace

from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import text

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
