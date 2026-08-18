"""Alembic async migration environment."""

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy.engine import Connection

import langley.infrastructure.models  # noqa: F401
from langley.infrastructure.database import (
    Base,
    create_database_engine,
    dispose_database_engine,
    require_test_database_url,
)
from langley.settings import Settings

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def get_database_url() -> str:
    """Read the explicitly selected migration database URL from Settings."""

    settings = Settings()
    options = context.get_x_argument(as_dictionary=True)
    if options.get("use_test_database") == "true":
        return require_test_database_url(settings)

    if settings.database_url is None:
        raise RuntimeError("LANGLEY_DATABASE_URL must be configured for migrations")
    return settings.database_url


def run_migrations_offline() -> None:
    """Run migrations without a live connection."""

    context.configure(
        url=get_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Configure Alembic against a synchronous connection bridge."""

    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Run migrations through the async MySQL engine."""

    engine = create_database_engine(get_database_url())
    try:
        async with engine.begin() as connection:
            await connection.run_sync(do_run_migrations)
    finally:
        await dispose_database_engine(engine)


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
