"""Explicit local/demo User bootstrap for Slice 2 development environments."""

import asyncio

from sqlalchemy.ext.asyncio import AsyncSession

from langley.business_time import utc_now
from langley.infrastructure.database import (
    create_database_engine,
    create_session_factory,
    dispose_database_engine,
)
from langley.infrastructure.models import User
from langley.settings import Settings


async def bootstrap_local_user(settings: Settings) -> bool:
    """Create the configured local User when absent."""

    if settings.database_url is None:
        raise RuntimeError("LANGLEY_DATABASE_URL must be configured for user bootstrap")
    if settings.local_user_id is None:
        raise RuntimeError(
            "LANGLEY_LOCAL_USER_ID must be configured for user bootstrap"
        )

    engine = create_database_engine(settings.database_url)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory() as session:
            return await _ensure_user(session, settings.local_user_id)
    finally:
        await dispose_database_engine(engine)


async def _ensure_user(session: AsyncSession, user_id: int) -> bool:
    """Persist one bootstrap User without request-time implicit creation."""

    async with session.begin():
        if await session.get(User, user_id) is not None:
            return False
        session.add(User(id=user_id, created_at=utc_now()))
        return True


def main() -> None:
    """Run the explicit bootstrap command from the configured environment."""

    created = asyncio.run(bootstrap_local_user(Settings()))
    outcome = "created" if created else "already exists"
    print(f"Local user bootstrap: {outcome}")


if __name__ == "__main__":
    main()
