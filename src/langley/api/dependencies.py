"""Shared FastAPI dependencies for database-backed API routes."""

from collections.abc import AsyncIterator

from fastapi import HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from langley.answer_execution import AnswerExecutionManager
from langley.infrastructure.models import User
from langley.memory_events import MemoryEventSubscribers
from langley.memory_policy import MemoryPolicy


def _get_session_factory(request: Request) -> async_sessionmaker[AsyncSession]:
    """Return the configured session factory without opening a connection at import."""

    session_factory = getattr(request.app.state, "session_factory", None)
    if session_factory is None:
        raise HTTPException(
            status_code=500,
            detail={"code": "DATABASE_NOT_CONFIGURED"},
        )
    return session_factory


def get_session_factory(request: Request) -> async_sessionmaker[AsyncSession]:
    """Provide the configured factory for execution phases needing separate sessions."""

    return _get_session_factory(request)


def get_execution_manager(request: Request) -> AnswerExecutionManager:
    """Return the application-scoped process-local execution manager."""

    execution_manager = getattr(request.app.state, "execution_manager", None)
    if execution_manager is None:
        raise HTTPException(
            status_code=500,
            detail={"code": "DATABASE_NOT_CONFIGURED"},
        )
    return execution_manager


def get_settings(request: Request):
    return request.app.state.settings


def get_memory_policy(request: Request) -> MemoryPolicy | None:
    return getattr(request.app.state, "memory_policy", None)


def get_memory_lane(request: Request):
    lane = getattr(request.app.state, "memory_lane", None)
    if lane is None:
        raise HTTPException(status_code=500, detail={"code": "DATABASE_NOT_CONFIGURED"})
    return lane


def get_memory_subscribers(request: Request) -> MemoryEventSubscribers:
    subscribers = getattr(request.app.state, "memory_subscribers", None)
    if subscribers is None:
        raise HTTPException(status_code=500, detail={"code": "DATABASE_NOT_CONFIGURED"})
    return subscribers


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    """Yield one request-local database session."""

    session_factory = _get_session_factory(request)
    async with session_factory() as session:
        yield session


async def get_current_user_id(request: Request) -> int:
    """Resolve the configured local identity without creating or upserting a user."""

    local_user_id = request.app.state.settings.local_user_id
    if local_user_id is None:
        raise HTTPException(
            status_code=500,
            detail={"code": "LOCAL_USER_NOT_CONFIGURED"},
        )

    session_factory = _get_session_factory(request)
    async with session_factory() as session:
        user = await session.get(User, local_user_id)

    if user is None:
        raise HTTPException(
            status_code=503,
            detail={"code": "LOCAL_USER_NOT_BOOTSTRAPPED"},
        )
    return local_user_id
