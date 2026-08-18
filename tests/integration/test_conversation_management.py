"""Real MySQL integration tests for Slice 4 Conversation usability commands."""

import asyncio
from argparse import Namespace
from collections.abc import Awaitable, Callable
from typing import TypeVar

import pytest
from alembic import command
from alembic.config import Config
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from langley.answer_execution import (
    _commit_success,
    _start_running,
    mark_run_failed_if_running,
)
from langley.bootstrap import bootstrap_local_user
from langley.conversation_commands import (
    ConversationNotFoundError,
    NewQuestionAdmission,
    RegenerateAdmission,
    RetryAdmission,
    admit_new_question,
    admit_regenerate,
    admit_retry,
)
from langley.conversations import (
    ConversationHasActiveRunError,
    create_conversation,
    delete_conversation,
)
from langley.infrastructure.database import (
    create_database_engine,
    create_session_factory,
    dispose_database_engine,
)
from langley.infrastructure.models import Conversation, Message, Run
from langley.main import create_app
from langley.settings import Settings

Result = TypeVar("Result")


@pytest.fixture
def migrated_database(test_database_url: str, reset_database) -> str:
    """Provide an empty real MySQL database migrated to the current head."""

    reset_database()
    config = Config("alembic.ini")
    config.cmd_opts = Namespace(x=["use_test_database=true"])
    command.upgrade(config, "head")
    return test_database_url


def _settings(database_url: str) -> Settings:
    return Settings(environment="test", database_url=database_url, local_user_id=1)


def _bootstrap(database_url: str) -> None:
    assert asyncio.run(bootstrap_local_user(_settings(database_url)))


async def _exercise(
    database_url: str,
    operation: Callable[[async_sessionmaker[AsyncSession]], Awaitable[Result]],
) -> Result:
    engine = create_database_engine(database_url)
    try:
        return await operation(create_session_factory(engine))
    finally:
        await dispose_database_engine(engine)


async def _create_conversation(
    session_factory: async_sessionmaker[AsyncSession],
) -> Conversation:
    async with session_factory() as session:
        return await create_conversation(session, user_id=1, title="原始标题")


async def _admit_new(
    session_factory: async_sessionmaker[AsyncSession], *, conversation_id: int
) -> NewQuestionAdmission:
    async with session_factory() as session:
        return await admit_new_question(
            session,
            user_id=1,
            conversation_id=conversation_id,
            content="问题",
            client_request_id="management-question",
        )


def test_rename_updates_only_conversation_title_and_keeps_answer_facts(
    migrated_database: str,
) -> None:
    _bootstrap(migrated_database)

    async def operation(session_factory: async_sessionmaker[AsyncSession]) -> None:
        conversation = await _create_conversation(session_factory)
        admission = await _admit_new(session_factory, conversation_id=conversation.id)
        await _start_running(
            session_factory,
            conversation_id=conversation.id,
            run_id=admission.run.id,
        )
        await _commit_success(
            session_factory,
            conversation_id=conversation.id,
            run_id=admission.run.id,
            content="回答",
        )
        app = create_app(_settings(migrated_database))
        async with app.router.lifespan_context(app):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.patch(
                    f"/api/conversations/{conversation.id}", json={"title": "新标题"}
                )

        assert response.status_code == 200
        assert response.json()["title"] == "新标题"
        async with session_factory() as session:
            persisted = await session.get(Conversation, conversation.id)
            assert persisted is not None
            assert persisted.title == "新标题"
            assert (
                await session.scalar(
                    select(func.count())
                    .select_from(Message)
                    .where(Message.conversation_id == conversation.id)
                )
                == 2
            )
            assert (
                await session.scalar(
                    select(func.count())
                    .select_from(Run)
                    .where(Run.conversation_id == conversation.id)
                )
                == 1
            )

    asyncio.run(_exercise(migrated_database, operation))


def test_delete_is_logical_and_hides_the_conversation_from_active_reads(
    migrated_database: str,
) -> None:
    _bootstrap(migrated_database)

    async def operation(session_factory: async_sessionmaker[AsyncSession]) -> None:
        conversation = await _create_conversation(session_factory)
        app = create_app(_settings(migrated_database))
        async with app.router.lifespan_context(app):
            admission = await _admit_new(
                session_factory, conversation_id=conversation.id
            )
            await _start_running(
                session_factory,
                conversation_id=conversation.id,
                run_id=admission.run.id,
            )
            await _commit_success(
                session_factory,
                conversation_id=conversation.id,
                run_id=admission.run.id,
                content="回答",
            )
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.delete(f"/api/conversations/{conversation.id}")
                listed = await client.get("/api/conversations")
                messages = await client.get(
                    f"/api/conversations/{conversation.id}/messages"
                )

        assert response.status_code == 204
        assert listed.status_code == 200
        assert listed.json() == []
        assert messages.status_code == 404
        async with session_factory() as session:
            persisted = await session.get(Conversation, conversation.id)
            assert persisted is not None
            assert persisted.deleted_at is not None
            assert (
                await session.scalar(
                    select(func.count())
                    .select_from(Message)
                    .where(Message.conversation_id == conversation.id)
                )
                == 2
            )
            assert (
                await session.scalar(
                    select(func.count())
                    .select_from(Run)
                    .where(Run.conversation_id == conversation.id)
                )
                == 1
            )

    asyncio.run(_exercise(migrated_database, operation))


def test_delete_active_conversation_returns_conflict_without_cancelling_run(
    migrated_database: str,
) -> None:
    _bootstrap(migrated_database)

    async def operation(session_factory: async_sessionmaker[AsyncSession]) -> None:
        conversation = await _create_conversation(session_factory)
        app = create_app(_settings(migrated_database))
        async with app.router.lifespan_context(app):
            admission = await _admit_new(
                session_factory, conversation_id=conversation.id
            )
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.delete(f"/api/conversations/{conversation.id}")

            assert response.status_code == 409
            assert response.json()["detail"] == {"code": "ACTIVE_RUN_EXISTS"}
            async with session_factory() as session:
                persisted = await session.get(Conversation, conversation.id)
                run = await session.get(Run, admission.run.id)
                assert persisted is not None and persisted.deleted_at is None
                assert run is not None and run.status == "PENDING"

    asyncio.run(_exercise(migrated_database, operation))


def test_delete_and_admission_serialize_to_one_legal_outcome(
    migrated_database: str,
) -> None:
    _bootstrap(migrated_database)

    async def operation(session_factory: async_sessionmaker[AsyncSession]) -> None:
        conversation = await _create_conversation(session_factory)
        start = asyncio.Event()

        async def delete() -> bool:
            await start.wait()
            async with session_factory() as session:
                return await delete_conversation(
                    session, user_id=1, conversation_id=conversation.id
                )

        async def admit() -> NewQuestionAdmission:
            await start.wait()
            async with session_factory() as session:
                return await admit_new_question(
                    session,
                    user_id=1,
                    conversation_id=conversation.id,
                    content="并发问题",
                    client_request_id="delete-admission-race",
                )

        delete_task = asyncio.create_task(delete())
        admit_task = asyncio.create_task(admit())
        start.set()
        deleted, admission = await asyncio.gather(
            delete_task, admit_task, return_exceptions=True
        )

        async with session_factory() as session:
            persisted = await session.get(Conversation, conversation.id)
            assert persisted is not None
            active_run_count = await session.scalar(
                select(func.count())
                .select_from(Run)
                .where(
                    Run.conversation_id == conversation.id,
                    Run.status.in_(("PENDING", "RUNNING")),
                )
            )

        if deleted is True:
            assert isinstance(admission, ConversationNotFoundError)
            assert persisted.deleted_at is not None
            assert active_run_count == 0
        else:
            assert isinstance(deleted, ConversationHasActiveRunError)
            assert isinstance(admission, NewQuestionAdmission)
            assert persisted.deleted_at is None
            assert active_run_count == 1

    asyncio.run(_exercise(migrated_database, operation))


async def _assert_delete_command_race(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    conversation_id: int,
    command: Callable[[], Awaitable[object]],
    expected_admission_type: type[object],
) -> None:
    """Prove Delete serializes with every admission command at Conversation."""

    start = asyncio.Event()

    async def delete() -> bool:
        await start.wait()
        async with session_factory() as session:
            return await delete_conversation(
                session, user_id=1, conversation_id=conversation_id
            )

    async def admit() -> object:
        await start.wait()
        return await command()

    delete_task = asyncio.create_task(delete())
    admit_task = asyncio.create_task(admit())
    start.set()
    deleted, admission = await asyncio.gather(
        delete_task, admit_task, return_exceptions=True
    )

    async with session_factory() as session:
        persisted = await session.get(Conversation, conversation_id)
        assert persisted is not None
        active_run_count = await session.scalar(
            select(func.count())
            .select_from(Run)
            .where(
                Run.conversation_id == conversation_id,
                Run.status.in_(("PENDING", "RUNNING")),
            )
        )

    if deleted is True:
        assert isinstance(admission, ConversationNotFoundError)
        assert persisted.deleted_at is not None
        assert active_run_count == 0
    else:
        assert isinstance(deleted, ConversationHasActiveRunError)
        assert isinstance(admission, expected_admission_type)
        assert persisted.deleted_at is None
        assert active_run_count == 1


def test_delete_and_retry_serialize_to_one_legal_outcome(
    migrated_database: str,
) -> None:
    _bootstrap(migrated_database)

    async def operation(session_factory: async_sessionmaker[AsyncSession]) -> None:
        conversation = await _create_conversation(session_factory)
        original = await _admit_new(session_factory, conversation_id=conversation.id)
        await _start_running(
            session_factory, conversation_id=conversation.id, run_id=original.run.id
        )
        failed = await mark_run_failed_if_running(
            session_factory,
            conversation_id=conversation.id,
            run_id=original.run.id,
            error_code="LLM_PROVIDER_FAILED",
        )
        assert failed is True

        async def retry() -> RetryAdmission:
            async with session_factory() as session:
                return await admit_retry(
                    session,
                    user_id=1,
                    conversation_id=conversation.id,
                    client_request_id="delete-retry-race",
                )

        await _assert_delete_command_race(
            session_factory,
            conversation_id=conversation.id,
            command=retry,
            expected_admission_type=RetryAdmission,
        )

    asyncio.run(_exercise(migrated_database, operation))


def test_delete_and_regenerate_serialize_to_one_legal_outcome(
    migrated_database: str,
) -> None:
    _bootstrap(migrated_database)

    async def operation(session_factory: async_sessionmaker[AsyncSession]) -> None:
        conversation = await _create_conversation(session_factory)
        original = await _admit_new(session_factory, conversation_id=conversation.id)
        await _start_running(
            session_factory, conversation_id=conversation.id, run_id=original.run.id
        )
        await _commit_success(
            session_factory,
            conversation_id=conversation.id,
            run_id=original.run.id,
            content="回答",
        )

        async def regenerate() -> RegenerateAdmission:
            async with session_factory() as session:
                return await admit_regenerate(
                    session,
                    user_id=1,
                    conversation_id=conversation.id,
                    client_request_id="delete-regenerate-race",
                )

        await _assert_delete_command_race(
            session_factory,
            conversation_id=conversation.id,
            command=regenerate,
            expected_admission_type=RegenerateAdmission,
        )

    asyncio.run(_exercise(migrated_database, operation))
