"""Narrow shared helpers for real MySQL Memory integration tests."""

import asyncio
import json
from argparse import Namespace

from alembic import command
from alembic.config import Config
from sqlalchemy import text

from langley.answering.contracts import LLMFinishReason, LLMResponseCompleted
from langley.answering.fake_provider import FakeProvider, ScriptedProviderRound
from langley.business_time import utc_now
from langley.infrastructure.database import (
    create_database_engine,
    create_session_factory,
    dispose_database_engine,
)
from langley.infrastructure.models import Conversation, Memory, Message, User
from langley.memory.policy import MemoryPolicy, MemoryPolicyResult
from langley.memory.processing import process_memory_through


def migrated_database(test_database_url: str, reset_database) -> str:
    reset_database()
    config = Config("alembic.ini")
    config.cmd_opts = Namespace(x=["use_test_database=true"])
    command.upgrade(config, "head")
    return test_database_url


def _completion(payload: dict[str, object] | str) -> LLMResponseCompleted:
    return LLMResponseCompleted(
        assistant_content=payload if isinstance(payload, str) else json.dumps(payload),
        tool_calls=(),
        finish_reason=LLMFinishReason.STOP,
        usage=None,
    )


def _policy(rounds: list[ScriptedProviderRound]) -> MemoryPolicy:
    return MemoryPolicy(
        provider=FakeProvider(rounds), memory_policy_estimated_token_budget=10_000
    )


def _seed_evidence(database_url: str, contents: list[str]) -> tuple[int, list[int]]:
    async def seed() -> tuple[int, list[int]]:
        engine = create_database_engine(database_url)
        session_factory = create_session_factory(engine)
        try:
            async with session_factory() as session, session.begin():
                now = utc_now()
                user = User(id=1, created_at=now)
                conversation = Conversation(
                    user_id=1,
                    title=None,
                    created_at=now,
                    updated_at=now,
                    last_message_at=now,
                    deleted_at=None,
                )
                session.add(user)
                await session.flush()
                session.add(conversation)
                await session.flush()
                messages = [
                    Message(
                        conversation_id=conversation.id,
                        sequence_no=index,
                        role="USER",
                        content=content,
                        run_id=None,
                        regenerated_from_message_id=None,
                        created_at=now,
                    )
                    for index, content in enumerate(contents, start=1)
                ]
                session.add_all(messages)
                await session.flush()
                return conversation.id, [message.id for message in messages]
        finally:
            await dispose_database_engine(engine)

    return asyncio.run(seed())


def _process(
    database_url: str,
    *,
    through_message_id: int,
    policy: MemoryPolicy,
    limit: int = 4,
    outcome_callback=None,
):
    async def process():
        engine = create_database_engine(database_url)
        session_factory = create_session_factory(engine)
        try:
            return await process_memory_through(
                session_factory,
                user_id=1,
                through_message_id=through_message_id,
                policy=policy,
                local_timezone="Asia/Shanghai",
                lane=asyncio.Lock(),
                limit=limit,
                outcome_callback=outcome_callback,
            )
        finally:
            await dispose_database_engine(engine)

    return asyncio.run(process())


def _scalar(database_url: str, statement: str) -> object:
    async def execute() -> object:
        engine = create_database_engine(database_url)
        try:
            async with engine.connect() as connection:
                return (await connection.execute(text(statement))).scalar_one()
        finally:
            await dispose_database_engine(engine)

    return asyncio.run(execute())


def _run(coroutine):
    return asyncio.run(coroutine)


def _result(payload: dict[str, object]) -> MemoryPolicyResult:
    return MemoryPolicyResult.model_validate(payload)


def _insert_memory(database_url: str, *, content: str = "existing") -> int:
    async def insert() -> int:
        engine = create_database_engine(database_url)
        session_factory = create_session_factory(engine)
        try:
            async with session_factory() as session, session.begin():
                memory = Memory(
                    user_id=1,
                    content=content,
                    source_message_id=None,
                    valid_until=None,
                    created_at=utc_now(),
                    updated_at=utc_now(),
                )
                session.add(memory)
                await session.flush()
                return memory.id
        finally:
            await dispose_database_engine(engine)

    return _run(insert())
