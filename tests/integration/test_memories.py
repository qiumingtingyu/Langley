"""Real MySQL tests for current Personal Context Memory reads."""

import asyncio
from argparse import Namespace
from dataclasses import dataclass
from datetime import datetime, timedelta

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from langley.infrastructure.database import (
    create_database_engine,
    create_session_factory,
    dispose_database_engine,
)
from langley.infrastructure.models import Conversation, Memory, Message, Run, User
from langley.memories import (
    get_memory_source_context,
    get_owned_current_memory,
    list_current_memories,
)


@dataclass(frozen=True)
class SeededMemoryFacts:
    user_id: int
    other_user_id: int
    direct_memory_id: int
    future_memory_id: int
    source_memory_id: int
    expired_memory_id: int
    regenerated_source_memory_id: int
    assistant_source_memory_id: int
    conversation_id: int


@pytest.fixture
def migrated_database(test_database_url: str, reset_database) -> str:
    """Provide an empty real MySQL database migrated to the current head."""
    reset_database()
    config = Config("alembic.ini")
    config.cmd_opts = Namespace(x=["use_test_database=true"])
    command.upgrade(config, "head")
    return test_database_url


async def _exercise(
    database_url: str,
    operation,
):
    engine = create_database_engine(database_url)
    try:
        return await operation(create_session_factory(engine))
    finally:
        await dispose_database_engine(engine)


async def _seed_memory_facts(
    session_factory: async_sessionmaker[AsyncSession],
) -> SeededMemoryFacts:
    now = datetime(2026, 8, 20, 2, 0)
    async with session_factory() as session:
        async with session.begin():
            user = User(created_at=now)
            other_user = User(created_at=now)
            session.add_all((user, other_user))
            await session.flush()

            conversation = Conversation(
                user_id=user.id,
                title="已删除的来源会话",
                created_at=now,
                updated_at=now,
                last_message_at=now,
                deleted_at=now,
            )
            session.add(conversation)
            await session.flush()

            previous_message = Message(
                conversation_id=conversation.id,
                sequence_no=1,
                role="USER",
                content="前一条",
                run_id=None,
                regenerated_from_message_id=None,
                created_at=now,
                memory_processed_at=None,
            )
            source_message = Message(
                conversation_id=conversation.id,
                sequence_no=2,
                role="USER",
                content="来源用户消息",
                run_id=None,
                regenerated_from_message_id=None,
                created_at=now,
                memory_processed_at=None,
            )
            next_message = Message(
                conversation_id=conversation.id,
                sequence_no=3,
                role="USER",
                content="后一条",
                run_id=None,
                regenerated_from_message_id=None,
                created_at=now,
                memory_processed_at=None,
            )
            session.add_all((previous_message, source_message, next_message))
            await session.flush()

            regenerated_message = Message(
                conversation_id=conversation.id,
                sequence_no=4,
                role="USER",
                content="重新生成的复制消息",
                run_id=None,
                regenerated_from_message_id=source_message.id,
                created_at=now,
                memory_processed_at=None,
            )
            run = Run(
                conversation_id=conversation.id,
                input_message_id=previous_message.id,
                client_request_id="memory-source-role-check",
                attempt_no=1,
                status="SUCCEEDED",
                started_at=now,
                finished_at=now,
                error_code=None,
                created_at=now,
                updated_at=now,
            )
            session.add_all((regenerated_message, run))
            await session.flush()

            assistant_message = Message(
                conversation_id=conversation.id,
                sequence_no=5,
                role="ASSISTANT",
                content="Assistant 不能作为 Memory provenance",
                run_id=run.id,
                regenerated_from_message_id=None,
                created_at=now,
                memory_processed_at=None,
            )
            session.add(assistant_message)
            await session.flush()

            source_memory = Memory(
                user_id=user.id,
                content="来源记忆",
                source_message_id=source_message.id,
                valid_until=None,
                created_at=now,
                updated_at=now + timedelta(minutes=1),
            )
            future_memory = Memory(
                user_id=user.id,
                content="未来有效记忆",
                source_message_id=None,
                valid_until=now + timedelta(hours=1),
                created_at=now,
                updated_at=now + timedelta(minutes=2),
            )
            direct_memory = Memory(
                user_id=user.id,
                content="直接设置记忆",
                source_message_id=None,
                valid_until=None,
                created_at=now,
                updated_at=now + timedelta(minutes=3),
            )
            expired_memory = Memory(
                user_id=user.id,
                content="已过期记忆",
                source_message_id=None,
                valid_until=now,
                created_at=now,
                updated_at=now + timedelta(minutes=4),
            )
            regenerated_source_memory = Memory(
                user_id=user.id,
                content="无效 regenerate provenance",
                source_message_id=regenerated_message.id,
                valid_until=None,
                created_at=now,
                updated_at=now,
            )
            assistant_source_memory = Memory(
                user_id=user.id,
                content="无效 assistant provenance",
                source_message_id=assistant_message.id,
                valid_until=None,
                created_at=now,
                updated_at=now,
            )
            other_memory = Memory(
                user_id=other_user.id,
                content="其他用户记忆",
                source_message_id=None,
                valid_until=None,
                created_at=now,
                updated_at=now + timedelta(minutes=5),
            )
            session.add_all(
                (
                    source_memory,
                    future_memory,
                    direct_memory,
                    expired_memory,
                    regenerated_source_memory,
                    assistant_source_memory,
                    other_memory,
                )
            )
            await session.flush()

            return SeededMemoryFacts(
                user_id=user.id,
                other_user_id=other_user.id,
                direct_memory_id=direct_memory.id,
                future_memory_id=future_memory.id,
                source_memory_id=source_memory.id,
                expired_memory_id=expired_memory.id,
                regenerated_source_memory_id=regenerated_source_memory.id,
                assistant_source_memory_id=assistant_source_memory.id,
                conversation_id=conversation.id,
            )


def test_current_memory_reads_filter_expiry_ownership_and_sorting(
    migrated_database: str,
) -> None:
    now = datetime(2026, 8, 20, 2, 0)

    async def operation(
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        facts = await _seed_memory_facts(session_factory)
        async with session_factory() as session:
            current = await list_current_memories(
                session, user_id=facts.user_id, now=now
            )
            owned_direct = await get_owned_current_memory(
                session,
                user_id=facts.user_id,
                memory_id=facts.direct_memory_id,
                now=now,
            )
            expired = await get_owned_current_memory(
                session,
                user_id=facts.user_id,
                memory_id=facts.expired_memory_id,
                now=now,
            )
            wrong_user = await get_owned_current_memory(
                session,
                user_id=facts.other_user_id,
                memory_id=facts.direct_memory_id,
                now=now,
            )
            nonexistent = await get_owned_current_memory(
                session,
                user_id=facts.user_id,
                memory_id=999_999,
                now=now,
            )

        assert [memory.id for memory in current] == [
            facts.direct_memory_id,
            facts.future_memory_id,
            facts.source_memory_id,
            facts.assistant_source_memory_id,
            facts.regenerated_source_memory_id,
        ]
        assert owned_direct is not None and owned_direct.id == facts.direct_memory_id
        assert expired is None
        assert wrong_user is None
        assert nonexistent is None

    asyncio.run(_exercise(migrated_database, operation))


def test_memory_source_context_uses_same_conversation_sequence_window(
    migrated_database: str,
) -> None:
    async def operation(
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        facts = await _seed_memory_facts(session_factory)
        async with session_factory() as session:
            source_memory = await session.get(Memory, facts.source_memory_id)
            direct_memory = await session.get(Memory, facts.direct_memory_id)
            regenerated_source_memory = await session.get(
                Memory, facts.regenerated_source_memory_id
            )
            assistant_source_memory = await session.get(
                Memory, facts.assistant_source_memory_id
            )
            assert source_memory is not None
            assert direct_memory is not None
            assert regenerated_source_memory is not None
            assert assistant_source_memory is not None

            context = await get_memory_source_context(
                session, user_id=facts.user_id, memory=source_memory
            )
            direct_context = await get_memory_source_context(
                session, user_id=facts.user_id, memory=direct_memory
            )
            regenerated_context = await get_memory_source_context(
                session, user_id=facts.user_id, memory=regenerated_source_memory
            )
            assistant_context = await get_memory_source_context(
                session, user_id=facts.user_id, memory=assistant_source_memory
            )

        assert context is not None
        assert context.conversation.id == facts.conversation_id
        assert context.conversation.deleted_at is not None
        assert context.source_message.content == "来源用户消息"
        assert [message.content for message in context.context_messages] == [
            "前一条",
            "来源用户消息",
            "后一条",
        ]
        assert direct_context is None
        assert regenerated_context is None
        assert assistant_context is None

    asyncio.run(_exercise(migrated_database, operation))
