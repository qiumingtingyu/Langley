"""Real MySQL coverage for Slice 5 Generation Recall selection."""

import asyncio
from argparse import Namespace
from datetime import timedelta

from alembic import command
from alembic.config import Config

from langley.answering.conversation_context_builder import ConversationContextBuilder
from langley.business_time import utc_now
from langley.infrastructure.database import (
    create_database_engine,
    create_session_factory,
    dispose_database_engine,
)
from langley.infrastructure.models import Conversation, Memory, Message, Run, User


def test_context_builder_loads_only_owned_current_memories(
    test_database_url: str, reset_database
) -> None:
    reset_database()
    config = Config("alembic.ini")
    config.cmd_opts = Namespace(x=["use_test_database=true"])
    command.upgrade(config, "head")

    async def exercise() -> None:
        engine = create_database_engine(test_database_url)
        session_factory = create_session_factory(engine)
        try:
            now = utc_now()
            async with session_factory() as session, session.begin():
                owner = User(id=1, created_at=now)
                other_user = User(id=2, created_at=now)
                conversation = Conversation(
                    user_id=1,
                    title=None,
                    created_at=now,
                    updated_at=now,
                    last_message_at=now,
                    deleted_at=None,
                )
                session.add_all((owner, other_user))
                await session.flush()
                session.add(conversation)
                await session.flush()
                current = Message(
                    conversation_id=conversation.id,
                    sequence_no=1,
                    role="USER",
                    content="current request",
                    run_id=None,
                    regenerated_from_message_id=None,
                    created_at=now,
                )
                session.add(current)
                await session.flush()
                session.add(
                    Run(
                        conversation_id=conversation.id,
                        input_message_id=current.id,
                        client_request_id="request",
                        attempt_no=1,
                        status="PENDING",
                        started_at=None,
                        finished_at=None,
                        error_code=None,
                        created_at=now,
                        updated_at=now,
                    )
                )
                session.add_all(
                    (
                        Memory(
                            user_id=1,
                            content="owned current",
                            source_message_id=None,
                            valid_until=None,
                            created_at=now,
                            updated_at=now,
                        ),
                        Memory(
                            user_id=1,
                            content="expired",
                            source_message_id=None,
                            valid_until=now - timedelta(seconds=1),
                            created_at=now,
                            updated_at=now,
                        ),
                        Memory(
                            user_id=2,
                            content="other user",
                            source_message_id=None,
                            valid_until=None,
                            created_at=now,
                            updated_at=now,
                        ),
                    )
                )
                conversation_id = conversation.id
                current_message_id = current.id

            context = await ConversationContextBuilder(
                working_context_budget_estimate=16_000,
                conversation_compaction_trigger_estimate=12_000,
                recent_raw_target_estimate=6_000,
                compact_state_target_estimate=2_000,
                memory_estimated_token_budget=8_192,
            ).build(
                session_factory,
                conversation_id=conversation_id,
                current_user_message_id=current_message_id,
            )

            assert [item.content for item in context.personal_context or ()] == [
                "owned current"
            ]
        finally:
            await dispose_database_engine(engine)

    asyncio.run(exercise())
