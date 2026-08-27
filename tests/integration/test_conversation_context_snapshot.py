"""Real-MySQL lifecycle for the rebuildable Conversation Context snapshot."""

import asyncio
import json
from argparse import Namespace

from alembic import command
from alembic.config import Config
from sqlalchemy import select

from langley.answering.contracts import LLMFinishReason, LLMResponseCompleted
from langley.answering.conversation_context import LLMConversationCompactor
from langley.answering.conversation_context_builder import ConversationContextBuilder
from langley.answering.fake_provider import FakeProvider, ScriptedProviderRound
from langley.business_time import utc_now
from langley.infrastructure.database import (
    create_database_engine,
    create_session_factory,
    dispose_database_engine,
)
from langley.infrastructure.models import (
    Conversation,
    ConversationContextSnapshot,
    Message,
    Run,
    User,
)


def _payload(decisions: list[tuple[str, list[int]]]) -> str:
    return json.dumps(
        {
            "current_goals": [],
            "active_decisions": [
                {"content": content, "source_message_ids": source_ids}
                for content, source_ids in decisions
            ],
            "active_constraints": [],
            "open_loops": [],
            "important_facts": [],
            "artifacts": [],
        }
    )


def _completion(content: str) -> LLMResponseCompleted:
    return LLMResponseCompleted(
        assistant_content=content,
        tool_calls=(),
        finish_reason=LLMFinishReason.STOP,
        usage=None,
    )


def test_snapshot_is_created_then_incrementally_upserted_without_mutating_messages(
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
            async with session_factory() as session, session.begin():
                now = utc_now()
                session.add(User(id=1, created_at=now))
                await session.flush()
                conversation = Conversation(
                    user_id=1,
                    title=None,
                    created_at=now,
                    updated_at=now,
                    last_message_at=now,
                    deleted_at=None,
                )
                session.add(conversation)
                await session.flush()

                orion_user = Message(
                    conversation_id=conversation.id,
                    sequence_no=1,
                    role="USER",
                    content=(
                        "The current project codename is ORION; APOLLO is rejected."
                    ),
                    run_id=None,
                    regenerated_from_message_id=None,
                    created_at=now,
                )
                session.add(orion_user)
                await session.flush()
                orion_run = Run(
                    conversation_id=conversation.id,
                    input_message_id=orion_user.id,
                    client_request_id="orion",
                    attempt_no=1,
                    status="SUCCEEDED",
                    started_at=now,
                    finished_at=now,
                    error_code=None,
                    created_at=now,
                    updated_at=now,
                )
                session.add(orion_run)
                await session.flush()
                orion_assistant = Message(
                    conversation_id=conversation.id,
                    sequence_no=2,
                    role="ASSISTANT",
                    content="Recorded ORION.",
                    run_id=orion_run.id,
                    regenerated_from_message_id=None,
                    created_at=now,
                )
                session.add(orion_assistant)

                canary_user = Message(
                    conversation_id=conversation.id,
                    sequence_no=3,
                    role="USER",
                    content="CANARY is final; BLUE-GREEN was rejected.",
                    run_id=None,
                    regenerated_from_message_id=None,
                    created_at=now,
                )
                session.add(canary_user)
                await session.flush()
                canary_run = Run(
                    conversation_id=conversation.id,
                    input_message_id=canary_user.id,
                    client_request_id="canary",
                    attempt_no=1,
                    status="SUCCEEDED",
                    started_at=now,
                    finished_at=now,
                    error_code=None,
                    created_at=now,
                    updated_at=now,
                )
                session.add(canary_run)
                await session.flush()
                canary_assistant = Message(
                    conversation_id=conversation.id,
                    sequence_no=4,
                    role="ASSISTANT",
                    content="Recorded CANARY.",
                    run_id=canary_run.id,
                    regenerated_from_message_id=None,
                    created_at=now,
                )
                question_one = Message(
                    conversation_id=conversation.id,
                    sequence_no=5,
                    role="USER",
                    content="What decisions are current?",
                    run_id=None,
                    regenerated_from_message_id=None,
                    created_at=now,
                )
                session.add_all((canary_assistant, question_one))
                await session.flush()
                ids = {
                    "conversation": conversation.id,
                    "orion_user": orion_user.id,
                    "orion_assistant": orion_assistant.id,
                    "canary_user": canary_user.id,
                    "canary_assistant": canary_assistant.id,
                    "question_one": question_one.id,
                }

            provider = FakeProvider(
                [
                    ScriptedProviderRound(
                        events=(
                            _completion(
                                _payload(
                                    [
                                        (
                                            "Project codename is ORION; "
                                            "APOLLO was rejected.",
                                            [ids["orion_user"]],
                                        )
                                    ]
                                )
                            ),
                        )
                    ),
                    ScriptedProviderRound(
                        events=(
                            _completion(
                                _payload(
                                    [
                                        (
                                            "Project codename is ORION; "
                                            "APOLLO was rejected.",
                                            [ids["orion_user"]],
                                        ),
                                        (
                                            "Production deployment strategy is CANARY; "
                                            "BLUE-GREEN was rejected.",
                                            [ids["canary_user"]],
                                        ),
                                    ]
                                )
                            ),
                        )
                    ),
                ]
            )
            builder = ConversationContextBuilder(
                working_context_budget_estimate=80,
                conversation_compaction_trigger_estimate=25,
                recent_raw_target_estimate=12,
                compact_state_target_estimate=10,
                compactor=LLMConversationCompactor(
                    provider=provider,
                    model="fake-compactor",
                    compact_state_target_estimate=10,
                ),
            )

            first = await builder.build(
                session_factory,
                conversation_id=ids["conversation"],
                current_user_message_id=ids["question_one"],
            )
            assert "ORION" in (first.conversation_compact_context or "")
            assert [turn.user_content for turn in first.completed_turns] == [
                "CANARY is final; BLUE-GREEN was rejected."
            ]
            async with session_factory() as session:
                first_snapshot = await session.get(
                    ConversationContextSnapshot, ids["conversation"]
                )
                assert first_snapshot is not None
                assert first_snapshot.through_message_id == ids["orion_assistant"]

            async with session_factory() as session, session.begin():
                now = utc_now()
                question_run = Run(
                    conversation_id=ids["conversation"],
                    input_message_id=ids["question_one"],
                    client_request_id="question-one",
                    attempt_no=1,
                    status="SUCCEEDED",
                    started_at=now,
                    finished_at=now,
                    error_code=None,
                    created_at=now,
                    updated_at=now,
                )
                session.add(question_run)
                await session.flush()
                session.add(
                    Message(
                        conversation_id=ids["conversation"],
                        sequence_no=6,
                        role="ASSISTANT",
                        content="ORION and CANARY are current.",
                        run_id=question_run.id,
                        regenerated_from_message_id=None,
                        created_at=now,
                    )
                )
                question_two = Message(
                    conversation_id=ids["conversation"],
                    sequence_no=7,
                    role="USER",
                    content="Repeat the deployment strategy.",
                    run_id=None,
                    regenerated_from_message_id=None,
                    created_at=now,
                )
                session.add(question_two)
                await session.flush()
                question_two_id = question_two.id

            second = await builder.build(
                session_factory,
                conversation_id=ids["conversation"],
                current_user_message_id=question_two_id,
            )
            assert "ORION" in (second.conversation_compact_context or "")
            assert "CANARY" in (second.conversation_compact_context or "")
            assert [turn.user_content for turn in second.completed_turns] == [
                "What decisions are current?"
            ]

            async with session_factory() as session:
                updated_snapshot = await session.get(
                    ConversationContextSnapshot, ids["conversation"]
                )
                assert updated_snapshot is not None
                assert updated_snapshot.through_message_id == ids["canary_assistant"]
                messages = tuple(
                    (
                        await session.scalars(
                            select(Message).where(
                                Message.conversation_id == ids["conversation"]
                            )
                        )
                    ).all()
                )
                assert len(messages) == 7
                assert {message.content for message in messages} >= {
                    "The current project codename is ORION; APOLLO is rejected.",
                    "CANARY is final; BLUE-GREEN was rejected.",
                    "What decisions are current?",
                }
        finally:
            await dispose_database_engine(engine)

    asyncio.run(exercise())
