"""Focused contracts for structured Conversation compact state and estimation."""

import asyncio
import json

import pytest

from langley.answering.contracts import (
    LLMFinishReason,
    LLMResponseCompleted,
    LLMUsage,
)
from langley.answering.conversation_context import (
    ConversationCompactionInvalidOutputError,
    ConversationContextMessage,
    LLMConversationCompactor,
    estimate_message_tokens,
    estimate_multilingual_tokens,
    render_conversation_compact_context,
)
from langley.answering.fake_provider import FakeProvider, ScriptedProviderRound


def _state_payload(
    *,
    fact: str = "The project codename is ORION; APOLLO was rejected.",
    source_message_ids: list[int] | None = None,
) -> dict[str, object]:
    return {
        "current_goals": [],
        "active_decisions": [
            {
                "content": fact,
                "source_message_ids": source_message_ids or [1],
            }
        ],
        "active_constraints": [],
        "open_loops": [],
        "important_facts": [],
        "artifacts": [],
    }


def _completion(payload: dict[str, object]) -> LLMResponseCompleted:
    return LLMResponseCompleted(
        assistant_content=json.dumps(payload),
        tool_calls=(),
        finish_reason=LLMFinishReason.STOP,
        usage=LLMUsage(input_tokens=91, output_tokens=23),
        provider_model="qwen-compactor-observed",
    )


def test_multilingual_estimator_is_explicit_and_conservative() -> None:
    assert estimate_multilingual_tokens("abcdefgh") == 2
    assert estimate_multilingual_tokens("你好世界") == 4
    assert estimate_multilingual_tokens("abcd你好") == 3
    assert estimate_message_tokens("abcd") == 5


def test_compactor_sends_only_structured_detached_history_and_validates_usage() -> None:
    provider = FakeProvider(
        [ScriptedProviderRound(events=(_completion(_state_payload()),))]
    )
    compactor = LLMConversationCompactor(
        provider=provider,
        model="qwen-compactor-configured",
        compact_state_target_estimate=2_000,
    )

    result = asyncio.run(
        compactor.compact(
            previous_state=None,
            newly_aged_out_messages=(
                ConversationContextMessage(1, "USER", "Codename is ORION."),
                ConversationContextMessage(2, "ASSISTANT", "Recorded."),
            ),
        )
    )

    assert result.usage == LLMUsage(input_tokens=91, output_tokens=23)
    assert result.provider_model == "qwen-compactor-observed"
    assert result.state.source_message_ids == frozenset({1})
    request = provider.requests[0]
    assert request.allowed_tools == ()
    assert request.current_user_message_index is None
    assert "Do not\nanswer the user" in request.system_input
    assert "Assistant suggestion alone" in request.system_input
    assert "rejected, no longer, correction" in request.system_input
    payload = json.loads(request.transcript[0].content)
    assert payload["previous_compact_state"] is None
    assert [
        message["message_id"] for message in payload["newly_aged_out_messages"]
    ] == [
        1,
        2,
    ]


def test_compactor_rejects_invented_provenance_as_one_invalid_result() -> None:
    provider = FakeProvider(
        [
            ScriptedProviderRound(
                events=(_completion(_state_payload(source_message_ids=[999])),)
            )
        ]
    )
    compactor = LLMConversationCompactor(
        provider=provider,
        model="qwen-compactor",
        compact_state_target_estimate=2_000,
    )

    with pytest.raises(
        ConversationCompactionInvalidOutputError,
        match="unsupported source_message_ids",
    ):
        asyncio.run(
            compactor.compact(
                previous_state=None,
                newly_aged_out_messages=(
                    ConversationContextMessage(1, "USER", "fact"),
                    ConversationContextMessage(2, "ASSISTANT", "ack"),
                ),
            )
        )


def test_rendered_state_keeps_supersession_but_hides_harness_provenance() -> None:
    state = LLMConversationCompactor._parse_state(
        json.dumps(
            _state_payload(
                fact="Deployment is Tokyo; Singapore is no longer current.",
                source_message_ids=[3],
            )
        ),
        allowed_source_ids={1, 2, 3, 4},
    )

    rendered = render_conversation_compact_context(state)

    assert "Deployment is Tokyo" in rendered
    assert "Singapore is no longer current" in rendered
    assert "source_message_ids" not in rendered
    assert "[3]" not in rendered
