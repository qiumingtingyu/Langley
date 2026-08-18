"""Deterministic contract tests for the Slice 4 FakeProvider."""

import asyncio

import pytest

from langley.answering.contracts import (
    AssistantContentDelta,
    LLMFinishReason,
    LLMRequest,
    LLMResponseCompleted,
    LLMUsage,
    ToolCall,
    UserRuntimeMessage,
)
from langley.answering.errors import RunErrorCode, WorkflowFailure
from langley.answering.fake_provider import FakeProvider, ScriptedProviderRound


async def _collect(provider: FakeProvider, request: LLMRequest) -> list[object]:
    return [event async for event in provider.stream(request)]


def _request() -> LLMRequest:
    return LLMRequest(
        system_input="system",
        transcript=(UserRuntimeMessage(content="question"),),
        allowed_tools=(),
    )


def _completion(content: str) -> LLMResponseCompleted:
    return LLMResponseCompleted(
        assistant_content=content,
        tool_calls=(),
        finish_reason=LLMFinishReason.STOP,
        usage=None,
    )


def test_fake_provider_consumes_scripted_rounds_and_captures_requests() -> None:
    request = _request()
    provider = FakeProvider(
        [
            ScriptedProviderRound(
                events=(AssistantContentDelta(content="hel"), _completion("hello"))
            )
        ]
    )

    events = asyncio.run(_collect(provider, request))

    assert events == [AssistantContentDelta(content="hel"), _completion("hello")]
    assert provider.requests == [request]


def test_fake_provider_fails_loudly_when_the_script_is_exhausted() -> None:
    provider = FakeProvider([])

    with pytest.raises(AssertionError, match="script exhausted"):
        asyncio.run(_collect(provider, _request()))


def test_fake_provider_can_fail_after_a_partial_delta() -> None:
    provider = FakeProvider(
        [
            ScriptedProviderRound(
                events=(AssistantContentDelta(content="partial"),),
                failure=WorkflowFailure(RunErrorCode.LLM_PROVIDER_FAILED),
            )
        ]
    )

    async def exercise() -> list[object]:
        events: list[object] = []
        with pytest.raises(WorkflowFailure) as raised:
            async for event in provider.stream(_request()):
                events.append(event)
        assert raised.value.error_code is RunErrorCode.LLM_PROVIDER_FAILED
        return events

    assert asyncio.run(exercise()) == [AssistantContentDelta(content="partial")]


def test_fake_provider_can_script_tool_calls_and_usage() -> None:
    provider = FakeProvider(
        [
            ScriptedProviderRound(
                events=(
                    LLMResponseCompleted(
                        assistant_content="",
                        tool_calls=(
                            ToolCall(
                                call_id="call-1",
                                name="get_current_time",
                                raw_arguments='{"timezone":"UTC"}',
                            ),
                        ),
                        finish_reason=LLMFinishReason.TOOL_CALLS,
                        usage=LLMUsage(input_tokens=10, output_tokens=2),
                    ),
                )
            )
        ]
    )

    events = asyncio.run(_collect(provider, _request()))

    assert isinstance(events[0], LLMResponseCompleted)
    assert events[0].tool_calls[0].call_id == "call-1"
    assert events[0].usage == LLMUsage(input_tokens=10, output_tokens=2)


def test_fake_provider_propagates_cancellation_while_blocked() -> None:
    started = asyncio.Event()
    blocked_until = asyncio.Event()
    provider = FakeProvider(
        [
            ScriptedProviderRound(
                events=(_completion("never"),),
                started=started,
                blocked_until=blocked_until,
            )
        ]
    )

    async def exercise() -> None:
        task = asyncio.create_task(_collect(provider, _request()))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())
