"""Offline protocol tests for the Qwen OpenAI-compatible adapter."""

import asyncio
import json
from collections.abc import Callable

import httpx
import pytest
from pydantic import SecretStr

from langley.answering.contracts import (
    AssistantContentDelta,
    AssistantRuntimeMessage,
    LLMFinishReason,
    LLMRequest,
    LLMResponseCompleted,
    ToolCall,
    ToolResult,
    ToolResultKind,
    ToolSpec,
    UserRuntimeMessage,
)
from langley.answering.errors import RunErrorCode, WorkflowFailure
from langley.infrastructure.qwen_provider import QwenProvider


def _sse(*events: object) -> bytes:
    lines = [
        f"data: {json.dumps(event, separators=(',', ':'))}\n\n" for event in events
    ]
    return ("".join(lines) + "data: [DONE]\n\n").encode()


def _request() -> LLMRequest:
    call = ToolCall(
        call_id="prior-call",
        name="get_current_time",
        raw_arguments='{"timezone":"UTC"}',
    )
    result = ToolResult(
        call_id="prior-call",
        name="get_current_time",
        kind=ToolResultKind.SUCCESS,
        content='{"timezone":"UTC","datetime":"2026-08-14T00:00:00+00:00"}',
    )
    return LLMRequest(
        system_input="system prompt",
        transcript=(
            UserRuntimeMessage(content="question"),
            AssistantRuntimeMessage(content="checking", tool_calls=(call,)),
            result,
        ),
        allowed_tools=(
            ToolSpec(
                name="get_current_time",
                description="Get time.",
                arguments_schema={"type": "object"},
            ),
        ),
    )


def _provider(handler: Callable[[httpx.Request], httpx.Response]) -> QwenProvider:
    return QwenProvider(
        api_key=SecretStr("test-key"),
        base_url="https://qwen.example.test/v1/",
        model="qwen-test-model",
        client_factory=lambda: httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ),
    )


async def _collect(provider: QwenProvider, request: LLMRequest) -> list[object]:
    return [event async for event in provider.stream(request)]


class _BlockingStream(httpx.AsyncByteStream):
    def __init__(self, started: asyncio.Event, released: asyncio.Event) -> None:
        self._started = started
        self._released = released

    async def __aiter__(self):
        self._started.set()
        await self._released.wait()
        yield b"data: [DONE]\n\n"

    async def aclose(self) -> None:
        return None


def test_qwen_provider_maps_normalized_request_and_disables_thinking() -> None:
    received: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        received.append(request)
        return httpx.Response(
            200,
            content=_sse(
                {"choices": [{"delta": {"content": "answer"}, "finish_reason": "stop"}]}
            ),
        )

    events = asyncio.run(_collect(_provider(handler), _request()))

    payload = json.loads(received[0].content)
    assert received[0].url == "https://qwen.example.test/v1/chat/completions"
    assert received[0].headers["authorization"] == "Bearer test-key"
    assert payload["model"] == "qwen-test-model"
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}
    assert payload["enable_thinking"] is False
    assert payload["parallel_tool_calls"] is False
    assert payload["messages"][1] == {"role": "user", "content": "question"}
    assert payload["messages"][2]["tool_calls"][0]["id"] == "prior-call"
    assert payload["messages"][3] == {
        "role": "tool",
        "tool_call_id": "prior-call",
        "content": '{"timezone":"UTC","datetime":"2026-08-14T00:00:00+00:00"}',
    }
    assert payload["tools"][0]["function"]["name"] == "get_current_time"
    assert events == [
        AssistantContentDelta(content="answer"),
        LLMResponseCompleted(
            assistant_content="answer",
            tool_calls=(),
            finish_reason=LLMFinishReason.STOP,
            usage=None,
        ),
    ]


@pytest.mark.parametrize(
    ("personal_context", "status", "expected_context"),
    [
        (("prefers concise answers",), "available", ["prefers concise answers"]),
        ((), "available", []),
        (None, "unavailable", None),
    ],
)
def test_qwen_provider_wraps_only_the_current_user_with_personal_context(
    personal_context: tuple[str, ...] | None,
    status: str,
    expected_context: list[str] | None,
) -> None:
    request = LLMRequest(
        system_input="system prompt",
        transcript=(
            UserRuntimeMessage(content="historical user"),
            AssistantRuntimeMessage(content="historical assistant", tool_calls=()),
            UserRuntimeMessage(content="current user"),
        ),
        allowed_tools=(),
        personal_context=personal_context,
        current_user_message_index=2,
    )

    payload = _provider(lambda _: httpx.Response(200))._request_payload(request)

    assert payload["messages"][1] == {"role": "user", "content": "historical user"}
    envelope = json.loads(payload["messages"][3]["content"])
    assert envelope == {
        "personal_context_status": status,
        "personal_context": expected_context,
        "current_user_request": "current user",
    }


def test_qwen_provider_losslessly_aggregates_fragmented_malformed_tool_arguments() -> (
    None
):
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_sse(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call-1",
                                        "function": {
                                            "name": "get_current_time",
                                            "arguments": '{"time',
                                        },
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {"arguments": 'zone":}'},
                                    }
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                },
            ),
        )

    events = asyncio.run(_collect(_provider(handler), _request()))

    assert events == [
        LLMResponseCompleted(
            assistant_content="",
            tool_calls=(
                ToolCall(
                    call_id="call-1",
                    name="get_current_time",
                    raw_arguments='{"timezone":}',
                ),
            ),
            finish_reason=LLMFinishReason.TOOL_CALLS,
            usage=None,
        )
    ]


def test_qwen_provider_accepts_qwen_empty_id_tool_fragments() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_sse(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call-1",
                                        "function": {
                                            "name": "get_current_time",
                                            "arguments": '{"time',
                                        },
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "",
                                        "function": {"arguments": 'zone":"UTC"}'},
                                    }
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                },
            ),
        )

    events = asyncio.run(_collect(_provider(handler), _request()))

    assert events[-1] == LLMResponseCompleted(
        assistant_content="",
        tool_calls=(
            ToolCall(
                call_id="call-1",
                name="get_current_time",
                raw_arguments='{"timezone":"UTC"}',
            ),
        ),
        finish_reason=LLMFinishReason.TOOL_CALLS,
        usage=None,
    )


def test_qwen_provider_normalizes_optional_usage() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_sse(
                {
                    "choices": [
                        {"delta": {"content": "answer"}, "finish_reason": "stop"}
                    ]
                },
                {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 2}},
            ),
        )

    events = asyncio.run(_collect(_provider(handler), _request()))

    assert isinstance(events[-1], LLMResponseCompleted)
    assert events[-1].usage is not None
    assert events[-1].usage.input_tokens == 10
    assert events[-1].usage.output_tokens == 2


@pytest.mark.parametrize(
    "usage", [{"prompt_tokens": "ten"}, [], {"completion_tokens": 1.5}]
)
def test_qwen_provider_ignores_malformed_optional_usage(usage: object) -> None:
    events = asyncio.run(
        _collect(
            _provider(
                lambda _: httpx.Response(
                    200,
                    content=_sse(
                        {
                            "choices": [
                                {
                                    "delta": {"content": "answer"},
                                    "finish_reason": "stop",
                                }
                            ]
                        },
                        {"choices": [], "usage": usage},
                    ),
                )
            ),
            _request(),
        )
    )

    assert isinstance(events[-1], LLMResponseCompleted)
    assert events[-1].usage is None


def test_qwen_provider_normalizes_midstream_failure_without_replaying() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            content=(
                'data: {"choices":[{"delta":{"content":"partial"},'
                '"finish_reason":null}]}\n\n'
                'data: {"error":{"message":"unavailable"}}\n\n'
            ).encode(),
        )

    async def exercise() -> list[object]:
        events: list[object] = []
        with pytest.raises(WorkflowFailure) as raised:
            async for event in _provider(handler).stream(_request()):
                events.append(event)
        assert raised.value.error_code is RunErrorCode.LLM_PROVIDER_FAILED
        return events

    assert asyncio.run(exercise()) == [AssistantContentDelta(content="partial")]
    assert calls == 1


def test_qwen_provider_normalizes_failure_before_the_first_delta() -> None:
    provider = _provider(lambda _: httpx.Response(503, text="unavailable"))

    async def exercise() -> None:
        with pytest.raises(WorkflowFailure) as raised:
            await _collect(provider, _request())
        assert raised.value.error_code is RunErrorCode.LLM_PROVIDER_FAILED

    asyncio.run(exercise())


def test_qwen_provider_maps_explicit_context_capacity_rejection() -> None:
    provider = _provider(
        lambda _: httpx.Response(
            400,
            json={
                "error": {
                    "code": "context_length_exceeded",
                    "message": "input exceeds maximum context length",
                }
            },
        )
    )

    async def exercise() -> None:
        with pytest.raises(WorkflowFailure) as raised:
            await _collect(provider, _request())
        assert raised.value.error_code is RunErrorCode.ANSWER_CONTEXT_TOO_LARGE
        assert "context_length_exceeded" not in str(raised.value)

    asyncio.run(exercise())


def test_qwen_provider_rejects_duplicate_canonical_tool_call_identity() -> None:
    def tool_delta(index: int) -> dict[str, object]:
        return {
            "index": index,
            "id": "duplicate-call",
            "function": {
                "name": "get_current_time",
                "arguments": '{"timezone":"UTC"}',
            },
        }

    provider = _provider(
        lambda _: httpx.Response(
            200,
            content=_sse(
                {
                    "choices": [
                        {
                            "delta": {"tool_calls": [tool_delta(0), tool_delta(1)]},
                            "finish_reason": "tool_calls",
                        }
                    ]
                }
            ),
        )
    )

    async def exercise() -> None:
        with pytest.raises(WorkflowFailure) as raised:
            await _collect(provider, _request())
        assert raised.value.error_code is RunErrorCode.LLM_RESPONSE_INVALID

    asyncio.run(exercise())


def test_qwen_provider_propagates_cancellation_while_the_stream_is_blocked() -> None:
    started = asyncio.Event()
    released = asyncio.Event()
    provider = _provider(
        lambda _: httpx.Response(200, stream=_BlockingStream(started, released))
    )

    async def exercise() -> None:
        task = asyncio.create_task(_collect(provider, _request()))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())
