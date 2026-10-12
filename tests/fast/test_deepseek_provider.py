"""Offline DeepSeek protocol and application composition checks."""

import asyncio
import json

import httpx
import pytest
from pydantic import SecretStr

from langley.answering.context_builder import AnswerContext
from langley.answering.contracts import (
    AssistantContentDelta,
    AssistantRuntimeMessage,
    LLMFinishReason,
    LLMRequest,
    LLMResponseCompleted,
    ToolResult,
    ToolResultKind,
    ToolSpec,
    UserRuntimeMessage,
)
from langley.answering.errors import RunErrorCode, WorkflowFailure
from langley.answering.tools import CurrentTimeTool, ToolExecutor
from langley.answering.workflow import LearningAssistantWorkflow
from langley.infrastructure.deepseek_provider import DeepSeekProvider
from langley.infrastructure.qwen_provider import QwenProvider
from langley.main import (
    _conversation_compactor_provider_for,
    _memory_provider_for,
    _provider_for,
)
from langley.settings import Settings


def _sse(*events):
    return (
        "".join("data: " + json.dumps(event) + "\n\n" for event in events)
        + "data: [DONE]\n\n"
    ).encode()


def _request(tools=()):
    return LLMRequest(
        system_input="system",
        transcript=(UserRuntimeMessage("question"),),
        allowed_tools=tools,
        current_user_message_index=0,
        evidence_context="evidence",
        personal_context=("preference",),
    )


def _provider(handler):
    return DeepSeekProvider(
        api_key=SecretStr("deepseek-test-key"),
        base_url="https://deepseek.example.test/",
        model="deepseek-test-model",
        client_factory=lambda: httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ),
    )


def test_stream_and_tool_result_followup():
    received = []

    def handler(request):
        received.append(request)
        if len(received) == 1:
            return httpx.Response(
                200,
                content=_sse(
                    {
                        "model": "served-model",
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
                        ],
                    },
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
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
        return httpx.Response(
            200,
            content=_sse(
                {"choices": [{"delta": {"content": "answer "}, "finish_reason": None}]},
                {"choices": [{"delta": {"content": "done"}, "finish_reason": None}]},
                {
                    "choices": [{"delta": {}, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": 20,
                        "completion_tokens": 3,
                        "prompt_cache_hit_tokens": 5,
                    },
                },
            ),
        )

    async def exercise():
        provider = _provider(handler)
        first_request = _request(
            (ToolSpec("get_current_time", "Time", {"type": "object"}),)
        )
        first = [e async for e in provider.stream(first_request)]
        completed = first[-1]
        assert isinstance(completed, LLMResponseCompleted)
        assert completed.finish_reason is LLMFinishReason.TOOL_CALLS
        assert completed.provider_model == "served-model"
        assert completed.tool_calls[0].raw_arguments == '{"timezone":"UTC"}'
        followup = LLMRequest(
            system_input=first_request.system_input,
            transcript=first_request.transcript
            + (
                AssistantRuntimeMessage(
                    completed.assistant_content, completed.tool_calls
                ),
                ToolResult(
                    "call-1", "get_current_time", ToolResultKind.SUCCESS, "time"
                ),
            ),
            allowed_tools=first_request.allowed_tools,
            current_user_message_index=0,
        )
        return [e async for e in provider.stream(followup)]

    events = asyncio.run(exercise())
    assert events[:2] == [
        AssistantContentDelta("answer "),
        AssistantContentDelta("done"),
    ]
    final = events[-1]
    assert final.assistant_content == "answer done"
    assert final.finish_reason is LLMFinishReason.STOP
    assert final.usage.input_tokens == 20
    assert final.usage.output_tokens == 3
    assert len(received) == 2
    for request in received:
        assert str(request.url) == "https://deepseek.example.test/chat/completions"
        assert request.headers["authorization"] == "Bearer deepseek-test-key"
        payload = json.loads(request.content)
        assert payload["thinking"] == {"type": "disabled"}
        assert "enable_thinking" not in payload
        assert "parallel_tool_calls" not in payload
        assert payload["stream_options"] == {"include_usage": True}
        assert payload["model"] == "deepseek-test-model"
    first = json.loads(received[0].content)
    envelope = json.loads(first["messages"][1]["content"])
    assert envelope["evidence_context"] == "evidence"
    assert envelope["personal_context"] == ["preference"]
    second = json.loads(received[1].content)
    assert second["messages"][2]["tool_calls"][0]["id"] == "call-1"
    assert second["messages"][3] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": "time",
    }
    assert all("reasoning_content" not in m for m in second["messages"])


def test_no_tool_request_omits_tools():
    received = []

    def handler(request):
        received.append(json.loads(request.content))
        return httpx.Response(
            200,
            content=_sse(
                {
                    "choices": [
                        {
                            "delta": {"content": "ok", "reasoning_content": ""},
                            "finish_reason": "stop",
                        }
                    ]
                },
            ),
        )

    async def exercise():
        return [e async for e in _provider(handler).stream(_request())]

    assert asyncio.run(exercise())[-1].assistant_content == "ok"
    assert "tools" not in received[0]


@pytest.mark.parametrize(
    ("response", "code"),
    [
        (httpx.Response(401), RunErrorCode.LLM_PROVIDER_FAILED),
        (httpx.Response(429), RunErrorCode.LLM_PROVIDER_FAILED),
        (httpx.Response(503), RunErrorCode.LLM_PROVIDER_FAILED),
        (
            httpx.Response(400, json={"error": {"code": "context_length_exceeded"}}),
            RunErrorCode.ANSWER_CONTEXT_TOO_LARGE,
        ),
        (
            httpx.Response(200, content=b"data: invalid\n\n"),
            RunErrorCode.LLM_RESPONSE_INVALID,
        ),
        (
            httpx.Response(200, content=b'data: {"choices":[]}\n\n'),
            RunErrorCode.LLM_RESPONSE_INVALID,
        ),
        (
            httpx.Response(200, content=_sse({"error": {"message": "unavailable"}})),
            RunErrorCode.LLM_PROVIDER_FAILED,
        ),
        (
            httpx.Response(
                200,
                content=_sse(
                    {
                        "choices": [
                            {
                                "delta": {"reasoning_content": "unsupported reasoning"},
                                "finish_reason": "stop",
                            }
                        ]
                    }
                ),
            ),
            RunErrorCode.LLM_RESPONSE_INVALID,
        ),
        *[
            (
                httpx.Response(
                    200,
                    content=_sse({"choices": [{"delta": {}, "finish_reason": reason}]}),
                ),
                RunErrorCode.LLM_PROVIDER_FAILED,
            )
            for reason in ("insufficient_system_resource", "aborted")
        ],
    ],
)
def test_errors(response, code):
    async def exercise():
        with pytest.raises(WorkflowFailure) as raised:
            _ = [e async for e in _provider(lambda _: response).stream(_request())]
        assert raised.value.error_code is code

    asyncio.run(exercise())


def test_network_failure_after_delta_does_not_retry():
    calls = []

    class BrokenStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
            raise httpx.ReadError("connection lost")

    def handler(request):
        calls.append(request)
        return httpx.Response(200, stream=BrokenStream())

    async def exercise():
        events = []
        with pytest.raises(WorkflowFailure) as raised:
            async for event in _provider(handler).stream(_request()):
                events.append(event)
        assert raised.value.error_code is RunErrorCode.LLM_PROVIDER_FAILED
        assert events == [AssistantContentDelta("partial")]

    asyncio.run(exercise())
    assert len(calls) == 1


def test_cancel_closes_stream():
    async def exercise():
        started = asyncio.Event()
        closed = asyncio.Event()

        class BlockingStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                started.set()
                await asyncio.Event().wait()
                yield b""

            async def aclose(self):
                closed.set()

        provider = _provider(lambda _: httpx.Response(200, stream=BlockingStream()))

        async def collect():
            return [e async for e in provider.stream(_request())]

        task = asyncio.create_task(collect())
        await asyncio.wait_for(started.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert closed.is_set()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("vendor", "provider_type"),
    [
        ("qwen", QwenProvider),
        ("deepseek", DeepSeekProvider),
    ],
)
def test_three_model_purposes_use_selected_credentials(vendor, provider_type):
    settings = Settings(
        _env_file=None,
        llm_provider=vendor,
        qwen_api_key="qwen-key",
        qwen_base_url="https://qwen.example.test",
        deepseek_api_key="deepseek-key",
        deepseek_base_url="https://deepseek.example.test",
        llm_model="answer-model",
        conversation_compactor_model="compact-model",
        memory_policy_model="memory-model",
        memory_policy_estimated_token_budget=1000,
    )
    for factory, model in [
        (_provider_for, "answer-model"),
        (_conversation_compactor_provider_for, "compact-model"),
        (_memory_provider_for, "memory-model"),
    ]:
        provider = factory(settings, None)
        assert isinstance(provider, provider_type)
        assert provider._api_key.get_secret_value() == vendor + "-key"
        assert provider._base_url == "https://" + vendor + ".example.test"
        assert provider._request_payload(_request())["model"] == model
        assert factory(settings, provider) is provider


def test_missing_deepseek_key_never_falls_back_to_qwen():
    settings = Settings(
        _env_file=None,
        llm_provider="deepseek",
        deepseek_api_key=None,
        qwen_api_key="qwen-key",
        qwen_base_url="https://qwen.example.test",
        memory_policy_model="memory-model",
        memory_policy_estimated_token_budget=1000,
    )

    async def exercise():
        for factory in (
            _provider_for,
            _memory_provider_for,
            _conversation_compactor_provider_for,
        ):
            with pytest.raises(WorkflowFailure) as raised:
                _ = [e async for e in factory(settings, None).stream(_request())]
            assert raised.value.error_code is RunErrorCode.LLM_PROVIDER_FAILED

    asyncio.run(exercise())


def test_deepseek_environment_defaults_and_explicit_models(monkeypatch):
    for variable in (
        "LANGLEY_LLM_MODEL",
        "LANGLEY_CONVERSATION_COMPACTOR_MODEL",
        "LANGLEY_MEMORY_POLICY_MODEL",
    ):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("LANGLEY_LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("LANGLEY_DEEPSEEK_API_KEY", "secret-deepseek")
    monkeypatch.setenv("LANGLEY_DEEPSEEK_BASE_URL", "https://custom.example.test")
    defaults = Settings()
    assert defaults.llm_model == "deepseek-flash"
    assert defaults.conversation_compactor_model == "deepseek-flash"
    assert defaults.deepseek_base_url == "https://custom.example.test"
    assert defaults.deepseek_api_key.get_secret_value() == "secret-deepseek"
    assert "secret-deepseek" not in repr(defaults)
    assert (
        _memory_provider_for(defaults, _provider(lambda _: httpx.Response(200))) is None
    )
    monkeypatch.setenv("LANGLEY_LLM_MODEL", "custom-answer")
    assert Settings().conversation_compactor_model == "custom-answer"
    monkeypatch.setenv("LANGLEY_CONVERSATION_COMPACTOR_MODEL", "custom-compact")
    assert Settings().conversation_compactor_model == "custom-compact"


def test_existing_workflow_executes_tool_then_uses_deepseek_followup():
    received = []

    def handler(request):
        payload = json.loads(request.content)
        received.append(payload)
        if len(received) == 1:
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
                                            "id": "time-call",
                                            "function": {
                                                "name": "get_current_time",
                                                "arguments": '{"timezone":"UTC"}',
                                            },
                                        }
                                    ]
                                },
                                "finish_reason": "tool_calls",
                            }
                        ]
                    }
                ),
            )
        tool_message = payload["messages"][-1]
        assert tool_message["role"] == "tool"
        assert tool_message["tool_call_id"] == "time-call"
        observation = json.loads(tool_message["content"])
        assert observation["timezone"] == "UTC"
        return httpx.Response(
            200,
            content=_sse(
                {
                    "choices": [
                        {
                            "delta": {"content": "The time is available."},
                            "finish_reason": "stop",
                        }
                    ]
                }
            ),
        )

    class ContextBuilder:
        async def build(self, *args, **kwargs):
            return AnswerContext(
                completed_turns=(), current_user_content="Time in UTC?"
            )

    async def exercise():
        workflow = LearningAssistantWorkflow(
            context_builder=ContextBuilder(),
            provider=_provider(handler),
            tool_executor=ToolExecutor(tools=(CurrentTimeTool(),)),
            max_llm_rounds=3,
            max_tool_calls=2,
            overall_deadline_seconds=10,
            provider_name="deepseek",
            model="deepseek-test-model",
        )
        deltas = []

        async def on_delta(content):
            deltas.append(content)

        completion = await workflow.execute(
            None,
            run_id=1,
            user_id=1,
            conversation_id=1,
            input_message_id=1,
            knowledge_base_id=None,
            on_assistant_delta=on_delta,
        )
        assert completion.content == "The time is available."
        assert "".join(deltas) == completion.content

    asyncio.run(exercise())
    assert len(received) == 2
    assert all(p["thinking"] == {"type": "disabled"} for p in received)
