"""Offline connection-only retry boundary, accounting and Workflow safety."""

import asyncio
import inspect
import json

import httpx
import pytest
import structlog
from pydantic import SecretStr, ValidationError
from test_deepseek_provider import _request, _sse

from langley.answering.contracts import AssistantContentDelta, LLMResponseCompleted
from langley.answering.errors import RunErrorCode, WorkflowFailure
from langley.infrastructure import openai_compatible_provider as adapter
from langley.infrastructure.deepseek_provider import DeepSeekProvider
from langley.infrastructure.qwen_provider import QwenProvider
from langley.settings import Settings


def response(text="done", tools=None):
    return httpx.Response(
        200,
        content=_sse(
            {
                "choices": [
                    {
                        "delta": {
                            "content": text,
                            **({"tool_calls": tools} if tools else {}),
                        },
                        "finish_reason": "tool_calls" if tools else "stop",
                    }
                ]
            },
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 42,
                    "completion_tokens": 8,
                    "prompt_cache_hit_tokens": 10,
                    "prompt_cache_miss_tokens": 32,
                },
            },
        ),
    )


def provider(handler, retries=2, cls=DeepSeekProvider, hooks=None, evidence=True):
    clients = []

    async def transport(request):
        # Explicit fake TCP phase; a bare MockTransport supplies no such proof.
        trace = request.extensions["trace"]
        if evidence:
            await trace(
                "connection.connect_tcp.started", {"host": "provider.test", "port": 443}
            )
        try:
            result = handler(request)
            if inspect.isawaitable(result):
                result = await result
        except (httpx.ConnectError, httpx.ConnectTimeout) as error:
            if evidence:
                await trace("connection.connect_tcp.failed", {"exception": error})
            raise
        if evidence:
            await trace("connection.connect_tcp.complete", {"return_value": None})
        return result

    def factory():
        c = httpx.AsyncClient(
            transport=httpx.MockTransport(transport), event_hooks=hooks, trust_env=False
        )
        clients.append(c)
        return c

    return cls(
        api_key=SecretStr("private-test-secret"),
        base_url="https://provider.test",
        model="test-model",
        client_factory=factory,
        connect_max_retries=retries,
    ), clients


async def collect(p):
    return [e async for e in p.stream(_request())]


@pytest.mark.parametrize("cls", [DeepSeekProvider, QwenProvider])
def test_two_connect_errors_then_one_completion_same_payload_and_unknown_total(
    monkeypatch, cls
):
    requests = []
    waits = []

    def handler(r):
        requests.append(r.content)
        if len(requests) < 3:
            raise httpx.ConnectError("do not log private-test-secret", request=r)
        return response()

    async def sleep(seconds):
        waits.append(seconds)

    monkeypatch.setattr(adapter.asyncio, "sleep", sleep)
    p, clients = provider(handler, cls=cls)
    with structlog.testing.capture_logs() as logs:
        events = asyncio.run(collect(p))
    assert len(clients) == 3 and all(c.is_closed for c in clients)
    assert len(set(requests)) == 1 and waits == [1, 2]
    assert events[0] == AssistantContentDelta("done")
    assert len([e for e in events if isinstance(e, LLMResponseCompleted)]) == 1
    assert events[-1].usage is None  # Unknown failed attempts never become zero.
    assert [(x["attempt"], x["outcome"]) for x in logs] == [
        (1, "RETRYING"),
        (2, "RETRYING"),
        (3, "RECOVERED"),
    ]
    assert logs[-1]["successful_attempt_input_tokens"] == 42
    assert logs[-1]["successful_attempt_output_tokens"] == 8
    assert logs[-1]["logical_usage_complete"] is False
    assert "private-test-secret" not in json.dumps(logs)


@pytest.mark.parametrize("retries,expected", [(0, 1), (2, 3)])
def test_exhaustion_and_default_off(monkeypatch, retries, expected):
    async def sleep(_):
        pass

    monkeypatch.setattr(adapter.asyncio, "sleep", sleep)

    def handler(r):
        raise httpx.ConnectError("unavailable", request=r)

    p, clients = provider(handler, retries)
    with pytest.raises(WorkflowFailure) as exc:
        asyncio.run(collect(p))
    assert exc.value.error_code is RunErrorCode.LLM_PROVIDER_FAILED
    assert isinstance(exc.value.__cause__, httpx.ConnectError)
    assert len(clients) == expected and all(c.is_closed for c in clients)


def test_connect_timeout_then_success(monkeypatch):
    attempts = []

    def handler(r):
        attempts.append(r)
        if len(attempts) == 1:
            raise httpx.ConnectTimeout("timeout", request=r)
        return response()

    async def sleep(_):
        pass

    monkeypatch.setattr(adapter.asyncio, "sleep", sleep)
    p, clients = provider(handler)
    assert asyncio.run(collect(p))[-1].assistant_content == "done"
    assert len(clients) == 2


class InterruptedStream(httpx.AsyncByteStream):
    def __init__(self, error, fragment):
        self.error, self.fragment = error, fragment

    async def __aiter__(self):
        yield (
            b"data: "
            + json.dumps({"choices": [{"delta": self.fragment}]}).encode()
            + b"\n\n"
        )
        raise self.error("sensitive body must not be logged")


@pytest.mark.parametrize(
    "error",
    [httpx.ReadError, httpx.ReadTimeout, httpx.ConnectError, httpx.RemoteProtocolError],
)
@pytest.mark.parametrize(
    "fragment",
    [
        {"content": "one"},
        {
            "tool_calls": [
                {"index": 0, "id": "c", "function": {"name": "x", "arguments": "{"}}
            ]
        },
    ],
)
def test_response_started_never_retries_even_for_connect_class(error, fragment):
    p, clients = provider(
        lambda r: httpx.Response(200, stream=InterruptedStream(error, fragment))
    )
    seen = []

    async def consume():
        async for e in p.stream(_request()):
            seen.append(e)

    with pytest.raises(WorkflowFailure):
        asyncio.run(consume())
    assert len(clients) == 1
    assert seen == ([AssistantContentDelta("one")] if "content" in fragment else [])


@pytest.mark.parametrize(
    "status,body",
    [
        (401, b"private"),
        (403, b"private"),
        (400, b"private"),
        (200, b"data: invalid\n\n"),
    ],
)
def test_status_and_invalid_sse_no_retry(status, body):
    p, clients = provider(lambda r: httpx.Response(status, content=body))
    with pytest.raises(WorkflowFailure):
        asyncio.run(collect(p))
    assert len(clients) == 1


@pytest.mark.parametrize("hook_kind", ["request", "response", "sent"])
def test_uncertain_hook_or_already_sent_does_not_retry(hook_kind):
    async def bad_hook(_):
        raise httpx.ConnectError("hook failure")

    async def handler(r):
        if hook_kind == "sent":
            await r.extensions["trace"]("http11.send_request_headers.started", {})
            raise httpx.ConnectError("after send", request=r)
        return response()

    hooks = {hook_kind: [bad_hook]} if hook_kind != "sent" else None
    p, clients = provider(handler, hooks=hooks)
    with pytest.raises(WorkflowFailure):
        asyncio.run(collect(p))
    assert len(clients) == 1


@pytest.mark.parametrize("deadline", [False, True])
def test_cancel_and_deadline_interrupt_backoff(deadline):
    def handler(r):
        raise httpx.ConnectError("unavailable", request=r)

    p, clients = provider(handler)

    async def run():
        if deadline:
            with pytest.raises(TimeoutError):
                async with asyncio.timeout(0.02):
                    await collect(p)
        else:
            task = asyncio.create_task(collect(p))
            # Wait deterministically until the first HTTP context is closed.
            while not clients or not clients[0].is_closed:
                await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(run())
    assert len(clients) == 1


def test_settings_composition_and_bounds():
    from langley.main import _configured_provider

    assert Settings(_env_file=None).llm_connect_max_retries == 0
    for name, cls in [("deepseek", DeepSeekProvider), ("qwen", QwenProvider)]:
        s = Settings(
            _env_file=None,
            llm_provider=name,
            llm_connect_max_retries=2,
            qwen_api_key="test",
            qwen_base_url="https://provider.test",
            deepseek_api_key="test",
        )
        p = _configured_provider(s, "model")
        assert isinstance(p, cls) and p._connect_max_retries == 2
    for value in (-1, 3):
        with pytest.raises(ValidationError):
            Settings(_env_file=None, llm_connect_max_retries=value)


def test_real_workflow_deadline_interrupts_retry_wait():
    from test_run_context_workflow import Context, discard

    from langley.answering.tools import ToolExecutor
    from langley.answering.workflow import LearningAssistantWorkflow

    def handler(request):
        raise httpx.ConnectError("unavailable", request=request)

    p, clients = provider(handler)
    flow = LearningAssistantWorkflow(
        context_builder=Context(),
        provider=p,
        tool_executor=ToolExecutor(tools=()),
        max_llm_rounds=2,
        max_tool_calls=0,
        overall_deadline_seconds=0.05,
        provider_name="fake",
        model="model",
    )
    with pytest.raises(WorkflowFailure) as exc:
        asyncio.run(
            flow.execute(
                None,
                run_id=1,
                user_id=1,
                conversation_id=1,
                input_message_id=1,
                knowledge_base_id=None,
                on_assistant_delta=discard,
            )
        )
    assert exc.value.error_code is RunErrorCode.AGENT_EXECUTION_TIMEOUT
    assert len(clients) == 1


def test_workflow_keeps_successful_tool_once_when_next_request_recovers(
    monkeypatch, tmp_path
):
    from test_run_context_workflow import Context, LargeTool, discard

    from langley.answering.context_builder import AnswerContext
    from langley.answering.local_tracing import LocalJsonTracer
    from langley.answering.tools import ToolContext, ToolExecutor
    from langley.answering.workflow import LearningAssistantWorkflow
    from langley.observatory.store import ObservatoryStore
    from langley.observatory.usage import usage_for_run

    attempts = []

    def handler(r):
        attempts.append(json.loads(r.content))
        if len(attempts) == 1:
            return response(
                "",
                [
                    {
                        "index": 0,
                        "id": "side-effect",
                        "function": {"name": "get_current_time", "arguments": "{}"},
                    }
                ],
            )
        if len(attempts) == 2:
            raise httpx.ConnectError("transient", request=r)
        return response("finished")

    async def sleep(_):
        pass

    monkeypatch.setattr(adapter.asyncio, "sleep", sleep)
    p, _ = provider(handler)
    tool = LargeTool()
    flow = LearningAssistantWorkflow(
        context_builder=Context(),
        provider=p,
        tool_executor=ToolExecutor(tools=(tool,)),
        max_llm_rounds=2,
        max_tool_calls=1,
        overall_deadline_seconds=5,
        provider_name="fake",
        model="model",
    )
    trace_root = tmp_path / "traces"
    trace = LocalJsonTracer(trace_root, include_content=False).start(
        1, "fake", "model", False
    )
    state = asyncio.run(
        flow._run_graph(
            AnswerContext((), "task"),
            ToolContext(run_id=1, user_id=1, knowledge_base_id=None),
            discard,
            trace,
        )
    )
    assert state["llm_rounds"] == 2 and state["tool_calls_used"] == tool.calls == 1
    assert attempts[1] == attempts[2] and len(state["transcript"]) == 4
    assert state["last_completion"].assistant_content == "finished"
    trace.success("finished")
    store = ObservatoryStore(tmp_path / "store.sqlite", trace_root)
    store.initialize()
    store.refresh()
    usage = usage_for_run(store, store.get_run(1))
    assert usage.generation.call_count == 2
    assert usage.all_run.known_input_tokens == 42
    assert usage.all_run.known_output_tokens == 8
    assert usage.all_run.complete is False
    assert store.list_rounds(1)[1].input_tokens is None
