"""Fast checks for optional LangSmith tracing."""

import asyncio
import threading
from typing import cast

import pytest
from langsmith import Client

from langley.answering.contracts import (
    AssistantContentDelta,
    LLMFinishReason,
    LLMRequest,
    LLMResponseCompleted,
    LLMUsage,
    ToolCall,
    ToolResult,
    ToolResultKind,
    UserRuntimeMessage,
)
from langley.answering.tracing import LangSmithTracer


def test_disabled_tracing_does_not_create_a_client() -> None:
    def forbidden() -> Client:
        raise AssertionError("client should not be created")

    trace = LangSmithTracer(
        enabled=False, project=None, client_factory=forbidden
    ).start(1, "qwen", "test", False)
    trace.success("answer")


@pytest.mark.anyio
async def test_content_disabled_trace_does_not_export_personal_context() -> None:
    class RecordingClient:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []
            self.updates: list[tuple[tuple[object, ...], dict[str, object]]] = []
            self.ready = threading.Event()

        def create_run(self, **kwargs: object) -> None:
            self.calls.append(kwargs)
            if len(self.calls) == 2 and self.updates:
                self.ready.set()

        def update_run(self, *args: object, **kwargs: object) -> None:
            self.updates.append((args, kwargs))
            if len(self.calls) == 2:
                self.ready.set()

    client = RecordingClient()
    trace = LangSmithTracer(
        enabled=True,
        project=None,
        client_factory=lambda: client,  # type: ignore[arg-type]
    ).start(1, "qwen", "test", False)
    llm = trace.begin_llm(
        LLMRequest(
            system_input="system",
            transcript=(UserRuntimeMessage(content="request"),),
            allowed_tools=(),
            personal_context=("private preference",),
            current_user_message_index=0,
            conversation_compact_context="private compact conversation",
            evidence_context="private evidence",
        ),
        1,
    )
    llm.finish(
        LLMResponseCompleted(
            assistant_content="answer",
            tool_calls=(),
            finish_reason=LLMFinishReason.STOP,
            usage=None,
        ),
    )
    assert await asyncio.to_thread(client.ready.wait, 1)

    assert len(client.calls) == 2
    assert client.calls[1]["inputs"] == {}
    assert client.updates[0][1]["outputs"] == {}


@pytest.mark.anyio
async def test_content_enabled_trace_does_not_expand_to_compact_conversation() -> None:
    class RecordingClient:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []
            self.ready = threading.Event()

        def create_run(self, **kwargs: object) -> None:
            self.calls.append(kwargs)
            if len(self.calls) == 2:
                self.ready.set()

        def update_run(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

    client = RecordingClient()
    trace = LangSmithTracer(
        enabled=True,
        project=None,
        client_factory=lambda: client,  # type: ignore[arg-type]
    ).start(1, "qwen", "test", True)
    trace.begin_llm(
        LLMRequest(
            system_input="system",
            transcript=(UserRuntimeMessage(content="request"),),
            allowed_tools=(),
            conversation_compact_context="private compact conversation",
        ),
        1,
    )
    assert await asyncio.to_thread(client.ready.wait, 1)

    assert "conversation_compact_context" not in client.calls[1]["inputs"]


@pytest.mark.anyio
async def test_content_enabled_trace_may_record_evidence_context() -> None:
    class RecordingClient:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []
            self.ready = threading.Event()

        def create_run(self, **kwargs: object) -> None:
            self.calls.append(kwargs)
            if len(self.calls) == 2:
                self.ready.set()

        def update_run(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

    client = RecordingClient()
    trace = LangSmithTracer(
        enabled=True,
        project=None,
        client_factory=lambda: client,  # type: ignore[arg-type]
    ).start(1, "qwen", "test", True)
    trace.begin_llm(
        LLMRequest(
            system_input="system",
            transcript=(UserRuntimeMessage(content="request"),),
            allowed_tools=(),
            evidence_context="[K1] private evidence",
        ),
        1,
    )

    assert await asyncio.to_thread(client.ready.wait, 1)
    assert client.calls[1]["inputs"]["evidence_context"] == (  # type: ignore[index]
        "[K1] private evidence"
    )


@pytest.mark.anyio
async def test_root_failure_exports_invalid_response_subtype_metadata() -> None:
    class RecordingClient:
        def __init__(self) -> None:
            self.updates: list[dict[str, object]] = []
            self.ready = threading.Event()

        def create_run(self, **kwargs: object) -> None:
            del kwargs

        def update_run(self, *args: object, **kwargs: object) -> None:
            del args
            self.updates.append(kwargs)
            self.ready.set()

    client = RecordingClient()
    trace = LangSmithTracer(
        enabled=True,
        project=None,
        client_factory=lambda: client,  # type: ignore[arg-type]
    ).start(1, "qwen", "test", False)

    trace.failure("LLM_RESPONSE_INVALID", "MISSING_REQUIRED_CITATION")

    assert await asyncio.to_thread(client.ready.wait, 1)
    update = client.updates[0]
    assert update["error"] == "LLM_RESPONSE_INVALID"
    metadata = cast(dict[str, object], update["extra"])["metadata"]
    assert cast(dict[str, object], metadata)["invalid_response_subtype"] == (
        "MISSING_REQUIRED_CITATION"
    )


@pytest.mark.anyio
async def test_llm_span_records_real_monotonic_timing_ttft_and_provider_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RecordingClient:
        def __init__(self) -> None:
            self.creates: list[dict[str, object]] = []
            self.updates: list[tuple[tuple[object, ...], dict[str, object]]] = []
            self.ready = threading.Event()

        def create_run(self, **kwargs: object) -> None:
            self.creates.append(kwargs)

        def update_run(self, *args: object, **kwargs: object) -> None:
            self.updates.append((args, kwargs))
            self.ready.set()

    ticks = iter((10.0, 10.125, 10.5))
    monkeypatch.setattr("langley.answering.tracing.monotonic", lambda: next(ticks))
    client = RecordingClient()
    trace = LangSmithTracer(
        enabled=True,
        project=None,
        client_factory=lambda: client,  # type: ignore[arg-type]
    ).start(2, "qwen", "requested-model", False)
    llm = trace.begin_llm(
        LLMRequest(
            system_input="private system",
            transcript=(UserRuntimeMessage(content="private request"),),
            allowed_tools=(),
        ),
        1,
    )
    llm.content_delta(AssistantContentDelta("first"))
    llm.finish(
        LLMResponseCompleted(
            assistant_content="private answer",
            tool_calls=(),
            finish_reason=LLMFinishReason.STOP,
            usage=LLMUsage(input_tokens=10, output_tokens=2),
            provider_model="provider-model",
        )
    )

    assert await asyncio.to_thread(client.ready.wait, 1)
    llm_create = next(
        item for item in client.creates if item["name"] == "learning_assistant_llm"
    )
    assert "end_time" not in llm_create
    assert llm_create["inputs"] == {}
    _, update = client.updates[0]
    assert update["end_time"] is not None
    assert update["outputs"] == {}
    metadata = cast(dict[str, object], update["extra"])["metadata"]
    assert metadata == {
        "llm_round": 1,
        "finish_reason": "STOP",
        "tool_call_count": 0,
        "llm_duration_ms": 500.0,
        "ttft_ms": 125.0,
        "input_tokens": 10,
        "output_tokens": 2,
        "total_tokens": 12,
        "provider_model": "provider-model",
    }


@pytest.mark.anyio
async def test_tool_call_round_without_content_records_unavailable_ttft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RecordingClient:
        def __init__(self) -> None:
            self.updates: list[dict[str, object]] = []
            self.ready = threading.Event()

        def create_run(self, **kwargs: object) -> None:
            del kwargs

        def update_run(self, *args: object, **kwargs: object) -> None:
            del args
            self.updates.append(kwargs)
            self.ready.set()

    ticks = iter((20.0, 20.25))
    monkeypatch.setattr("langley.answering.tracing.monotonic", lambda: next(ticks))
    client = RecordingClient()
    trace = LangSmithTracer(
        enabled=True,
        project=None,
        client_factory=lambda: client,  # type: ignore[arg-type]
    ).start(3, "qwen", "test", False)
    llm = trace.begin_llm(
        LLMRequest(
            system_input="system",
            transcript=(UserRuntimeMessage(content="request"),),
            allowed_tools=(),
        ),
        1,
    )
    llm.finish(
        LLMResponseCompleted(
            assistant_content="",
            tool_calls=(ToolCall("call-1", "search_knowledge", "{}"),),
            finish_reason=LLMFinishReason.TOOL_CALLS,
            usage=None,
        )
    )

    assert await asyncio.to_thread(client.ready.wait, 1)
    metadata = cast(dict[str, object], client.updates[0]["extra"])["metadata"]
    assert cast(dict[str, object], metadata)["ttft_ms"] is None


@pytest.mark.anyio
async def test_agent_tool_and_citation_spans_export_only_authorized_metadata() -> None:
    class RecordingClient:
        def __init__(self) -> None:
            self.creates: list[dict[str, object]] = []
            self.updates: list[tuple[tuple[object, ...], dict[str, object]]] = []
            self.ready = threading.Event()
            self.updates_ready = threading.Event()

        def create_run(self, **kwargs: object) -> None:
            self.creates.append(kwargs)
            if len(self.creates) == 4:
                self.ready.set()

        def update_run(self, *args: object, **kwargs: object) -> None:
            self.updates.append((args, kwargs))
            if len(self.updates) == 2:
                self.updates_ready.set()

    client = RecordingClient()
    trace = LangSmithTracer(
        enabled=True,
        project=None,
        client_factory=lambda: client,  # type: ignore[arg-type]
    ).start(7, "qwen", "test", False)
    tool = trace.begin_tool(ToolCall("search-1", "search_knowledge", "{}"), 1)
    search = tool.begin_knowledge_search(
        knowledge_base_id=44,
        top_k=5,
        query="private query",
    )
    search.finish(hit_count=1)
    tool.finish(
        ToolResult("search-1", "search_knowledge", ToolResultKind.SUCCESS, "secret")
    )
    trace.citation_validate(
        available_evidence_count=1,
        cited_handles=(1,),
        cited_document_version_ids=(13,),
        abstained=False,
        error_code=None,
    )

    assert await asyncio.to_thread(client.ready.wait, 1)
    assert await asyncio.to_thread(client.updates_ready.wait, 1)
    tool_span = client.creates[1]
    knowledge_span = client.creates[2]
    citation_span = client.creates[3]
    updates_by_run_id = {args[0]: kwargs for args, kwargs in client.updates}
    tool_update = updates_by_run_id[tool_span["id"]]
    knowledge_update = updates_by_run_id[knowledge_span["id"]]
    assert tool_span["name"] == "tool.search_knowledge"
    assert tool_span["extra"] == {
        "metadata": {
            "langley_run_id": 7,
            "workflow": "learning_assistant",
            "provider": "qwen",
            "model": "test",
            "tool_name": "search_knowledge",
            "tool_call_id": "search-1",
            "tool_calls_used": 1,
        }
    }
    assert knowledge_span["name"] == "knowledge.search"
    assert knowledge_span["parent_run_id"] == tool_span["id"]
    assert knowledge_span["inputs"] == {}
    assert knowledge_span["extra"] == {
        "metadata": {
            "langley_run_id": 7,
            "workflow": "learning_assistant",
            "provider": "qwen",
            "model": "test",
            "knowledge_base_id": 44,
            "top_k": 5,
            "origin": "AGENT_TOOL",
        }
    }
    assert tool_update["end_time"] is not None
    assert tool_update["error"] is None
    assert tool_update["outputs"] == {}
    tool_metadata = cast(dict[str, object], tool_update["extra"])["metadata"]
    assert cast(dict[str, object], tool_metadata)["tool_result_kind"] == (
        ToolResultKind.SUCCESS.value
    )
    assert knowledge_update["end_time"] is not None
    assert knowledge_update["error"] is None
    assert knowledge_update["outputs"] == {}
    knowledge_metadata = cast(dict[str, object], knowledge_update["extra"])["metadata"]
    assert cast(dict[str, object], knowledge_metadata)["success"] is True
    assert cast(dict[str, object], knowledge_metadata)["hit_count"] == 1
    assert citation_span["extra"] == {
        "metadata": {
            "langley_run_id": 7,
            "workflow": "learning_assistant",
            "provider": "qwen",
            "model": "test",
            "available_evidence_count": 1,
            "citation_count": 1,
            "evidence_handles": [1],
            "document_version_ids": [13],
            "abstained": False,
            "success": True,
        }
    }


@pytest.mark.anyio
async def test_web_tool_trace_exports_metadata_without_query_or_evidence_content() -> (
    None
):
    class RecordingClient:
        def __init__(self) -> None:
            self.creates: list[dict[str, object]] = []
            self.updates: list[dict[str, object]] = []
            self.ready = threading.Event()

        def create_run(self, **kwargs: object) -> None:
            self.creates.append(kwargs)

        def update_run(self, *args: object, **kwargs: object) -> None:
            del args
            self.updates.append(kwargs)
            self.ready.set()

    client = RecordingClient()
    trace = LangSmithTracer(
        enabled=True,
        project=None,
        client_factory=lambda: client,  # type: ignore[arg-type]
    ).start(8, "qwen", "test", False)
    tool = trace.begin_tool(
        ToolCall("web-1", "search_web", '{"query":"private query"}'), 1
    )
    tool.finish(
        ToolResult(
            "web-1",
            "search_web",
            ToolResultKind.SUCCESS,
            "private snippets",
        ),
        metadata={
            "hit_count": 5,
            "provider_response_time_ms": 200.0,
            "tavily_provider_reported_credits": 1.0,
            "provider_request_id": "request-1",
        },
    )

    assert await asyncio.to_thread(client.ready.wait, 1)
    tool_create = next(
        item for item in client.creates if item["name"] == "tool.search_web"
    )
    update = client.updates[0]
    assert tool_create["inputs"] == {}
    assert update["outputs"] == {}
    metadata = cast(dict[str, object], update["extra"])["metadata"]
    assert cast(dict[str, object], metadata)["hit_count"] == 5
    assert cast(dict[str, object], metadata)["tavily_provider_reported_credits"] == 1.0
    assert "private query" not in repr((client.creates, client.updates))
    assert "private snippets" not in repr((client.creates, client.updates))
