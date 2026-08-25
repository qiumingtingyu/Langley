"""Fast checks for optional LangSmith tracing."""

import asyncio
import threading
from typing import cast

import pytest
from langsmith import Client

from langley.answering.contracts import (
    LLMFinishReason,
    LLMRequest,
    LLMResponseCompleted,
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
    ).start(1, "qwen", "test", False)
    trace.llm(
        LLMRequest(
            system_input="system",
            transcript=(UserRuntimeMessage(content="request"),),
            allowed_tools=(),
            personal_context=("private preference",),
            current_user_message_index=0,
        ),
        LLMResponseCompleted(
            assistant_content="answer",
            tool_calls=(),
            finish_reason=LLMFinishReason.STOP,
            usage=None,
        ),
        1,
    )
    assert await asyncio.to_thread(client.ready.wait, 1)

    assert len(client.calls) == 2
    assert client.calls[1]["inputs"] == {}


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
