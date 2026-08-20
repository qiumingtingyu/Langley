"""Optional, fail-open LangSmith tracing."""

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID, uuid4

import structlog
from langsmith import Client

from langley.answering.contracts import (
    AssistantRuntimeMessage,
    JSONValue,
    LLMRequest,
    LLMResponseCompleted,
    RuntimeTranscriptItem,
    ToolCall,
    ToolResult,
    UserRuntimeMessage,
)

logger = structlog.get_logger(__name__)
LangSmithClientFactory = Callable[[], Client]


class ExecutionTrace(Protocol):
    def llm(
        self, request: LLMRequest, response: LLMResponseCompleted, round_: int
    ) -> None: ...

    def tool(
        self, calls: tuple[ToolCall, ...], results: tuple[ToolResult, ...], total: int
    ) -> None: ...

    def success(self, answer: str) -> None: ...

    def failure(self, error_code: str) -> None: ...


class Tracer(Protocol):
    def start(
        self, run_id: int, provider: str, model: str, include_content: bool
    ) -> ExecutionTrace: ...


class LangSmithTracer:
    """Send safe trace data to LangSmith without affecting answers."""

    def __init__(
        self,
        *,
        enabled: bool,
        project: str | None,
        client_factory: LangSmithClientFactory | None = None,
    ) -> None:
        self._enabled = enabled
        self._project = project
        self._client_factory = client_factory or Client

    def start(
        self, run_id: int, provider: str, model: str, include_content: bool
    ) -> ExecutionTrace:
        if not self._enabled:
            return _NoopTrace()
        try:
            trace = _LangSmithTrace(
                self._client_factory(),
                self._project,
                uuid4(),
                {
                    "langley_run_id": run_id,
                    "workflow": "learning_assistant",
                    "provider": provider,
                    "model": model,
                },
                include_content,
            )
            trace._create("learning_assistant", "chain", {}, {}, {}, None, False)
            return trace
        except Exception:
            logger.warning("langsmith_tracing_start_failed")
            return _NoopTrace()


class _NoopTrace:
    def llm(
        self, request: LLMRequest, response: LLMResponseCompleted, round_: int
    ) -> None:
        del request, response, round_

    def tool(
        self, calls: tuple[ToolCall, ...], results: tuple[ToolResult, ...], total: int
    ) -> None:
        del calls, results, total

    def success(self, answer: str) -> None:
        del answer

    def failure(self, error_code: str) -> None:
        del error_code


class _LangSmithTrace:
    def __init__(
        self,
        client: Client,
        project: str | None,
        root_id: UUID,
        metadata: dict[str, JSONValue],
        include_content: bool,
    ) -> None:
        self._client = client
        self._project = project
        self._root_id = root_id
        self._metadata = metadata
        self._include_content = include_content

    def llm(
        self, request: LLMRequest, response: LLMResponseCompleted, round_: int
    ) -> None:
        inputs: dict[str, JSONValue] = {}
        outputs: dict[str, JSONValue] = {}
        if self._include_content:
            inputs = {
                "system_input": request.system_input,
                "transcript": [_transcript_item(item) for item in request.transcript],
                "personal_context": (
                    None
                    if request.personal_context is None
                    else list(request.personal_context)
                ),
            }
            outputs = {
                "assistant_content": response.assistant_content,
                "tool_calls": [_tool_call(call) for call in response.tool_calls],
            }
        self._create(
            "learning_assistant_llm",
            "llm",
            inputs,
            outputs,
            {
                "llm_round": round_,
                "finish_reason": response.finish_reason.value,
                "tool_call_count": len(response.tool_calls),
            },
            self._root_id,
            True,
        )

    def tool(
        self, calls: tuple[ToolCall, ...], results: tuple[ToolResult, ...], total: int
    ) -> None:
        inputs: dict[str, JSONValue] = {}
        outputs: dict[str, JSONValue] = {}
        if self._include_content:
            inputs = {"tool_calls": [_tool_call(call) for call in calls]}
            outputs = {"tool_results": [_tool_result(result) for result in results]}
        self._create(
            "learning_assistant_tool",
            "tool",
            inputs,
            outputs,
            {
                "tool_call_count": len(calls),
                "tool_calls_used": total,
                "tool_result_kinds": [result.kind.value for result in results],
            },
            self._root_id,
            True,
        )

    def success(self, answer: str) -> None:
        self._update(
            {"assistant_content": answer} if self._include_content else {}, None
        )

    def failure(self, error_code: str) -> None:
        self._update({}, error_code)

    def _create(
        self,
        name: str,
        run_type: str,
        inputs: dict[str, JSONValue],
        outputs: dict[str, JSONValue],
        metadata: dict[str, JSONValue],
        parent_run_id: UUID | None,
        completed: bool,
    ) -> None:
        now = datetime.now(UTC)
        kwargs: dict[str, object] = {
            "id": self._root_id if parent_run_id is None else uuid4(),
            "name": name,
            "run_type": run_type,
            "inputs": inputs,
            "outputs": outputs,
            "project_name": self._project,
            "parent_run_id": parent_run_id,
            "extra": {"metadata": self._metadata | metadata},
            "start_time": now,
        }
        if completed:
            kwargs["end_time"] = now
        self._submit("create", self._client.create_run, **kwargs)

    def _update(self, outputs: dict[str, JSONValue], error: str | None) -> None:
        self._submit(
            "update",
            self._client.update_run,
            self._root_id,
            outputs=outputs,
            error=error,
            end_time=datetime.now(UTC),
        )

    def _submit(
        self,
        operation: str,
        action: Callable[..., None],
        *args: object,
        **kwargs: object,
    ) -> None:
        async def export() -> None:
            try:
                await asyncio.to_thread(action, *args, **kwargs)
            except Exception:
                logger.warning("langsmith_tracing_export_failed", operation=operation)

        asyncio.create_task(export())


def _transcript_item(item: RuntimeTranscriptItem) -> dict[str, JSONValue]:
    if isinstance(item, UserRuntimeMessage):
        return {"role": "user", "content": item.content}
    if isinstance(item, AssistantRuntimeMessage):
        return {
            "role": "assistant",
            "content": item.content,
            "tool_calls": [_tool_call(call) for call in item.tool_calls],
        }
    if isinstance(item, ToolResult):
        return _tool_result(item)
    raise TypeError("unsupported runtime transcript item")


def _tool_call(call: ToolCall) -> dict[str, JSONValue]:
    return {
        "call_id": call.call_id,
        "name": call.name,
        "raw_arguments": call.raw_arguments,
    }


def _tool_result(result: ToolResult) -> dict[str, JSONValue]:
    return {
        "call_id": result.call_id,
        "name": result.name,
        "kind": result.kind.value,
        "content": result.content,
    }
