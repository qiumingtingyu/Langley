"""Optional, fail-open LangSmith tracing."""

import asyncio
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from enum import StrEnum
from time import monotonic
from typing import Protocol
from uuid import UUID, uuid4

import structlog
from langsmith import Client

from langley.answering.contracts import (
    AssistantContentDelta,
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


class KnowledgeSearchOrigin(StrEnum):
    """Why one real knowledge retrieval attempt was performed."""

    AGENT_TOOL = "AGENT_TOOL"
    HARNESS_REQUIRED = "HARNESS_REQUIRED"


class ExecutionTrace(Protocol):
    def begin_llm(self, request: LLMRequest, round_: int) -> "LLMTrace": ...

    def begin_tool(self, call: ToolCall, tool_calls_used: int) -> "ToolTrace": ...

    def begin_knowledge_search(
        self,
        *,
        origin: "KnowledgeSearchOrigin",
        knowledge_base_id: int,
        top_k: int,
        query: str,
    ) -> "KnowledgeSearchTrace": ...

    def rejected_response(
        self, response: LLMResponseCompleted, subtype: str | None
    ) -> None: ...

    def citation_validate(
        self,
        *,
        available_evidence_count: int,
        cited_handles: tuple[int, ...],
        cited_document_version_ids: tuple[int, ...],
        abstained: bool,
        error_code: str | None,
        abstention_control_token_leaked: bool = False,
    ) -> None: ...

    def success(self, answer: str, stop_reason: str = "FINAL_ANSWER") -> None: ...

    def failure(
        self, error_code: str, invalid_response_subtype: str | None = None
    ) -> None: ...


class Tracer(Protocol):
    def start(
        self, run_id: int, provider: str, model: str, include_content: bool
    ) -> ExecutionTrace: ...


class LLMTrace(Protocol):
    def content_delta(self, delta: AssistantContentDelta) -> None: ...

    def finish(self, response: LLMResponseCompleted) -> None: ...

    def failure(self, error_code: str) -> None: ...


class ToolTrace(Protocol):
    def begin_knowledge_search(
        self,
        *,
        knowledge_base_id: int,
        top_k: int,
        query: str,
        origin: "KnowledgeSearchOrigin" = KnowledgeSearchOrigin.AGENT_TOOL,
    ) -> "KnowledgeSearchTrace": ...

    def finish(
        self,
        result: ToolResult | None,
        error_code: str | None = None,
        metadata: dict[str, JSONValue] | None = None,
    ) -> None: ...


class KnowledgeSearchTrace(Protocol):
    def finish(self, hit_count: int | None, error_code: str | None = None) -> None: ...


class KnowledgeSearchTraceParent(Protocol):
    """Narrow parent seam that contains no Agent trace topology."""

    def begin_knowledge_search(
        self,
        *,
        origin: KnowledgeSearchOrigin,
        knowledge_base_id: int,
        top_k: int,
        query: str,
    ) -> KnowledgeSearchTrace: ...


_current_retrieval_trace_parent: ContextVar[KnowledgeSearchTraceParent | None] = (
    ContextVar("current_retrieval_trace_parent", default=None)
)


@contextmanager
def tool_trace_context(trace: ToolTrace | None):
    """Keep Agent-tool trace parenting out of business context and DTOs."""

    if trace is None:
        yield
        return
    token = _current_retrieval_trace_parent.set(trace)
    try:
        yield
    finally:
        _current_retrieval_trace_parent.reset(token)


def current_retrieval_trace_parent() -> KnowledgeSearchTraceParent | None:
    return _current_retrieval_trace_parent.get()


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
    def begin_llm(self, request: LLMRequest, round_: int) -> LLMTrace:
        del request, round_
        return _NoopLLMTrace()

    def begin_tool(self, call: ToolCall, tool_calls_used: int) -> ToolTrace:
        del call, tool_calls_used
        return _NoopToolTrace()

    def begin_knowledge_search(
        self,
        *,
        origin: KnowledgeSearchOrigin,
        knowledge_base_id: int,
        top_k: int,
        query: str,
    ) -> KnowledgeSearchTrace:
        del origin, knowledge_base_id, top_k, query
        return _NoopKnowledgeSearchTrace()

    def rejected_response(
        self, response: LLMResponseCompleted, subtype: str | None
    ) -> None:
        del response, subtype

    def citation_validate(
        self,
        *,
        available_evidence_count: int,
        cited_handles: tuple[int, ...],
        cited_document_version_ids: tuple[int, ...],
        abstained: bool,
        error_code: str | None,
        abstention_control_token_leaked: bool = False,
    ) -> None:
        del (
            available_evidence_count,
            cited_handles,
            cited_document_version_ids,
            abstained,
            error_code,
            abstention_control_token_leaked,
        )

    def success(self, answer: str, stop_reason: str = "FINAL_ANSWER") -> None:
        del answer, stop_reason

    def failure(
        self, error_code: str, invalid_response_subtype: str | None = None
    ) -> None:
        del error_code, invalid_response_subtype


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

    def begin_llm(self, request: LLMRequest, round_: int) -> LLMTrace:
        return _LangSmithLLMTrace(self, request, round_)

    def begin_tool(self, call: ToolCall, tool_calls_used: int) -> ToolTrace:
        return _LangSmithToolTrace(self, call, tool_calls_used)

    def begin_knowledge_search(
        self,
        *,
        origin: KnowledgeSearchOrigin,
        knowledge_base_id: int,
        top_k: int,
        query: str,
    ) -> KnowledgeSearchTrace:
        return _LangSmithKnowledgeSearchTrace(
            self,
            parent_id=self._root_id,
            origin=origin,
            knowledge_base_id=knowledge_base_id,
            top_k=top_k,
            query=query,
        )

    def rejected_response(
        self, response: LLMResponseCompleted, subtype: str | None
    ) -> None:
        outputs: dict[str, JSONValue] = {}
        if self._include_content:
            outputs = {
                "assistant_content": response.assistant_content,
                "tool_calls": [_tool_call(call) for call in response.tool_calls],
                "finish_reason": response.finish_reason.value,
            }
        metadata: dict[str, JSONValue] = {}
        if subtype is not None:
            metadata["invalid_response_subtype"] = subtype
        self._create(
            "llm.rejected_response",
            "chain",
            {},
            outputs,
            metadata,
            self._root_id,
            True,
        )

    def citation_validate(
        self,
        *,
        available_evidence_count: int,
        cited_handles: tuple[int, ...],
        cited_document_version_ids: tuple[int, ...],
        abstained: bool,
        error_code: str | None,
        abstention_control_token_leaked: bool = False,
    ) -> None:
        metadata: dict[str, JSONValue] = {
            "available_evidence_count": available_evidence_count,
            "citation_count": len(cited_handles),
            "evidence_handles": list(cited_handles),
            "document_version_ids": list(cited_document_version_ids),
            "abstained": abstained,
            "success": error_code is None,
        }
        if error_code is not None:
            metadata["error_code"] = error_code
        if abstention_control_token_leaked:
            metadata["abstention_control_token_leaked"] = True
        self._create(
            "citation.validate",
            "chain",
            {},
            {},
            metadata,
            self._root_id,
            True,
        )

    def success(self, answer: str, stop_reason: str = "FINAL_ANSWER") -> None:
        self._update(
            (
                {"assistant_content": answer, "stop_reason": stop_reason}
                if self._include_content
                else {"stop_reason": stop_reason}
            ),
            None,
        )

    def failure(
        self, error_code: str, invalid_response_subtype: str | None = None
    ) -> None:
        metadata: dict[str, JSONValue] = {}
        if invalid_response_subtype is not None:
            metadata["invalid_response_subtype"] = invalid_response_subtype
        self._update({}, error_code, metadata)

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
        self._create_with_id(
            self._root_id if parent_run_id is None else uuid4(),
            name,
            run_type,
            inputs,
            outputs,
            metadata,
            parent_run_id,
            completed,
            datetime.now(UTC),
        )

    def _create_with_id(
        self,
        run_id: UUID,
        name: str,
        run_type: str,
        inputs: dict[str, JSONValue],
        outputs: dict[str, JSONValue],
        metadata: dict[str, JSONValue],
        parent_run_id: UUID | None,
        completed: bool,
        started_at: datetime,
    ) -> None:
        kwargs: dict[str, object] = {
            "id": run_id,
            "name": name,
            "run_type": run_type,
            "inputs": inputs,
            "outputs": outputs,
            "project_name": self._project,
            "parent_run_id": parent_run_id,
            "extra": {"metadata": self._metadata | metadata},
            "start_time": started_at,
        }
        if completed:
            kwargs["end_time"] = datetime.now(UTC)
        self._submit("create", self._client.create_run, **kwargs)

    def _update(
        self,
        outputs: dict[str, JSONValue],
        error: str | None,
        metadata: dict[str, JSONValue] | None = None,
    ) -> None:
        kwargs: dict[str, object] = {
            "outputs": outputs,
            "error": error,
            "end_time": datetime.now(UTC),
        }
        if metadata:
            kwargs["extra"] = {"metadata": self._metadata | metadata}
        self._submit(
            "update",
            self._client.update_run,
            self._root_id,
            **kwargs,
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


class _NoopToolTrace:
    def begin_knowledge_search(
        self,
        *,
        knowledge_base_id: int,
        top_k: int,
        query: str,
        origin: KnowledgeSearchOrigin = KnowledgeSearchOrigin.AGENT_TOOL,
    ) -> KnowledgeSearchTrace:
        del origin, knowledge_base_id, top_k, query
        return _NoopKnowledgeSearchTrace()

    def finish(
        self,
        result: ToolResult | None,
        error_code: str | None = None,
        metadata: dict[str, JSONValue] | None = None,
    ) -> None:
        del result, error_code, metadata


class _NoopKnowledgeSearchTrace:
    def finish(self, hit_count: int | None, error_code: str | None = None) -> None:
        del hit_count, error_code


class _NoopLLMTrace:
    def content_delta(self, delta: AssistantContentDelta) -> None:
        del delta

    def finish(self, response: LLMResponseCompleted) -> None:
        del response

    def failure(self, error_code: str) -> None:
        del error_code


class _LangSmithLLMTrace:
    def __init__(
        self, trace: _LangSmithTrace, request: LLMRequest, round_: int
    ) -> None:
        self._trace = trace
        self._id = uuid4()
        self._round = round_
        self._started_at = datetime.now(UTC)
        self._started_monotonic = monotonic()
        self._ttft_ms: float | None = None
        self._finished = False
        inputs: dict[str, JSONValue] = {}
        if trace._include_content:
            inputs = {
                "system_input": request.system_input,
                "transcript": [_transcript_item(item) for item in request.transcript],
                "personal_context": (
                    None
                    if request.personal_context is None
                    else list(request.personal_context)
                ),
            }
            if request.evidence_context is not None:
                inputs["evidence_context"] = request.evidence_context
        trace._create_with_id(
            self._id,
            "learning_assistant_llm",
            "llm",
            inputs,
            {},
            {"llm_round": round_},
            trace._root_id,
            False,
            self._started_at,
        )

    def content_delta(self, delta: AssistantContentDelta) -> None:
        if self._ttft_ms is None and delta.content:
            self._ttft_ms = round(
                (monotonic() - self._started_monotonic) * 1000,
                3,
            )

    def finish(self, response: LLMResponseCompleted) -> None:
        self._require_open()
        self._finished = True
        input_tokens = None if response.usage is None else response.usage.input_tokens
        output_tokens = None if response.usage is None else response.usage.output_tokens
        total_tokens = (
            input_tokens + output_tokens
            if input_tokens is not None and output_tokens is not None
            else None
        )
        metadata: dict[str, JSONValue] = {
            "llm_round": self._round,
            "finish_reason": response.finish_reason.value,
            "tool_call_count": len(response.tool_calls),
            "llm_duration_ms": self._duration_ms(),
            "ttft_ms": self._ttft_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "provider_model": response.provider_model,
        }
        outputs: dict[str, JSONValue] = {}
        if self._trace._include_content:
            outputs = {
                "assistant_content": response.assistant_content,
                "tool_calls": [_tool_call(call) for call in response.tool_calls],
            }
        self._trace._submit(
            "update",
            self._trace._client.update_run,
            self._id,
            outputs=outputs,
            error=None,
            end_time=datetime.now(UTC),
            extra={"metadata": metadata},
        )

    def failure(self, error_code: str) -> None:
        self._require_open()
        self._finished = True
        self._trace._submit(
            "update",
            self._trace._client.update_run,
            self._id,
            outputs={},
            error=error_code,
            end_time=datetime.now(UTC),
            extra={
                "metadata": {
                    "llm_round": self._round,
                    "llm_duration_ms": self._duration_ms(),
                    "ttft_ms": self._ttft_ms,
                }
            },
        )

    def _duration_ms(self) -> float:
        return round((monotonic() - self._started_monotonic) * 1000, 3)

    def _require_open(self) -> None:
        if self._finished:
            raise RuntimeError("LLM trace is already closed")


class _LangSmithToolTrace:
    def __init__(
        self, trace: _LangSmithTrace, call: ToolCall, tool_calls_used: int
    ) -> None:
        self._trace = trace
        self._call = call
        self._id = uuid4()
        self._started_at = datetime.now(UTC)
        inputs: dict[str, JSONValue] = (
            {"tool_call": _tool_call(call)} if trace._include_content else {}
        )
        trace._create_with_id(
            self._id,
            f"tool.{call.name}",
            "tool",
            inputs,
            {},
            {
                "tool_name": call.name,
                "tool_call_id": call.call_id,
                "tool_calls_used": tool_calls_used,
            },
            trace._root_id,
            False,
            self._started_at,
        )

    def begin_knowledge_search(
        self,
        *,
        knowledge_base_id: int,
        top_k: int,
        query: str,
        origin: KnowledgeSearchOrigin = KnowledgeSearchOrigin.AGENT_TOOL,
    ) -> KnowledgeSearchTrace:
        return _LangSmithKnowledgeSearchTrace(
            self._trace,
            parent_id=self._id,
            origin=origin,
            knowledge_base_id=knowledge_base_id,
            top_k=top_k,
            query=query,
        )

    def finish(
        self,
        result: ToolResult | None,
        error_code: str | None = None,
        metadata: dict[str, JSONValue] | None = None,
    ) -> None:
        trace_metadata: dict[str, JSONValue] = {
            "tool_duration_ms": round(
                (datetime.now(UTC) - self._started_at).total_seconds() * 1000,
                3,
            )
        }
        if metadata is not None:
            trace_metadata.update(metadata)
        if result is not None:
            trace_metadata["tool_result_kind"] = result.kind.value
        if error_code is not None:
            trace_metadata["error_code"] = error_code
        self._trace._submit(
            "update",
            self._trace._client.update_run,
            self._id,
            outputs=(
                {"tool_result": _tool_result(result)}
                if self._trace._include_content and result is not None
                else {}
            ),
            error=error_code,
            end_time=datetime.now(UTC),
            extra={"metadata": trace_metadata},
        )


class _LangSmithKnowledgeSearchTrace:
    def __init__(
        self,
        trace: _LangSmithTrace,
        *,
        parent_id: UUID,
        origin: KnowledgeSearchOrigin,
        knowledge_base_id: int,
        top_k: int,
        query: str,
    ) -> None:
        self._trace = trace
        self._id = uuid4()
        self._started_at = datetime.now(UTC)
        metadata: dict[str, JSONValue] = {
            "knowledge_base_id": knowledge_base_id,
            "top_k": top_k,
            "origin": origin.value,
        }
        self._trace._create_with_id(
            self._id,
            "knowledge.search",
            "retriever",
            ({"query": query} if self._trace._include_content else {}),
            {},
            metadata,
            parent_id,
            False,
            self._started_at,
        )

    def finish(self, hit_count: int | None, error_code: str | None = None) -> None:
        metadata: dict[str, JSONValue] = {
            "success": error_code is None,
            "knowledge_search_duration_ms": round(
                (datetime.now(UTC) - self._started_at).total_seconds() * 1000,
                3,
            ),
        }
        if hit_count is not None:
            metadata["hit_count"] = hit_count
        if error_code is not None:
            metadata["error_code"] = error_code
        self._trace._submit(
            "update",
            self._trace._client.update_run,
            self._id,
            outputs={},
            error=error_code,
            end_time=datetime.now(UTC),
            extra={"metadata": metadata},
        )


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
