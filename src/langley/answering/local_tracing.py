"""Local Run diagnostics through the existing tracing seam, with isolated fan-out."""

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from time import monotonic
from typing import Any

import structlog

from langley.answering.contracts import (
    AssistantContentDelta,
    JSONValue,
    LLMRequest,
    LLMResponseCompleted,
    ToolCall,
    ToolResult,
)
from langley.answering.tracing import (
    ExecutionTrace,
    KnowledgeSearchTrace,
    LLMTrace,
    ToolTrace,
    Tracer,
    _NoopLLMTrace,
    _NoopToolTrace,
    _NoopTrace,
    _tool_call,
    _transcript_item,
)

logger = structlog.get_logger(__name__)


class LocalJsonTracer:
    """Append one UTF-8 JSON object per event; content policy is local-only."""

    def __init__(self, root: Path, *, include_content: bool) -> None:
        self.root = root
        self.include_content = include_content
        self._lock = Lock()

    def start(
        self, run_id: int, provider: str, model: str, include_content: bool
    ) -> ExecutionTrace:
        # The seam's flag belongs to external tracing. Never reuse it locally.
        del include_content
        trace = _LocalExecutionTrace(self, run_id)
        trace.event("run.start", provider=provider, configured_model=model)
        return trace


class _LocalExecutionTrace(_NoopTrace):
    def __init__(self, tracer: LocalJsonTracer, run_id: int) -> None:
        self.tracer = tracer
        self.run_id = run_id
        self.write_failed = False

    def event(self, kind: str, **fields: object) -> None:
        if self.write_failed:
            return
        try:
            payload = (
                json.dumps(
                    {
                        "timestamp": datetime.now(UTC).isoformat(),
                        "run_id": self.run_id,
                        "kind": kind,
                        **fields,
                    },
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
                + b"\n"
            )
            with self.tracer._lock:
                self.tracer.root.mkdir(parents=True, exist_ok=True)
                with (self.tracer.root / f"run-{self.run_id}.jsonl").open(
                    "ab"
                ) as stream:
                    stream.write(payload)
        except Exception:
            # Diagnostics are best effort; do not retry or log private payloads.
            self.write_failed = True
            logger.warning("local_run_diagnostics_write_failed", run_id=self.run_id)

    def begin_llm(self, request: LLMRequest, round_: int) -> LLMTrace:
        return _LocalLLMTrace(self, request, round_)

    def begin_tool(self, call: ToolCall, tool_calls_used: int) -> ToolTrace:
        return _LocalToolTrace(self, call, tool_calls_used)

    def context_compact(self, **fields: Any) -> None:
        self.event("context.compact", **fields)

    def success(self, answer: str, stop_reason: str = "FINAL_ANSWER") -> None:
        content = {"assistant_content": answer} if self.tracer.include_content else {}
        self.event("run.success", stop_reason=stop_reason, **content)

    def failure(
        self, error_code: str, invalid_response_subtype: str | None = None
    ) -> None:
        self.event(
            "run.failure",
            error_code=error_code,
            invalid_response_subtype=invalid_response_subtype,
        )


class _LocalLLMTrace(_NoopLLMTrace):
    def __init__(self, trace: _LocalExecutionTrace, request: LLMRequest, round_: int):
        self.trace = trace
        self.started = monotonic()
        self.ttft_ms: float | None = None
        self.fields: dict[str, object] = {"round": round_}
        if trace.tracer.include_content:
            normalized = asdict(request)
            normalized["transcript"] = [
                _transcript_item(item) for item in request.transcript
            ]
            self.fields["request"] = normalized

    def content_delta(self, delta: AssistantContentDelta) -> None:
        if self.ttft_ms is None and delta.content:
            self.ttft_ms = round((monotonic() - self.started) * 1000, 3)

    def finish(self, response: LLMResponseCompleted) -> None:
        input_tokens = response.usage.input_tokens if response.usage else None
        output_tokens = response.usage.output_tokens if response.usage else None
        content = (
            {
                "assistant_content": response.assistant_content,
                "tool_calls": [_tool_call(call) for call in response.tool_calls],
            }
            if self.trace.tracer.include_content
            else {}
        )
        self.trace.event(
            "llm",
            **self.fields,
            duration_ms=round((monotonic() - self.started) * 1000, 3),
            ttft_ms=self.ttft_ms,
            finish_reason=response.finish_reason.value,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens
            if input_tokens is not None and output_tokens is not None
            else None,
            provider_model=response.provider_model,
            tool_call_count=len(response.tool_calls),
            **content,
        )

    def failure(self, error_code: str) -> None:
        self.trace.event(
            "llm",
            **self.fields,
            duration_ms=round((monotonic() - self.started) * 1000, 3),
            ttft_ms=self.ttft_ms,
            error_code=error_code,
        )


class _LocalToolTrace(_NoopToolTrace):
    def __init__(self, trace: _LocalExecutionTrace, call: ToolCall, ordinal: int):
        self.trace = trace
        self.call = call
        self.ordinal = ordinal
        self.started = monotonic()

    def finish(
        self,
        result: ToolResult | None,
        error_code: str | None = None,
        metadata: dict[str, JSONValue] | None = None,
    ) -> None:
        content = (
            {
                "raw_arguments": self.call.raw_arguments,
                "result_content": result.content if result is not None else None,
            }
            if self.trace.tracer.include_content
            else {}
        )
        self.trace.event(
            "tool",
            ordinal=self.ordinal,
            tool_calls_used=self.ordinal,
            call_id=self.call.call_id,
            tool_name=self.call.name,
            duration_ms=round((monotonic() - self.started) * 1000, 3),
            result_kind=result.kind.value if result is not None else None,
            error_code=error_code,
            metadata=metadata or {},
            **content,
        )


class _TraceFanout:
    """Forward existing seam methods, isolating each sink's exceptions."""

    def __init__(self, children: list[Any]) -> None:
        self.children = children

    def call(self, method: str, *args: object, **kwargs: object) -> list[Any]:
        results = []
        for child in self.children:
            try:
                results.append(getattr(child, method)(*args, **kwargs))
            except Exception:
                logger.warning("tracing_sink_failed", operation=method)
        return results


class CompositeTracer:
    def __init__(self, *tracers: Tracer) -> None:
        self._tracers = _TraceFanout(list(tracers))

    def start(
        self, run_id: int, provider: str, model: str, include_content: bool
    ) -> ExecutionTrace:
        return _CompositeExecutionTrace(
            self._tracers.call("start", run_id, provider, model, include_content)
        )


class _CompositeExecutionTrace(_TraceFanout, _NoopTrace):
    def begin_llm(self, request: LLMRequest, round_: int) -> LLMTrace:
        return _CompositeLLMTrace(self.call("begin_llm", request, round_))

    def begin_tool(self, call: ToolCall, tool_calls_used: int) -> ToolTrace:
        return _CompositeToolTrace(self.call("begin_tool", call, tool_calls_used))

    def begin_knowledge_search(self, **fields: Any) -> KnowledgeSearchTrace:
        return _CompositeKnowledgeSearchTrace(
            self.call("begin_knowledge_search", **fields)
        )

    def rejected_response(
        self, response: LLMResponseCompleted, subtype: str | None
    ) -> None:
        self.call("rejected_response", response, subtype)

    def citation_validate(self, **fields: Any) -> None:
        self.call("citation_validate", **fields)

    def context_compact(self, **fields: Any) -> None:
        self.call("context_compact", **fields)

    def success(self, answer: str, stop_reason: str = "FINAL_ANSWER") -> None:
        self.call("success", answer, stop_reason)

    def failure(
        self, error_code: str, invalid_response_subtype: str | None = None
    ) -> None:
        self.call("failure", error_code, invalid_response_subtype)


class _CompositeLLMTrace(_TraceFanout):
    def content_delta(self, delta: AssistantContentDelta) -> None:
        self.call("content_delta", delta)

    def finish(self, response: LLMResponseCompleted) -> None:
        self.call("finish", response)

    def failure(self, error_code: str) -> None:
        self.call("failure", error_code)


class _CompositeToolTrace(_TraceFanout):
    def begin_knowledge_search(self, **fields: Any) -> KnowledgeSearchTrace:
        return _CompositeKnowledgeSearchTrace(
            self.call("begin_knowledge_search", **fields)
        )

    def finish(
        self,
        result: ToolResult | None,
        error_code: str | None = None,
        metadata: dict[str, JSONValue] | None = None,
    ) -> None:
        self.call("finish", result, error_code, metadata)


class _CompositeKnowledgeSearchTrace(_TraceFanout):
    def finish(self, hit_count: int | None, error_code: str | None = None) -> None:
        self.call("finish", hit_count, error_code)
