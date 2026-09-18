"""Local Run diagnostics through the existing tracing seam, with isolated fan-out."""

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from time import monotonic
from typing import Any

import structlog

from langley.answering.context_frame import build_context_frame
from langley.answering.contracts import (
    AssistantContentDelta,
    JSONValue,
    LLMRequest,
    LLMResponseCompleted,
    ToolCall,
    ToolResult,
)
from langley.answering.tracing import (
    CitationNamespace,
    ExecutionTrace,
    KnowledgeSearchOrigin,
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
        self._seq_by_run: dict[int, int] = {}

    def _sequence_floor(self, run_id: int) -> int:
        path = self.root / f"run-{run_id}.jsonl"
        if not path.exists():
            return 0
        line_count = 0
        observed_max = 0
        with path.open("rb") as stream:
            for raw_line in stream:
                if not raw_line.strip():
                    continue
                line_count += 1
                try:
                    value = json.loads(raw_line)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                seq = value.get("seq") if isinstance(value, dict) else None
                if isinstance(seq, int) and not isinstance(seq, bool) and seq > 0:
                    observed_max = max(observed_max, seq)
        return max(line_count, observed_max)

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
        self._current_llm_round: int | None = None
        self._last_completed_llm_round: int | None = None

    @property
    def capture_mode(self) -> str:
        return "FULL_CONTENT" if self.tracer.include_content else "METADATA_ONLY"

    @property
    def correlated_round(self) -> int | None:
        if self._current_llm_round is not None:
            return self._current_llm_round
        return self._last_completed_llm_round

    def event(self, event_kind: str, **fields: object) -> bool:
        if self.write_failed:
            return False
        try:
            with self.tracer._lock:
                if self.run_id not in self.tracer._seq_by_run:
                    self.tracer._seq_by_run[self.run_id] = self.tracer._sequence_floor(
                        self.run_id
                    )
                seq = self.tracer._seq_by_run[self.run_id] + 1
                payload = (
                    json.dumps(
                        {
                            **fields,
                            "schema_version": 1,
                            "timestamp": datetime.now(UTC).isoformat(),
                            "run_id": self.run_id,
                            "seq": seq,
                            "kind": event_kind,
                            "capture_mode": self.capture_mode,
                        },
                        ensure_ascii=False,
                        allow_nan=False,
                    ).encode("utf-8")
                    + b"\n"
                )
                self.tracer.root.mkdir(parents=True, exist_ok=True)
                with (self.tracer.root / f"run-{self.run_id}.jsonl").open(
                    "ab"
                ) as stream:
                    stream.write(payload)
                self.tracer._seq_by_run[self.run_id] = seq
            return True
        except Exception:
            # Diagnostics are best effort; do not retry or log private payloads.
            self.write_failed = True
            logger.warning("local_run_diagnostics_write_failed", run_id=self.run_id)
            return False

    def begin_llm(self, request: LLMRequest, round_: int) -> LLMTrace:
        llm_trace = _LocalLLMTrace(self, request, round_)
        self._current_llm_round = round_
        return llm_trace

    def begin_tool(self, call: ToolCall, tool_calls_used: int) -> ToolTrace:
        return _LocalToolTrace(
            self, call, tool_calls_used, round_=self.correlated_round
        )

    def begin_knowledge_search(
        self,
        *,
        origin: KnowledgeSearchOrigin,
        knowledge_base_id: int,
        top_k: int,
        query: str,
    ) -> KnowledgeSearchTrace:
        return _LocalKnowledgeSearchTrace(
            self,
            origin=origin,
            knowledge_base_id=knowledge_base_id,
            top_k=top_k,
            query=query,
            round_=self.correlated_round,
        )

    def rejected_response(
        self, response: LLMResponseCompleted, subtype: str | None
    ) -> None:
        content = (
            {
                "assistant_content": response.assistant_content,
                "tool_calls": [_tool_call(call) for call in response.tool_calls],
                "finish_reason": response.finish_reason.value,
            }
            if self.tracer.include_content
            else {}
        )
        self.event(
            "response.rejected",
            round=self.correlated_round,
            invalid_response_subtype=subtype,
            **content,
        )

    def citation_validate(
        self,
        *,
        namespace: CitationNamespace,
        available_evidence_count: int,
        cited_handles: tuple[int | str, ...],
        cited_document_version_ids: tuple[int, ...],
        abstained: bool,
        error_code: str | None,
        abstention_control_token_leaked: bool = False,
    ) -> None:
        fields: dict[str, object] = {
            "round": self.correlated_round,
            "namespace": namespace.value,
            "available_evidence_count": available_evidence_count,
            "citation_count": len(cited_handles),
            "abstained": abstained,
            "success": error_code is None,
            "error_code": error_code,
        }
        if namespace is CitationNamespace.KNOWLEDGE:
            fields["evidence_handles"] = list(cited_handles)
            fields["document_version_ids"] = list(cited_document_version_ids)
        if abstention_control_token_leaked:
            fields["abstention_control_token_leaked"] = True
        self.event("citation.validate", **fields)

    def context_compact(self, **fields: Any) -> None:
        self.event("context.compact", **fields)

    def success(self, answer: str, stop_reason: str = "FINAL_ANSWER") -> None:
        content = {"assistant_content": answer} if self.tracer.include_content else {}
        if self.event("run.success", stop_reason=stop_reason, **content):
            self._release_sequence_cache()

    def failure(
        self, error_code: str, invalid_response_subtype: str | None = None
    ) -> None:
        if self.event(
            "run.failure",
            error_code=error_code,
            invalid_response_subtype=invalid_response_subtype,
        ):
            self._release_sequence_cache()

    def _release_sequence_cache(self) -> None:
        with self.tracer._lock:
            self.tracer._seq_by_run.pop(self.run_id, None)


class _LocalLLMTrace(_NoopLLMTrace):
    def __init__(self, trace: _LocalExecutionTrace, request: LLMRequest, round_: int):
        self.trace = trace
        self.started = monotonic()
        self.ttft_ms: float | None = None
        self.round = round_
        self.fields: dict[str, object] = {
            "round": round_,
            "context_frame": build_context_frame(
                request,
                include_content_hashes=trace.tracer.include_content,
            ).as_json(),
        }
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
        self.trace._last_completed_llm_round = self.round
        self.trace._current_llm_round = None
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
        self.trace._current_llm_round = None
        self.trace.event(
            "llm",
            **self.fields,
            duration_ms=round((monotonic() - self.started) * 1000, 3),
            ttft_ms=self.ttft_ms,
            error_code=error_code,
        )


class _LocalToolTrace(_NoopToolTrace):
    def __init__(
        self,
        trace: _LocalExecutionTrace,
        call: ToolCall,
        ordinal: int,
        *,
        round_: int | None,
    ) -> None:
        self.trace = trace
        self.call = call
        self.ordinal = ordinal
        self.round = round_
        self.started = monotonic()

    def begin_knowledge_search(
        self,
        *,
        knowledge_base_id: int,
        top_k: int,
        query: str,
        origin: KnowledgeSearchOrigin = KnowledgeSearchOrigin.AGENT_TOOL,
    ) -> KnowledgeSearchTrace:
        return _LocalKnowledgeSearchTrace(
            self.trace,
            origin=origin,
            knowledge_base_id=knowledge_base_id,
            top_k=top_k,
            query=query,
            round_=self.round,
            tool_call_id=self.call.call_id,
            tool_ordinal=self.ordinal,
        )

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
            round=self.round,
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


class _LocalKnowledgeSearchTrace:
    def __init__(
        self,
        trace: _LocalExecutionTrace,
        *,
        origin: KnowledgeSearchOrigin,
        knowledge_base_id: int,
        top_k: int,
        query: str,
        round_: int | None,
        tool_call_id: str | None = None,
        tool_ordinal: int | None = None,
    ) -> None:
        self.trace = trace
        self.origin = origin
        self.knowledge_base_id = knowledge_base_id
        self.top_k = top_k
        self.query = query
        self.round = round_
        self.tool_call_id = tool_call_id
        self.tool_ordinal = tool_ordinal
        self.started = monotonic()

    def finish(self, hit_count: int | None, error_code: str | None = None) -> None:
        content = {"query": self.query} if self.trace.tracer.include_content else {}
        self.trace.event(
            "knowledge.search",
            round=self.round,
            tool_call_id=self.tool_call_id,
            tool_ordinal=self.tool_ordinal,
            origin=self.origin.value,
            knowledge_base_id=self.knowledge_base_id,
            top_k=self.top_k,
            duration_ms=round((monotonic() - self.started) * 1000, 3),
            success=error_code is None,
            hit_count=hit_count,
            error_code=error_code,
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
