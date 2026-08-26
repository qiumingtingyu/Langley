"""Focused runtime observations and artifacts for Knowledge Casebook V1 B0."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Any
from uuid import uuid4

from langley.answering.contracts import (
    AssistantContentDelta,
    LLMRequest,
    LLMResponseCompleted,
    ToolCall,
    ToolResult,
)

FORMAL_B0_MAX_CHUNK_CHARS = 1200
FORMAL_B0_CHUNK_OVERLAP = 0


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _duration_ms(started: float) -> float:
    return round((monotonic() - started) * 1000, 3)


class CasebookRuntimeBindingError(ValueError):
    """The supplied runtime KB cannot be bound exactly to the frozen Manifest."""


@dataclass(frozen=True)
class RuntimeDocumentFact:
    """One authoritative current DocumentVersion fact for the Eval KB."""

    document_version_id: int
    source_sha256: str
    chunk_count: int
    chunk_max_chars: int | None


@dataclass(frozen=True)
class RuntimeIndexFact:
    """The active index facts that must describe the exact current corpus."""

    knowledge_base_id: int
    index_status: str
    active_generation_id: str | None
    active_chunk_snapshot_sha256: str | None
    active_embedding_model: str | None
    active_embedding_revision: str | None
    active_embedding_dimension: int | None
    active_embedding_representation: str | None
    index_job_generation_id: str | None
    index_job_status: str | None
    index_job_chunk_snapshot_sha256: str | None
    index_job_processed_chunk_count: int | None
    index_job_total_chunk_count: int | None


def verify_runtime_corpus_binding(
    manifest_documents: list[dict[str, Any]],
    runtime_documents: list[RuntimeDocumentFact],
    index: RuntimeIndexFact,
    *,
    expected_embedding_model: str,
    expected_embedding_revision: str,
    expected_embedding_dimension: int,
    expected_embedding_representation: str,
    expected_chunk_max_chars: int,
) -> dict[str, Any]:
    """Bind Manifest keys to current version IDs using unique exact source hashes."""

    if len(manifest_documents) != 40:
        raise CasebookRuntimeBindingError(
            "runtime binding requires exactly 40 Manifest documents"
        )
    if len(runtime_documents) != 40:
        raise CasebookRuntimeBindingError(
            "runtime KB must contain exactly 40 eligible DocumentVersions"
        )

    manifest_by_sha: dict[str, tuple[str, str]] = {}
    seen_keys: set[str] = set()
    for position, manifest_document in enumerate(manifest_documents):
        key = manifest_document.get("key")
        source_sha = manifest_document.get("sha256")
        if not isinstance(key, str) or not key or key in seen_keys:
            raise CasebookRuntimeBindingError(
                f"Manifest document {position} has a missing or duplicate key"
            )
        if (
            not isinstance(source_sha, str)
            or len(source_sha) != 64
            or any(character not in "0123456789abcdef" for character in source_sha)
        ):
            raise CasebookRuntimeBindingError(
                f"Manifest document {key!r} has an invalid source SHA-256"
            )
        if source_sha in manifest_by_sha:
            raise CasebookRuntimeBindingError(
                "Manifest source SHA-256 values must be unambiguous"
            )
        seen_keys.add(key)
        manifest_by_sha[source_sha] = (key, source_sha)

    runtime_by_sha: dict[str, RuntimeDocumentFact] = {}
    seen_version_ids: set[int] = set()
    for runtime_document in runtime_documents:
        if (
            runtime_document.document_version_id <= 0
            or runtime_document.document_version_id in seen_version_ids
        ):
            raise CasebookRuntimeBindingError(
                "runtime DocumentVersion IDs must be positive and unambiguous"
            )
        if runtime_document.source_sha256 in runtime_by_sha:
            raise CasebookRuntimeBindingError(
                "runtime source SHA-256 values must be unambiguous"
            )
        if runtime_document.chunk_count <= 0:
            raise CasebookRuntimeBindingError(
                f"runtime DocumentVersion {runtime_document.document_version_id} "
                "has no chunks"
            )
        if runtime_document.chunk_max_chars != expected_chunk_max_chars:
            raise CasebookRuntimeBindingError(
                f"runtime DocumentVersion {runtime_document.document_version_id} "
                "chunk_max_chars does not match Formal B0: "
                f"expected {expected_chunk_max_chars}, "
                f"got {runtime_document.chunk_max_chars!r}"
            )
        seen_version_ids.add(runtime_document.document_version_id)
        runtime_by_sha[runtime_document.source_sha256] = runtime_document

    missing = sorted(
        key for sha, (key, _) in manifest_by_sha.items() if sha not in runtime_by_sha
    )
    unexpected = sorted(
        runtime_document.document_version_id
        for sha, runtime_document in runtime_by_sha.items()
        if sha not in manifest_by_sha
    )
    if missing or unexpected:
        raise CasebookRuntimeBindingError(
            "runtime corpus does not exactly match the frozen Manifest; "
            f"missing_keys={missing!r}, unexpected_document_version_ids={unexpected!r}"
        )

    if index.index_status != "READY":
        raise CasebookRuntimeBindingError(
            f"runtime index is not READY: {index.index_status!r}"
        )
    if not index.active_generation_id:
        raise CasebookRuntimeBindingError("runtime index has no active generation")
    if (
        index.index_job_generation_id != index.active_generation_id
        or index.index_job_status != "SUCCEEDED"
    ):
        raise CasebookRuntimeBindingError(
            "active generation has no matching successful index job"
        )
    if (
        index.index_job_total_chunk_count is None
        or index.index_job_total_chunk_count <= 0
        or index.index_job_processed_chunk_count != index.index_job_total_chunk_count
        or index.index_job_total_chunk_count
        != sum(document.chunk_count for document in runtime_documents)
    ):
        raise CasebookRuntimeBindingError(
            "active generation does not cover the exact current chunk corpus"
        )
    if (
        not index.active_chunk_snapshot_sha256
        or index.index_job_chunk_snapshot_sha256 != index.active_chunk_snapshot_sha256
    ):
        raise CasebookRuntimeBindingError(
            "active generation chunk snapshot does not match its index job"
        )
    configured_embedding = (
        index.active_embedding_model,
        index.active_embedding_revision,
        index.active_embedding_dimension,
        index.active_embedding_representation,
    )
    expected_embedding = (
        expected_embedding_model,
        expected_embedding_revision,
        expected_embedding_dimension,
        expected_embedding_representation,
    )
    if configured_embedding != expected_embedding:
        raise CasebookRuntimeBindingError(
            "active generation embedding configuration does not match Formal B0"
        )

    by_key: dict[str, dict[str, object]] = {}
    reverse: dict[str, str] = {}
    for source_sha, (key, _) in sorted(
        manifest_by_sha.items(), key=lambda item: item[1][0]
    ):
        runtime = runtime_by_sha[source_sha]
        by_key[key] = {
            "document_version_id": runtime.document_version_id,
            "source_sha256": source_sha,
            "chunk_max_chars": runtime.chunk_max_chars,
        }
        reverse[str(runtime.document_version_id)] = key
    return {
        "status": "VERIFIED",
        "knowledge_base_id": index.knowledge_base_id,
        "eligible_runtime_document_count": len(runtime_documents),
        "active_generation_id": index.active_generation_id,
        "active_chunk_snapshot_sha256": index.active_chunk_snapshot_sha256,
        "active_generation_chunk_count": index.index_job_total_chunk_count,
        "index_state": {
            "index_status": index.index_status,
            "index_job_generation_id": index.index_job_generation_id,
            "index_job_status": index.index_job_status,
        },
        "embedding_configuration": {
            "model": index.active_embedding_model,
            "revision": index.active_embedding_revision,
            "dimension": index.active_embedding_dimension,
            "representation": index.active_embedding_representation,
        },
        "chunking_configuration": {
            "max_chunk_chars": expected_chunk_max_chars,
            "overlap": FORMAL_B0_CHUNK_OVERLAP,
        },
        "document_key_to_runtime": by_key,
        "document_version_id_to_document_key": reverse,
    }


class CasebookBaselineTracer:
    """In-process, non-durable Trace projection for one formal B0 run."""

    def __init__(self) -> None:
        self._traces: dict[int, _CasebookExecutionTrace] = {}

    def start(
        self, run_id: int, provider: str, model: str, include_content: bool
    ) -> _CasebookExecutionTrace:
        if run_id in self._traces:
            raise ValueError(f"duplicate traced run_id {run_id}")
        if include_content:
            raise ValueError("Formal B0 local Eval trace forbids full content capture")
        trace = _CasebookExecutionTrace(
            run_id=run_id,
            provider=provider,
            model=model,
            include_content=include_content,
        )
        self._traces[run_id] = trace
        return trace

    def snapshot(self, run_id: int) -> dict[str, Any] | None:
        trace = self._traces.get(run_id)
        return None if trace is None else trace.snapshot()


class _CasebookExecutionTrace:
    def __init__(
        self,
        *,
        run_id: int,
        provider: str,
        model: str,
        include_content: bool,
    ) -> None:
        self._include_content = include_content
        self._started_monotonic = monotonic()
        self._closed = False
        self._value: dict[str, Any] = {
            "trace_id": str(uuid4()),
            "langley_run_id": run_id,
            "provider": provider,
            "requested_model": model,
            "content_included": include_content,
            "started_at": _utc_now(),
            "finished_at": None,
            "workflow_duration_ms": None,
            "final_status": "RUNNING",
            "error_code": None,
            "llm_rounds": [],
            "tool_calls": [],
            "knowledge_searches": [],
            "citation_validations": [],
        }

    def begin_llm(self, request: LLMRequest, round_: int) -> _CasebookLLMTrace:
        observation: dict[str, Any] = {
            "round": round_,
            "started_at": _utc_now(),
            "finished_at": None,
            "duration_ms": None,
            "ttft_ms": None,
            "finish_reason": None,
            "tool_call_count": None,
            "provider_model": None,
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "error_code": None,
        }
        if self._include_content:
            observation["request"] = {
                "system_input": request.system_input,
                "transcript_item_count": len(request.transcript),
                "personal_context": (
                    None
                    if request.personal_context is None
                    else list(request.personal_context)
                ),
            }
        self._value["llm_rounds"].append(observation)
        return _CasebookLLMTrace(
            observation=observation,
            include_content=self._include_content,
        )

    def begin_tool(self, call: ToolCall, tool_calls_used: int) -> _CasebookToolTrace:
        observation: dict[str, Any] = {
            "tool_name": call.name,
            "tool_call_id": call.call_id,
            "tool_calls_used": tool_calls_used,
            "started_at": _utc_now(),
            "finished_at": None,
            "duration_ms": None,
            "result_kind": None,
            "error_code": None,
        }
        if self._include_content:
            observation["raw_arguments"] = call.raw_arguments
        self._value["tool_calls"].append(observation)
        return _CasebookToolTrace(
            execution_trace=self,
            observation=observation,
            include_content=self._include_content,
        )

    def citation_validate(
        self,
        *,
        available_evidence_count: int,
        cited_handles: tuple[int, ...],
        cited_document_version_ids: tuple[int, ...],
        abstained: bool,
        error_code: str | None,
    ) -> None:
        self._value["citation_validations"].append(
            {
                "available_evidence_count": available_evidence_count,
                "cited_handles": list(cited_handles),
                "cited_document_version_ids": list(cited_document_version_ids),
                "abstained": abstained,
                "error_code": error_code,
            }
        )

    def success(self, answer: str, stop_reason: str = "FINAL_ANSWER") -> None:
        self._finish("SUCCEEDED", None)
        self._value["stop_reason"] = stop_reason
        if self._include_content:
            self._value["answer"] = answer

    def failure(self, error_code: str) -> None:
        self._finish("FAILED", error_code)

    def snapshot(self) -> dict[str, Any]:
        return deepcopy(self._value)

    def _finish(self, status: str, error_code: str | None) -> None:
        if self._closed:
            raise RuntimeError("execution trace is already closed")
        self._closed = True
        self._value["finished_at"] = _utc_now()
        self._value["workflow_duration_ms"] = _duration_ms(self._started_monotonic)
        self._value["final_status"] = status
        self._value["error_code"] = error_code


class _CasebookLLMTrace:
    def __init__(self, *, observation: dict[str, Any], include_content: bool) -> None:
        self._observation = observation
        self._include_content = include_content
        self._started_monotonic = monotonic()
        self._closed = False

    def content_delta(self, delta: AssistantContentDelta) -> None:
        if self._observation["ttft_ms"] is None and delta.content:
            self._observation["ttft_ms"] = _duration_ms(self._started_monotonic)

    def finish(self, response: LLMResponseCompleted) -> None:
        self._require_open()
        self._closed = True
        usage = response.usage
        input_tokens = None if usage is None else usage.input_tokens
        output_tokens = None if usage is None else usage.output_tokens
        self._observation.update(
            {
                "finished_at": _utc_now(),
                "duration_ms": _duration_ms(self._started_monotonic),
                "finish_reason": response.finish_reason.value,
                "tool_call_count": len(response.tool_calls),
                "provider_model": response.provider_model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": (
                    input_tokens + output_tokens
                    if input_tokens is not None and output_tokens is not None
                    else None
                ),
            }
        )
        if self._include_content:
            self._observation["response"] = {
                "assistant_content": response.assistant_content,
                "tool_call_names": [call.name for call in response.tool_calls],
            }

    def failure(self, error_code: str) -> None:
        self._require_open()
        self._closed = True
        self._observation.update(
            {
                "finished_at": _utc_now(),
                "duration_ms": _duration_ms(self._started_monotonic),
                "error_code": error_code,
            }
        )

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("LLM observation is already closed")


class _CasebookToolTrace:
    def __init__(
        self,
        *,
        execution_trace: _CasebookExecutionTrace,
        observation: dict[str, Any],
        include_content: bool,
    ) -> None:
        self._execution_trace = execution_trace
        self._observation = observation
        self._include_content = include_content
        self._started_monotonic = monotonic()
        self._closed = False

    def begin_knowledge_search(
        self,
        *,
        knowledge_base_id: int,
        top_k: int,
        query: str,
    ) -> _CasebookKnowledgeSearchTrace:
        observation: dict[str, Any] = {
            "knowledge_base_id": knowledge_base_id,
            "top_k": top_k,
            "query": query,
            "started_at": _utc_now(),
            "finished_at": None,
            "duration_ms": None,
            "hit_count": None,
            "error_code": None,
        }
        self._execution_trace._value["knowledge_searches"].append(observation)
        return _CasebookKnowledgeSearchTrace(observation)

    def finish(self, result: ToolResult | None, error_code: str | None = None) -> None:
        if self._closed:
            raise RuntimeError("Tool observation is already closed")
        self._closed = True
        self._observation.update(
            {
                "finished_at": _utc_now(),
                "duration_ms": _duration_ms(self._started_monotonic),
                "result_kind": None if result is None else result.kind.value,
                "error_code": error_code,
            }
        )
        if self._include_content and result is not None:
            self._observation["result"] = result.content


class _CasebookKnowledgeSearchTrace:
    def __init__(self, observation: dict[str, Any]) -> None:
        self._observation = observation
        self._started_monotonic = monotonic()
        self._closed = False

    def finish(self, hit_count: int | None, error_code: str | None = None) -> None:
        if self._closed:
            raise RuntimeError("Knowledge search observation is already closed")
        self._closed = True
        self._observation.update(
            {
                "finished_at": _utc_now(),
                "duration_ms": _duration_ms(self._started_monotonic),
                "hit_count": hit_count,
                "error_code": error_code,
            }
        )


def small_workload_distribution(values: list[float]) -> dict[str, object]:
    """Return deterministic nearest-rank summaries for the fixed Eval workload."""

    ordered = sorted(values)
    if not ordered:
        return {
            "sample_count": 0,
            "method": "nearest_rank",
            "p50": None,
            "p95": None,
            "max": None,
        }

    def percentile(fraction: float) -> float:
        index = max(0, math.ceil(fraction * len(ordered)) - 1)
        return round(ordered[index], 3)

    return {
        "sample_count": len(ordered),
        "method": "nearest_rank",
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": round(ordered[-1], 3),
    }


def _markdown_cell(value: object) -> str:
    if value is None:
        return "—"
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def render_observation_markdown(observation: dict[str, Any]) -> str:
    """Render only precomputed observation.json fields; never recompute metrics."""

    metrics = observation["metrics"]
    cases = observation["cases"]
    provenance = observation["provenance"]
    configuration = observation["configuration"]
    lines = [
        f"# Formal Observation {observation['experiment_id']}",
        "",
        f"- Run ID: `{observation['run_id']}`",
        f"- Status: `{observation['status']}`",
        f"- Started: `{observation['started_at']}`",
        f"- Finished: `{observation['finished_at']}`",
        "- Answer Point assessment: `PENDING_HUMAN_REVIEW`",
        "- Faithfulness: `PENDING_HUMAN_REVIEW`",
        "",
        "## Configuration and provenance",
        "",
        "```json",
        json.dumps(
            {
                "provenance": provenance,
                "configuration": configuration,
                "postflight": observation.get("postflight"),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        "```",
        "",
        "## Quality",
        "",
        "```json",
        json.dumps(metrics["quality"], ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "## Efficiency",
        "",
        (
            "Latency distributions below are measurements over this small fixed "
            "25-case Eval workload, not production SLA."
        ),
        "",
        "```json",
        json.dumps(metrics["efficiency"], ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "## Reliability",
        "",
        "```json",
        json.dumps(
            metrics["reliability"], ensure_ascii=False, indent=2, sort_keys=True
        ),
        "```",
        "",
        "## All-case summary",
        "",
        (
            "| Case | Final status | Abstained | Searches | Retrieval | Tokens | "
            "E2E ms | Flags |"
        ),
        "| --- | --- | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for case in cases:
        summary = case["summary"]
        lines.append(
            "| "
            + " | ".join(
                _markdown_cell(value)
                for value in (
                    case["case_id"],
                    summary["final_status"],
                    summary["abstained"],
                    summary["knowledge_search_count"],
                    summary["retrieval_supporting_evidence"],
                    summary["total_tokens"],
                    summary["e2e_workflow_duration_ms"],
                    ", ".join(case["bad_case_flags"]),
                )
            )
            + " |"
        )

    flagged_cases = observation["deterministic_flagged_cases"]
    lines.extend(("", "## Deterministic flags", ""))
    if not flagged_cases:
        lines.append(
            "No deterministic flags were emitted. This does not mean there are no "
            "Bad Cases; semantic Human Review is still pending."
        )
    for bad_case in flagged_cases:
        lines.extend(
            (
                f"### {bad_case['case_id']}",
                "",
                f"- Flags: `{', '.join(bad_case['bad_case_flags'])}`",
                f"- Query: {bad_case['query']}",
                f"- Actual answer: {bad_case.get('actual_answer') or '—'}",
                f"- Evidence gap: {bad_case.get('evidence_gap') or '—'}",
                "",
            )
        )
    lines.extend(
        (
            "",
            "## Human semantic review queue",
            "",
            "All ANSWERABLE cases require Answer Point and Faithfulness adjudication. ",
            "The deterministic metrics above are not full Generation/E2E correctness.",
        )
    )
    for case in cases:
        if case.get("answerability") != "ANSWERABLE":
            continue
        quality = case["quality"]
        lines.extend(
            (
                "",
                f"### {case['case_id']}",
                "",
                f"- Query: {case['query']}",
                f"- Actual answer: {case.get('actual_answer') or '—'}",
                f"- Reference answer: {case.get('reference_answer') or '—'}",
                (f"- Answer Point assessment: `{quality['answer_point_assessment']}`"),
                f"- Faithfulness: `{quality['faithfulness_assessment']}`",
                "",
                "```json",
                json.dumps(
                    {
                        "required_answer_points": case["required_answer_points"],
                        "golden_supporting_evidence": case[
                            "golden_supporting_evidence"
                        ],
                        "runtime_retrieval": case["retrieval"],
                        "citations": case["citations"],
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                ),
                "```",
            )
        )
    return "\n".join(lines).rstrip() + "\n"


def write_formal_observation(
    observation: dict[str, Any], output_directory: Path
) -> None:
    """Atomically persist one Formal Observation with exactly JSON and Markdown."""

    if output_directory.exists():
        raise FileExistsError(f"observation output already exists: {output_directory}")
    if observation.get("status") != "OBSERVATION_COMPLETE_PENDING_HUMAN_REVIEW":
        raise ValueError("Formal Observation must stop pending Human Review")
    temporary = output_directory.with_name(f".{output_directory.name}.part")
    if temporary.exists():
        raise FileExistsError(
            f"observation temporary output already exists: {temporary}"
        )
    temporary.mkdir(parents=True)
    json_path = temporary / "observation.json"
    markdown_path = temporary / "observation.md"
    json_path.write_text(
        json.dumps(observation, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    markdown_path.write_text(
        render_observation_markdown(observation),
        encoding="utf-8",
        newline="\n",
    )
    if sorted(path.name for path in temporary.iterdir()) != [
        "observation.json",
        "observation.md",
    ]:
        raise RuntimeError("Formal Observation directory has unexpected entries")
    temporary.replace(output_directory)
