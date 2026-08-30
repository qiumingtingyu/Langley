"""Focused Formal Observation, runtime binding, and local trace contracts."""

import json
from pathlib import Path

import pytest

from langley.answering.contracts import ToolCall, ToolResult, ToolResultKind
from langley.knowledge.casebook_baseline import (
    CasebookBaselineTracer,
    CasebookRuntimeBindingError,
    RuntimeDocumentFact,
    RuntimeIndexFact,
    verify_runtime_corpus_binding,
    write_formal_observation,
)


def _manifest_documents() -> list[dict[str, object]]:
    return [
        {"key": f"document-{index:02d}", "sha256": f"{index:064x}"}
        for index in range(40)
    ]


def _runtime_documents() -> list[RuntimeDocumentFact]:
    return [
        RuntimeDocumentFact(
            document_version_id=1000 + index,
            source_sha256=f"{index:064x}",
            chunk_count=1,
            chunk_max_chars=1200,
        )
        for index in range(40)
    ]


def _index_fact(*, status: str = "READY") -> RuntimeIndexFact:
    return RuntimeIndexFact(
        knowledge_base_id=7,
        index_status=status,
        active_embedding_model="BAAI/bge-m3",
        active_embedding_revision="revision",
        active_embedding_dimension=1024,
        active_embedding_representation="source_context_v1",
        index_job_status="SUCCEEDED",
        index_job_chunk_snapshot_sha256="a" * 64,
        index_job_processed_chunk_count=40,
        index_job_total_chunk_count=40,
    )


def _binding(
    manifest: list[dict[str, object]],
    runtime: list[RuntimeDocumentFact],
    *,
    index: RuntimeIndexFact | None = None,
) -> dict[str, object]:
    return verify_runtime_corpus_binding(
        manifest,
        runtime,
        index or _index_fact(),
        expected_embedding_model="BAAI/bge-m3",
        expected_embedding_revision="revision",
        expected_embedding_dimension=1024,
        expected_embedding_representation="source_context_v1",
        expected_chunk_max_chars=1200,
    )


def test_exact_runtime_corpus_binding_persists_both_identity_directions() -> None:
    result = _binding(_manifest_documents(), _runtime_documents())

    assert result["status"] == "VERIFIED"
    assert result["eligible_runtime_document_count"] == 40
    by_key = result["document_key_to_runtime"]
    reverse = result["document_version_id_to_document_key"]
    assert isinstance(by_key, dict)
    assert isinstance(reverse, dict)
    assert by_key["document-07"] == {
        "document_version_id": 1007,
        "source_sha256": f"{7:064x}",
        "chunk_max_chars": 1200,
    }
    assert reverse["1007"] == "document-07"
    assert result["chunking_configuration"] == {
        "max_chunk_chars": 1200,
        "overlap": 0,
    }
    assert result["index_state"] == {
        "index_status": "READY",
        "index_job_status": "SUCCEEDED",
    }


@pytest.mark.parametrize("mutation", ["missing", "extra", "mismatch", "ambiguous"])
def test_runtime_corpus_binding_blocks_non_exact_corpora(mutation: str) -> None:
    manifest = _manifest_documents()
    runtime = _runtime_documents()
    if mutation == "missing":
        runtime.pop()
    elif mutation == "extra":
        runtime.append(RuntimeDocumentFact(9999, "f" * 64, 1, 1200))
    elif mutation == "mismatch":
        runtime[-1] = RuntimeDocumentFact(
            runtime[-1].document_version_id, "f" * 64, 1, 1200
        )
    else:
        runtime[-1] = RuntimeDocumentFact(
            runtime[-1].document_version_id,
            runtime[0].source_sha256,
            1,
            1200,
        )

    with pytest.raises(CasebookRuntimeBindingError):
        _binding(manifest, runtime)


def test_runtime_corpus_binding_rejects_wrong_chunk_max_chars() -> None:
    runtime = _runtime_documents()
    runtime[17] = RuntimeDocumentFact(
        document_version_id=runtime[17].document_version_id,
        source_sha256=runtime[17].source_sha256,
        chunk_count=runtime[17].chunk_count,
        chunk_max_chars=1199,
    )

    with pytest.raises(CasebookRuntimeBindingError, match="chunk_max_chars"):
        _binding(_manifest_documents(), runtime)


def test_runtime_corpus_binding_requires_ready_exact_generation() -> None:
    with pytest.raises(CasebookRuntimeBindingError, match="not READY"):
        _binding(
            _manifest_documents(),
            _runtime_documents(),
            index=_index_fact(status="STALE"),
        )


def test_local_eval_trace_keeps_only_minimal_agent_search_diagnostic() -> None:
    tracer = CasebookBaselineTracer()
    execution = tracer.start(11, "qwen", "model", include_content=False)
    tool = execution.begin_tool(
        ToolCall("search-1", "search_knowledge", '{"query":"private raw"}'), 1
    )
    search = tool.begin_knowledge_search(
        knowledge_base_id=7, top_k=5, query="actual model search"
    )
    search.finish(3)
    tool.finish(
        ToolResult(
            "search-1", "search_knowledge", ToolResultKind.SUCCESS, "full result"
        )
    )
    execution.success("private answer")

    snapshot = tracer.snapshot(11)
    assert snapshot is not None
    assert snapshot["content_included"] is False
    search_observation = snapshot["knowledge_searches"][0]
    assert search_observation["query"] == "actual model search"
    assert search_observation["top_k"] == 5
    assert search_observation["hit_count"] == 3
    assert "raw_arguments" not in snapshot["tool_calls"][0]
    assert "result" not in snapshot["tool_calls"][0]
    assert "answer" not in snapshot


def test_local_eval_trace_refuses_full_content_capture() -> None:
    tracer = CasebookBaselineTracer()

    with pytest.raises(ValueError, match="forbids full content"):
        tracer.start(11, "qwen", "model", include_content=True)


def test_observation_json_and_markdown_share_one_pending_human_authority(
    tmp_path: Path,
) -> None:
    observation = {
        "experiment_id": "knowledge-casebook-v1-b0",
        "run_id": "run-1",
        "status": "OBSERVATION_COMPLETE_PENDING_HUMAN_REVIEW",
        "started_at": "2026-08-26T00:00:00+00:00",
        "finished_at": "2026-08-26T00:01:00+00:00",
        "provenance": {"git_commit": "abc", "casebook_id": "knowledge-casebook-v1"},
        "configuration": {"provider": "qwen", "latency_policy": "WARM_STEADY_STATE"},
        "metrics": {
            # Deliberately differs from len(cases); Markdown must not recompute it.
            "quality": {
                "evaluated_case_count": 999,
                "answer_point_assessment": "PENDING_HUMAN_REVIEW",
                "faithfulness_assessment": "PENDING_HUMAN_REVIEW",
            },
            "efficiency": {"e2e_workflow_duration_ms": {"p50": 12.5}},
            "reliability": {"succeeded_case_count": 1},
        },
        "cases": [
            {
                "case_id": "case-1",
                "summary": {
                    "final_status": "SUCCEEDED",
                    "abstained": False,
                    "knowledge_search_count": 1,
                    "retrieval_supporting_evidence": "ALL_RETRIEVED",
                    "total_tokens": 12,
                    "e2e_workflow_duration_ms": 25.0,
                },
                "bad_case_flags": [],
            }
        ],
        "deterministic_flagged_cases": [],
    }
    output = tmp_path / "formal-observation"

    write_formal_observation(observation, output)

    assert sorted(path.name for path in output.iterdir()) == [
        "observation.json",
        "observation.md",
    ]
    assert json.loads((output / "observation.json").read_text(encoding="utf-8")) == (
        observation
    )
    markdown = (output / "observation.md").read_text(encoding="utf-8")
    assert '"evaluated_case_count": 999' in markdown
    assert "PENDING_HUMAN_REVIEW" in markdown
    assert "does not mean there are no Bad Cases" in markdown
    assert "not full Generation/E2E correctness" in markdown
