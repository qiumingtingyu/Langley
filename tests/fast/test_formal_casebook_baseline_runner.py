"""Focused repair contracts for the private Formal B0 Observation runner."""

from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from eval.slice6 import run_formal_casebook_baseline as runner


class _Response:
    status_code = 200
    headers = {"X-Request-ID": "request-1"}

    def __init__(
        self, generation_id: str = "11111111-1111-4111-8111-111111111111"
    ) -> None:
        self._generation_id = generation_id

    def json(self) -> dict[str, object]:
        return {
            "generation_id": self._generation_id,
            "hits": [
                {
                    "rank": 1,
                    "document_version_id": 101,
                    "source_sha256": "a" * 64,
                }
            ],
        }


class _WarmupClient:
    def __init__(
        self, generation_id: str = "11111111-1111-4111-8111-111111111111"
    ) -> None:
        self.requests: list[tuple[str, str, object]] = []
        self.generation_id = generation_id

    async def request(self, method: str, path: str, *, json: object) -> _Response:
        self.requests.append((method, path, json))
        return _Response(self.generation_id)


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        casebook=Path("casebook.json"),
        manifest=Path("manifest.json"),
        corpus_root=Path("corpus"),
        eval_user_id=77,
        knowledge_base_id=7,
        timeout_per_case=10.0,
    )


def test_formal_eval_settings_use_explicit_user_and_disable_memory_processing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGLEY_LOCAL_USER_ID", "999")
    monkeypatch.setenv("LANGLEY_MEMORY_POLICY_MODEL", "memory-model")
    monkeypatch.setenv("LANGLEY_TRACING_ENABLED", "true")
    monkeypatch.setenv("LANGLEY_TRACE_CONTENT_ENABLED", "true")

    settings = runner._settings_for_eval(77)

    assert settings.local_user_id == 77
    assert settings.memory_policy_model is None
    assert settings.tracing_enabled is False
    assert settings.trace_content_enabled is False


def test_eval_user_with_active_memory_blocks_without_mutation() -> None:
    with pytest.raises(runner.BaselineInputError, match="found 2"):
        runner._verified_eval_user_isolation(
            eval_user_id=77,
            user_exists=True,
            active_memory_count=2,
            auto_memory_enabled=True,
        )


def _retrieval_case() -> dict[str, object]:
    return {
        "supporting_evidence": [
            {
                "evidence_id": "E1",
                "document_key": "doc-a",
                "start_byte": 100,
                "end_byte": 200,
            }
        ]
    }


def _runtime_binding() -> dict[str, object]:
    return {
        "document_key_to_runtime": {
            "doc-a": {"document_version_id": 101, "source_sha256": "a" * 64}
        },
        "document_version_id_to_document_key": {"101": "doc-a"},
    }


def test_adjacent_partial_hits_cannot_be_unioned_into_one_golden_hit() -> None:
    hits = [
        {
            "rank": 1,
            "document_version_id": 101,
            "source_sha256": "a" * 64,
            "source_regions": [{"start_byte": 100, "end_byte": 150}],
        },
        {
            "rank": 2,
            "document_version_id": 101,
            "source_sha256": "a" * 64,
            "source_regions": [{"start_byte": 150, "end_byte": 200}],
        },
    ]

    assessment = runner._retrieval_assessment(
        _retrieval_case(), hits, _runtime_binding()
    )

    assert assessment["all_supporting_evidence_retrieved"] is False
    assert assessment["first_complete_rank"] is None


def test_one_individual_region_fully_containing_golden_is_a_hit() -> None:
    hits = [
        {
            "rank": 3,
            "document_version_id": 101,
            "source_sha256": "a" * 64,
            "source_regions": [{"start_byte": 90, "end_byte": 210}],
        }
    ]

    assessment = runner._retrieval_assessment(
        _retrieval_case(), hits, _runtime_binding()
    )

    assert assessment["all_supporting_evidence_retrieved"] is True
    assert assessment["first_complete_rank"] == 3


def test_unknown_runtime_document_version_id_is_execution_error() -> None:
    with pytest.raises(runner.BaselineInputError, match="unknown runtime"):
        runner._annotate_retrieval_hits(
            [
                {
                    "rank": 1,
                    "document_version_id": 202,
                    "source_sha256": "a" * 64,
                }
            ],
            _runtime_binding(),
        )


def test_verified_runtime_version_with_sha_mismatch_is_execution_error() -> None:
    with pytest.raises(runner.BaselineInputError, match="source SHA-256"):
        runner._annotate_retrieval_hits(
            [
                {
                    "rank": 1,
                    "document_version_id": 101,
                    "source_sha256": "b" * 64,
                }
            ],
            _runtime_binding(),
        )


def test_search_decision_uses_tool_call_even_when_arguments_or_result_fail() -> None:
    tool_calls = [
        {
            "tool_name": "search_knowledge",
            "result_kind": "ERROR",
            "error_code": "TOOL_ARGUMENTS_INVALID",
        }
    ]

    count = runner._search_tool_call_count(tool_calls)

    assert count == 1
    assert runner._search_expectation_met("REQUIRED", count) is True


def test_required_search_decision_fails_without_search_tool_call() -> None:
    tool_calls = [{"tool_name": "get_current_time", "result_kind": "SUCCESS"}]

    count = runner._search_tool_call_count(tool_calls)

    assert count == 0
    assert runner._search_expectation_met("REQUIRED", count) is False


def test_verified_retrieval_hits_receive_document_key_annotation() -> None:
    hits = [
        {
            "rank": 1,
            "document_version_id": 101,
            "source_sha256": "a" * 64,
        }
    ]

    annotated = runner._annotate_retrieval_hits(hits, _runtime_binding())

    assert annotated[0]["verified_document_key"] == "doc-a"
    assert annotated[0]["source_sha256_matches_verified_binding"] is True


@pytest.mark.anyio
async def test_explicit_warmup_occurs_once_and_is_not_a_scored_sample() -> None:
    client = _WarmupClient()

    warmup = await runner._run_retrieval_warmup(
        client,
        knowledge_base_id=7,
        expected_generation_id="11111111-1111-4111-8111-111111111111",
        runtime_corpus_binding=_runtime_binding(),
    )

    assert client.requests == [
        (
            "POST",
            "/api/knowledge-bases/7/retrieval",
            {"query": runner.RETRIEVAL_WARMUP_QUERY, "top_k": 5},
        )
    ]
    assert warmup["status"] == "SUCCEEDED"
    assert warmup["hit_count"] == 1
    assert warmup["excluded_from_scored_metrics"] is True

    case = {
        "answerability": "ANSWERABLE",
        "quality": {
            "answerability_abstention_structure_matches": True,
            "knowledge_search_expectation_met": True,
            "supporting_evidence_retrieved": True,
        },
        "e2e_workflow_duration_ms": 20.0,
        "llm_rounds": [],
        "knowledge_searches": [],
        "total_tokens": None,
        "final_status": "SUCCEEDED",
        "failure": {"run_error_code": None},
        "bad_case_flags": [],
        "llm_round_count": 0,
        "tool_call_count": 0,
        "knowledge_search_tool_call_count": 0,
        "knowledge_search_count": 0,
        "retrieval": {"call": {"elapsed_ms": 10.0}},
    }
    metrics = runner._aggregate([case])
    efficiency = metrics["efficiency"]
    assert isinstance(efficiency, dict)
    assert efficiency["latency_policy"] == "WARM_STEADY_STATE"
    assert efficiency["direct_retrieval_duration_ms"]["p50"] == 10.0


@pytest.mark.anyio
async def test_warmup_generation_mismatch_aborts() -> None:
    client = _WarmupClient("22222222-2222-4222-8222-222222222222")

    with pytest.raises(runner.BaselineInputError, match="warm-up.*generation drift"):
        await runner._run_retrieval_warmup(
            client,
            knowledge_base_id=7,
            expected_generation_id="11111111-1111-4111-8111-111111111111",
            runtime_corpus_binding=_runtime_binding(),
        )


@pytest.mark.anyio
async def test_case_direct_retrieval_generation_mismatch_aborts() -> None:
    client = _WarmupClient("22222222-2222-4222-8222-222222222222")
    tracer = runner.CasebookBaselineTracer()
    case = {
        "case_id": "case-1",
        "query": "query",
    }

    with pytest.raises(
        runner.BaselineInputError, match="case case-1.*generation drift"
    ):
        await runner._run_case(
            client,
            tracer,
            case,
            knowledge_base_id=7,
            formal_run_id="formal-run",
            timeout_seconds=1.0,
            runtime_corpus_binding=_runtime_binding(),
            expected_generation_id="11111111-1111-4111-8111-111111111111",
        )


@pytest.mark.anyio
async def test_dirty_gate_stops_before_app_or_model_composition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    composed = False

    def compose(_: int) -> Any:
        nonlocal composed
        composed = True
        raise AssertionError("settings must not be composed")

    monkeypatch.setattr(runner, "validate_casebook", lambda *args: None)
    monkeypatch.setattr(runner, "_git_dirty_state", lambda: (True, True))
    monkeypatch.setattr(runner, "_settings_for_eval", compose)

    with pytest.raises(runner.BaselineInputError, match="tracked worktree is dirty"):
        await runner._run(_args())

    assert composed is False


@pytest.mark.anyio
async def test_invalid_runtime_binding_stops_before_warmup_or_provider_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"warmup": 0, "case": 0}
    settings = SimpleNamespace(
        memory_policy_model=None,
        trace_content_enabled=False,
        tracing_enabled=False,
    )

    @asynccontextmanager
    async def lifespan_context(app: object):
        del app
        yield

    app = SimpleNamespace(
        state=SimpleNamespace(session_factory=object()),
        router=SimpleNamespace(lifespan_context=lifespan_context),
    )

    async def invalid_preflight(*args: object, **kwargs: object) -> dict[str, Any]:
        del args, kwargs
        raise runner.BaselineInputError("runtime corpus mismatch")

    async def warmup(*args: object, **kwargs: object) -> dict[str, object]:
        del args, kwargs
        calls["warmup"] += 1
        return {}

    async def run_case(*args: object, **kwargs: object) -> dict[str, object]:
        del args, kwargs
        calls["case"] += 1
        return {}

    monkeypatch.setattr(runner, "validate_casebook", lambda *args: None)
    monkeypatch.setattr(runner, "_git_dirty_state", lambda: (False, False))
    monkeypatch.setattr(runner, "_settings_for_eval", lambda _: settings)
    monkeypatch.setattr(runner, "_require_live_settings", lambda _: None)
    monkeypatch.setattr(
        runner,
        "_load_object",
        lambda _path, label: (
            {"cases": [{} for _ in range(25)]}
            if label == "Casebook"
            else {"documents": [{} for _ in range(40)]}
        ),
    )
    monkeypatch.setattr(runner, "create_app", lambda *args, **kwargs: app)
    monkeypatch.setattr(runner, "_load_runtime_preflight", invalid_preflight)
    monkeypatch.setattr(runner, "_run_retrieval_warmup", warmup)
    monkeypatch.setattr(runner, "_run_case", run_case)

    with pytest.raises(runner.BaselineInputError, match="runtime corpus mismatch"):
        await runner._run(_args())

    assert calls == {"warmup": 0, "case": 0}


@pytest.mark.anyio
@pytest.mark.parametrize("drift", ["generation", "corpus", "chunking"])
async def test_final_postflight_runtime_drift_aborts(
    monkeypatch: pytest.MonkeyPatch, drift: str
) -> None:
    initial = {
        "active_generation_id": "11111111-1111-4111-8111-111111111111",
        "active_chunk_snapshot_sha256": "a" * 64,
        "chunking_configuration": {"max_chunk_chars": 1200, "overlap": 0},
        "document_key_to_runtime": {"doc-a": {"document_version_id": 101}},
    }
    final = {
        **initial,
        "chunking_configuration": dict(initial["chunking_configuration"]),
        "document_key_to_runtime": dict(initial["document_key_to_runtime"]),
    }
    if drift == "generation":
        final["active_generation_id"] = "22222222-2222-4222-8222-222222222222"
    elif drift == "corpus":
        final["document_key_to_runtime"] = {"doc-a": {"document_version_id": 202}}
    else:
        final["chunking_configuration"] = {"max_chunk_chars": 1199, "overlap": 0}

    async def load_final(*args: object, **kwargs: object) -> dict[str, Any]:
        del args, kwargs
        return {"eval_user": {"exists": True}, "runtime_corpus_binding": final}

    monkeypatch.setattr(runner, "_load_runtime_preflight", load_final)

    with pytest.raises(runner.BaselineInputError, match="postflight"):
        await runner._load_runtime_postflight(
            object(),
            eval_user_id=77,
            knowledge_base_id=7,
            manifest_documents=[],
            settings=SimpleNamespace(),  # type: ignore[arg-type]
            initial_binding=initial,
        )


def test_observation_contract_stops_for_human_semantic_review() -> None:
    assert runner.OBSERVATION_ROOT.as_posix().endswith(
        "eval/runs/knowledge_casebook_v1/b0"
    )
    assert runner.OBSERVATION_STATUS == "OBSERVATION_COMPLETE_PENDING_HUMAN_REVIEW"
    assert runner._semantic_human_review_fields("ANSWERABLE") == {
        "answer_point_assessment": "PENDING_HUMAN_REVIEW",
        "faithfulness_assessment": "PENDING_HUMAN_REVIEW",
    }
    assert runner.FORMAL_B0_MAX_CHUNK_CHARS == 1200
    assert runner.FORMAL_B0_CHUNK_OVERLAP == 0
    settings = SimpleNamespace(
        llm_provider="qwen",
        llm_model="qwen-model",
        max_llm_rounds=4,
        max_tool_calls=4,
        overall_workflow_deadline_seconds=120.0,
        knowledge_embedding_model="BAAI/bge-m3",
        knowledge_embedding_revision="revision",
        knowledge_embedding_device="cpu",
        knowledge_embedding_dimension=1024,
        knowledge_embedding_representation="content_only",
    )
    configuration = runner._configuration(settings, "runner-sha")  # type: ignore[arg-type]
    assert configuration["chunking"] == {"max_chunk_chars": 1200, "overlap": 0}
