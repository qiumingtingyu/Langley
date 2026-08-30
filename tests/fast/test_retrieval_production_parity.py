"""Pure deterministic coverage for Experiment #1 parity method mechanics."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

import langley.knowledge.retrieval_production_parity as parity
from langley.knowledge.chunking import CandidateChunk
from langley.knowledge.contracts import TextSpanRegion
from langley.knowledge.retrieval_eval import (
    CorpusPreflight,
    DetachedCandidate,
    GoldenCase,
    GoldenEvidence,
    LoadedGoldenCorpus,
)
from langley.knowledge.retrieval_production_parity import (
    CandidateIdentity,
    CaseTop5,
    DocumentMapping,
    ExperimentExecutionError,
    ExperimentSetupError,
    PersistedDocument,
    ProductionChunk,
    RankedIdentity,
    ReadyIndexFacts,
    aggregate_metrics,
    candidate_from_production_hit,
    compare_case,
    first_hit_top5,
    load_experiment_0_reference,
    poll_ready,
    render_result_markdown,
    run_production_parity,
    validate_candidate_parity,
    validate_document_mapping_parity,
    validate_ready_embedding_baseline,
    validate_source_parity,
)


def _candidate(
    document_key: str, ordinal: int, region: tuple[int, int], content: str | None = None
) -> DetachedCandidate:
    return DetachedCandidate(
        document_key=document_key,
        chunk=CandidateChunk(
            ordinal=ordinal,
            content=content or f"{document_key}-{ordinal}",
            heading_path=("Heading",),
            source_regions=(TextSpanRegion(*region),),
        ),
    )


def _corpus() -> LoadedGoldenCorpus:
    source = b"abcdefghij"
    case = GoldenCase("case", "exact query", "right", GoldenEvidence(3, 7, "defg"))
    return LoadedGoldenCorpus(
        "Test Golden",
        "a" * 64,
        {"right": source, "other": source},
        (_candidate("right", 1, (2, 8)), _candidate("other", 1, (0, 2))),
        (case,),
    )


def _top5(
    case_id: str, identities: list[tuple[str, int]], hit_rank: int | None
) -> CaseTop5:
    return CaseTop5(
        case_id,
        tuple(
            RankedIdentity(CandidateIdentity(*identity), 1.0) for identity in identities
        ),
        hit_rank,
    )


def _reference(
    corpus: LoadedGoldenCorpus, *, manifest: str | None = None
) -> dict[str, object]:
    ranking = [
        {
            "rank": 1,
            "document_key": "right",
            "ordinal": 1,
            "raw_score": 0.9,
            "hit": True,
        },
        {
            "rank": 2,
            "document_key": "other",
            "ordinal": 1,
            "raw_score": 0.8,
            "hit": False,
        },
    ]
    return {
        "dataset_manifest_sha256": manifest or corpus.dataset_manifest_sha256,
        "code_commit": "reference-commit",
        "embedding": {
            "model_id": "BAAI/bge-m3",
            "model_revision": "5617a9f61b028005a4858fdac845db406aefb181",
            "representation": "content_only",
        },
        "observed_embedding_dimension": 1024,
        "aggregate": {
            "hit_at_1": {"value": 1.0},
            "hit_at_3": {"value": 1.0},
            "hit_at_5": {"value": 1.0},
        },
        "cases": [{"case_id": "case", "ranking": ranking}],
    }


def test_reference_load_validates_manifest_and_derives_complete_ranking_top5(
    tmp_path: Path,
) -> None:
    corpus = _corpus()
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(_reference(corpus)), encoding="utf-8")

    provenance, cases = load_experiment_0_reference(path, corpus)

    assert provenance["artifact_sha256"] == sha256(path.read_bytes()).hexdigest()
    assert cases["case"].ranking == (
        RankedIdentity(CandidateIdentity("right", 1), 0.9),
        RankedIdentity(CandidateIdentity("other", 1), 0.8),
    )
    assert cases["case"].reciprocal_rank_at_5 == 1.0

    path.write_text(json.dumps(_reference(corpus, manifest="b" * 64)), encoding="utf-8")
    with pytest.raises(ExperimentSetupError, match="manifest"):
        load_experiment_0_reference(path, corpus)
    bad_embedding = _reference(corpus)
    bad_embedding["observed_embedding_dimension"] = 768
    path.write_text(json.dumps(bad_embedding), encoding="utf-8")
    with pytest.raises(ExperimentSetupError, match="embedding baseline"):
        load_experiment_0_reference(path, corpus)


def test_metrics_preserve_full_rank_semantics_but_cap_experiment_rr_at_five() -> None:
    rank_six = _top5(
        "six",
        [("wrong", 1), ("wrong", 2), ("wrong", 3), ("wrong", 4), ("wrong", 5)],
        None,
    )
    assert rank_six.reciprocal_rank_at_5 == 0.0
    aggregate = aggregate_metrics((_top5("one", [("right", 1)], 1), rank_six))
    assert aggregate.hit_at_1 == pytest.approx(1 / 2)
    assert aggregate.hit_at_3 == pytest.approx(1 / 2)
    assert aggregate.hit_at_5 == pytest.approx(1 / 2)
    assert aggregate.mrr_at_5 == pytest.approx(1 / 2)


def test_source_parity_rejects_sha_missing_and_extra_documents() -> None:
    corpus = _corpus()
    expected = {
        key: sha256(value).hexdigest()
        for key, value in corpus.source_bytes_by_document.items()
    }
    validate_source_parity(corpus, expected)
    for actual in (
        {"right": expected["right"], "other": "0" * 64},
        {"right": expected["right"]},
        {**expected, "extra": "a" * 64},
    ):
        with pytest.raises(ExperimentSetupError, match="source parity"):
            validate_source_parity(corpus, actual)


def test_candidate_parity_compares_every_field_not_just_counts() -> None:
    corpus = _corpus()
    version_mapping = {11: "right", 12: "other"}
    chunks = (
        ProductionChunk(11, 1, "right-1", ("Heading",), (TextSpanRegion(2, 8),)),
        ProductionChunk(12, 1, "other-1", ("Heading",), (TextSpanRegion(0, 2),)),
    )
    config = {11: 1200, 12: 1200}
    validate_candidate_parity(corpus, chunks, version_mapping, config)
    variants = (
        (
            ProductionChunk(11, 1, "changed", ("Heading",), (TextSpanRegion(2, 8),)),
            chunks[1],
        ),
        (
            ProductionChunk(11, 1, "right-1", ("Other",), (TextSpanRegion(2, 8),)),
            chunks[1],
        ),
        (
            ProductionChunk(11, 1, "right-1", ("Heading",), (TextSpanRegion(3, 8),)),
            chunks[1],
        ),
        (
            ProductionChunk(11, 2, "right-1", ("Heading",), (TextSpanRegion(2, 8),)),
            chunks[1],
        ),
    )
    for variant in variants:
        with pytest.raises(ExperimentSetupError, match="candidate parity"):
            validate_candidate_parity(corpus, variant, version_mapping, config)


def test_document_mapping_parity_requires_exact_persisted_kb_universe() -> None:
    corpus = _corpus()
    mappings = (
        DocumentMapping("right", 1, 11, sha256(b"abcdefghij").hexdigest()),
        DocumentMapping("other", 2, 12, sha256(b"abcdefghij").hexdigest()),
    )
    persisted = (
        PersistedDocument(1, 11, mappings[0].source_sha256),
        PersistedDocument(2, 12, mappings[1].source_sha256),
    )
    assert validate_document_mapping_parity(corpus, mappings, persisted) == {
        "right": mappings[0].source_sha256,
        "other": mappings[1].source_sha256,
    }
    for actual in (
        persisted[:1],
        (*persisted, PersistedDocument(3, 13, mappings[0].source_sha256)),
        (
            PersistedDocument(1, 11, mappings[0].source_sha256),
            PersistedDocument(3, 12, mappings[1].source_sha256),
        ),
        (PersistedDocument(1, 11, "0" * 64), persisted[1]),
    ):
        with pytest.raises(ExperimentSetupError):
            validate_document_mapping_parity(corpus, mappings, actual)


def test_retrieval_hit_mapping_reuses_full_containment_matcher() -> None:
    corpus = _corpus()
    hit = {
        "document_version_id": 11,
        "chunk_ordinal": 1,
        "content": "right-1",
        "heading_path": ["Heading"],
        "source_regions": [{"kind": "text_span", "start_byte": 2, "end_byte": 8}],
    }
    candidate = candidate_from_production_hit(hit, {11: "right"})
    result = first_hit_top5(corpus, corpus.cases[0], (candidate,))
    assert result.hit_at_5 is True
    partial = candidate_from_production_hit(
        {
            **hit,
            "source_regions": [{"kind": "text_span", "start_byte": 3, "end_byte": 6}],
        },
        {11: "right"},
    )
    assert first_hit_top5(corpus, corpus.cases[0], (partial,)).hit_at_5 is False
    with pytest.raises(ExperimentExecutionError, match="unknown"):
        candidate_from_production_hit(hit, {})
    with pytest.raises(ExperimentExecutionError, match="integer"):
        candidate_from_production_hit(
            {**hit, "document_version_id": "11"}, {11: "right"}
        )


def test_production_score_requires_finite_numeric_value() -> None:
    assert parity._production_score(0.5) == 0.5
    for value in (None, "0.5", True, float("nan"), float("inf")):
        with pytest.raises(ExperimentExecutionError, match="hit.score"):
            parity._production_score(value)


def test_comparison_records_overlap_entered_left_rank_changes_and_classification() -> (
    None
):
    baseline = _top5("case", [("a", 1), ("b", 1)], 1)
    assert (
        compare_case(baseline, _top5("case", [("a", 1), ("b", 1)], 1)).classification
        == "PARITY"
    )
    swapped = compare_case(baseline, _top5("case", [("b", 1), ("a", 1)], 2))
    assert swapped.classification == "RANKING_DIFFERENCE"
    assert swapped.overlap_count == 2
    assert swapped.rank_changes == (
        (CandidateIdentity("b", 1), 2, 1),
        (CandidateIdentity("a", 1), 1, 2),
    )
    changed = compare_case(baseline, _top5("case", [("a", 1), ("c", 1)], 1))
    assert changed.entered == (CandidateIdentity("c", 1),)
    assert changed.left == (CandidateIdentity("b", 1),)
    assert (
        compare_case(baseline, _top5("case", [("c", 1)], None)).classification
        == "REGRESSION_EVENT"
    )


def test_poll_ready_is_finite_and_never_retries_a_build() -> None:
    calls = 0

    def status() -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"index_status": "INDEXING", "latest_job": {"status": "RUNNING"}}

    clock = iter((0.0, 0.0, 1.0, 2.0))
    with pytest.raises(ExperimentExecutionError, match="TIMEOUT"):
        poll_ready(
            status,
            deadline_seconds=1.0,
            interval_seconds=0.0,
            now=lambda: next(clock),
            sleep=lambda _: None,
        )
    assert calls == 2
    with pytest.raises(ExperimentExecutionError, match="TERMINAL"):
        poll_ready(
            lambda: {"index_status": "INDEXING", "latest_job": {"status": "FAILED"}},
            sleep=lambda _: None,
        )


class _Response:
    def __init__(self, status_code: int, payload: dict[str, object]) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict[str, object]:
        return self._payload


class _CandidateMismatchClient:
    def __init__(self, source_sha: str) -> None:
        self.source_sha = source_sha
        self.posts: list[str] = []
        self.uploads = 0

    def post(self, url: str, **kwargs: object) -> _Response:
        self.posts.append(url)
        if url == "/api/knowledge-bases":
            return _Response(201, {"id": 7})
        if url.endswith("/documents"):
            self.uploads += 1
            return _Response(
                201,
                {
                    "id": 7 + self.uploads,
                    "source": {
                        "document_version_id": 8 + self.uploads,
                        "sha256": self.source_sha,
                    },
                },
            )
        if url.endswith("/chunks/rebuild"):
            return _Response(200, {})
        raise AssertionError(f"index build must not be reached: {url}")

    def get(self, url: str, **kwargs: object) -> _Response:
        if url == "/api/knowledge-bases/7/documents":
            return _Response(
                200,
                [
                    {
                        "id": 8,
                        "source": {"document_version_id": 9, "sha256": self.source_sha},
                    },
                    {
                        "id": 9,
                        "source": {
                            "document_version_id": 10,
                            "sha256": self.source_sha,
                        },
                    },
                ],
            )
        assert url in {
            "/api/document-versions/9/chunks",
            "/api/document-versions/10/chunks",
        }
        return _Response(
            200,
            {
                "successful_chunk_max_chars": 1200,
                "chunk_count": 1,
                "chunks": [
                    {
                        "ordinal": 1,
                        "content": "mismatch",
                        "heading_path": ["Heading"],
                        "source_regions": [
                            {"kind": "text_span", "start_byte": 2, "end_byte": 8}
                        ],
                    }
                ],
            },
        )


def test_candidate_setup_failure_prevents_index_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = _corpus()
    monkeypatch.setattr(
        parity,
        "corpus_preflight",
        lambda _: CorpusPreflight(40, 1003, 1, 1.0, 1, 13, 13, ()),
    )
    client = _CandidateMismatchClient(
        sha256(corpus.source_bytes_by_document["right"]).hexdigest()
    )
    with pytest.raises(ExperimentSetupError, match="candidate parity"):
        run_production_parity(client, corpus=corpus, reference={}, code_commit="test")
    assert not any(url.endswith("/index-build") for url in client.posts)


def test_source_setup_failure_prevents_processing_or_index_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = _corpus()
    monkeypatch.setattr(
        parity,
        "corpus_preflight",
        lambda _: CorpusPreflight(40, 1003, 1, 1.0, 1, 13, 13, ()),
    )
    client = _CandidateMismatchClient("0" * 64)
    with pytest.raises(ExperimentSetupError, match="source parity"):
        run_production_parity(client, corpus=corpus, reference={}, code_commit="test")
    assert not any(
        "chunks/rebuild" in url or "index-build" in url for url in client.posts
    )


class _ReadyBaselineClient:
    def __init__(self) -> None:
        self.query_count = 0
        self.uploads = 0

    def post(self, url: str, **kwargs: object) -> _Response:
        if url == "/api/knowledge-bases":
            return _Response(201, {"id": 7})
        if url.endswith("/documents"):
            self.uploads += 1
            return _Response(
                201,
                {
                    "id": self.uploads,
                    "source": {
                        "document_version_id": self.uploads,
                        "sha256": sha256(b"abcdefghij").hexdigest(),
                    },
                },
            )
        if url.endswith("/chunks/rebuild"):
            return _Response(200, {})
        if url.endswith("/index-build"):
            return _Response(202, {"job_id": 5})
        if url.endswith("/retrieval"):
            self.query_count += 1
            raise AssertionError("wrong READY baseline must prevent retrieval")
        raise AssertionError(url)

    def get(self, url: str, **kwargs: object) -> _Response:
        if url.endswith("/documents"):
            return _Response(200, [])
        raise AssertionError(url)


def test_wrong_ready_baseline_prevents_retrieval_queries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = _corpus()
    monkeypatch.setattr(
        parity,
        "corpus_preflight",
        lambda _: CorpusPreflight(40, 1003, 1, 1.0, 1, 13, 13, ()),
    )
    monkeypatch.setattr(parity, "validate_document_mapping_parity", lambda *_: {})
    monkeypatch.setattr(parity, "_read_all_chunks", lambda *_: ())
    monkeypatch.setattr(parity, "validate_candidate_parity", lambda *_: None)
    monkeypatch.setattr(parity, "poll_ready", lambda *_: {"index_status": "READY"})
    client = _ReadyBaselineClient()
    wrong = ReadyIndexFacts(
        active_embedding_model="wrong-model",
        active_embedding_revision="5617a9f61b028005a4858fdac845db406aefb181",
        active_embedding_dimension=1024,
        active_embedding_representation="source_context_v1",
    )
    with pytest.raises(ExperimentSetupError, match="READY embedding baseline"):
        run_production_parity(
            client,
            corpus=corpus,
            reference={},
            code_commit="test",
            ready_facts_reader=lambda _: wrong,
        )
    assert client.query_count == 0


@pytest.mark.parametrize(
    "facts",
    (
        ReadyIndexFacts(
            "wrong",
            "5617a9f61b028005a4858fdac845db406aefb181",
            1024,
            "source_context_v1",
        ),
        ReadyIndexFacts("BAAI/bge-m3", "0" * 40, 1024, "source_context_v1"),
        ReadyIndexFacts(
            "BAAI/bge-m3",
            "5617a9f61b028005a4858fdac845db406aefb181",
            768,
            "source_context_v1",
        ),
        ReadyIndexFacts(
            "BAAI/bge-m3",
            "5617a9f61b028005a4858fdac845db406aefb181",
            1024,
            "other",
        ),
    ),
)
def test_ready_baseline_rejects_each_frozen_fact(facts: ReadyIndexFacts) -> None:
    with pytest.raises(ExperimentSetupError, match="READY embedding baseline"):
        validate_ready_embedding_baseline(facts)


def test_rendered_report_exposes_computed_human_review_facts() -> None:
    result = {
        "experiment": "Experiment #1",
        "status": "COMPUTED_STOP_FOR_HUMAN",
        "code_commit": "method-commit",
        "golden_manifest_sha256": "a" * 64,
        "reference_experiment": {
            "path": "reference.json",
            "artifact_sha256": "b" * 64,
            "code_commit": "reference-commit",
            "dataset_manifest_sha256": "a" * 64,
        },
        "setup_parity": {"source": "PASS", "candidate": "PASS"},
        "ready_index_facts": {},
        "embedding": {
            "model": "BAAI/bge-m3",
            "revision": "d" * 40,
            "dimension": 1024,
            "representation": "source_context_v1",
            "device": "cuda:0",
        },
        "qdrant": {"collection": "langley_knowledge_dense_v2", "distance": "COSINE"},
        "aggregate": {
            "experiment_0": {
                "hit_at_1": 1.0,
                "hit_at_3": 1.0,
                "hit_at_5": 1.0,
                "mrr_at_5": 1.0,
            },
            "experiment_1": {
                "hit_at_1": 0.0,
                "hit_at_3": 0.0,
                "hit_at_5": 0.0,
                "mrr_at_5": 0.0,
            },
            "deltas": {
                "hit_at_1": -1.0,
                "hit_at_3": -1.0,
                "hit_at_5": -1.0,
                "mrr_at_5": -1.0,
            },
        },
        "cases": [
            {
                "case_id": "case",
                "classification": "REGRESSION_EVENT",
                "overlap_count": 0,
                "entered": [],
                "left": [],
                "rank_changes": [],
                "experiment_0": {
                    "first_hit_rank": 1,
                    "top5": [
                        {
                            "identity": {"document_key": "old", "ordinal": 1},
                            "score": 0.9,
                        }
                    ],
                },
                "experiment_1": {
                    "first_hit_rank": None,
                    "top5": [
                        {
                            "identity": {"document_key": "new", "ordinal": 1},
                            "score": 0.8,
                        }
                    ],
                },
            }
        ],
    }
    rendered = render_result_markdown(result)
    for expected in (
        "method-commit",
        "reference.json",
        "Source Parity",
        "READY projection",
        "BGE model",
        "Qdrant collection",
        "REGRESSION_EVENT",
        "#0 Top5",
        "#1 Top5",
        "Rank changes",
    ):
        assert expected in rendered
