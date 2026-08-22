"""Fast deterministic coverage for the offline Task 4 Retrieval Eval mechanics."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

import langley.knowledge.retrieval_eval_entrypoint as retrieval_eval_entrypoint
from langley.knowledge.chunking import CandidateChunk, ChunkingConfig
from langley.knowledge.contracts import TextSpanRegion
from langley.knowledge.retrieval_eval import (
    DetachedCandidate,
    EmbeddingMetadata,
    GoldenCase,
    GoldenEvidence,
    LoadedGoldenCorpus,
    RetrievalEvalError,
    corpus_preflight,
    evaluate_case,
    load_golden_corpus,
    normalize_embeddings,
    rank_candidates,
    render_result_json,
    render_result_markdown,
    run_evaluation,
)
from langley.knowledge.retrieval_eval_entrypoint import (
    _operational_vector_facts,
    _require_immutable_model_revision,
    _run_experiment,
    _run_preflight,
)


class FixedEmbedding:
    def __init__(self, documents: object, queries: object) -> None:
        self.documents = documents
        self.queries = queries
        self.document_inputs: tuple[str, ...] | None = None
        self.query_inputs: tuple[str, ...] | None = None

    def embed_documents(self, values: object) -> object:
        self.document_inputs = tuple(values)
        return self.documents

    def embed_queries(self, values: object) -> object:
        self.query_inputs = tuple(values)
        return self.queries


def _write_dataset(root: Path, source: bytes = b"# Heading\nexact evidence\n") -> Path:
    documents = root / "documents"
    documents.mkdir(parents=True)
    (documents / "one.md").write_bytes(source)
    evidence_start = source.index(b"exact") if b"exact evidence" in source else 0
    evidence_bytes = b"exact evidence" if b"exact evidence" in source else source[:1]
    payload = {
        "name": "Test Golden",
        "documents": [
            {
                "key": "one",
                "path": "documents/one.md",
                "sha256": sha256(source).hexdigest(),
            }
        ],
        "cases": [
            {
                "case_id": "case-1",
                "query": "exact query",
                "document_key": "one",
                "evidence": {
                    "start_byte": evidence_start,
                    "end_byte": evidence_start + len(evidence_bytes),
                    "text": evidence_bytes.decode("utf-8", errors="replace"),
                },
            }
        ],
    }
    (root / "dataset.json").write_text(json.dumps(payload), encoding="utf-8")
    return root


def _candidate(
    document_key: str, ordinal: int, region: tuple[int, int]
) -> DetachedCandidate:
    return DetachedCandidate(
        document_key=document_key,
        chunk=CandidateChunk(
            ordinal=ordinal,
            content=f"content-{document_key}-{ordinal}",
            heading_path=("Heading",),
            source_regions=(TextSpanRegion(*region),),
        ),
    )


def _metadata() -> EmbeddingMetadata:
    return EmbeddingMetadata(
        model_id="fixed-test",
        model_revision="test-revision",
        sentence_transformers_version=None,
        device="cpu",
    )


def test_loader_reuses_production_parser_and_chunker_and_preserves_fields(
    tmp_path: Path,
) -> None:
    corpus = load_golden_corpus(
        _write_dataset(tmp_path), config=ChunkingConfig(max_chunk_chars=321)
    )

    assert corpus.candidates[0].document_key == "one"
    assert corpus.candidates[0].identity == ("one", 1)
    assert corpus.candidates[0].chunk.content == "exact evidence\n"
    assert corpus.candidates[0].chunk.heading_path == ("Heading",)
    assert corpus.candidates[0].chunk.source_regions == (TextSpanRegion(10, 25),)
    assert corpus.chunking_config == ChunkingConfig(max_chunk_chars=321)


def test_loader_orders_detached_candidates_by_document_key_then_ordinal(
    tmp_path: Path,
) -> None:
    documents = tmp_path / "documents"
    documents.mkdir()
    sources = {"b": b"# B\nbody b\n", "a": b"# A\nbody a\n"}
    for key, source in sources.items():
        (documents / f"{key}.md").write_bytes(source)
    payload = {
        "name": "Unsorted Golden",
        "documents": [
            {
                "key": key,
                "path": f"documents/{key}.md",
                "sha256": sha256(sources[key]).hexdigest(),
            }
            for key in ("b", "a")
        ],
        "cases": [
            {
                "case_id": "case-a",
                "query": "query",
                "document_key": "a",
                "evidence": {"start_byte": 4, "end_byte": 10, "text": "body a"},
            }
        ],
    }
    (tmp_path / "dataset.json").write_text(json.dumps(payload), encoding="utf-8")

    corpus = load_golden_corpus(tmp_path)

    assert [candidate.identity for candidate in corpus.candidates] == [
        ("a", 1),
        ("b", 1),
    ]


@pytest.mark.parametrize(
    ("path", "source", "sha", "message"),
    [
        ("../escape.md", b"x", None, "unsafe"),
        ("documents/missing.md", b"x", None, "missing"),
        ("documents/one.md", b"\xff", None, "strict UTF-8"),
        ("documents/one.md", b"safe", "0" * 64, "SHA-256 mismatch"),
    ],
)
def test_loader_fails_closed_for_fixture_integrity(
    tmp_path: Path, path: str, source: bytes, sha: str | None, message: str
) -> None:
    root = _write_dataset(tmp_path, source)
    manifest_path = root / "dataset.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["documents"][0]["path"] = path
    if sha is not None:
        payload["documents"][0]["sha256"] = sha
    if path.endswith("missing.md"):
        (root / "documents" / "one.md").unlink()
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RetrievalEvalError, match=message):
        load_golden_corpus(root)


def test_normalization_rejects_malformed_rows_values_and_zero_norms() -> None:
    for values in (
        [[1.0, 0.0]],
        [[float("nan"), 0.0], [1.0, 0.0]],
        [[float("inf"), 0.0], [1.0, 0.0]],
        [[0.0, 0.0], [1.0, 0.0]],
    ):
        with pytest.raises(RetrievalEvalError):
            normalize_embeddings(values, expected_rows=2, role="document")

    normalized = normalize_embeddings([[3.0, 4.0]], expected_rows=1, role="query")
    assert normalized.dtype == np.float32
    assert np.linalg.norm(normalized[0]) == pytest.approx(1.0)


def test_complete_ranking_uses_raw_exact_tie_breaks() -> None:
    candidates = (
        _candidate("b", 2, (0, 1)),
        _candidate("a", 3, (1, 2)),
        _candidate("a", 1, (2, 3)),
    )

    ranked = rank_candidates(candidates, [0.99999999, 1.0, 1.0])

    assert [(candidate.identity, score, tied) for candidate, score, tied in ranked] == [
        (("a", 1), 1.0, True),
        (("a", 3), 1.0, True),
        (("b", 2), 0.99999999, False),
    ]


def test_matcher_requires_full_containment_and_allows_any_valid_region() -> None:
    source = b"abcdefghij"
    case = GoldenCase("case", "query", "right", GoldenEvidence(3, 7, "defg"))
    candidates = (
        _candidate("right", 1, (2, 6)),
        DetachedCandidate(
            "right",
            CandidateChunk(
                ordinal=2,
                content="multiple",
                heading_path=(),
                source_regions=(TextSpanRegion(0, 1), TextSpanRegion(3, 7)),
            ),
        ),
        _candidate("wrong", 1, (3, 7)),
    )
    corpus = LoadedGoldenCorpus(
        "golden", "sha", {"right": source, "wrong": source}, candidates, (case,)
    )
    ranked = rank_candidates(candidates, [3.0, 2.0, 1.0])

    result = evaluate_case(corpus, case, ranked)

    assert result.candidate_pool_has_valid_match is True
    assert result.first_valid_hit_rank == 2
    assert [item.is_hit for item in result.ranking] == [False, True, False]


def test_evaluation_keeps_exact_embedding_inputs_and_full_rank_metrics() -> None:
    source = b"abcdefghi"
    candidates = tuple(
        _candidate("doc", ordinal, (ordinal - 1, ordinal)) for ordinal in range(1, 7)
    )
    cases = (
        GoldenCase("rank-1", "q1", "doc", GoldenEvidence(0, 1, "a")),
        GoldenCase("rank-2", "q2", "doc", GoldenEvidence(1, 2, "b")),
        GoldenCase("rank-6", "q6", "doc", GoldenEvidence(5, 6, "f")),
        GoldenCase("no-match", "q0", "doc", GoldenEvidence(8, 9, "i")),
    )
    corpus = LoadedGoldenCorpus(
        "golden",
        "sha",
        {"doc": source},
        candidates,
        cases,
        chunking_config=ChunkingConfig(max_chunk_chars=321),
    )
    embedder = FixedEmbedding(
        [[6.0], [5.0], [4.0], [3.0], [2.0], [1.0]], [[1.0], [1.0], [1.0], [1.0]]
    )

    result = run_evaluation(
        corpus,
        embedder,
        code_commit="deadbeef",
        tracked_worktree_dirty=True,
        worktree_dirty=True,
        embedding_metadata=_metadata(),
    )

    assert embedder.document_inputs == tuple(
        candidate.chunk.content for candidate in candidates
    )
    assert embedder.query_inputs == tuple(case.query for case in cases)
    assert [case.first_valid_hit_rank for case in result.cases] == [1, 2, 6, None]
    assert result.hit_at_1.value == pytest.approx(1 / 4)
    assert result.hit_at_3.value == pytest.approx(2 / 4)
    assert result.hit_at_5.value == pytest.approx(2 / 4)
    assert result.mean_reciprocal_rank == pytest.approx((1 + 1 / 2 + 1 / 6) / 4)

    result_again = run_evaluation(
        corpus,
        embedder,
        code_commit="deadbeef",
        tracked_worktree_dirty=True,
        worktree_dirty=True,
        embedding_metadata=_metadata(),
    )
    assert render_result_json(result_again) == render_result_json(result)
    rendered = render_result_markdown(result)
    assert "Hit@1" in rendered
    assert "| 5 |" in rendered
    assert "| 6 |" in rendered
    assert "Heading" in rendered
    assert "Documents: 1" in rendered
    assert "Candidate chunks: 6" in rendered
    assert "Chunks/document (min / median / max): 6 / 6.0 / 6" in rendered
    assert "Query: q1" in rendered
    assert "Golden evidence: [0,1): a" in rendered
    assert "## NO_MATCHABLE_CANDIDATE" in rendered
    assert "no-match" in rendered
    rendered_json = render_result_json(result)
    assert '"max_chunk_chars": 321' in rendered_json
    assert '"raw_score"' in rendered_json
    assert '"tracked_worktree_dirty": true' in rendered_json
    assert '"worktree_dirty": true' in rendered_json


def test_evaluation_rejects_dimension_mismatch_and_duplicate_ranking_candidates() -> (
    None
):
    source = b"ab"
    candidates = (_candidate("doc", 1, (0, 1)), _candidate("doc", 2, (1, 2)))
    case = GoldenCase("case", "q", "doc", GoldenEvidence(0, 1, "a"))
    corpus = LoadedGoldenCorpus("golden", "sha", {"doc": source}, candidates, (case,))
    with pytest.raises(RetrievalEvalError, match="dimensions differ"):
        run_evaluation(
            corpus,
            FixedEmbedding([[1.0, 0.0], [0.0, 1.0]], [[1.0]]),
            code_commit="deadbeef",
            tracked_worktree_dirty=False,
            worktree_dirty=False,
            embedding_metadata=_metadata(),
        )
    with pytest.raises(RetrievalEvalError, match="every candidate exactly once"):
        evaluate_case(corpus, case, ((_candidate("doc", 1, (0, 1)), 1.0, False),) * 2)
    malformed = DetachedCandidate(
        "doc",
        CandidateChunk(ordinal=3, content="bad", heading_path=(), source_regions=()),
    )
    malformed_corpus = LoadedGoldenCorpus(
        "golden", "sha", {"doc": source}, (malformed,), (case,)
    )
    with pytest.raises(RetrievalEvalError, match="no source regions"):
        evaluate_case(malformed_corpus, case, ((malformed, 1.0, False),))


def test_corpus_preflight_reports_matchability_without_embedding(
    tmp_path: Path,
) -> None:
    corpus = load_golden_corpus(_write_dataset(tmp_path))

    preflight = corpus_preflight(corpus)

    assert preflight.document_count == 1
    assert preflight.candidate_chunk_count == 1
    assert preflight.approved_case_count == 1
    assert preflight.matchable_case_count == 1
    assert preflight.no_matchable_case_ids == ()


def test_preflight_entrypoint_writes_reproducible_no_model_artifact(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_root = _write_dataset(tmp_path / "dataset")
    output = tmp_path / "results" / "preflight.json"

    _run_preflight(dataset_root, output)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["model_activity"] == "none"
    assert payload["chunking_config"] == {"max_chunk_chars": 1200}
    assert payload["document_count"] == 1
    assert payload["matchable_case_count"] == 1
    assert "retrieval-corpus-preflight" in capsys.readouterr().out


def test_operational_preflight_exercises_tiny_query_and_document_batches() -> None:
    embedder = FixedEmbedding(
        documents=np.array([[3.0, 4.0], [5.0, 12.0]], dtype=np.float32),
        queries=np.array([[8.0, 6.0], [7.0, 24.0]], dtype=np.float32),
    )

    facts = _operational_vector_facts(embedder)

    assert embedder.query_inputs is not None
    assert embedder.document_inputs is not None
    assert facts == {
        "observed_dimension": 2,
        "dtype": "float32",
        "query_row_count": 2,
        "document_row_count": 2,
        "finite": True,
        "non_zero_norm": True,
        "normalized_norm_check": "pass",
    }


def test_experiment_rejects_floating_revision_before_model_activity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    valid_revision = "5617a9f61b028005a4858fdac845db406aefb181"
    assert _require_immutable_model_revision(valid_revision) == valid_revision

    def unexpected_model_activity(*args: object, **kwargs: object) -> object:
        raise AssertionError("invalid revision must not reach model activity")

    monkeypatch.setattr(
        retrieval_eval_entrypoint, "load_golden_corpus", unexpected_model_activity
    )
    monkeypatch.setattr(
        retrieval_eval_entrypoint,
        "SentenceTransformersEmbedding",
        unexpected_model_activity,
    )

    with pytest.raises(RetrievalEvalError, match="immutable 40-character"):
        _run_experiment(
            tmp_path / "dataset",
            tmp_path / "result.json",
            tmp_path / "result.md",
            model_revision="main",
            device=None,
        )
