"""Offline, deterministic mechanics for the Slice 6 Retrieval Golden Eval.

This module deliberately evaluates detached Task 2 chunks.  It has no database,
index, API, or model-loading side effects at import time.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from statistics import median
from typing import Any, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from langley.knowledge.chunking import (
    CandidateChunk,
    ChunkingConfig,
    build_candidate_chunks,
)
from langley.knowledge.contracts import TextSpanRegion
from langley.knowledge.markdown import parse_markdown


class RetrievalEvalError(RuntimeError):
    """Raised when a Golden Eval fact or its mechanics is malformed."""


@dataclass(frozen=True)
class GoldenEvidence:
    start_byte: int
    end_byte: int
    text: str


@dataclass(frozen=True)
class GoldenCase:
    case_id: str
    query: str
    document_key: str
    evidence: GoldenEvidence


@dataclass(frozen=True)
class DetachedCandidate:
    document_key: str
    chunk: CandidateChunk

    @property
    def identity(self) -> tuple[str, int]:
        return (self.document_key, self.chunk.ordinal)


@dataclass(frozen=True)
class LoadedGoldenCorpus:
    dataset_identity: str
    dataset_manifest_sha256: str
    source_bytes_by_document: dict[str, bytes]
    candidates: tuple[DetachedCandidate, ...]
    cases: tuple[GoldenCase, ...]
    chunking_config: ChunkingConfig = ChunkingConfig(max_chunk_chars=1200)


@dataclass(frozen=True)
class EmbeddingMetadata:
    model_id: str
    model_revision: str | None
    sentence_transformers_version: str | None
    device: str
    representation: str = "content_only"
    mode: str = "dense_only"
    query_prompt: str = "none"
    normalization: str = "l2"
    similarity: str = "normalized_dot_product_cosine"
    dtype: str = "float32"


class EmbeddingBoundary(Protocol):
    """The two roles needed by the frozen Experiment #0 input contract."""

    def embed_queries(self, queries: Sequence[str]) -> NDArray[np.float32]: ...

    def embed_documents(self, documents: Sequence[str]) -> NDArray[np.float32]: ...


@dataclass(frozen=True)
class RankedCandidate:
    rank: int
    candidate: DetachedCandidate
    raw_score: float
    is_hit: bool
    exact_tie: bool


@dataclass(frozen=True)
class Metric:
    numerator: int
    denominator: int
    value: float


@dataclass(frozen=True)
class CaseResult:
    case: GoldenCase
    candidate_pool_has_valid_match: bool
    first_valid_hit_rank: int | None
    hit_at_1: bool
    hit_at_3: bool
    hit_at_5: bool
    reciprocal_rank: float
    ranking: tuple[RankedCandidate, ...]


@dataclass(frozen=True)
class EvalResult:
    corpus: LoadedGoldenCorpus
    code_commit: str
    tracked_worktree_dirty: bool
    worktree_dirty: bool
    embedding_metadata: EmbeddingMetadata
    embedding_dimension: int
    cases: tuple[CaseResult, ...]
    hit_at_1: Metric
    hit_at_3: Metric
    hit_at_5: Metric
    mean_reciprocal_rank: float


@dataclass(frozen=True)
class CorpusPreflight:
    document_count: int
    candidate_chunk_count: int
    chunks_per_document_min: int
    chunks_per_document_median: float
    chunks_per_document_max: int
    approved_case_count: int
    matchable_case_count: int
    no_matchable_case_ids: tuple[str, ...]


def _nonblank_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RetrievalEvalError(f"{label} must be a nonblank string")
    return value


def _safe_document_path(root: Path, value: object, document_key: str) -> Path:
    path_value = _nonblank_string(value, f"document {document_key}.path")
    relative = PurePosixPath(path_value)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or relative.parts[:1] != ("documents",)
        or relative.suffix != ".md"
    ):
        raise RetrievalEvalError(f"document {document_key} has unsafe fixture path")
    documents_root = (root / "documents").resolve()
    resolved = (root / Path(*relative.parts)).resolve()
    if not resolved.is_relative_to(documents_root):
        raise RetrievalEvalError(
            f"document {document_key} fixture path escapes Golden root"
        )
    return resolved


def _parse_cases(raw_cases: object) -> tuple[GoldenCase, ...]:
    if not isinstance(raw_cases, list) or not raw_cases:
        raise RetrievalEvalError("dataset cases must be a non-empty list")
    cases: list[GoldenCase] = []
    case_ids: set[str] = set()
    for raw_case in raw_cases:
        if not isinstance(raw_case, dict):
            raise RetrievalEvalError("each Golden case must be an object")
        case_id = _nonblank_string(raw_case.get("case_id"), "case_id")
        if case_id in case_ids:
            raise RetrievalEvalError(f"duplicate case_id {case_id!r}")
        case_ids.add(case_id)
        evidence = raw_case.get("evidence")
        if not isinstance(evidence, dict):
            raise RetrievalEvalError(f"case {case_id} evidence must be an object")
        start, end = evidence.get("start_byte"), evidence.get("end_byte")
        if type(start) is not int or type(end) is not int:
            raise RetrievalEvalError(f"case {case_id} evidence bounds must be integers")
        cases.append(
            GoldenCase(
                case_id=case_id,
                query=_nonblank_string(raw_case.get("query"), f"case {case_id}.query"),
                document_key=_nonblank_string(
                    raw_case.get("document_key"), f"case {case_id}.document_key"
                ),
                evidence=GoldenEvidence(
                    start_byte=start,
                    end_byte=end,
                    text=_nonblank_string(
                        evidence.get("text"), f"case {case_id}.evidence.text"
                    ),
                ),
            )
        )
    return tuple(cases)


def load_golden_corpus(
    root: Path, config: ChunkingConfig = ChunkingConfig(max_chunk_chars=1200)
) -> LoadedGoldenCorpus:
    """Load only the source facts required by this offline Eval run, fail closed."""

    manifest_path = root / "dataset.json"
    try:
        manifest_bytes = manifest_path.read_bytes()
        raw_manifest = json.loads(manifest_bytes.decode("utf-8", errors="strict"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RetrievalEvalError(
            "Golden manifest is unreadable strict UTF-8 JSON"
        ) from error
    if not isinstance(raw_manifest, dict):
        raise RetrievalEvalError("Golden manifest must be an object")
    documents = raw_manifest.get("documents")
    if not isinstance(documents, list) or not documents:
        raise RetrievalEvalError("Golden manifest documents must be a non-empty list")

    source_bytes_by_document: dict[str, bytes] = {}
    candidates: list[DetachedCandidate] = []
    identities: set[tuple[str, int]] = set()
    for raw_document in documents:
        if not isinstance(raw_document, dict):
            raise RetrievalEvalError("each Golden document must be an object")
        document_key = _nonblank_string(raw_document.get("key"), "document key")
        if document_key in source_bytes_by_document:
            raise RetrievalEvalError(f"duplicate document_key {document_key!r}")
        fixture_path = _safe_document_path(root, raw_document.get("path"), document_key)
        try:
            source_bytes = fixture_path.read_bytes()
            source_bytes.decode("utf-8", errors="strict")
        except OSError as error:
            raise RetrievalEvalError(
                f"document {document_key} fixture is missing"
            ) from error
        except UnicodeDecodeError as error:
            raise RetrievalEvalError(
                f"document {document_key} is not strict UTF-8"
            ) from error
        expected_sha256 = _nonblank_string(
            raw_document.get("sha256"), f"document {document_key}.sha256"
        )
        if sha256(source_bytes).hexdigest() != expected_sha256:
            raise RetrievalEvalError(f"document {document_key} SHA-256 mismatch")
        source_bytes_by_document[document_key] = source_bytes
        for chunk in build_candidate_chunks(parse_markdown(source_bytes), config):
            candidate = DetachedCandidate(document_key=document_key, chunk=chunk)
            if candidate.identity in identities:
                raise RetrievalEvalError(
                    f"duplicate candidate identity {candidate.identity!r}"
                )
            identities.add(candidate.identity)
            candidates.append(candidate)

    ordered_candidates = tuple(
        sorted(candidates, key=lambda candidate: candidate.identity)
    )
    return LoadedGoldenCorpus(
        dataset_identity=_nonblank_string(raw_manifest.get("name"), "dataset name"),
        dataset_manifest_sha256=sha256(manifest_bytes).hexdigest(),
        source_bytes_by_document=source_bytes_by_document,
        candidates=ordered_candidates,
        cases=_parse_cases(raw_manifest.get("cases")),
        chunking_config=config,
    )


def _validate_case(corpus: LoadedGoldenCorpus, case: GoldenCase) -> None:
    source_bytes = corpus.source_bytes_by_document.get(case.document_key)
    if source_bytes is None:
        raise RetrievalEvalError(f"case {case.case_id} references an unknown document")
    if not case.query.strip():
        raise RetrievalEvalError(f"case {case.case_id} query is blank")
    evidence = case.evidence
    if not 0 <= evidence.start_byte < evidence.end_byte <= len(source_bytes):
        raise RetrievalEvalError(f"case {case.case_id} evidence bounds are invalid")
    try:
        actual_text = source_bytes[evidence.start_byte : evidence.end_byte].decode(
            "utf-8", errors="strict"
        )
    except UnicodeDecodeError as error:
        raise RetrievalEvalError(
            f"case {case.case_id} evidence is not UTF-8 aligned"
        ) from error
    if actual_text != evidence.text:
        raise RetrievalEvalError(
            f"case {case.case_id} evidence text does not match source"
        )


def candidate_is_golden_hit(
    corpus: LoadedGoldenCorpus, case: GoldenCase, candidate: DetachedCandidate
) -> bool:
    """Apply the frozen Golden full-containment relevance definition."""

    if candidate.document_key not in corpus.source_bytes_by_document:
        raise RetrievalEvalError(
            f"candidate {candidate.identity!r} has no source document"
        )
    source_length = len(corpus.source_bytes_by_document[candidate.document_key])
    if not candidate.chunk.source_regions:
        raise RetrievalEvalError(
            f"candidate {candidate.identity!r} has no source regions"
        )
    for region in candidate.chunk.source_regions:
        if not isinstance(region, TextSpanRegion) or not (
            0 <= region.start_byte < region.end_byte <= source_length
        ):
            raise RetrievalEvalError(
                f"candidate {candidate.identity!r} has malformed source region"
            )
    return candidate.document_key == case.document_key and any(
        region.start_byte <= case.evidence.start_byte
        and region.end_byte >= case.evidence.end_byte
        for region in candidate.chunk.source_regions
    )


def _validate_candidate_identities(candidates: Sequence[DetachedCandidate]) -> None:
    identities = [candidate.identity for candidate in candidates]
    if len(identities) != len(set(identities)):
        raise RetrievalEvalError("candidate identities must be unique")


def normalize_embeddings(
    values: object, *, expected_rows: int, role: str
) -> NDArray[np.float32]:
    """Validate a 2-D embedding matrix then explicitly L2-normalize every row."""

    try:
        matrix = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise RetrievalEvalError(
            f"{role} embeddings must be numeric float32 values"
        ) from error
    if matrix.ndim != 2 or matrix.shape[0] != expected_rows or matrix.shape[1] == 0:
        raise RetrievalEvalError(f"{role} embedding shape does not match inputs")
    if not np.isfinite(matrix).all():
        raise RetrievalEvalError(f"{role} embeddings contain NaN or Inf")
    norms = np.linalg.norm(matrix, axis=1)
    if not np.isfinite(norms).all() or np.any(norms == 0):
        raise RetrievalEvalError(f"{role} embeddings contain a zero-norm row")
    normalized = np.asarray(matrix / norms[:, np.newaxis], dtype=np.float32)
    normalized_norms = np.linalg.norm(normalized, axis=1)
    if not np.allclose(normalized_norms, 1.0, rtol=1e-5, atol=1e-6):
        raise RetrievalEvalError(f"{role} embeddings failed L2 normalization")
    return normalized


def rank_candidates(
    candidates: Sequence[DetachedCandidate], scores: Sequence[float]
) -> tuple[tuple[DetachedCandidate, float, bool], ...]:
    """Return the whole exact-score ordering before any metric is derived."""

    _validate_candidate_identities(candidates)
    if len(candidates) != len(scores):
        raise RetrievalEvalError("ranking score count does not match candidate count")
    normalized_scores: list[float] = []
    for score in scores:
        value = float(score)
        if not np.isfinite(value):
            raise RetrievalEvalError("ranking score contains NaN or Inf")
        normalized_scores.append(value)
    ordered = sorted(
        zip(candidates, normalized_scores, strict=True),
        key=lambda item: (-item[1], item[0].document_key, item[0].chunk.ordinal),
    )
    score_counts = Counter(score for _, score in ordered)
    return tuple(
        (candidate, score, score_counts[score] > 1) for candidate, score in ordered
    )


def evaluate_case(
    corpus: LoadedGoldenCorpus,
    case: GoldenCase,
    ranked: Sequence[tuple[DetachedCandidate, float, bool]],
) -> CaseResult:
    """Mechanically match one full candidate ranking against its exact evidence."""

    _validate_case(corpus, case)
    _validate_candidate_identities(corpus.candidates)
    if len(ranked) != len(corpus.candidates):
        raise RetrievalEvalError("ranking length does not match candidate pool")
    expected_identities = {candidate.identity for candidate in corpus.candidates}
    actual_identities = [candidate.identity for candidate, _, _ in ranked]
    if set(actual_identities) != expected_identities or len(actual_identities) != len(
        set(actual_identities)
    ):
        raise RetrievalEvalError("ranking must contain every candidate exactly once")
    is_matchable = any(
        candidate_is_golden_hit(corpus, case, candidate)
        for candidate in corpus.candidates
    )
    result_ranking: list[RankedCandidate] = []
    for rank, (candidate, raw_score, exact_tie) in enumerate(ranked, start=1):
        if not np.isfinite(raw_score):
            raise RetrievalEvalError("ranking score contains NaN or Inf")
        result_ranking.append(
            RankedCandidate(
                rank=rank,
                candidate=candidate,
                raw_score=raw_score,
                is_hit=candidate_is_golden_hit(corpus, case, candidate),
                exact_tie=exact_tie,
            )
        )
    first_hit = next((item.rank for item in result_ranking if item.is_hit), None)
    return CaseResult(
        case=case,
        candidate_pool_has_valid_match=is_matchable,
        first_valid_hit_rank=first_hit,
        hit_at_1=first_hit is not None and first_hit <= 1,
        hit_at_3=first_hit is not None and first_hit <= 3,
        hit_at_5=first_hit is not None and first_hit <= 5,
        reciprocal_rank=0.0 if first_hit is None else 1.0 / first_hit,
        ranking=tuple(result_ranking),
    )


def _metric(results: Sequence[CaseResult], predicate: str) -> Metric:
    numerator = sum(bool(getattr(result, predicate)) for result in results)
    denominator = len(results)
    if denominator == 0:
        raise RetrievalEvalError("cannot aggregate an empty case set")
    return Metric(numerator, denominator, numerator / denominator)


def run_evaluation(
    corpus: LoadedGoldenCorpus,
    embedder: EmbeddingBoundary,
    *,
    code_commit: str,
    tracked_worktree_dirty: bool,
    worktree_dirty: bool,
    embedding_metadata: EmbeddingMetadata,
) -> EvalResult:
    """Compute rankings, matches, and all metrics once from exact frozen inputs."""

    if not corpus.candidates:
        raise RetrievalEvalError("candidate pool must not be empty")
    _validate_candidate_identities(corpus.candidates)
    for case in corpus.cases:
        _validate_case(corpus, case)
    document_inputs = tuple(candidate.chunk.content for candidate in corpus.candidates)
    query_inputs = tuple(case.query for case in corpus.cases)
    document_vectors = normalize_embeddings(
        embedder.embed_documents(document_inputs),
        expected_rows=len(document_inputs),
        role="document",
    )
    query_vectors = normalize_embeddings(
        embedder.embed_queries(query_inputs),
        expected_rows=len(query_inputs),
        role="query",
    )
    if document_vectors.shape[1] != query_vectors.shape[1]:
        raise RetrievalEvalError("query and document embedding dimensions differ")
    case_results: list[CaseResult] = []
    for case, query_vector in zip(corpus.cases, query_vectors, strict=True):
        scores = np.dot(document_vectors, query_vector)
        ranked = rank_candidates(corpus.candidates, scores.tolist())
        case_results.append(evaluate_case(corpus, case, ranked))
    return EvalResult(
        corpus=corpus,
        code_commit=_nonblank_string(code_commit, "code_commit"),
        tracked_worktree_dirty=tracked_worktree_dirty,
        worktree_dirty=worktree_dirty,
        embedding_metadata=embedding_metadata,
        embedding_dimension=int(document_vectors.shape[1]),
        cases=tuple(case_results),
        hit_at_1=_metric(case_results, "hit_at_1"),
        hit_at_3=_metric(case_results, "hit_at_3"),
        hit_at_5=_metric(case_results, "hit_at_5"),
        mean_reciprocal_rank=sum(result.reciprocal_rank for result in case_results)
        / len(case_results),
    )


def corpus_preflight(corpus: LoadedGoldenCorpus) -> CorpusPreflight:
    """Report real-corpus chunk facts and matchability without embedding anything."""

    counts = Counter(candidate.document_key for candidate in corpus.candidates)
    for case in corpus.cases:
        _validate_case(corpus, case)
    no_matchable = tuple(
        case.case_id
        for case in corpus.cases
        if not any(
            candidate_is_golden_hit(corpus, case, candidate)
            for candidate in corpus.candidates
        )
    )
    per_document = [counts[key] for key in corpus.source_bytes_by_document]
    return CorpusPreflight(
        document_count=len(corpus.source_bytes_by_document),
        candidate_chunk_count=len(corpus.candidates),
        chunks_per_document_min=min(per_document),
        chunks_per_document_median=float(median(per_document)),
        chunks_per_document_max=max(per_document),
        approved_case_count=len(corpus.cases),
        matchable_case_count=len(corpus.cases) - len(no_matchable),
        no_matchable_case_ids=no_matchable,
    )


def _region_dict(region: TextSpanRegion) -> dict[str, int | str]:
    return {
        "kind": region.kind,
        "start_byte": region.start_byte,
        "end_byte": region.end_byte,
    }


def result_as_dict(result: EvalResult) -> dict[str, object]:
    """Make a JSON-safe artifact from the already-computed Eval result."""

    candidates_by_document = Counter(
        candidate.document_key for candidate in result.corpus.candidates
    )
    per_case: list[dict[str, object]] = []
    for case_result in result.cases:
        ranking = [
            {
                "rank": item.rank,
                "document_key": item.candidate.document_key,
                "ordinal": item.candidate.chunk.ordinal,
                "raw_score": item.raw_score,
                "hit": item.is_hit,
                "exact_tie": item.exact_tie,
                "heading_path": list(item.candidate.chunk.heading_path),
                "source_regions": [
                    _region_dict(region)
                    for region in item.candidate.chunk.source_regions
                ],
            }
            for item in case_result.ranking
        ]
        per_case.append(
            {
                "case_id": case_result.case.case_id,
                "query": case_result.case.query,
                "golden_document_key": case_result.case.document_key,
                "evidence": asdict(case_result.case.evidence),
                "candidate_pool_has_valid_match": (
                    case_result.candidate_pool_has_valid_match
                ),
                "first_valid_hit_rank": case_result.first_valid_hit_rank,
                "hit_at_1": case_result.hit_at_1,
                "hit_at_3": case_result.hit_at_3,
                "hit_at_5": case_result.hit_at_5,
                "reciprocal_rank": case_result.reciprocal_rank,
                "ranking": ranking,
            }
        )
    return {
        "dataset_identity": result.corpus.dataset_identity,
        "dataset_manifest_sha256": result.corpus.dataset_manifest_sha256,
        "code_commit": result.code_commit,
        "tracked_worktree_dirty": result.tracked_worktree_dirty,
        "worktree_dirty": result.worktree_dirty,
        "chunking_config": {
            "max_chunk_chars": result.corpus.chunking_config.max_chunk_chars
        },
        "representation": result.embedding_metadata.representation,
        "embedding": asdict(result.embedding_metadata),
        "observed_embedding_dimension": result.embedding_dimension,
        "corpus": {
            "document_count": len(result.corpus.source_bytes_by_document),
            "candidate_chunk_count": len(result.corpus.candidates),
            "chunks_per_document": {
                "min": min(candidates_by_document.values()),
                "median": float(median(candidates_by_document.values())),
                "max": max(candidates_by_document.values()),
            },
        },
        "aggregate": {
            "case_count": len(result.cases),
            "hit_at_1": asdict(result.hit_at_1),
            "hit_at_3": asdict(result.hit_at_3),
            "hit_at_5": asdict(result.hit_at_5),
            "mrr": result.mean_reciprocal_rank,
            "matchable_case_count": sum(
                case.candidate_pool_has_valid_match for case in result.cases
            ),
            "no_matchable_case_count": sum(
                not case.candidate_pool_has_valid_match for case in result.cases
            ),
            "first_hit_rank_gt_5_count": sum(
                case.first_valid_hit_rank is not None and case.first_valid_hit_rank > 5
                for case in result.cases
            ),
        },
        "cases": per_case,
    }


def render_result_json(result: EvalResult) -> str:
    return (
        json.dumps(result_as_dict(result), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    )


def _preview(content: str, limit: int = 180) -> str:
    compact = " ".join(content.split())
    return compact if len(compact) <= limit else f"{compact[: limit - 1]}…"


def _metric_line(label: str, metric: Metric) -> str:
    return f"- {label}: {metric.numerator}/{metric.denominator} ({metric.value:.6f})"


def render_result_markdown(result: EvalResult) -> str:
    """Render diagnostics only from ``result``; it performs no scoring or matching."""

    payload = result_as_dict(result)
    aggregate = payload["aggregate"]
    corpus_facts = payload["corpus"]
    assert isinstance(aggregate, dict)
    assert isinstance(corpus_facts, dict)
    chunk_counts = corpus_facts["chunks_per_document"]
    assert isinstance(chunk_counts, dict)
    no_matchable_case_ids = [
        case.case.case_id
        for case in result.cases
        if not case.candidate_pool_has_valid_match
    ]
    lines = [
        "# Langley Retrieval Eval Result",
        "",
        "## Configuration",
        "",
        f"- Dataset: `{result.corpus.dataset_identity}`",
        f"- Dataset manifest SHA-256: `{result.corpus.dataset_manifest_sha256}`",
        f"- Code commit: `{result.code_commit}`",
        f"- Tracked worktree dirty: `{result.tracked_worktree_dirty}`",
        f"- Worktree dirty: `{result.worktree_dirty}`",
        f"- Representation: `{result.embedding_metadata.representation}`",
        f"- Model: `{result.embedding_metadata.model_id}`",
        f"- Model revision: `{result.embedding_metadata.model_revision}`",
        (
            "- Dimension / dtype / device: "
            f"`{result.embedding_dimension}` / `{result.embedding_metadata.dtype}` / "
            f"`{result.embedding_metadata.device}`"
        ),
        "",
        "## Aggregate",
        "",
        _metric_line("Hit@1", result.hit_at_1),
        _metric_line("Hit@3", result.hit_at_3),
        _metric_line("Hit@5", result.hit_at_5),
        f"- MRR: {result.mean_reciprocal_rank:.6f}",
        (
            "- Matchable / no-matchable: "
            f"{aggregate['matchable_case_count']} / "
            f"{aggregate['no_matchable_case_count']}"
        ),
    ]
    lines.extend(
        [
            "",
            "## Candidate Pool",
            "",
            f"- Documents: {corpus_facts['document_count']}",
            f"- Candidate chunks: {corpus_facts['candidate_chunk_count']}",
            (
                "- Chunks/document (min / median / max): "
                f"{chunk_counts['min']} / {chunk_counts['median']} / "
                f"{chunk_counts['max']}"
            ),
            "",
            "## NO_MATCHABLE_CANDIDATE",
            "",
            (
                "- none"
                if not no_matchable_case_ids
                else "- " + ", ".join(no_matchable_case_ids)
            ),
            "",
            "## Cases",
        ]
    )
    for case_result in result.cases:
        lines.extend(
            [
                "",
                f"### {case_result.case.case_id}",
                "",
                f"- Query: {case_result.case.query}",
                (
                    "- Golden evidence: "
                    f"[{case_result.case.evidence.start_byte},"
                    f"{case_result.case.evidence.end_byte}): "
                    f"{case_result.case.evidence.text}"
                ),
                (
                    "- Candidate pool has valid match: "
                    f"`{case_result.candidate_pool_has_valid_match}`"
                ),
                f"- First valid HIT rank: `{case_result.first_valid_hit_rank}`",
                f"- RR: `{case_result.reciprocal_rank:.6f}`",
                "",
                (
                    "| Rank | Score | Document | Ordinal | Heading path | Preview | "
                    "Regions | Result | Tie |"
                ),
                "| ---: | ---: | --- | ---: | --- | --- | --- | --- | --- |",
            ]
        )
        highlighted = list(case_result.ranking[:5])
        if (
            case_result.first_valid_hit_rank is not None
            and case_result.first_valid_hit_rank > 5
        ):
            highlighted.append(
                case_result.ranking[case_result.first_valid_hit_rank - 1]
            )
        for item in highlighted:
            regions = ", ".join(
                f"[{region.start_byte},{region.end_byte})"
                for region in item.candidate.chunk.source_regions
            )
            lines.append(
                (
                    "| {rank} | {score:.6f} | {document} | {ordinal} | {heading} | "
                    "{preview} | {regions} | {result} | {tie} |"
                ).format(
                    rank=item.rank,
                    score=item.raw_score,
                    document=item.candidate.document_key,
                    ordinal=item.candidate.chunk.ordinal,
                    heading=" / ".join(item.candidate.chunk.heading_path) or "(root)",
                    preview=_preview(item.candidate.chunk.content).replace("|", "\\|"),
                    regions=regions,
                    result="HIT" if item.is_hit else "MISS",
                    tie="exact" if item.exact_tie else "",
                )
            )
    return "\n".join(lines) + "\n"


def concise_stdout(result: EvalResult) -> str:
    return (
        f"{result.corpus.dataset_identity}: {len(result.cases)} cases, "
        f"Hit@1={result.hit_at_1.value:.6f}, Hit@3={result.hit_at_3.value:.6f}, "
        f"Hit@5={result.hit_at_5.value:.6f}, MRR={result.mean_reciprocal_rank:.6f}"
    )


class SentenceTransformersEmbedding:
    """Lazy Experiment #0 adapter; model import/load happens only on encode."""

    def __init__(
        self,
        model_id: str = "BAAI/bge-m3",
        *,
        revision: str | None = None,
        device: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.device = device
        self._model: Any | None = None

    def _get_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_id, revision=self.revision, device=self.device
            )
        return self._model

    @property
    def observed_device(self) -> str:
        if self._model is None:
            raise RetrievalEvalError("embedding model has not been loaded")
        return str(self._model.device)

    def _embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        # ``encode`` is used for both roles: Experiment #0 has no query prompt.
        values = self._get_model().encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return np.asarray(values, dtype=np.float32)

    def embed_queries(self, queries: Sequence[str]) -> NDArray[np.float32]:
        return self._embed(queries)

    def embed_documents(self, documents: Sequence[str]) -> NDArray[np.float32]:
        return self._embed(documents)
