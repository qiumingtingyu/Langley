"""Experiment #1 production dense-retrieval regression/parity mechanics.

The formal runner deliberately drives the existing Knowledge HTTP routes.  The
pure reference, parity, metric, and rendering functions are kept independent of
model, Qdrant, MySQL, and network activity so they can be covered deterministically.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Protocol, Sequence

from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from langley.infrastructure.models import KnowledgeBase
from langley.knowledge.chunking import CandidateChunk, ChunkingConfig
from langley.knowledge.contracts import TextSpanRegion
from langley.knowledge.index_build import COLLECTION_NAME
from langley.knowledge.retrieval_eval import (
    DetachedCandidate,
    GoldenCase,
    LoadedGoldenCorpus,
    candidate_is_golden_hit,
    corpus_preflight,
    load_golden_corpus,
)
from langley.main import create_app

_DEFAULT_DATASET_ROOT = Path("tests/fixtures/knowledge/retrieval")
_MAX_CHUNK_CHARS = 1200
_TOP_K = 5
_INDEX_READY_DEADLINE_SECONDS = 30 * 60
_POLL_INTERVAL_SECONDS = 2.0


@dataclass(frozen=True)
class EmbeddingBaseline:
    model: str
    revision: str
    dimension: int
    representation: str


_FROZEN_EMBEDDING_BASELINE = EmbeddingBaseline(
    model="BAAI/bge-m3",
    revision="5617a9f61b028005a4858fdac845db406aefb181",
    dimension=1024,
    representation="content_only",
)


class ExperimentParityError(RuntimeError):
    """Base error with an explicit setup or execution boundary."""


class ExperimentSetupError(ExperimentParityError):
    """Golden/reference/source/candidate parity is invalid before index build."""


class ExperimentExecutionError(ExperimentParityError):
    """A real production operation failed after setup parity had passed."""


@dataclass(frozen=True)
class CandidateIdentity:
    document_key: str
    ordinal: int


@dataclass(frozen=True)
class RankedIdentity:
    identity: CandidateIdentity
    score: float | None = None


@dataclass(frozen=True)
class Top5Metrics:
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    mrr_at_5: float


@dataclass(frozen=True)
class CaseTop5:
    case_id: str
    ranking: tuple[RankedIdentity, ...]
    first_hit_rank: int | None

    @property
    def hit_at_1(self) -> bool:
        return self.first_hit_rank == 1

    @property
    def hit_at_3(self) -> bool:
        return self.first_hit_rank is not None and self.first_hit_rank <= 3

    @property
    def hit_at_5(self) -> bool:
        return self.first_hit_rank is not None and self.first_hit_rank <= 5

    @property
    def reciprocal_rank_at_5(self) -> float:
        rank = self.first_hit_rank
        return 0.0 if rank is None or rank > _TOP_K else 1.0 / rank


@dataclass(frozen=True)
class ProductionChunk:
    document_version_id: int
    ordinal: int
    content: str
    heading_path: tuple[str, ...]
    source_regions: tuple[TextSpanRegion, ...]


@dataclass(frozen=True)
class DocumentMapping:
    document_key: str
    document_id: int
    document_version_id: int
    source_sha256: str


@dataclass(frozen=True)
class PersistedDocument:
    document_id: int
    document_version_id: int
    source_sha256: str


@dataclass(frozen=True)
class ReadyIndexFacts:
    active_generation_id: str
    active_chunk_snapshot_sha256: str
    active_embedding_model: str
    active_embedding_revision: str
    active_embedding_dimension: int
    active_embedding_representation: str


@dataclass(frozen=True)
class CaseComparison:
    case_id: str
    experiment_0: CaseTop5
    experiment_1: CaseTop5
    overlap_count: int
    entered: tuple[CandidateIdentity, ...]
    left: tuple[CandidateIdentity, ...]
    rank_changes: tuple[tuple[CandidateIdentity, int, int], ...]
    classification: str


class _HttpResponse(Protocol):
    status_code: int

    def json(self) -> Any: ...


class _HttpClient(Protocol):
    def get(self, url: str, **kwargs: Any) -> _HttpResponse: ...

    def post(self, url: str, **kwargs: Any) -> _HttpResponse: ...


def _require_object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ExperimentSetupError(f"{label} must be an object")
    return value


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ExperimentSetupError(f"{label} must be a nonblank string")
    return value


def _require_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ExperimentSetupError(f"{label} must be an integer")
    return value


def _execution_object(value: object, label: str) -> dict[str, Any]:
    try:
        return _require_object(value, label)
    except ExperimentSetupError as error:
        raise ExperimentExecutionError(str(error)) from error


def _execution_string(value: object, label: str) -> str:
    try:
        return _require_string(value, label)
    except ExperimentSetupError as error:
        raise ExperimentExecutionError(str(error)) from error


def _execution_int(value: object, label: str) -> int:
    try:
        return _require_int(value, label)
    except ExperimentSetupError as error:
        raise ExperimentExecutionError(str(error)) from error


def _identity(value: object, label: str) -> CandidateIdentity:
    raw = _require_object(value, label)
    return CandidateIdentity(
        document_key=_require_string(raw.get("document_key"), f"{label}.document_key"),
        ordinal=_require_int(raw.get("ordinal"), f"{label}.ordinal"),
    )


def _ranking_from_reference(value: object, label: str) -> tuple[RankedIdentity, ...]:
    if not isinstance(value, list) or not value:
        raise ExperimentSetupError(f"{label} must be a non-empty list")
    ranking: list[RankedIdentity] = []
    for expected_rank, raw in enumerate(value, start=1):
        item = _require_object(raw, f"{label}[{expected_rank}]")
        if (
            _require_int(item.get("rank"), f"{label}[{expected_rank}].rank")
            != expected_rank
        ):
            raise ExperimentSetupError(f"{label} ranks must be contiguous from one")
        score = item.get("raw_score")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise ExperimentSetupError(f"{label}[{expected_rank}].raw_score invalid")
        ranking.append(
            RankedIdentity(_identity(item, f"{label}[{expected_rank}]"), float(score))
        )
    if len({item.identity for item in ranking}) != len(ranking):
        raise ExperimentSetupError(f"{label} contains duplicate candidate identities")
    return tuple(ranking)


def _reference_embedding_baseline(payload: dict[str, Any]) -> EmbeddingBaseline:
    embedding = _require_object(payload.get("embedding"), "reference.embedding")
    baseline = EmbeddingBaseline(
        model=_require_string(embedding.get("model_id"), "reference embedding model"),
        revision=_require_string(
            embedding.get("model_revision"), "reference embedding revision"
        ),
        dimension=_require_int(
            payload.get("observed_embedding_dimension"),
            "reference embedding dimension",
        ),
        representation=_require_string(
            embedding.get("representation"), "reference embedding representation"
        ),
    )
    if baseline != _FROZEN_EMBEDDING_BASELINE:
        raise ExperimentSetupError("reference embedding baseline mismatch")
    return baseline


def load_experiment_0_reference(
    path: Path, corpus: LoadedGoldenCorpus
) -> tuple[dict[str, Any], dict[str, CaseTop5]]:
    """Load the accepted corrected full ranking and derive its Top5 facts."""

    try:
        artifact_bytes = path.read_bytes()
        payload = json.loads(artifact_bytes.decode("utf-8", errors="strict"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExperimentSetupError(
            "accepted Experiment #0 reference is unreadable"
        ) from error
    raw = _require_object(payload, "reference")
    if raw.get("dataset_manifest_sha256") != corpus.dataset_manifest_sha256:
        raise ExperimentSetupError("reference Golden manifest identity mismatch")
    reference_baseline = _reference_embedding_baseline(raw)
    cases = raw.get("cases")
    if not isinstance(cases, list):
        raise ExperimentSetupError("reference.cases must be a list")
    expected_by_id = {case.case_id: case for case in corpus.cases}
    derived: dict[str, CaseTop5] = {}
    for case_payload in cases:
        case_raw = _require_object(case_payload, "reference case")
        case_id = _require_string(case_raw.get("case_id"), "reference case.case_id")
        if case_id in derived or case_id not in expected_by_id:
            raise ExperimentSetupError("reference case identity mismatch")
        ranking = _ranking_from_reference(
            case_raw.get("ranking"), f"reference {case_id}.ranking"
        )
        top5 = ranking[:_TOP_K]
        first = next(
            (
                rank
                for rank, item in enumerate(ranking, start=1)
                if _reference_hit(case_raw, item.identity)
            ),
            None,
        )
        derived[case_id] = CaseTop5(case_id, top5, first)
    if set(derived) != set(expected_by_id):
        raise ExperimentSetupError(
            "reference cases do not exactly match corrected Golden"
        )
    _validate_reference_metrics(raw, derived)
    provenance = {
        "path": str(path),
        "artifact_sha256": sha256(artifact_bytes).hexdigest(),
        "code_commit": raw.get("code_commit"),
        "dataset_manifest_sha256": raw.get("dataset_manifest_sha256"),
        "embedding": asdict(reference_baseline),
    }
    return provenance, derived


def _reference_hit(case_payload: dict[str, Any], identity: CandidateIdentity) -> bool:
    ranking = case_payload.get("ranking")
    assert isinstance(ranking, list)
    for item in ranking:
        raw = _require_object(item, "reference ranking item")
        if _identity(raw, "reference ranking item") == identity:
            value = raw.get("hit")
            if not isinstance(value, bool):
                raise ExperimentSetupError("reference ranking hit must be boolean")
            return value
    raise ExperimentSetupError("reference ranking identity disappeared")


def _validate_reference_metrics(
    payload: dict[str, Any], cases: dict[str, CaseTop5]
) -> None:
    aggregate = _require_object(payload.get("aggregate"), "reference.aggregate")
    derived = aggregate_metrics(tuple(cases.values()))
    for key, actual in (
        ("hit_at_1", derived.hit_at_1),
        ("hit_at_3", derived.hit_at_3),
        ("hit_at_5", derived.hit_at_5),
    ):
        metric = _require_object(aggregate.get(key), f"reference.aggregate.{key}")
        value = metric.get("value")
        if not isinstance(value, (int, float)) or float(value) != actual:
            raise ExperimentSetupError(f"reference aggregate {key} contradicts ranking")


def candidate_from_production_hit(
    hit: dict[str, Any], version_to_document_key: dict[int, str]
) -> DetachedCandidate:
    """Adapt a Task 5.2 HTTP hit to the existing Task 4 matcher input."""

    version_id = _execution_int(hit.get("document_version_id"), "retrieval hit version")
    document_key = version_to_document_key.get(version_id)
    if document_key is None:
        raise ExperimentExecutionError("retrieval hit has unknown document_version_id")
    regions = _regions(hit.get("source_regions"), "retrieval hit.source_regions")
    heading = hit.get("heading_path")
    if not isinstance(heading, list) or not all(
        isinstance(value, str) for value in heading
    ):
        raise ExperimentExecutionError("retrieval hit.heading_path invalid")
    return DetachedCandidate(
        document_key=document_key,
        chunk=CandidateChunk(
            ordinal=_execution_int(
                hit.get("chunk_ordinal"), "retrieval hit.chunk_ordinal"
            ),
            content=_execution_string(hit.get("content"), "retrieval hit.content"),
            heading_path=tuple(heading),
            source_regions=regions,
        ),
    )


def _regions(value: object, label: str) -> tuple[TextSpanRegion, ...]:
    if not isinstance(value, list) or not value:
        raise ExperimentExecutionError(f"{label} must be a non-empty list")
    regions: list[TextSpanRegion] = []
    for index, item in enumerate(value):
        raw = _execution_object(item, f"{label}[{index}]")
        if raw.get("kind") != "text_span":
            raise ExperimentExecutionError(f"{label}[{index}].kind invalid")
        try:
            regions.append(
                TextSpanRegion(
                    _execution_int(
                        raw.get("start_byte"), f"{label}[{index}].start_byte"
                    ),
                    _execution_int(raw.get("end_byte"), f"{label}[{index}].end_byte"),
                )
            )
        except ValueError as error:
            raise ExperimentExecutionError(f"{label}[{index}] invalid") from error
    return tuple(regions)


def first_hit_top5(
    corpus: LoadedGoldenCorpus,
    case: GoldenCase,
    candidates: Sequence[DetachedCandidate],
    *,
    scores: Sequence[float | None] | None = None,
) -> CaseTop5:
    if len(candidates) > _TOP_K:
        raise ExperimentExecutionError("production retrieval returned more than Top5")
    ranking = tuple(
        RankedIdentity(
            CandidateIdentity(candidate.document_key, candidate.chunk.ordinal),
            None if scores is None else scores[index],
        )
        for index, candidate in enumerate(candidates)
    )
    if len({item.identity for item in ranking}) != len(ranking):
        raise ExperimentExecutionError(
            "production retrieval contains duplicate identities"
        )
    first = next(
        (
            rank
            for rank, candidate in enumerate(candidates, start=1)
            if candidate_is_golden_hit(corpus, case, candidate)
        ),
        None,
    )
    return CaseTop5(case.case_id, ranking, first)


def aggregate_metrics(cases: Sequence[CaseTop5]) -> Top5Metrics:
    if not cases:
        raise ExperimentSetupError("cannot aggregate an empty case set")
    denominator = len(cases)
    return Top5Metrics(
        hit_at_1=sum(case.hit_at_1 for case in cases) / denominator,
        hit_at_3=sum(case.hit_at_3 for case in cases) / denominator,
        hit_at_5=sum(case.hit_at_5 for case in cases) / denominator,
        mrr_at_5=mean(case.reciprocal_rank_at_5 for case in cases),
    )


def compare_case(experiment_0: CaseTop5, experiment_1: CaseTop5) -> CaseComparison:
    if experiment_0.case_id != experiment_1.case_id:
        raise ExperimentSetupError("cannot compare different Golden cases")
    old = tuple(item.identity for item in experiment_0.ranking)
    new = tuple(item.identity for item in experiment_1.ranking)
    old_set, new_set = set(old), set(new)
    old_rank = {identity: rank for rank, identity in enumerate(old, start=1)}
    new_rank = {identity: rank for rank, identity in enumerate(new, start=1)}
    if experiment_0.hit_at_5 and not experiment_1.hit_at_5:
        classification = "REGRESSION_EVENT"
    elif old == new:
        classification = "PARITY"
    else:
        classification = "RANKING_DIFFERENCE"
    return CaseComparison(
        case_id=experiment_0.case_id,
        experiment_0=experiment_0,
        experiment_1=experiment_1,
        overlap_count=len(old_set & new_set),
        entered=tuple(identity for identity in new if identity not in old_set),
        left=tuple(identity for identity in old if identity not in new_set),
        rank_changes=tuple(
            (identity, old_rank[identity], new_rank[identity])
            for identity in new
            if identity in old_rank and old_rank[identity] != new_rank[identity]
        ),
        classification=classification,
    )


def validate_source_parity(
    corpus: LoadedGoldenCorpus, actual_sha256_by_document: dict[str, str]
) -> None:
    expected = {
        key: sha256(source).hexdigest()
        for key, source in corpus.source_bytes_by_document.items()
    }
    if set(actual_sha256_by_document) != set(expected):
        raise ExperimentSetupError("source parity document identities mismatch")
    for document_key, expected_sha in expected.items():
        if actual_sha256_by_document[document_key] != expected_sha:
            raise ExperimentSetupError(f"source parity SHA mismatch for {document_key}")


def validate_document_mapping_parity(
    corpus: LoadedGoldenCorpus,
    mappings: Sequence[DocumentMapping],
    persisted_documents: Sequence[PersistedDocument],
) -> dict[str, str]:
    """Prove the fresh KB's authoritative document/version universe is exact."""

    expected_keys = set(corpus.source_bytes_by_document)
    if (
        len(mappings) != len(expected_keys)
        or {item.document_key for item in mappings} != expected_keys
    ):
        raise ExperimentSetupError("mapped Golden document identities mismatch")
    expected_identities = {
        (item.document_id, item.document_version_id): item for item in mappings
    }
    if len(expected_identities) != len(mappings):
        raise ExperimentSetupError("mapped document/version identities are not unique")
    actual_identities = {
        (item.document_id, item.document_version_id): item
        for item in persisted_documents
    }
    if len(actual_identities) != len(persisted_documents):
        raise ExperimentSetupError(
            "persisted document/version identities are not unique"
        )
    if len(persisted_documents) != len(expected_keys) or set(actual_identities) != set(
        expected_identities
    ):
        raise ExperimentSetupError("fresh KB document/version universe mismatch")
    source_sha_by_document: dict[str, str] = {}
    for identity, mapping in expected_identities.items():
        actual = actual_identities[identity]
        if actual.source_sha256 != mapping.source_sha256:
            raise ExperimentSetupError(
                "persisted source SHA differs from upload mapping"
            )
        source_sha_by_document[mapping.document_key] = actual.source_sha256
    validate_source_parity(corpus, source_sha_by_document)
    return source_sha_by_document


def validate_candidate_parity(
    corpus: LoadedGoldenCorpus,
    actual_chunks: Sequence[ProductionChunk],
    version_to_document_key: dict[int, str],
    successful_chunk_max_chars_by_version: dict[int, int | None],
) -> None:
    expected = {
        CandidateIdentity(
            candidate.document_key, candidate.chunk.ordinal
        ): candidate.chunk
        for candidate in corpus.candidates
    }
    actual: dict[CandidateIdentity, ProductionChunk] = {}
    for chunk in actual_chunks:
        document_key = version_to_document_key.get(chunk.document_version_id)
        if document_key is None:
            raise ExperimentSetupError("candidate parity has unknown document version")
        identity = CandidateIdentity(document_key, chunk.ordinal)
        if identity in actual:
            raise ExperimentSetupError("candidate parity has duplicate identity")
        actual[identity] = chunk
    if set(actual) != set(expected):
        raise ExperimentSetupError("candidate parity identities mismatch")
    for identity, expected_chunk in expected.items():
        actual_chunk = actual[identity]
        if actual_chunk.content != expected_chunk.content:
            raise ExperimentSetupError(
                f"candidate parity content mismatch for {identity}"
            )
        if actual_chunk.heading_path != expected_chunk.heading_path:
            raise ExperimentSetupError(
                f"candidate parity heading_path mismatch for {identity}"
            )
        if actual_chunk.source_regions != expected_chunk.source_regions:
            raise ExperimentSetupError(
                f"candidate parity source_regions mismatch for {identity}"
            )
        if (
            successful_chunk_max_chars_by_version.get(actual_chunk.document_version_id)
            != _MAX_CHUNK_CHARS
        ):
            raise ExperimentSetupError(
                "candidate parity successful_chunk_max_chars mismatch"
            )


def poll_ready(
    read_status: Callable[[], dict[str, Any]],
    *,
    deadline_seconds: float = _INDEX_READY_DEADLINE_SECONDS,
    interval_seconds: float = _POLL_INTERVAL_SECONDS,
    now: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Observe one admitted build until READY; never admit or retry another build."""

    deadline = now() + deadline_seconds
    while now() <= deadline:
        payload = read_status()
        status = payload.get("index_status")
        job = payload.get("latest_job")
        job_status = job.get("status") if isinstance(job, dict) else None
        if status == "READY":
            return payload
        if job_status in {"FAILED", "INTERRUPTED"}:
            raise ExperimentExecutionError("INDEX_BUILD_TERMINAL_FAILURE")
        sleep(interval_seconds)
    raise ExperimentExecutionError("INDEX_READY_TIMEOUT")


def validate_ready_embedding_baseline(facts: ReadyIndexFacts) -> None:
    observed = EmbeddingBaseline(
        model=facts.active_embedding_model,
        revision=facts.active_embedding_revision,
        dimension=facts.active_embedding_dimension,
        representation=facts.active_embedding_representation,
    )
    if observed != _FROZEN_EMBEDDING_BASELINE:
        raise ExperimentSetupError("READY embedding baseline mismatch")


def _production_score(value: object) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ExperimentExecutionError("retrieval hit.score invalid")
    return float(value)


async def read_ready_index_facts(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    user_id: int,
    knowledge_base_id: int,
) -> ReadyIndexFacts:
    """Read already-persisted READY facts for formal-result provenance only."""

    async with session_factory() as session:
        knowledge_base = await session.scalar(
            select(KnowledgeBase).where(
                KnowledgeBase.id == knowledge_base_id,
                KnowledgeBase.user_id == user_id,
            )
        )
    if knowledge_base is None or knowledge_base.index_status != "READY":
        raise ExperimentExecutionError("READY facts unavailable")
    values = (
        knowledge_base.active_generation_id,
        knowledge_base.active_chunk_snapshot_sha256,
        knowledge_base.active_embedding_model,
        knowledge_base.active_embedding_revision,
        knowledge_base.active_embedding_dimension,
        knowledge_base.active_embedding_representation,
    )
    if any(value is None for value in values):
        raise ExperimentExecutionError("READY facts incomplete")
    assert isinstance(knowledge_base.active_generation_id, str)
    assert isinstance(knowledge_base.active_chunk_snapshot_sha256, str)
    assert isinstance(knowledge_base.active_embedding_model, str)
    assert isinstance(knowledge_base.active_embedding_revision, str)
    assert isinstance(knowledge_base.active_embedding_dimension, int)
    assert isinstance(knowledge_base.active_embedding_representation, str)
    return ReadyIndexFacts(
        active_generation_id=knowledge_base.active_generation_id,
        active_chunk_snapshot_sha256=knowledge_base.active_chunk_snapshot_sha256,
        active_embedding_model=knowledge_base.active_embedding_model,
        active_embedding_revision=knowledge_base.active_embedding_revision,
        active_embedding_dimension=knowledge_base.active_embedding_dimension,
        active_embedding_representation=knowledge_base.active_embedding_representation,
    )


def _read_all_chunks(
    client: _HttpClient, version_id: int
) -> tuple[ProductionChunk, ...]:
    offset = 0
    chunks: list[ProductionChunk] = []
    successful_max: int | None = None
    while True:
        response = client.get(
            f"/api/document-versions/{version_id}/chunks",
            params={"offset": offset, "limit": 100},
        )
        if response.status_code != 200:
            raise ExperimentExecutionError("production chunk read failed")
        payload = _execution_object(response.json(), "chunk read")
        reported_max = payload.get("successful_chunk_max_chars")
        if reported_max != _MAX_CHUNK_CHARS:
            raise ExperimentSetupError("production successful_chunk_max_chars mismatch")
        successful_max = reported_max
        raw_chunks = payload.get("chunks")
        if not isinstance(raw_chunks, list):
            raise ExperimentExecutionError("production chunk page malformed")
        for raw_chunk in raw_chunks:
            chunk = _execution_object(raw_chunk, "production chunk")
            chunks.append(
                ProductionChunk(
                    document_version_id=version_id,
                    ordinal=_execution_int(
                        chunk.get("ordinal"), "production chunk.ordinal"
                    ),
                    content=_execution_string(
                        chunk.get("content"), "production chunk.content"
                    ),
                    heading_path=tuple(chunk.get("heading_path", [])),
                    source_regions=_regions(
                        chunk.get("source_regions"), "production chunk.source_regions"
                    ),
                )
            )
        offset += len(raw_chunks)
        if offset >= _execution_int(
            payload.get("chunk_count"), "production chunk_count"
        ):
            break
        if not raw_chunks:
            raise ExperimentExecutionError("production chunk pagination stalled")
    if successful_max is None:
        raise ExperimentExecutionError("production chunk read missing configuration")
    return tuple(chunks)


def _read_authoritative_documents(
    client: _HttpClient, knowledge_base_id: int
) -> tuple[PersistedDocument, ...]:
    response = client.get(f"/api/knowledge-bases/{knowledge_base_id}/documents")
    if response.status_code != 200:
        raise ExperimentExecutionError("authoritative document list read failed")
    payload = response.json()
    if not isinstance(payload, list):
        raise ExperimentExecutionError("authoritative document list is malformed")
    documents: list[PersistedDocument] = []
    for raw_document in payload:
        document = _execution_object(raw_document, "persisted document")
        source = _execution_object(document.get("source"), "persisted document.source")
        documents.append(
            PersistedDocument(
                document_id=_execution_int(document.get("id"), "persisted document.id"),
                document_version_id=_execution_int(
                    source.get("document_version_id"),
                    "persisted document.source.document_version_id",
                ),
                source_sha256=_execution_string(
                    source.get("sha256"), "persisted document.source.sha256"
                ),
            )
        )
    return tuple(documents)


def run_production_parity(
    client: _HttpClient,
    *,
    corpus: LoadedGoldenCorpus,
    reference: dict[str, CaseTop5],
    code_commit: str,
    ready_facts_reader: Callable[[int], ReadyIndexFacts] | None = None,
    embedding_device: str | None = None,
) -> dict[str, Any]:
    """Execute the one formal sequence through production HTTP routes exactly once."""

    preflight = corpus_preflight(corpus)
    if (
        preflight.document_count != 40
        or preflight.candidate_chunk_count != 1003
        or preflight.approved_case_count != 13
        or preflight.matchable_case_count != 13
    ):
        raise ExperimentSetupError("corrected Golden preflight facts mismatch")
    created = client.post("/api/knowledge-bases", json={"name": "Experiment #1 Eval"})
    if created.status_code != 201:
        raise ExperimentExecutionError("fresh Eval KnowledgeBase creation failed")
    knowledge_base_id = _execution_int(
        _execution_object(created.json(), "Eval KB").get("id"), "Eval KB.id"
    )
    mappings: list[DocumentMapping] = []
    for document_key, source in corpus.source_bytes_by_document.items():
        response = client.post(
            f"/api/knowledge-bases/{knowledge_base_id}/documents",
            files={"file": (f"{document_key}.md", source, "text/markdown")},
            data={"document_name": document_key},
        )
        if response.status_code != 201:
            raise ExperimentExecutionError("Golden source upload failed")
        uploaded = _execution_object(response.json(), "uploaded document")
        source_facts = _execution_object(
            uploaded.get("source"), "uploaded document.source"
        )
        mappings.append(
            DocumentMapping(
                document_key=document_key,
                document_id=_execution_int(uploaded.get("id"), "uploaded document.id"),
                document_version_id=_execution_int(
                    source_facts.get("document_version_id"), "uploaded version id"
                ),
                source_sha256=_execution_string(
                    source_facts.get("sha256"), "uploaded source sha256"
                ),
            )
        )
    persisted_documents = _read_authoritative_documents(client, knowledge_base_id)
    validate_document_mapping_parity(corpus, mappings, persisted_documents)
    for mapping in mappings:
        rebuilt = client.post(
            f"/api/document-versions/{mapping.document_version_id}/chunks/rebuild",
            json={"max_chunk_chars": _MAX_CHUNK_CHARS},
        )
        if rebuilt.status_code != 200:
            raise ExperimentExecutionError("production chunk processing failed")
    version_to_key = {
        mapping.document_version_id: mapping.document_key for mapping in mappings
    }
    chunks = tuple(
        chunk
        for mapping in mappings
        for chunk in _read_all_chunks(client, mapping.document_version_id)
    )
    validate_candidate_parity(
        corpus,
        chunks,
        version_to_key,
        {mapping.document_version_id: _MAX_CHUNK_CHARS for mapping in mappings},
    )
    admitted = client.post(f"/api/knowledge-bases/{knowledge_base_id}/index-build")
    if admitted.status_code != 202:
        raise ExperimentExecutionError("Task 5.1 index-build admission failed")
    job_id = _execution_int(
        _execution_object(admitted.json(), "index admission").get("job_id"),
        "index job_id",
    )
    status = poll_ready(lambda: _status_payload(client, knowledge_base_id))
    ready_facts = (
        ready_facts_reader(knowledge_base_id)
        if ready_facts_reader is not None
        else None
    )
    if ready_facts is None:
        raise ExperimentExecutionError("READY facts reader is unavailable")
    validate_ready_embedding_baseline(ready_facts)
    production: dict[str, CaseTop5] = {}
    generation_id: str | None = None
    for case in corpus.cases:
        response = client.post(
            f"/api/knowledge-bases/{knowledge_base_id}/retrieval",
            json={"query": case.query, "top_k": _TOP_K},
        )
        if response.status_code != 200:
            raise ExperimentExecutionError("Task 5.2 retrieval route failed")
        payload = _execution_object(response.json(), "retrieval response")
        observed_generation = _execution_string(
            payload.get("generation_id"), "retrieval generation_id"
        )
        if generation_id is None:
            generation_id = observed_generation
        elif generation_id != observed_generation:
            raise ExperimentExecutionError(
                "retrieval generation changed during experiment"
            )
        if (
            ready_facts is not None
            and observed_generation != ready_facts.active_generation_id
        ):
            raise ExperimentExecutionError(
                "retrieval generation differs from READY facts"
            )
        hits = payload.get("hits")
        if not isinstance(hits, list) or not 1 <= len(hits) <= _TOP_K:
            raise ExperimentExecutionError("retrieval hit count invalid")
        if [item.get("rank") for item in hits if isinstance(item, dict)] != list(
            range(1, len(hits) + 1)
        ):
            raise ExperimentExecutionError("retrieval ranks invalid")
        raw_hits = [_execution_object(hit, "retrieval hit") for hit in hits]
        candidates = tuple(
            candidate_from_production_hit(hit, version_to_key) for hit in raw_hits
        )
        scores = tuple(_production_score(hit.get("score")) for hit in raw_hits)
        production[case.case_id] = first_hit_top5(
            corpus, case, candidates, scores=scores
        )
    comparisons = tuple(
        compare_case(reference[case.case_id], production[case.case_id])
        for case in corpus.cases
    )
    experiment_0_metrics = aggregate_metrics(tuple(reference.values()))
    experiment_1_metrics = aggregate_metrics(tuple(production.values()))
    return {
        "experiment": (
            "Slice 6 Experiment #1 Production Dense Retrieval Regression / Parity"
        ),
        "status": "COMPUTED_STOP_FOR_HUMAN",
        "code_commit": code_commit,
        "golden_manifest_sha256": corpus.dataset_manifest_sha256,
        "chunk_config": {"max_chunk_chars": _MAX_CHUNK_CHARS, "overlap": 0},
        "eval_kb_id": knowledge_base_id,
        "index_job_id": job_id,
        "index_status": status.get("index_status"),
        "active_generation_id": generation_id,
        "ready_index_facts": None if ready_facts is None else asdict(ready_facts),
        "embedding": (
            None
            if ready_facts is None
            else {
                "model": ready_facts.active_embedding_model,
                "revision": ready_facts.active_embedding_revision,
                "dimension": ready_facts.active_embedding_dimension,
                "representation": ready_facts.active_embedding_representation,
                "device": embedding_device,
            }
        ),
        "qdrant": {"collection": COLLECTION_NAME, "distance": "COSINE"},
        "document_mapping": [asdict(mapping) for mapping in mappings],
        "setup_parity": {"source": "PASS", "candidate": "PASS"},
        "aggregate": {
            "experiment_0": asdict(experiment_0_metrics),
            "experiment_1": asdict(experiment_1_metrics),
            "deltas": {
                key: getattr(experiment_1_metrics, key)
                - getattr(experiment_0_metrics, key)
                for key in ("hit_at_1", "hit_at_3", "hit_at_5", "mrr_at_5")
            },
        },
        "cases": [_comparison_dict(value) for value in comparisons],
    }


def _status_payload(client: _HttpClient, knowledge_base_id: int) -> dict[str, Any]:
    response = client.get(f"/api/knowledge-bases/{knowledge_base_id}/index-status")
    if response.status_code != 200:
        raise ExperimentExecutionError("index status read failed")
    return _execution_object(response.json(), "index status")


def _comparison_dict(value: CaseComparison) -> dict[str, Any]:
    def case_dict(case: CaseTop5) -> dict[str, Any]:
        return {
            "first_hit_rank": case.first_hit_rank,
            "hit_at_1": case.hit_at_1,
            "hit_at_3": case.hit_at_3,
            "hit_at_5": case.hit_at_5,
            "rr_at_5": case.reciprocal_rank_at_5,
            "top5": [asdict(item) for item in case.ranking],
        }

    return {
        "case_id": value.case_id,
        "experiment_0": case_dict(value.experiment_0),
        "experiment_1": case_dict(value.experiment_1),
        "overlap_count": value.overlap_count,
        "entered": [asdict(item) for item in value.entered],
        "left": [asdict(item) for item in value.left],
        "rank_changes": [
            {
                "candidate": asdict(identity),
                "experiment_0_rank": old,
                "experiment_1_rank": new,
            }
            for identity, old, new in value.rank_changes
        ],
        "classification": value.classification,
    }


def render_result_markdown(result: dict[str, Any]) -> str:
    """Render only an already-computed formal result; it never recomputes metrics."""

    aggregate = _require_object(result.get("aggregate"), "result.aggregate")
    experiment_0 = _require_object(aggregate.get("experiment_0"), "result experiment_0")
    experiment_1 = _require_object(aggregate.get("experiment_1"), "result experiment_1")
    deltas = _require_object(aggregate.get("deltas"), "result deltas")
    reference = _require_object(
        result.get("reference_experiment"), "result reference_experiment"
    )
    embedding = _require_object(result.get("embedding"), "result embedding")
    qdrant = _require_object(result.get("qdrant"), "result qdrant")
    ready = _require_object(result.get("ready_index_facts"), "result ready_index_facts")
    lines = ["# Experiment #1 Production Dense Retrieval Regression / Parity", ""]
    lines.extend(
        [
            "## Provenance",
            "",
            f"- Experiment: `{result['experiment']}`",
            f"- Status: `{result['status']}`",
            f"- Code commit: `{result['code_commit']}`",
            f"- Golden manifest SHA-256: `{result['golden_manifest_sha256']}`",
            f"- Experiment #0 artifact: `{reference['path']}`",
            f"- Experiment #0 SHA-256: `{reference['artifact_sha256']}`",
            f"- Experiment #0 code commit: `{reference['code_commit']}`",
            (
                "- Experiment #0 Golden manifest: "
                f"`{reference['dataset_manifest_sha256']}`"
            ),
            "",
            "## Setup Parity",
            "",
            f"- Source Parity: `{result['setup_parity']['source']}`",
            f"- Candidate Parity: `{result['setup_parity']['candidate']}`",
            f"- READY generation: `{result['active_generation_id']}`",
            f"- Active snapshot: `{ready['active_chunk_snapshot_sha256']}`",
            (
                "- BGE model / revision / dimension / representation / device: "
                f"`{embedding['model']}` / `{embedding['revision']}` / "
                f"`{embedding['dimension']}` / `{embedding['representation']}` / "
                f"`{embedding['device']}`"
            ),
            (
                "- Qdrant collection / distance: "
                f"`{qdrant['collection']}` / `{qdrant['distance']}`"
            ),
        ]
    )
    lines.extend(
        [
            "## Aggregate",
            "",
            "| Metric | Experiment #0 | Experiment #1 | Delta |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for key in ("hit_at_1", "hit_at_3", "hit_at_5", "mrr_at_5"):
        lines.append(
            f"| {key} | {float(experiment_0[key]):.6f} | "
            f"{float(experiment_1[key]):.6f} | {float(deltas[key]):+.6f} |"
        )
    cases = [_require_object(case, "result case") for case in result.get("cases", [])]
    lines.extend(["", "## Classifications", ""])
    for classification in ("REGRESSION_EVENT", "RANKING_DIFFERENCE", "PARITY"):
        case_ids = [
            _require_string(case.get("case_id"), "result case.case_id")
            for case in cases
            if case.get("classification") == classification
        ]
        lines.append(f"- {classification}: " + (", ".join(case_ids) or "none"))
    lines.extend(["", "## Differing Case Diagnostics", ""])
    differing_cases = [case for case in cases if case.get("classification") != "PARITY"]
    if not differing_cases:
        lines.append("- none")
    for raw in differing_cases:
        lines.append(f"### {raw['case_id']}")
        lines.append("")
        lines.append(f"- Classification: `{raw['classification']}`")
        lines.append(
            "- First HIT rank (#0 / #1): "
            f"`{raw['experiment_0']['first_hit_rank']}` / "
            f"`{raw['experiment_1']['first_hit_rank']}`"
        )
        lines.append(f"- Top5 overlap: `{raw['overlap_count']}`")
        lines.append(f"- Entered: `{raw['entered']}`")
        lines.append(f"- Left: `{raw['left']}`")
        lines.append(f"- Rank changes: `{raw['rank_changes']}`")
        lines.append(f"- #0 Top5: `{raw['experiment_0']['top5']}`")
        lines.append(f"- #1 Top5: `{raw['experiment_1']['top5']}`")
        lines.append("")
    return "\n".join(lines) + "\n"


def _write_result(output_dir: Path, result: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        render_result_markdown(result), encoding="utf-8"
    )


def _code_commit() -> str:
    import subprocess

    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Experiment #1 production parity")
    parser.add_argument("command", choices=("run",))
    parser.add_argument("--dataset-root", type=Path, default=_DEFAULT_DATASET_ROOT)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    corpus = load_golden_corpus(
        args.dataset_root, ChunkingConfig(max_chunk_chars=_MAX_CHUNK_CHARS)
    )
    provenance, reference = load_experiment_0_reference(args.reference, corpus)
    app = create_app()
    with TestClient(app) as client:
        session_factory = app.state.session_factory
        user_id = app.state.settings.local_user_id
        portal = client.portal
        if session_factory is None or user_id is None or portal is None:
            raise ExperimentExecutionError(
                "production database/local user is unavailable"
            )
        result = run_production_parity(
            client,
            corpus=corpus,
            reference=reference,
            code_commit=_code_commit(),
            ready_facts_reader=lambda knowledge_base_id: portal.call(
                lambda: read_ready_index_facts(
                    session_factory,
                    user_id=user_id,
                    knowledge_base_id=knowledge_base_id,
                )
            ),
            embedding_device=app.state.settings.knowledge_embedding_device,
        )
    result["reference_experiment"] = provenance
    _write_result(args.output_dir, result)


if __name__ == "__main__":
    main()
