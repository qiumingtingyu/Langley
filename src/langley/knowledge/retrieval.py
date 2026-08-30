"""Fail-closed production retrieval over authoritative dense-v2 candidates."""

import asyncio
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from langley.infrastructure.models import (
    Document,
    DocumentVersion,
    KnowledgeBase,
    KnowledgeChunk,
)
from langley.knowledge.index_build import (
    DenseSearchHit,
    DenseSearchResultError,
    IndexBuildFailure,
    KnowledgeIndexBuildRuntime,
    current_knowledge_chunks_statement,
)
from langley.knowledge.reranking import (
    Reranker,
    RerankerError,
    validate_reranker_scores,
)


class RetrievalError(Exception):
    """A small machine-readable retrieval failure for the API boundary."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class KnowledgeBaseRetrievalNotFoundError(RetrievalError):
    def __init__(self) -> None:
        super().__init__("KNOWLEDGE_BASE_NOT_FOUND")


class IndexNotReadyError(RetrievalError):
    def __init__(self) -> None:
        super().__init__("INDEX_NOT_READY")


class RetrievalIndexChangedError(RetrievalError):
    def __init__(self) -> None:
        super().__init__("RETRIEVAL_INDEX_CHANGED")


class RetrievalIndexInconsistentError(RetrievalError):
    def __init__(self) -> None:
        super().__init__("RETRIEVAL_INDEX_INCONSISTENT")


class RetrievalEmbeddingInvalidError(RetrievalError):
    def __init__(self) -> None:
        super().__init__("RETRIEVAL_EMBEDDING_INVALID")


class RetrievalEmbeddingUnavailableError(RetrievalError):
    def __init__(self) -> None:
        super().__init__("RETRIEVAL_EMBEDDING_UNAVAILABLE")


class RetrievalQdrantUnavailableError(RetrievalError):
    def __init__(self) -> None:
        super().__init__("RETRIEVAL_QDRANT_UNAVAILABLE")


@dataclass(frozen=True)
class ActiveRetrievalContext:
    """Detached global embedding facts allowed across slow runtime boundaries."""

    knowledge_base_id: int
    user_id: int
    model: str
    revision: str
    dimension: int
    representation: str


@dataclass(frozen=True)
class RetrievalHit:
    knowledge_chunk_id: int
    rank: int
    retrieval_rank: int
    score: float
    rerank_score: float | None
    chunk_ordinal: int
    content: str
    heading_path: tuple[str, ...]
    source_regions: tuple[object, ...]
    document_id: int
    document_version_id: int
    source_display_name: str
    source_sha256: str


@dataclass(frozen=True)
class RetrievalResult:
    knowledge_base_id: int
    hits: tuple[RetrievalHit, ...]


@dataclass(frozen=True)
class _AuthoritativeChunk:
    knowledge_chunk_id: int
    chunk_ordinal: int
    content: str
    heading_path: tuple[str, ...]
    source_regions: tuple[object, ...]
    document_id: int
    document_version_id: int
    source_display_name: str
    source_sha256: str


async def retrieve_dense(
    session_factory: async_sessionmaker[AsyncSession],
    runtime: KnowledgeIndexBuildRuntime,
    *,
    user_id: int,
    knowledge_base_id: int,
    query: str,
    top_k: int,
) -> RetrievalResult:
    """Retrieve once without retaining DB resources across slow local/remote work."""

    context, hits = await _retrieve_dense_candidates(
        session_factory,
        runtime,
        user_id=user_id,
        knowledge_base_id=knowledge_base_id,
        query=query,
        candidate_k=top_k,
    )
    return RetrievalResult(
        knowledge_base_id=context.knowledge_base_id,
        hits=hits,
    )


async def retrieve_reranked(
    session_factory: async_sessionmaker[AsyncSession],
    runtime: KnowledgeIndexBuildRuntime,
    reranker: Reranker,
    *,
    user_id: int,
    knowledge_base_id: int,
    query: str,
    candidate_k: int,
    top_k: int,
) -> RetrievalResult:
    """Over-retrieve, rerank detached candidates, then revalidate selected hits."""

    if top_k < 1 or candidate_k < top_k:
        raise ValueError("candidate_k must be greater than or equal to positive top_k")
    context, candidates = await _retrieve_dense_candidates(
        session_factory,
        runtime,
        user_id=user_id,
        knowledge_base_id=knowledge_base_id,
        query=query,
        candidate_k=candidate_k,
    )
    if not candidates:
        return RetrievalResult(knowledge_base_id=context.knowledge_base_id, hits=())
    try:
        raw_scores = await reranker.score(
            query=query,
            passages=tuple(candidate.content for candidate in candidates),
        )
    except asyncio.CancelledError:
        raise
    except RerankerError:
        raise
    except Exception as error:
        raise RerankerError("reranker execution failed") from error
    rerank_scores = validate_reranker_scores(raw_scores, expected_count=len(candidates))
    ordered = sorted(
        zip(candidates, rerank_scores, strict=True),
        key=lambda item: (-item[1], item[0].retrieval_rank),
    )
    selected = ordered[:top_k]

    final_chunks = await _final_revalidate(
        session_factory,
        context=context,
        returned_chunk_ids=tuple(
            candidate.knowledge_chunk_id for candidate, _score in selected
        ),
    )
    chunks_by_id = {chunk.knowledge_chunk_id: chunk for chunk in final_chunks}
    hits: list[RetrievalHit] = []
    for candidate, rerank_score in selected:
        chunk = chunks_by_id.get(candidate.knowledge_chunk_id)
        if chunk is None:
            continue
        hits.append(
            _retrieval_hit(
                chunk=chunk,
                rank=len(hits) + 1,
                retrieval_rank=candidate.retrieval_rank,
                score=candidate.score,
                rerank_score=rerank_score,
            )
        )
    return RetrievalResult(
        knowledge_base_id=context.knowledge_base_id,
        hits=tuple(hits),
    )


async def _retrieve_dense_candidates(
    session_factory: async_sessionmaker[AsyncSession],
    runtime: KnowledgeIndexBuildRuntime,
    *,
    user_id: int,
    knowledge_base_id: int,
    query: str,
    candidate_k: int,
) -> tuple[ActiveRetrievalContext, tuple[RetrievalHit, ...]]:
    """Return detached authoritative Dense candidates with their original rank."""

    context = await _read_active_context(
        session_factory, user_id=user_id, knowledge_base_id=knowledge_base_id
    )
    try:
        query_vector = await runtime.encode_query(
            query,
            model=context.model,
            revision=context.revision,
            dimension=context.dimension,
            representation=context.representation,
        )
    except IndexBuildFailure as error:
        if error.code == "INVALID_EMBEDDING":
            raise RetrievalEmbeddingInvalidError() from error
        raise RetrievalEmbeddingUnavailableError() from error
    except Exception as error:
        raise RetrievalEmbeddingUnavailableError() from error

    search_limit = min(max(candidate_k * 2, candidate_k), 100)
    search_hits: tuple[DenseSearchHit, ...] = ()
    try:
        search_hits = await runtime.search_dense(
            query_vector,
            user_id=context.user_id,
            knowledge_base_id=context.knowledge_base_id,
            top_k=search_limit,
            dimension=context.dimension,
        )
        _validate_search_hits(search_hits, top_k=search_limit)
    except DenseSearchResultError:
        raise RetrievalIndexInconsistentError() from None
    except IndexBuildFailure:
        raise RetrievalIndexInconsistentError() from None
    except Exception:
        raise RetrievalQdrantUnavailableError() from None

    chunks = await _final_revalidate(
        session_factory,
        context=context,
        returned_chunk_ids=tuple(hit.knowledge_chunk_id for hit in search_hits),
    )

    chunks_by_id = {chunk.knowledge_chunk_id: chunk for chunk in chunks}
    hits: list[RetrievalHit] = []
    for retrieval_rank, search_hit in enumerate(search_hits, start=1):
        chunk = chunks_by_id.get(search_hit.knowledge_chunk_id)
        if chunk is None:
            continue
        hits.append(
            _retrieval_hit(
                chunk=chunk,
                rank=len(hits) + 1,
                retrieval_rank=retrieval_rank,
                score=search_hit.score,
                rerank_score=None,
            )
        )
        if len(hits) == candidate_k:
            break
    return context, tuple(hits)


def _retrieval_hit(
    *,
    chunk: _AuthoritativeChunk,
    rank: int,
    retrieval_rank: int,
    score: float,
    rerank_score: float | None,
) -> RetrievalHit:
    return RetrievalHit(
        knowledge_chunk_id=chunk.knowledge_chunk_id,
        rank=rank,
        retrieval_rank=retrieval_rank,
        score=score,
        rerank_score=rerank_score,
        chunk_ordinal=chunk.chunk_ordinal,
        content=chunk.content,
        heading_path=chunk.heading_path,
        source_regions=chunk.source_regions,
        document_id=chunk.document_id,
        document_version_id=chunk.document_version_id,
        source_display_name=chunk.source_display_name,
        source_sha256=chunk.source_sha256,
    )


async def _read_active_context(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    user_id: int,
    knowledge_base_id: int,
) -> ActiveRetrievalContext:
    """Perform the initial short ownership and active-generation read."""

    async with session_factory() as session:
        knowledge_base = await session.scalar(
            select(KnowledgeBase).where(
                KnowledgeBase.id == knowledge_base_id,
                KnowledgeBase.user_id == user_id,
            )
        )
    if knowledge_base is None:
        raise KnowledgeBaseRetrievalNotFoundError()
    return _context_from_knowledge_base(
        knowledge_base, user_id=user_id, final_read=False
    )


def _context_from_knowledge_base(
    knowledge_base: KnowledgeBase,
    *,
    user_id: int,
    final_read: bool,
) -> ActiveRetrievalContext:
    if knowledge_base.index_status != "READY":
        if final_read:
            raise RetrievalIndexChangedError()
        raise IndexNotReadyError()
    values = (
        knowledge_base.active_embedding_model,
        knowledge_base.active_embedding_revision,
        knowledge_base.active_embedding_dimension,
        knowledge_base.active_embedding_representation,
    )
    if (
        any(value is None for value in values)
        or not _is_nonblank_string(knowledge_base.active_embedding_model)
        or not _is_pinned_revision(knowledge_base.active_embedding_revision)
        or not isinstance(knowledge_base.active_embedding_dimension, int)
        or isinstance(knowledge_base.active_embedding_dimension, bool)
        or knowledge_base.active_embedding_dimension <= 0
        or knowledge_base.active_embedding_representation != "source_context_v1"
    ):
        raise RetrievalIndexInconsistentError()
    assert knowledge_base.active_embedding_model is not None
    assert knowledge_base.active_embedding_revision is not None
    assert knowledge_base.active_embedding_dimension is not None
    assert knowledge_base.active_embedding_representation is not None
    return ActiveRetrievalContext(
        knowledge_base_id=knowledge_base.id,
        user_id=user_id,
        model=knowledge_base.active_embedding_model,
        revision=knowledge_base.active_embedding_revision,
        dimension=knowledge_base.active_embedding_dimension,
        representation=knowledge_base.active_embedding_representation,
    )


async def _final_revalidate(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    context: ActiveRetrievalContext,
    returned_chunk_ids: tuple[int, ...],
) -> tuple[_AuthoritativeChunk, ...]:
    """Use one fresh short MySQL snapshot for final state and exact-N validation."""

    async with session_factory() as session:
        async with session.begin():
            knowledge_base = await session.scalar(
                select(KnowledgeBase).where(
                    KnowledgeBase.id == context.knowledge_base_id,
                    KnowledgeBase.user_id == context.user_id,
                )
            )
            if knowledge_base is None:
                raise RetrievalIndexChangedError()
            final_context = _context_from_knowledge_base(
                knowledge_base, user_id=context.user_id, final_read=True
            )
            if final_context != context:
                raise RetrievalIndexChangedError()
            if not returned_chunk_ids:
                return ()
            rows = (
                await session.execute(
                    current_knowledge_chunks_statement(context.knowledge_base_id)
                    .where(
                        KnowledgeChunk.id.in_(returned_chunk_ids),
                        DocumentVersion.chunk_revision > 0,
                        DocumentVersion.indexed_chunk_revision
                        == DocumentVersion.chunk_revision,
                    )
                    .with_only_columns(
                        KnowledgeChunk.id,
                        KnowledgeChunk.ordinal,
                        KnowledgeChunk.content,
                        KnowledgeChunk.heading_path,
                        KnowledgeChunk.source_regions,
                        Document.id,
                        DocumentVersion.id,
                        Document.name,
                        DocumentVersion.source_sha256,
                    )
                )
            ).all()
    chunks_by_id = {
        row[0]: _AuthoritativeChunk(
            knowledge_chunk_id=row[0],
            chunk_ordinal=row[1],
            content=row[2],
            heading_path=tuple(row[3]),
            source_regions=tuple(row[4]),
            document_id=row[5],
            document_version_id=row[6],
            source_display_name=row[7],
            source_sha256=row[8],
        )
        for row in rows
    }
    return tuple(
        chunks_by_id[chunk_id]
        for chunk_id in returned_chunk_ids
        if chunk_id in chunks_by_id
    )


def _is_nonblank_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_pinned_revision(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_search_hits(values: tuple[DenseSearchHit, ...], *, top_k: int) -> None:
    """Keep fake/runtime adapters equally fail-closed at the service boundary."""

    if len(values) > top_k:
        raise DenseSearchResultError("dense search returned more results than top_k")
    seen_chunk_ids: set[int] = set()
    for value in values:
        if (
            not isinstance(value, DenseSearchHit)
            or isinstance(value.knowledge_chunk_id, bool)
            or not isinstance(value.knowledge_chunk_id, int)
            or value.knowledge_chunk_id in seen_chunk_ids
            or not isinstance(value.score, float)
            or value.score != value.score
            or value.score in {float("inf"), float("-inf")}
        ):
            raise DenseSearchResultError("malformed dense search result")
        seen_chunk_ids.add(value.knowledge_chunk_id)
