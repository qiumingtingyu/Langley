"""Detached application service for one scoped production dense retrieval."""

import asyncio

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from langley.answering.tracing import (
    KnowledgeSearchOrigin,
    KnowledgeSearchTrace,
    KnowledgeSearchTraceParent,
    current_retrieval_trace_parent,
)
from langley.knowledge.index_build import KnowledgeIndexBuildRuntime
from langley.knowledge.retrieval import (
    IndexNotReadyError,
    KnowledgeBaseRetrievalNotFoundError,
    RetrievalError,
    RetrievalGenerationChangedError,
    RetrievalResult,
    retrieve_dense,
)


class KnowledgeSearchError(Exception):
    """Stable, technology-neutral failure exposed by the knowledge service."""

    def __init__(self, code: str, *, retryable: bool) -> None:
        super().__init__(code)
        self.code = code
        self.retryable = retryable


class KnowledgeRetrievalService:
    """Reuse the frozen dense retrieval path without owning its algorithm."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        runtime: KnowledgeIndexBuildRuntime,
    ) -> None:
        self._session_factory = session_factory
        self._runtime = runtime

    async def search(
        self,
        *,
        user_id: int,
        knowledge_base_id: int,
        query: str,
        top_k: int,
        trace_parent: KnowledgeSearchTraceParent | None = None,
        origin: KnowledgeSearchOrigin = KnowledgeSearchOrigin.AGENT_TOOL,
    ) -> RetrievalResult:
        """Retrieve with the server-owned user and KB scope only."""

        search_trace = _start_search_trace(
            trace_parent or current_retrieval_trace_parent(),
            origin=origin,
            knowledge_base_id=knowledge_base_id,
            top_k=top_k,
            query=query,
        )
        try:
            result = await retrieve_dense(
                self._session_factory,
                self._runtime,
                user_id=user_id,
                knowledge_base_id=knowledge_base_id,
                query=query,
                top_k=top_k,
            )
        except asyncio.CancelledError:
            _finish_search_trace(search_trace, hit_count=None, error_code="CANCELLED")
            raise
        except RetrievalError as error:
            stable_error = _knowledge_search_error(error)
            _finish_search_trace(
                search_trace, hit_count=None, error_code=stable_error.code
            )
            raise stable_error from error
        except Exception:
            _finish_search_trace(search_trace, hit_count=None, error_code="UNEXPECTED")
            raise
        _finish_search_trace(search_trace, hit_count=len(result.hits), error_code=None)
        return result


def _knowledge_search_error(error: RetrievalError) -> KnowledgeSearchError:
    if isinstance(error, KnowledgeBaseRetrievalNotFoundError):
        return KnowledgeSearchError("KNOWLEDGE_SCOPE_UNAVAILABLE", retryable=False)
    if isinstance(error, IndexNotReadyError):
        return KnowledgeSearchError("KNOWLEDGE_INDEX_NOT_READY", retryable=False)
    if isinstance(error, RetrievalGenerationChangedError):
        return KnowledgeSearchError("KNOWLEDGE_SEARCH_CHANGED", retryable=True)
    return KnowledgeSearchError("KNOWLEDGE_SEARCH_UNAVAILABLE", retryable=False)


def _start_search_trace(
    trace: KnowledgeSearchTraceParent | None,
    *,
    origin: KnowledgeSearchOrigin,
    knowledge_base_id: int,
    top_k: int,
    query: str,
) -> KnowledgeSearchTrace | None:
    if trace is None:
        return None
    try:
        return trace.begin_knowledge_search(
            origin=origin,
            knowledge_base_id=knowledge_base_id,
            top_k=top_k,
            query=query,
        )
    except Exception:
        return None


def _finish_search_trace(
    trace: KnowledgeSearchTrace | None,
    *,
    hit_count: int | None,
    error_code: str | None,
) -> None:
    if trace is None:
        return
    try:
        trace.finish(hit_count, error_code)
    except Exception:
        return
