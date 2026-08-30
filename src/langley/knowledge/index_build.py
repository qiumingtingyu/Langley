"""Narrow durable lifecycle for manual Knowledge dense-index builds."""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from math import isfinite
from typing import TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from langley.business_time import utc_now
from langley.infrastructure.models import (
    Document,
    DocumentIndexJob,
    DocumentProcessingJob,
    DocumentVersion,
    KnowledgeBase,
    KnowledgeChunk,
    KnowledgeIndexJob,
)
from langley.knowledge.document_index_contract import build_source_context_v1
from langley.knowledge.embedding_runtime import (
    KnowledgeEmbeddingError,
    KnowledgeEmbeddingRuntime,
    normalize_embedding_rows,
    normalize_query_embedding,
)

if TYPE_CHECKING:
    from langley.settings import Settings


COLLECTION_NAME = "langley_knowledge_dense_v2"
_BATCH_SIZE = 64


class IndexBuildAdmissionError(Exception):
    """A manual build cannot be admitted for this owned KnowledgeBase."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class IndexBuildFailure(Exception):
    """A safe, machine-readable terminal failure for one build attempt."""

    def __init__(self, code: str, message: str, *, stale: bool = False) -> None:
        super().__init__(code)
        self.code = code
        self.message = message
        self.stale = stale


class DenseSearchResultError(Exception):
    """A Qdrant response cannot safely represent authoritative retrieval hits."""


@dataclass(frozen=True)
class IndexChunk:
    id: int
    document_version_id: int
    content: str
    heading_path: tuple[str, ...]
    chunk_revision: int
    chunk_set_sha256: str


@dataclass(frozen=True)
class DenseSearchHit:
    """One validated dense-search result, retaining Qdrant's rank order."""

    knowledge_chunk_id: int
    score: float


@dataclass(frozen=True)
class IndexBuildAdmission:
    job_id: int


@dataclass(frozen=True)
class IndexJobRead:
    id: int
    status: str
    stage: str | None
    processed_chunk_count: int
    total_chunk_count: int
    error_code: str | None
    error_message: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


@dataclass(frozen=True)
class IndexStatusRead:
    index_status: str
    latest_job: IndexJobRead | None


def _snapshot_sha256(chunks: tuple[IndexChunk, ...]) -> str:
    digest = sha256()
    documents = {
        (chunk.document_version_id, chunk.chunk_revision, chunk.chunk_set_sha256)
        for chunk in chunks
    }
    for document_version_id, chunk_revision, chunk_set_sha256 in sorted(documents):
        digest.update(
            f"{document_version_id}:{chunk_revision}:{chunk_set_sha256}\n".encode(
                "ascii"
            )
        )
    return digest.hexdigest()


def _normalize_embedding_rows(
    values: object, *, row_count: int, dimension: int
) -> list[list[float]]:
    """Reject malformed document embeddings before any Qdrant write."""

    try:
        return normalize_embedding_rows(
            values, row_count=row_count, dimension=dimension
        )
    except KnowledgeEmbeddingError as error:
        raise IndexBuildFailure(error.code, error.message) from error


def _normalize_query_embedding(values: object, *, dimension: int) -> list[float]:
    """Reject a malformed production query vector before Qdrant search."""

    try:
        return normalize_query_embedding(values, dimension=dimension)
    except KnowledgeEmbeddingError as error:
        raise IndexBuildFailure(error.code, error.message) from error


async def _current_chunks(
    session: AsyncSession, knowledge_base_id: int
) -> tuple[IndexChunk, ...]:
    rows = (
        await session.execute(
            current_knowledge_chunks_statement(knowledge_base_id).with_only_columns(
                KnowledgeChunk.id,
                KnowledgeChunk.document_version_id,
                KnowledgeChunk.content,
                KnowledgeChunk.heading_path,
                DocumentVersion.chunk_revision,
                DocumentVersion.chunk_set_sha256,
            )
        )
    ).all()
    return tuple(
        IndexChunk(
            id=row.id,
            document_version_id=row.document_version_id,
            content=row.content,
            heading_path=tuple(row.heading_path),
            chunk_revision=row.chunk_revision,
            chunk_set_sha256=row.chunk_set_sha256,
        )
        for row in rows
    )


def current_knowledge_chunks_statement(knowledge_base_id: int):
    """Select the exact KnowledgeChunk membership indexed for one KnowledgeBase."""

    return (
        select(KnowledgeChunk)
        .join(DocumentVersion, DocumentVersion.id == KnowledgeChunk.document_version_id)
        .join(Document, Document.id == DocumentVersion.document_id)
        .where(Document.knowledge_base_id == knowledge_base_id)
        .order_by(KnowledgeChunk.id.asc())
    )


async def admit_index_build(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    user_id: int,
    knowledge_base_id: int,
    settings: "Settings",
) -> IndexBuildAdmission:
    """Serialize full-build admission against processing and document claims."""

    async with session_factory() as session:
        async with session.begin():
            knowledge_base = await session.scalar(
                select(KnowledgeBase)
                .where(
                    KnowledgeBase.id == knowledge_base_id,
                    KnowledgeBase.user_id == user_id,
                )
                .with_for_update()
            )
            if knowledge_base is None:
                raise IndexBuildAdmissionError("KNOWLEDGE_BASE_NOT_FOUND")
            if knowledge_base.index_status == "INDEXING":
                raise IndexBuildAdmissionError("INDEX_BUILD_IN_PROGRESS")
            running_document_index_id = await session.scalar(
                select(DocumentIndexJob.id)
                .join(
                    DocumentVersion,
                    DocumentVersion.id == DocumentIndexJob.document_version_id,
                )
                .join(Document, Document.id == DocumentVersion.document_id)
                .where(
                    Document.knowledge_base_id == knowledge_base_id,
                    DocumentIndexJob.status == "RUNNING",
                )
                .limit(1)
            )
            if running_document_index_id is not None:
                raise IndexBuildAdmissionError("KNOWLEDGE_BASE_DOCUMENTS_INDEXING")
            pending_document_index_jobs = (
                await session.scalars(
                    select(DocumentIndexJob)
                    .join(
                        DocumentVersion,
                        DocumentVersion.id == DocumentIndexJob.document_version_id,
                    )
                    .join(Document, Document.id == DocumentVersion.document_id)
                    .where(
                        Document.knowledge_base_id == knowledge_base_id,
                        DocumentIndexJob.status == "PENDING",
                    )
                    .order_by(DocumentIndexJob.id)
                    .with_for_update()
                )
            ).all()
            superseded_at = utc_now()
            for pending_job in pending_document_index_jobs:
                _interrupt_pending_document_index_job(pending_job, now=superseded_at)
            active_processing_id = await session.scalar(
                select(DocumentProcessingJob.id)
                .join(
                    DocumentVersion,
                    DocumentVersion.id == DocumentProcessingJob.document_version_id,
                )
                .join(Document, Document.id == DocumentVersion.document_id)
                .where(
                    Document.knowledge_base_id == knowledge_base_id,
                    DocumentProcessingJob.status.in_(("PENDING", "RUNNING")),
                )
                .limit(1)
            )
            if active_processing_id is not None:
                raise IndexBuildAdmissionError("KNOWLEDGE_BASE_DOCUMENTS_PROCESSING")
            unprocessed_version_id = await session.scalar(
                select(DocumentVersion.id)
                .join(Document, Document.id == DocumentVersion.document_id)
                .where(
                    Document.knowledge_base_id == knowledge_base_id,
                    DocumentVersion.source_media_type == "text/markdown",
                    DocumentVersion.chunk_max_chars.is_(None),
                )
                .limit(1)
            )
            if unprocessed_version_id is not None:
                raise IndexBuildAdmissionError("KNOWLEDGE_BASE_DOCUMENTS_UNPROCESSED")
            chunks = await _current_chunks(session, knowledge_base_id)
            if not chunks:
                raise IndexBuildAdmissionError("KNOWLEDGE_BASE_NOT_CHUNKED")
            now = utc_now()
            job = KnowledgeIndexJob(
                knowledge_base_id=knowledge_base_id,
                status="PENDING",
                stage=None,
                processed_chunk_count=0,
                total_chunk_count=len(chunks),
                embedding_model=settings.knowledge_embedding_model,
                embedding_revision=settings.knowledge_embedding_revision,
                embedding_dimension=settings.knowledge_embedding_dimension,
                embedding_representation=settings.knowledge_embedding_representation,
                chunk_snapshot_sha256=_snapshot_sha256(chunks),
                error_code=None,
                error_message=None,
                created_at=now,
                started_at=None,
                finished_at=None,
            )
            knowledge_base.index_status = "INDEXING"
            session.add(job)
            await session.flush()
            return IndexBuildAdmission(job_id=job.id)


async def read_index_status(
    session: AsyncSession, *, user_id: int, knowledge_base_id: int
) -> IndexStatusRead | None:
    """Return the owned current status and its most recent build attempt."""

    knowledge_base = await session.scalar(
        select(KnowledgeBase).where(
            KnowledgeBase.id == knowledge_base_id, KnowledgeBase.user_id == user_id
        )
    )
    if knowledge_base is None:
        return None
    job = await session.scalar(
        select(KnowledgeIndexJob)
        .where(KnowledgeIndexJob.knowledge_base_id == knowledge_base_id)
        .order_by(KnowledgeIndexJob.created_at.desc(), KnowledgeIndexJob.id.desc())
        .limit(1)
    )
    return IndexStatusRead(
        index_status=knowledge_base.index_status,
        latest_job=None
        if job is None
        else IndexJobRead(
            id=job.id,
            status=job.status,
            stage=job.stage,
            processed_chunk_count=job.processed_chunk_count,
            total_chunk_count=job.total_chunk_count,
            error_code=job.error_code,
            error_message=job.error_message,
            created_at=job.created_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
        ),
    )


async def reconcile_interrupted_index_builds(
    session_factory: async_sessionmaker[AsyncSession],
) -> tuple[int, ...]:
    """Fail closed for jobs whose in-process task disappeared at restart."""

    async with session_factory() as session:
        async with session.begin():
            jobs = (
                await session.scalars(
                    select(KnowledgeIndexJob)
                    .where(KnowledgeIndexJob.status.in_(("PENDING", "RUNNING")))
                    .with_for_update()
                )
            ).all()
            now = utc_now()
            for job in jobs:
                job.status = "INTERRUPTED"
                job.stage = None
                job.error_code = "INDEX_BUILD_INTERRUPTED"
                job.error_message = "索引建立因应用重启而中断。"
                job.finished_at = now
                knowledge_base = await session.get(
                    KnowledgeBase, job.knowledge_base_id, with_for_update=True
                )
                if (
                    knowledge_base is not None
                    and knowledge_base.index_status == "INDEXING"
                ):
                    knowledge_base.index_status = "FAILED"
            return tuple(job.id for job in jobs)


async def reconcile_stale_ready_index_configurations(
    session_factory: async_sessionmaker[AsyncSession], *, settings: "Settings"
) -> tuple[int, ...]:
    """Fail closed after a restart whose current embedding configuration changed."""

    async with session_factory() as session:
        async with session.begin():
            knowledge_bases = (
                await session.scalars(
                    select(KnowledgeBase)
                    .where(KnowledgeBase.index_status.in_(("CHUNKED", "READY")))
                    .with_for_update()
                )
            ).all()
            stale_ids: list[int] = []
            interrupted_at = utc_now()
            for knowledge_base in knowledge_bases:
                ready_version_id = await session.scalar(
                    select(DocumentVersion.id)
                    .join(Document, Document.id == DocumentVersion.document_id)
                    .where(
                        Document.knowledge_base_id == knowledge_base.id,
                        DocumentVersion.chunk_revision > 0,
                        DocumentVersion.indexed_chunk_revision
                        == DocumentVersion.chunk_revision,
                    )
                    .limit(1)
                )
                configuration_matches = (
                    knowledge_base.active_embedding_model
                    == settings.knowledge_embedding_model
                    and knowledge_base.active_embedding_revision
                    == settings.knowledge_embedding_revision
                    and knowledge_base.active_embedding_dimension
                    == settings.knowledge_embedding_dimension
                    and knowledge_base.active_embedding_representation
                    == settings.knowledge_embedding_representation
                )
                if ready_version_id is not None and not configuration_matches:
                    knowledge_base.index_status = "STALE"
                    stale_ids.append(knowledge_base.id)
                    pending_jobs = (
                        await session.scalars(
                            select(DocumentIndexJob)
                            .join(
                                DocumentVersion,
                                DocumentVersion.id
                                == DocumentIndexJob.document_version_id,
                            )
                            .join(Document, Document.id == DocumentVersion.document_id)
                            .where(
                                Document.knowledge_base_id == knowledge_base.id,
                                DocumentIndexJob.status == "PENDING",
                            )
                            .order_by(DocumentIndexJob.id)
                            .with_for_update()
                        )
                    ).all()
                    for pending_job in pending_jobs:
                        if not _document_index_job_uses_settings(
                            pending_job, settings=settings
                        ):
                            _interrupt_pending_document_index_job(
                                pending_job, now=interrupted_at
                            )
                elif ready_version_id is not None:
                    knowledge_base.index_status = "READY"
                else:
                    knowledge_base.index_status = "CHUNKED"
                    knowledge_base.active_embedding_model = (
                        settings.knowledge_embedding_model
                    )
                    knowledge_base.active_embedding_revision = (
                        settings.knowledge_embedding_revision
                    )
                    knowledge_base.active_embedding_dimension = (
                        settings.knowledge_embedding_dimension
                    )
                    knowledge_base.active_embedding_representation = (
                        settings.knowledge_embedding_representation
                    )
            return tuple(stale_ids)


def _interrupt_pending_document_index_job(
    job: DocumentIndexJob, *, now: datetime
) -> None:
    if job.status != "PENDING":
        return
    job.status = "INTERRUPTED"
    job.error_code = "INDEX_INTERRUPTED"
    job.error_message = "整库索引重建取代了尚未开始的文档索引任务。"
    job.finished_at = now


def _document_index_job_uses_settings(
    job: DocumentIndexJob, *, settings: "Settings"
) -> bool:
    return (
        job.embedding_model == settings.knowledge_embedding_model
        and job.embedding_revision == settings.knowledge_embedding_revision
        and job.embedding_dimension == settings.knowledge_embedding_dimension
        and job.embedding_representation == settings.knowledge_embedding_representation
    )


class KnowledgeIndexBuildRuntime:
    """One application-local, bounded executor for Task 5.1 index builds."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        settings: "Settings",
        *,
        embedding_runtime: KnowledgeEmbeddingRuntime | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._settings = settings
        self._capacity = asyncio.Semaphore(settings.knowledge_index_build_concurrency)
        self._tasks: set[asyncio.Task[None]] = set()
        self._embedding_runtime = embedding_runtime or KnowledgeEmbeddingRuntime(
            device=settings.knowledge_embedding_device
        )

    @property
    def settings(self) -> "Settings":
        return self._settings

    @property
    def embedding_runtime(self) -> KnowledgeEmbeddingRuntime:
        return self._embedding_runtime

    def schedule(self, job_id: int) -> None:
        task = asyncio.create_task(self.run(job_id))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def run(self, job_id: int) -> None:
        """Run an admitted job, always persisting a terminal outcome."""

        barrier_crossed = False
        async with self._capacity:
            try:
                await self._mark_running(job_id)
                job, chunks = await self._load_snapshot(job_id)
                vectors = await self._embed(job, chunks)
                await self._upload(job, chunks, vectors)
                barrier_crossed = True
                await self._verify(job)
                await self._activate(job)
            except IndexBuildFailure as error:
                if barrier_crossed:
                    await self._cleanup_projection(job_id)
                await self._fail(job_id, error)
            except Exception:
                if barrier_crossed:
                    await self._cleanup_projection(job_id)
                await self._fail(
                    job_id,
                    IndexBuildFailure(
                        "INDEX_BUILD_FAILED", "索引建立失败，请检查本地索引服务后重试。"
                    ),
                )

    async def _cleanup_projection(self, job_id: int) -> None:
        client = None
        try:
            async with self._session_factory() as session:
                job = await session.get(KnowledgeIndexJob, job_id)
                if job is None:
                    return
                knowledge_base = await session.get(KnowledgeBase, job.knowledge_base_id)
                if knowledge_base is None:
                    return
                user_id = knowledge_base.user_id
            from qdrant_client.http import models as qmodels

            client = await self._qdrant_client()
            await client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=qmodels.FilterSelector(
                    filter=self._scope_filter(
                        user_id=user_id,
                        knowledge_base_id=job.knowledge_base_id,
                    )
                ),
                wait=True,
            )
        except Exception:
            return
        finally:
            if client is not None:
                try:
                    await client.close()
                except Exception:
                    pass

    async def _mark_running(self, job_id: int) -> None:
        async with self._session_factory() as session:
            async with session.begin():
                job = await session.get(KnowledgeIndexJob, job_id, with_for_update=True)
                if job is None or job.status != "PENDING":
                    raise IndexBuildFailure(
                        "INDEX_BUILD_NOT_PENDING", "索引任务已不可执行。"
                    )
                job.status = "RUNNING"
                job.stage = "SNAPSHOT"
                job.started_at = utc_now()

    async def _load_snapshot(
        self, job_id: int
    ) -> tuple[KnowledgeIndexJob, tuple[IndexChunk, ...]]:
        async with self._session_factory() as session:
            job = await session.get(KnowledgeIndexJob, job_id)
            if job is None:
                raise IndexBuildFailure("INDEX_BUILD_NOT_FOUND", "索引任务不存在。")
            chunks = await _current_chunks(session, job.knowledge_base_id)
            if not chunks or _snapshot_sha256(chunks) != job.chunk_snapshot_sha256:
                raise IndexBuildFailure(
                    "SOURCE_CHUNKS_CHANGED",
                    "知识分块已变化，请重新建立索引。",
                    stale=True,
                )
            return job, chunks

    async def _embed(
        self, job: KnowledgeIndexJob, chunks: tuple[IndexChunk, ...]
    ) -> list[list[float]]:
        await self._update_stage(job.id, "EMBEDDING")
        if job.embedding_representation != "source_context_v1":
            raise IndexBuildFailure(
                "EMBEDDING_REPRESENTATION_UNSUPPORTED", "嵌入表示配置不受支持。"
            )
        vectors = await asyncio.to_thread(
            self._encode_documents,
            [
                build_source_context_v1(chunk.content, chunk.heading_path)
                for chunk in chunks
            ],
            job,
        )
        if len(vectors) != len(chunks):
            raise IndexBuildFailure("EMBEDDING_COUNT_MISMATCH", "嵌入结果不完整。")
        return vectors

    def _encode_documents(
        self, contents: list[str], job: KnowledgeIndexJob
    ) -> list[list[float]]:
        """Perform heavy local BGE-M3 document encoding outside the event loop."""

        try:
            self._embedding_runtime.set_device(
                self._settings.knowledge_embedding_device
            )
            return self._embedding_runtime.encode_documents(
                contents,
                model=job.embedding_model,
                revision=job.embedding_revision,
                dimension=job.embedding_dimension,
            )
        except KnowledgeEmbeddingError as error:
            raise IndexBuildFailure(error.code, error.message) from error

    async def encode_query(
        self,
        query: str,
        *,
        model: str,
        revision: str,
        dimension: int,
        representation: str,
    ) -> list[float]:
        """Encode one exact query using the active generation's BGE configuration."""

        if representation != "source_context_v1":
            raise IndexBuildFailure(
                "EMBEDDING_REPRESENTATION_UNSUPPORTED", "嵌入表示配置不受支持。"
            )
        return await asyncio.to_thread(
            self._encode_query,
            query,
            model=model,
            revision=revision,
            dimension=dimension,
        )

    def _encode_query(
        self, query: str, *, model: str, revision: str, dimension: int
    ) -> list[float]:
        """Perform heavy local BGE-M3 query encoding outside the event loop."""

        try:
            self._embedding_runtime.set_device(
                self._settings.knowledge_embedding_device
            )
            return self._embedding_runtime.encode_query(
                query,
                model=model,
                revision=revision,
                dimension=dimension,
            )
        except KnowledgeEmbeddingError as error:
            raise IndexBuildFailure(error.code, error.message) from error

    async def _qdrant_client(self):
        from qdrant_client import AsyncQdrantClient

        return AsyncQdrantClient(url=self._settings.qdrant_url)

    @staticmethod
    def _scope_filter(*, user_id: int, knowledge_base_id: int):
        from qdrant_client.http import models as qmodels

        return qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="user_id", match=qmodels.MatchValue(value=user_id)
                ),
                qmodels.FieldCondition(
                    key="knowledge_base_id",
                    match=qmodels.MatchValue(value=knowledge_base_id),
                ),
            ]
        )

    async def search_dense(
        self,
        query_vector: list[float],
        *,
        user_id: int,
        knowledge_base_id: int,
        top_k: int,
        dimension: int,
    ) -> tuple[DenseSearchHit, ...]:
        """Search dense-v2 candidates using ownership scope only."""

        from qdrant_client.http import models as qmodels

        client = await self._qdrant_client()
        try:
            await self._require_collection(client, dimension=dimension)
            response = await client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                query_filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="user_id", match=qmodels.MatchValue(value=user_id)
                        ),
                        qmodels.FieldCondition(
                            key="knowledge_base_id",
                            match=qmodels.MatchValue(value=knowledge_base_id),
                        ),
                    ]
                ),
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )
            if len(response.points) > top_k:
                raise DenseSearchResultError("Qdrant returned more points than top_k")
            hits: list[DenseSearchHit] = []
            seen_chunk_ids: set[int] = set()
            for point in response.points:
                payload = point.payload
                chunk_id = (
                    None if payload is None else payload.get("knowledge_chunk_id")
                )
                if isinstance(chunk_id, bool) or not isinstance(chunk_id, int):
                    raise DenseSearchResultError("malformed knowledge_chunk_id")
                if chunk_id in seen_chunk_ids:
                    raise DenseSearchResultError("duplicate knowledge_chunk_id")
                score = float(point.score)
                if not isfinite(score):
                    raise DenseSearchResultError("non-finite Qdrant score")
                seen_chunk_ids.add(chunk_id)
                hits.append(DenseSearchHit(knowledge_chunk_id=chunk_id, score=score))
            return tuple(hits)
        finally:
            await client.close()

    @staticmethod
    async def _ensure_collection(client, *, dimension: int) -> None:
        from qdrant_client.http import models as qmodels

        if not await client.collection_exists(COLLECTION_NAME):
            await client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=qmodels.VectorParams(
                    size=dimension, distance=qmodels.Distance.COSINE
                ),
            )
            return
        await KnowledgeIndexBuildRuntime._require_collection(
            client, dimension=dimension
        )

    @staticmethod
    async def _require_collection(client, *, dimension: int) -> None:
        """Validate an existing collection without mutating the secondary index."""

        from qdrant_client.http import models as qmodels

        if not await client.collection_exists(COLLECTION_NAME):
            raise IndexBuildFailure("INDEX_COLLECTION_MISSING", "知识索引集合不存在。")
        vectors = (await client.get_collection(COLLECTION_NAME)).config.params.vectors
        if (
            not isinstance(vectors, qmodels.VectorParams)
            or vectors.size != dimension
            or vectors.distance != qmodels.Distance.COSINE
        ):
            raise IndexBuildFailure(
                "INDEX_COLLECTION_CONFIGURATION_MISMATCH",
                "现有索引集合配置不兼容。",
            )

    async def _upload(
        self,
        job: KnowledgeIndexJob,
        chunks: tuple[IndexChunk, ...],
        vectors: list[list[float]],
    ) -> None:
        from qdrant_client.http import models as qmodels

        async with self._session_factory() as session, session.begin():
            current_job = await session.get(
                KnowledgeIndexJob, job.id, with_for_update=True
            )
            knowledge_base = await session.get(
                KnowledgeBase, job.knowledge_base_id, with_for_update=True
            )
            current_chunks = await _current_chunks(session, job.knowledge_base_id)
            if (
                current_job is None
                or current_job.status != "RUNNING"
                or knowledge_base is None
                or knowledge_base.index_status != "INDEXING"
                or _snapshot_sha256(current_chunks) != current_job.chunk_snapshot_sha256
            ):
                raise IndexBuildFailure(
                    "SOURCE_CHUNKS_CHANGED", "知识分块已变化，请重新建立索引。"
                )
            for document_version_id in {
                chunk.document_version_id for chunk in current_chunks
            }:
                version = await session.get(
                    DocumentVersion, document_version_id, with_for_update=True
                )
                assert version is not None
                version.indexed_chunk_revision = None
            current_job.stage = "UPLOADING_INDEX"
            user_id = knowledge_base.user_id
        client = await self._qdrant_client()
        try:
            await self._ensure_collection(client, dimension=job.embedding_dimension)
            await client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=qmodels.FilterSelector(
                    filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key="user_id", match=qmodels.MatchValue(value=user_id)
                            ),
                            qmodels.FieldCondition(
                                key="knowledge_base_id",
                                match=qmodels.MatchValue(value=job.knowledge_base_id),
                            ),
                        ]
                    )
                ),
                wait=True,
            )
            for offset in range(0, len(chunks), _BATCH_SIZE):
                selected_chunks = chunks[offset : offset + _BATCH_SIZE]
                selected_vectors = vectors[offset : offset + _BATCH_SIZE]
                await client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=[
                        qmodels.PointStruct(
                            id=chunk.id,
                            vector=vector,
                            payload={
                                "knowledge_chunk_id": chunk.id,
                                "knowledge_base_id": job.knowledge_base_id,
                                "document_version_id": chunk.document_version_id,
                                "user_id": user_id,
                            },
                        )
                        for chunk, vector in zip(
                            selected_chunks, selected_vectors, strict=True
                        )
                    ],
                    wait=True,
                )
                await self._update_progress(job.id, offset + len(selected_chunks))
        except Exception:
            try:
                await client.delete(
                    collection_name=COLLECTION_NAME,
                    points_selector=qmodels.FilterSelector(
                        filter=qmodels.Filter(
                            must=[
                                qmodels.FieldCondition(
                                    key="user_id",
                                    match=qmodels.MatchValue(value=user_id),
                                ),
                                qmodels.FieldCondition(
                                    key="knowledge_base_id",
                                    match=qmodels.MatchValue(
                                        value=job.knowledge_base_id
                                    ),
                                ),
                            ]
                        )
                    ),
                    wait=True,
                )
            except Exception:
                pass
            raise
        finally:
            await client.close()

    async def _verify(self, job: KnowledgeIndexJob) -> None:
        await self._update_stage(job.id, "VERIFYING")
        async with self._session_factory() as session:
            knowledge_base = await session.get(KnowledgeBase, job.knowledge_base_id)
            if knowledge_base is None:
                raise IndexBuildFailure("KNOWLEDGE_BASE_NOT_FOUND", "知识库不存在。")
            user_id = knowledge_base.user_id
        client = await self._qdrant_client()
        try:
            result = await client.count(
                collection_name=COLLECTION_NAME,
                count_filter=self._scope_filter(
                    user_id=user_id, knowledge_base_id=job.knowledge_base_id
                ),
                exact=True,
            )
        finally:
            await client.close()
        if result.count != job.total_chunk_count:
            raise IndexBuildFailure("INDEX_VERIFICATION_MISMATCH", "索引验证未通过。")

    async def _activate(self, job: KnowledgeIndexJob) -> None:
        await self._update_stage(job.id, "ACTIVATING")
        async with self._session_factory() as session:
            async with session.begin():
                current_job = await session.get(
                    KnowledgeIndexJob, job.id, with_for_update=True
                )
                knowledge_base = await session.get(
                    KnowledgeBase, job.knowledge_base_id, with_for_update=True
                )
                if current_job is None or knowledge_base is None:
                    raise IndexBuildFailure("ACTIVATION_FAILED", "索引激活失败。")
                chunks = await _current_chunks(session, knowledge_base.id)
                if _snapshot_sha256(chunks) != current_job.chunk_snapshot_sha256:
                    raise IndexBuildFailure(
                        "SOURCE_CHUNKS_CHANGED",
                        "知识分块已变化，请重新建立索引。",
                        stale=True,
                    )
                if (
                    current_job.embedding_model
                    != self._settings.knowledge_embedding_model
                    or current_job.embedding_revision
                    != self._settings.knowledge_embedding_revision
                    or current_job.embedding_dimension
                    != self._settings.knowledge_embedding_dimension
                    or current_job.embedding_representation
                    != self._settings.knowledge_embedding_representation
                ):
                    raise IndexBuildFailure(
                        "INDEX_CONFIGURATION_CHANGED",
                        "索引配置已变化，请重新建立索引。",
                        stale=True,
                    )
                now = utc_now()
                for document_version_id, chunk_revision in {
                    (chunk.document_version_id, chunk.chunk_revision)
                    for chunk in chunks
                }:
                    version = await session.get(
                        DocumentVersion, document_version_id, with_for_update=True
                    )
                    assert version is not None
                    version.indexed_chunk_revision = chunk_revision
                knowledge_base.index_status = "READY" if chunks else "CHUNKED"
                knowledge_base.active_embedding_model = current_job.embedding_model
                knowledge_base.active_embedding_revision = (
                    current_job.embedding_revision
                )
                knowledge_base.active_embedding_dimension = (
                    current_job.embedding_dimension
                )
                knowledge_base.active_embedding_representation = (
                    current_job.embedding_representation
                )
                current_job.status = "SUCCEEDED"
                current_job.stage = "ACTIVATING"
                current_job.processed_chunk_count = current_job.total_chunk_count
                current_job.finished_at = now

    async def _update_stage(self, job_id: int, stage: str) -> None:
        async with self._session_factory() as session:
            async with session.begin():
                job = await session.get(KnowledgeIndexJob, job_id, with_for_update=True)
                if job is None or job.status != "RUNNING":
                    raise IndexBuildFailure(
                        "INDEX_BUILD_NOT_RUNNING", "索引任务已停止。"
                    )
                job.stage = stage

    async def _update_progress(self, job_id: int, processed_chunk_count: int) -> None:
        async with self._session_factory() as session:
            async with session.begin():
                job = await session.get(KnowledgeIndexJob, job_id, with_for_update=True)
                if job is None or job.status != "RUNNING":
                    raise IndexBuildFailure(
                        "INDEX_BUILD_NOT_RUNNING", "索引任务已停止。"
                    )
                job.processed_chunk_count = processed_chunk_count

    async def _fail(self, job_id: int, failure: IndexBuildFailure) -> None:
        async with self._session_factory() as session:
            async with session.begin():
                job = await session.get(KnowledgeIndexJob, job_id, with_for_update=True)
                if job is None or job.status in {"SUCCEEDED", "INTERRUPTED"}:
                    return
                knowledge_base = await session.get(
                    KnowledgeBase, job.knowledge_base_id, with_for_update=True
                )
                now = utc_now()
                job.status = "FAILED"
                job.stage = None
                job.error_code = failure.code
                job.error_message = failure.message
                job.finished_at = now
                if knowledge_base is not None:
                    knowledge_base.index_status = "FAILED"
