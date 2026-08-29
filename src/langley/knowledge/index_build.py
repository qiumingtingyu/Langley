"""Narrow durable lifecycle for manual Knowledge dense-index builds."""

import asyncio
import threading
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from math import isfinite
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from langley.business_time import utc_now
from langley.infrastructure.models import (
    Document,
    DocumentProcessingJob,
    DocumentVersion,
    KnowledgeBase,
    KnowledgeChunk,
    KnowledgeIndexJob,
)

if TYPE_CHECKING:
    from langley.settings import Settings


COLLECTION_NAME = "langley_knowledge_dense_v1"
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


@dataclass(frozen=True)
class DenseSearchHit:
    """One validated dense-search result, retaining Qdrant's rank order."""

    knowledge_chunk_id: int
    score: float


@dataclass(frozen=True)
class IndexBuildAdmission:
    job_id: int
    generation_id: str


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
    for chunk in sorted(
        chunks, key=lambda value: (value.id, value.document_version_id)
    ):
        digest.update(f"{chunk.id}:{chunk.document_version_id}:".encode("ascii"))
        digest.update(sha256(chunk.content.encode("utf-8")).hexdigest().encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _normalize_embedding_rows(
    values: object, *, row_count: int, dimension: int
) -> list[list[float]]:
    """Reject malformed document embeddings before any Qdrant write."""

    import numpy as np

    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape != (row_count, dimension):
        raise IndexBuildFailure("INVALID_EMBEDDING", "嵌入维度不符合索引配置。")
    if not np.isfinite(matrix).all():
        raise IndexBuildFailure("INVALID_EMBEDDING", "嵌入包含无效数值。")
    norms = np.linalg.norm(matrix, axis=1)
    if not np.isfinite(norms).all() or (norms <= 0).any():
        raise IndexBuildFailure("INVALID_EMBEDDING", "嵌入向量不能为空。")
    matrix /= norms[:, None]
    return matrix.tolist()


def _normalize_query_embedding(values: object, *, dimension: int) -> list[float]:
    """Reject a malformed production query vector before Qdrant search."""

    import numpy as np

    matrix = np.asarray(values)
    if matrix.dtype != np.float32 or matrix.shape != (1, dimension):
        raise IndexBuildFailure("INVALID_EMBEDDING", "嵌入维度不符合索引配置。")
    if not np.isfinite(matrix).all():
        raise IndexBuildFailure("INVALID_EMBEDDING", "嵌入包含无效数值。")
    norm = float(np.linalg.norm(matrix[0]))
    if not np.isfinite(norm) or norm <= 0:
        raise IndexBuildFailure("INVALID_EMBEDDING", "嵌入向量不能为空。")
    normalized = np.asarray(matrix[0] / norm, dtype=np.float32)
    if not np.isfinite(normalized).all():
        raise IndexBuildFailure("INVALID_EMBEDDING", "嵌入包含无效数值。")
    return normalized.tolist()


async def _current_chunks(
    session: AsyncSession, knowledge_base_id: int
) -> tuple[IndexChunk, ...]:
    rows = (
        await session.execute(
            current_knowledge_chunks_statement(knowledge_base_id).with_only_columns(
                KnowledgeChunk.id,
                KnowledgeChunk.document_version_id,
                KnowledgeChunk.content,
            )
        )
    ).all()
    return tuple(
        IndexChunk(
            id=row.id, document_version_id=row.document_version_id, content=row.content
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
    """Atomically admit one manual generation without doing slow work."""

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
            generation_id = str(uuid4())
            job = KnowledgeIndexJob(
                knowledge_base_id=knowledge_base_id,
                generation_id=generation_id,
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
            knowledge_base.building_generation_id = generation_id
            session.add(job)
            await session.flush()
            return IndexBuildAdmission(job_id=job.id, generation_id=generation_id)


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
                    and knowledge_base.building_generation_id == job.generation_id
                ):
                    knowledge_base.building_generation_id = None
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
                    .where(KnowledgeBase.index_status == "READY")
                    .with_for_update()
                )
            ).all()
            stale_ids: list[int] = []
            for knowledge_base in knowledge_bases:
                if (
                    knowledge_base.active_embedding_model
                    != settings.knowledge_embedding_model
                    or knowledge_base.active_embedding_revision
                    != settings.knowledge_embedding_revision
                    or knowledge_base.active_embedding_dimension
                    != settings.knowledge_embedding_dimension
                    or knowledge_base.active_embedding_representation
                    != settings.knowledge_embedding_representation
                ):
                    knowledge_base.index_status = "STALE"
                    stale_ids.append(knowledge_base.id)
            return tuple(stale_ids)


class KnowledgeIndexBuildRuntime:
    """One application-local, bounded executor for Task 5.1 index builds."""

    def __init__(
        self, session_factory: async_sessionmaker[AsyncSession], settings: "Settings"
    ) -> None:
        self._session_factory = session_factory
        self._settings = settings
        self._capacity = asyncio.Semaphore(settings.knowledge_index_build_concurrency)
        self._tasks: set[asyncio.Task[None]] = set()
        self._embedding_lock = threading.Lock()
        self._embedding_identity: tuple[str, str, str] | None = None
        self._embedding_model: Any | None = None

    @property
    def settings(self) -> "Settings":
        return self._settings

    def schedule(self, job_id: int) -> None:
        task = asyncio.create_task(self.run(job_id))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def run(self, job_id: int) -> None:
        """Run an admitted job, always persisting a terminal outcome."""

        async with self._capacity:
            try:
                await self._mark_running(job_id)
                job, chunks = await self._load_snapshot(job_id)
                vectors = await self._embed(job, chunks)
                await self._upload(job, chunks, vectors)
                await self._verify(job)
                await self._activate(job)
            except IndexBuildFailure as error:
                await self._fail(job_id, error)
            except Exception:
                await self._fail(
                    job_id,
                    IndexBuildFailure(
                        "INDEX_BUILD_FAILED", "索引建立失败，请检查本地索引服务后重试。"
                    ),
                )

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
        if job.embedding_representation != "content_only":
            raise IndexBuildFailure(
                "EMBEDDING_REPRESENTATION_UNSUPPORTED", "嵌入表示配置不受支持。"
            )
        vectors = await asyncio.to_thread(
            self._encode_documents, [chunk.content for chunk in chunks], job
        )
        if len(vectors) != len(chunks):
            raise IndexBuildFailure("EMBEDDING_COUNT_MISMATCH", "嵌入结果不完整。")
        return vectors

    def _encode_documents(
        self, contents: list[str], job: KnowledgeIndexJob
    ) -> list[list[float]]:
        """Perform heavy local BGE-M3 document encoding outside the event loop."""

        with self._embedding_lock:
            model = self._embedding_model_for(
                job.embedding_model, job.embedding_revision
            )
            values = model.encode_document(
                contents, convert_to_numpy=True, show_progress_bar=False
            )
        return _normalize_embedding_rows(
            values, row_count=len(contents), dimension=job.embedding_dimension
        )

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

        if representation != "content_only":
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

        with self._embedding_lock:
            embedding_model = self._embedding_model_for(model, revision)
            values = embedding_model.encode_query(
                [query], convert_to_numpy=True, show_progress_bar=False
            )
        return _normalize_query_embedding(values, dimension=dimension)

    def _embedding_model_for(self, model: str, revision: str) -> Any:
        """Load or reuse the one runtime-local BGE instance while the lock is held."""

        configured_device = self._settings.knowledge_embedding_device
        identity = (model, revision, configured_device)
        if self._embedding_identity == identity and self._embedding_model is not None:
            return self._embedding_model
        from sentence_transformers import SentenceTransformer

        embedding_model = SentenceTransformer(
            model, revision=revision, device=configured_device
        )
        if str(embedding_model.device) != configured_device:
            raise IndexBuildFailure(
                "EMBEDDING_DEVICE_UNAVAILABLE", "配置的嵌入设备不可用。"
            )
        self._embedding_identity = identity
        self._embedding_model = embedding_model
        return embedding_model

    async def _qdrant_client(self):
        from qdrant_client import AsyncQdrantClient

        return AsyncQdrantClient(url=self._settings.qdrant_url)

    async def search_dense(
        self,
        query_vector: list[float],
        *,
        user_id: int,
        knowledge_base_id: int,
        generation_id: str,
        top_k: int,
        dimension: int,
    ) -> tuple[DenseSearchHit, ...]:
        """Search one active dense generation with its complete ownership filter."""

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
                        qmodels.FieldCondition(
                            key="generation_id",
                            match=qmodels.MatchValue(value=generation_id),
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

        await self._update_stage(job.id, "UPLOADING_INDEX")
        async with self._session_factory() as session:
            knowledge_base = await session.get(KnowledgeBase, job.knowledge_base_id)
            if knowledge_base is None:
                raise IndexBuildFailure("KNOWLEDGE_BASE_NOT_FOUND", "知识库不存在。")
            user_id = knowledge_base.user_id
        client = await self._qdrant_client()
        try:
            await self._ensure_collection(client, dimension=job.embedding_dimension)
            for offset in range(0, len(chunks), _BATCH_SIZE):
                selected_chunks = chunks[offset : offset + _BATCH_SIZE]
                selected_vectors = vectors[offset : offset + _BATCH_SIZE]
                await client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=[
                        qmodels.PointStruct(
                            id=str(uuid4()),
                            vector=vector,
                            payload={
                                "knowledge_chunk_id": chunk.id,
                                "knowledge_base_id": job.knowledge_base_id,
                                "document_version_id": chunk.document_version_id,
                                "user_id": user_id,
                                "generation_id": job.generation_id,
                            },
                        )
                        for chunk, vector in zip(
                            selected_chunks, selected_vectors, strict=True
                        )
                    ],
                    wait=True,
                )
                await self._update_progress(job.id, offset + len(selected_chunks))
        finally:
            await client.close()

    async def _verify(self, job: KnowledgeIndexJob) -> None:
        from qdrant_client.http import models as qmodels

        await self._update_stage(job.id, "VERIFYING")
        client = await self._qdrant_client()
        try:
            result = await client.count(
                collection_name=COLLECTION_NAME,
                count_filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="generation_id",
                            match=qmodels.MatchValue(value=job.generation_id),
                        )
                    ]
                ),
                exact=True,
            )
        finally:
            await client.close()
        if result.count != job.total_chunk_count:
            raise IndexBuildFailure("INDEX_VERIFICATION_MISMATCH", "索引验证未通过。")

    async def _activate(self, job: KnowledgeIndexJob) -> None:
        await self._update_stage(job.id, "ACTIVATING")
        previous_generation: str | None = None
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
                if knowledge_base.building_generation_id != current_job.generation_id:
                    raise IndexBuildFailure(
                        "ACTIVATION_FAILED", "索引生成已不再是当前任务。"
                    )
                previous_generation = knowledge_base.active_generation_id
                now = utc_now()
                knowledge_base.index_status = "READY"
                knowledge_base.active_generation_id = current_job.generation_id
                knowledge_base.building_generation_id = None
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
                knowledge_base.active_chunk_snapshot_sha256 = (
                    current_job.chunk_snapshot_sha256
                )
                current_job.status = "SUCCEEDED"
                current_job.stage = "ACTIVATING"
                current_job.processed_chunk_count = current_job.total_chunk_count
                current_job.finished_at = now
        if previous_generation is not None and previous_generation != job.generation_id:
            await self._cleanup_generation(previous_generation)

    async def _cleanup_generation(self, generation_id: str) -> None:
        """Best-effort cleanup cannot affect an already activated generation."""

        client = None
        try:
            from qdrant_client.http import models as qmodels

            client = await self._qdrant_client()
            await client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=qmodels.FilterSelector(
                    filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key="generation_id",
                                match=qmodels.MatchValue(value=generation_id),
                            )
                        ]
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
                if (
                    knowledge_base is not None
                    and knowledge_base.building_generation_id == job.generation_id
                ):
                    knowledge_base.building_generation_id = None
                    knowledge_base.index_status = "STALE" if failure.stale else "FAILED"
