"""Document-scoped incremental dense-index publication and lifecycle."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import structlog
from sqlalchemy import delete, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from langley.business_time import utc_now
from langley.infrastructure.models import (
    Document,
    DocumentIndexJob,
    DocumentVersion,
    KnowledgeBase,
    KnowledgeChunk,
)
from langley.knowledge.contracts import (
    encode_source_region,
    validate_heading_path,
    validate_source_regions,
)
from langley.knowledge.document_index_contract import (
    SOURCE_CONTEXT_V1,
    build_source_context_v1,
    chunk_set_sha256,
)
from langley.knowledge.embedding_runtime import (
    KnowledgeEmbeddingError,
    KnowledgeEmbeddingRuntime,
)

if TYPE_CHECKING:
    from langley.settings import Settings


logger = structlog.get_logger(__name__)

DOCUMENT_INDEX_COLLECTION = "langley_knowledge_dense_v2"
DOCUMENT_INDEX_REPRESENTATION = SOURCE_CONTEXT_V1
_BATCH_SIZE = 64
_ACTIVE_JOB_STATUSES = ("PENDING", "RUNNING")


@dataclass(frozen=True)
class DocumentIndexConfiguration:
    model: str
    revision: str
    dimension: int
    representation: str
    qdrant_url: str

    @classmethod
    def from_settings(cls, settings: Settings) -> DocumentIndexConfiguration:
        return cls(
            model=settings.knowledge_embedding_model,
            revision=settings.knowledge_embedding_revision,
            dimension=settings.knowledge_embedding_dimension,
            representation=DOCUMENT_INDEX_REPRESENTATION,
            qdrant_url=settings.qdrant_url,
        )


@dataclass(frozen=True)
class DocumentChunkPublicationResult:
    document_version_id: int
    chunk_count: int
    index_status: str
    changed: bool
    job_created: bool


@dataclass(frozen=True)
class DocumentIndexClaim:
    job_id: int
    document_version_id: int
    attempt_no: int
    target_chunk_revision: int
    knowledge_base_id: int
    user_id: int
    model: str
    revision: str
    dimension: int
    representation: str


@dataclass(frozen=True)
class DocumentIndexChunk:
    id: int
    ordinal: int
    content: str
    heading_path: tuple[str, ...]
    source_regions: tuple[dict[str, object], ...]


class DocumentIndexFailure(Exception):
    """One safe terminal failure for a claimed document-index attempt."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(code)
        self.code = code
        self.message = message


async def publish_document_chunks(
    session: AsyncSession,
    *,
    knowledge_base: KnowledgeBase,
    version: DocumentVersion,
    prepared_rows: list[KnowledgeChunk],
    configuration: DocumentIndexConfiguration,
) -> DocumentChunkPublicationResult:
    """Publish one candidate chunk set and admit only the necessary index work."""

    if not prepared_rows:
        changed = version.chunk_revision != 0 or version.chunk_set_sha256 is not None
        if changed:
            await session.execute(
                delete(KnowledgeChunk).where(
                    KnowledgeChunk.document_version_id == version.id
                )
            )
            version.chunk_revision = 0
            version.chunk_set_sha256 = None
            version.indexed_chunk_revision = None
        await refresh_knowledge_base_readiness(session, knowledge_base)
        await session.flush()
        return DocumentChunkPublicationResult(
            document_version_id=version.id,
            chunk_count=0,
            index_status=knowledge_base.index_status,
            changed=changed,
            job_created=False,
        )

    fingerprint = chunk_set_sha256(prepared_rows)
    changed = version.chunk_set_sha256 != fingerprint
    if changed:
        await session.execute(
            delete(KnowledgeChunk).where(
                KnowledgeChunk.document_version_id == version.id
            )
        )
        session.add_all(prepared_rows)
        version.chunk_revision += 1
        version.chunk_set_sha256 = fingerprint
        await refresh_knowledge_base_readiness(session, knowledge_base)

    job_created = False
    if version.indexed_chunk_revision != version.chunk_revision:
        job_created = await _ensure_document_index_job(
            session,
            version=version,
            configuration=configuration,
        )
    await session.flush()
    return DocumentChunkPublicationResult(
        document_version_id=version.id,
        chunk_count=len(prepared_rows),
        index_status=knowledge_base.index_status,
        changed=changed,
        job_created=job_created,
    )


async def _ensure_document_index_job(
    session: AsyncSession,
    *,
    version: DocumentVersion,
    configuration: DocumentIndexConfiguration,
) -> bool:
    active_id = await session.scalar(
        select(DocumentIndexJob.id)
        .where(
            DocumentIndexJob.document_version_id == version.id,
            DocumentIndexJob.target_chunk_revision == version.chunk_revision,
            DocumentIndexJob.status.in_(_ACTIVE_JOB_STATUSES),
            DocumentIndexJob.embedding_model == configuration.model,
            DocumentIndexJob.embedding_revision == configuration.revision,
            DocumentIndexJob.embedding_dimension == configuration.dimension,
            DocumentIndexJob.embedding_representation == configuration.representation,
        )
        .limit(1)
    )
    if active_id is not None:
        return False
    latest_attempt = await session.scalar(
        select(func.max(DocumentIndexJob.attempt_no)).where(
            DocumentIndexJob.document_version_id == version.id
        )
    )
    session.add(
        DocumentIndexJob(
            document_version_id=version.id,
            attempt_no=(latest_attempt or 0) + 1,
            target_chunk_revision=version.chunk_revision,
            status="PENDING",
            stage=None,
            embedding_model=configuration.model,
            embedding_revision=configuration.revision,
            embedding_dimension=configuration.dimension,
            embedding_representation=configuration.representation,
            error_code=None,
            error_message=None,
            created_at=utc_now(),
            started_at=None,
            finished_at=None,
        )
    )
    return True


async def claim_next_document_index_job(
    session_factory: async_sessionmaker[AsyncSession],
    configuration: DocumentIndexConfiguration | None = None,
) -> DocumentIndexClaim | None:
    """Claim with KnowledgeBase-first locking against full-build admission."""

    async with session_factory() as session, session.begin():
        statement = (
            select(
                DocumentIndexJob.id,
                KnowledgeBase.id.label("knowledge_base_id"),
                KnowledgeBase.user_id,
            )
            .join(
                DocumentVersion,
                DocumentVersion.id == DocumentIndexJob.document_version_id,
            )
            .join(Document, Document.id == DocumentVersion.document_id)
            .join(KnowledgeBase, KnowledgeBase.id == Document.knowledge_base_id)
            .where(
                DocumentIndexJob.status == "PENDING",
                KnowledgeBase.index_status.in_(("CHUNKED", "READY")),
            )
            .order_by(DocumentIndexJob.id.asc())
            .limit(1)
        )
        if configuration is not None:
            statement = statement.where(
                DocumentIndexJob.embedding_model == configuration.model,
                DocumentIndexJob.embedding_revision == configuration.revision,
                DocumentIndexJob.embedding_dimension == configuration.dimension,
                DocumentIndexJob.embedding_representation
                == configuration.representation,
            )
        candidate = (await session.execute(statement)).one_or_none()
        if candidate is None:
            return None
        knowledge_base = await session.get(
            KnowledgeBase, candidate.knowledge_base_id, with_for_update=True
        )
        if knowledge_base is None or knowledge_base.index_status not in {
            "CHUNKED",
            "READY",
        }:
            return None
        job = await session.get(DocumentIndexJob, candidate.id, with_for_update=True)
        if job is None or job.status != "PENDING":
            return None
        mark_document_index_running(job, now=utc_now())
        return DocumentIndexClaim(
            job_id=job.id,
            document_version_id=job.document_version_id,
            attempt_no=job.attempt_no,
            target_chunk_revision=job.target_chunk_revision,
            knowledge_base_id=knowledge_base.id,
            user_id=knowledge_base.user_id,
            model=job.embedding_model,
            revision=job.embedding_revision,
            dimension=job.embedding_dimension,
            representation=job.embedding_representation,
        )


async def seed_document_index_backlog(
    session_factory: async_sessionmaker[AsyncSession],
    configuration: DocumentIndexConfiguration,
) -> tuple[int, ...]:
    """Seed current-config work for every non-ready current document revision."""

    async with session_factory() as session, session.begin():
        versions = tuple(
            (
                await session.scalars(
                    select(DocumentVersion)
                    .join(Document, Document.id == DocumentVersion.document_id)
                    .join(KnowledgeBase, KnowledgeBase.id == Document.knowledge_base_id)
                    .where(
                        DocumentVersion.chunk_revision > 0,
                        or_(
                            DocumentVersion.indexed_chunk_revision.is_(None),
                            DocumentVersion.indexed_chunk_revision
                            != DocumentVersion.chunk_revision,
                        ),
                        KnowledgeBase.index_status.in_(("CHUNKED", "READY")),
                    )
                    .order_by(DocumentVersion.id)
                )
            ).all()
        )
        created_ids: list[int] = []
        for version in versions:
            if await _ensure_document_index_job(
                session, version=version, configuration=configuration
            ):
                await session.flush()
                job_id = await session.scalar(
                    select(func.max(DocumentIndexJob.id)).where(
                        DocumentIndexJob.document_version_id == version.id
                    )
                )
                assert job_id is not None
                created_ids.append(job_id)
        return tuple(created_ids)


async def refresh_knowledge_base_readiness(
    session: AsyncSession, knowledge_base: KnowledgeBase
) -> None:
    """Adjust only CHUNKED/READY from current document revision authority."""

    if knowledge_base.index_status not in {"CHUNKED", "READY"}:
        return
    await session.flush()
    ready_exists = (
        await session.scalar(
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
    ) is not None
    knowledge_base.index_status = "READY" if ready_exists else "CHUNKED"


async def reconcile_interrupted_document_index_jobs(
    session_factory: async_sessionmaker[AsyncSession],
) -> tuple[int, ...]:
    """Make restart interruption durable without retrying or restoring readiness."""

    async with session_factory() as session, session.begin():
        jobs = tuple(
            (
                await session.scalars(
                    select(DocumentIndexJob)
                    .where(DocumentIndexJob.status == "RUNNING")
                    .order_by(DocumentIndexJob.id)
                    .with_for_update()
                )
            ).all()
        )
        now = utc_now()
        for job in jobs:
            interrupt_document_index_job(job, now=now)
        return tuple(job.id for job in jobs)


def mark_document_index_running(job: DocumentIndexJob, *, now: datetime) -> None:
    """Apply the exact PENDING to RUNNING/EMBEDDING transition."""

    if job.status != "PENDING":
        raise ValueError("document index job is not PENDING")
    job.status = "RUNNING"
    job.stage = "EMBEDDING"
    job.started_at = now


def interrupt_document_index_job(job: DocumentIndexJob, *, now: datetime) -> None:
    """Apply restart reconciliation while leaving PENDING jobs untouched."""

    if job.status != "RUNNING":
        return
    job.status = "INTERRUPTED"
    job.error_code = "INDEX_INTERRUPTED"
    job.error_message = _safe_error_message("INDEX_INTERRUPTED")
    job.finished_at = now


class DocumentIndexRuntime:
    """Execute one claimed document revision without retaining DB resources."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        configuration: DocumentIndexConfiguration,
        embedding_runtime: KnowledgeEmbeddingRuntime,
    ) -> None:
        self._session_factory = session_factory
        self._configuration = configuration
        self._embedding_runtime = embedding_runtime

    @property
    def embedding_runtime(self) -> KnowledgeEmbeddingRuntime:
        return self._embedding_runtime

    async def execute(self, claim: DocumentIndexClaim) -> None:
        barrier_crossed = False
        metadata = {
            "job_id": claim.job_id,
            "document_version_id": claim.document_version_id,
            "attempt_no": claim.attempt_no,
            "target_chunk_revision": claim.target_chunk_revision,
        }
        logger.info("knowledge.document_index.started", **metadata)
        try:
            chunks, expected_chunk_set_sha256 = await self._load_snapshot(claim)
            contexts = [
                build_source_context_v1(chunk.content, chunk.heading_path)
                for chunk in chunks
            ]
            try:
                vectors = await asyncio.to_thread(
                    self._embedding_runtime.encode_documents,
                    contexts,
                    model=claim.model,
                    revision=claim.revision,
                    dimension=claim.dimension,
                )
            except KnowledgeEmbeddingError as error:
                code = (
                    "INVALID_EMBEDDING"
                    if error.code == "INVALID_EMBEDDING"
                    else "EMBEDDING_FAILED"
                )
                raise DocumentIndexFailure(code, _safe_error_message(code)) from error
            except Exception as error:
                raise DocumentIndexFailure(
                    "EMBEDDING_FAILED", _safe_error_message("EMBEDDING_FAILED")
                ) from error
            if len(vectors) != len(chunks):
                raise DocumentIndexFailure(
                    "INVALID_EMBEDDING", _safe_error_message("INVALID_EMBEDDING")
                )

            await self._publication_barrier(claim, expected_chunk_set_sha256)
            barrier_crossed = True
            try:
                await self._publish_vectors(claim, chunks, vectors)
            except Exception as error:
                raise DocumentIndexFailure(
                    "INDEX_PUBLICATION_FAILED",
                    _safe_error_message("INDEX_PUBLICATION_FAILED"),
                ) from error
            await self._advance_to_verifying(claim, expected_chunk_set_sha256)
            try:
                count = await self._count_vectors(claim)
            except Exception as error:
                raise DocumentIndexFailure(
                    "INDEX_VERIFICATION_FAILED",
                    _safe_error_message("INDEX_VERIFICATION_FAILED"),
                ) from error
            if count != len(chunks):
                raise DocumentIndexFailure(
                    "INDEX_VERIFICATION_FAILED",
                    _safe_error_message("INDEX_VERIFICATION_FAILED"),
                )
            await self._complete(claim, expected_chunk_set_sha256)
        except DocumentIndexFailure as error:
            if barrier_crossed:
                await self._cleanup_vectors(claim)
            await self._fail(claim.job_id, error)
            logger.error(
                "knowledge.document_index.failed",
                **metadata,
                error_code=error.code,
            )
            return
        except Exception:
            unexpected_failure = DocumentIndexFailure(
                "INDEX_PUBLICATION_FAILED",
                _safe_error_message("INDEX_PUBLICATION_FAILED"),
            )
            if barrier_crossed:
                await self._cleanup_vectors(claim)
            await self._fail(claim.job_id, unexpected_failure)
            logger.exception(
                "knowledge.document_index.unexpected_failure",
                **metadata,
                error_code=unexpected_failure.code,
            )
            return
        logger.info("knowledge.document_index.succeeded", **metadata)

    async def _load_snapshot(
        self, claim: DocumentIndexClaim
    ) -> tuple[tuple[DocumentIndexChunk, ...], str]:
        self._require_configuration(claim)
        async with self._session_factory() as session:
            job = await session.get(DocumentIndexJob, claim.job_id)
            version = await session.get(DocumentVersion, claim.document_version_id)
            if (
                not _job_matches_claim(job, claim, stage="EMBEDDING")
                or version is None
                or version.chunk_revision != claim.target_chunk_revision
                or version.chunk_set_sha256 is None
            ):
                raise DocumentIndexFailure(
                    "SOURCE_CHUNKS_CHANGED",
                    _safe_error_message("SOURCE_CHUNKS_CHANGED"),
                )
            rows = tuple(
                (
                    await session.scalars(
                        select(KnowledgeChunk)
                        .where(
                            KnowledgeChunk.document_version_id
                            == claim.document_version_id
                        )
                        .order_by(KnowledgeChunk.ordinal)
                    )
                ).all()
            )
            chunks = tuple(
                DocumentIndexChunk(
                    id=row.id,
                    ordinal=row.ordinal,
                    content=row.content,
                    heading_path=tuple(validate_heading_path(row.heading_path)),
                    source_regions=tuple(
                        encode_source_region(region)
                        for region in validate_source_regions(row.source_regions)
                    ),
                )
                for row in rows
            )
            if not chunks or chunk_set_sha256(chunks) != version.chunk_set_sha256:
                raise DocumentIndexFailure(
                    "SOURCE_CHUNKS_CHANGED",
                    _safe_error_message("SOURCE_CHUNKS_CHANGED"),
                )
            return chunks, version.chunk_set_sha256

    async def _publication_barrier(
        self, claim: DocumentIndexClaim, expected_chunk_set_sha256: str
    ) -> None:
        configuration_changed = False
        async with self._session_factory() as session, session.begin():
            knowledge_base = await session.get(
                KnowledgeBase, claim.knowledge_base_id, with_for_update=True
            )
            job = await session.get(
                DocumentIndexJob, claim.job_id, with_for_update=True
            )
            version = await session.get(
                DocumentVersion, claim.document_version_id, with_for_update=True
            )
            if (
                knowledge_base is None
                or knowledge_base.index_status not in {"CHUNKED", "READY"}
                or not _job_matches_claim(job, claim, stage="EMBEDDING")
                or version is None
                or version.chunk_revision != claim.target_chunk_revision
                or version.chunk_set_sha256 != expected_chunk_set_sha256
            ):
                raise DocumentIndexFailure(
                    "SOURCE_CHUNKS_CHANGED",
                    _safe_error_message("SOURCE_CHUNKS_CHANGED"),
                )
            self._require_configuration(claim)
            assert job is not None
            if not _knowledge_base_uses_configuration(
                knowledge_base, self._configuration
            ):
                initialized = await _initialize_fresh_knowledge_base_configuration(
                    session,
                    knowledge_base=knowledge_base,
                    configuration=self._configuration,
                )
                if not initialized:
                    if await _ready_document_version_exists(
                        session, knowledge_base_id=knowledge_base.id
                    ):
                        knowledge_base.index_status = "STALE"
                    configuration_changed = True
            if not configuration_changed:
                version.indexed_chunk_revision = None
                job.stage = "PUBLISHING"
                await refresh_knowledge_base_readiness(session, knowledge_base)
        if configuration_changed:
            raise DocumentIndexFailure(
                "INDEX_CONFIGURATION_CHANGED",
                _safe_error_message("INDEX_CONFIGURATION_CHANGED"),
            )

    async def _advance_to_verifying(
        self, claim: DocumentIndexClaim, expected_chunk_set_sha256: str
    ) -> None:
        async with self._session_factory() as session, session.begin():
            job = await session.get(
                DocumentIndexJob, claim.job_id, with_for_update=True
            )
            version = await session.get(
                DocumentVersion, claim.document_version_id, with_for_update=True
            )
            if (
                not _job_matches_claim(job, claim, stage="PUBLISHING")
                or version is None
                or version.chunk_revision != claim.target_chunk_revision
                or version.chunk_set_sha256 != expected_chunk_set_sha256
            ):
                raise DocumentIndexFailure(
                    "SOURCE_CHUNKS_CHANGED",
                    _safe_error_message("SOURCE_CHUNKS_CHANGED"),
                )
            self._require_configuration(claim)
            assert job is not None
            job.stage = "VERIFYING"

    async def _complete(
        self, claim: DocumentIndexClaim, expected_chunk_set_sha256: str
    ) -> None:
        async with self._session_factory() as session, session.begin():
            knowledge_base = await session.get(
                KnowledgeBase, claim.knowledge_base_id, with_for_update=True
            )
            job = await session.get(
                DocumentIndexJob, claim.job_id, with_for_update=True
            )
            version = await session.get(
                DocumentVersion, claim.document_version_id, with_for_update=True
            )
            if (
                knowledge_base is None
                or knowledge_base.index_status not in {"CHUNKED", "READY"}
                or not _knowledge_base_uses_configuration(
                    knowledge_base, self._configuration
                )
                or not _job_matches_claim(job, claim, stage="VERIFYING")
                or version is None
                or version.chunk_revision != claim.target_chunk_revision
                or version.chunk_set_sha256 != expected_chunk_set_sha256
            ):
                raise DocumentIndexFailure(
                    "SOURCE_CHUNKS_CHANGED",
                    _safe_error_message("SOURCE_CHUNKS_CHANGED"),
                )
            self._require_configuration(claim)
            assert job is not None
            version.indexed_chunk_revision = claim.target_chunk_revision
            job.status = "SUCCEEDED"
            job.finished_at = utc_now()
            await refresh_knowledge_base_readiness(session, knowledge_base)

    def _require_configuration(self, claim: DocumentIndexClaim) -> None:
        if (
            claim.model != self._configuration.model
            or claim.revision != self._configuration.revision
            or claim.dimension != self._configuration.dimension
            or claim.representation != self._configuration.representation
        ):
            raise DocumentIndexFailure(
                "INDEX_CONFIGURATION_CHANGED",
                _safe_error_message("INDEX_CONFIGURATION_CHANGED"),
            )

    async def _qdrant_client(self):
        from qdrant_client import AsyncQdrantClient

        return AsyncQdrantClient(url=self._configuration.qdrant_url)

    @staticmethod
    def _scope_filter(claim: DocumentIndexClaim):
        from qdrant_client.http import models as qmodels

        return qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="user_id", match=qmodels.MatchValue(value=claim.user_id)
                ),
                qmodels.FieldCondition(
                    key="knowledge_base_id",
                    match=qmodels.MatchValue(value=claim.knowledge_base_id),
                ),
                qmodels.FieldCondition(
                    key="document_version_id",
                    match=qmodels.MatchValue(value=claim.document_version_id),
                ),
            ]
        )

    async def _publish_vectors(
        self,
        claim: DocumentIndexClaim,
        chunks: tuple[DocumentIndexChunk, ...],
        vectors: list[list[float]],
    ) -> None:
        from qdrant_client.http import models as qmodels

        client = await self._qdrant_client()
        try:
            await self._ensure_collection(client, dimension=claim.dimension)
            await client.delete(
                collection_name=DOCUMENT_INDEX_COLLECTION,
                points_selector=qmodels.FilterSelector(
                    filter=self._scope_filter(claim)
                ),
                wait=True,
            )
            for offset in range(0, len(chunks), _BATCH_SIZE):
                selected_chunks = chunks[offset : offset + _BATCH_SIZE]
                selected_vectors = vectors[offset : offset + _BATCH_SIZE]
                await client.upsert(
                    collection_name=DOCUMENT_INDEX_COLLECTION,
                    points=[
                        qmodels.PointStruct(
                            id=chunk.id,
                            vector=vector,
                            payload={
                                "knowledge_chunk_id": chunk.id,
                                "knowledge_base_id": claim.knowledge_base_id,
                                "document_version_id": claim.document_version_id,
                                "user_id": claim.user_id,
                            },
                        )
                        for chunk, vector in zip(
                            selected_chunks, selected_vectors, strict=True
                        )
                    ],
                    wait=True,
                )
        finally:
            await client.close()

    async def _count_vectors(self, claim: DocumentIndexClaim) -> int:
        client = await self._qdrant_client()
        try:
            result = await client.count(
                collection_name=DOCUMENT_INDEX_COLLECTION,
                count_filter=self._scope_filter(claim),
                exact=True,
            )
            return result.count
        finally:
            await client.close()

    async def _cleanup_vectors(self, claim: DocumentIndexClaim) -> None:
        client = None
        try:
            from qdrant_client.http import models as qmodels

            client = await self._qdrant_client()
            await client.delete(
                collection_name=DOCUMENT_INDEX_COLLECTION,
                points_selector=qmodels.FilterSelector(
                    filter=self._scope_filter(claim)
                ),
                wait=True,
            )
        except Exception:
            logger.warning(
                "knowledge.document_index.cleanup_failed",
                job_id=claim.job_id,
                document_version_id=claim.document_version_id,
            )
        finally:
            if client is not None:
                try:
                    await client.close()
                except Exception:
                    pass

    @staticmethod
    async def _ensure_collection(client, *, dimension: int) -> None:
        from qdrant_client.http import models as qmodels

        if not await client.collection_exists(DOCUMENT_INDEX_COLLECTION):
            await client.create_collection(
                collection_name=DOCUMENT_INDEX_COLLECTION,
                vectors_config=qmodels.VectorParams(
                    size=dimension, distance=qmodels.Distance.COSINE
                ),
            )
            return
        vectors = (
            await client.get_collection(DOCUMENT_INDEX_COLLECTION)
        ).config.params.vectors
        if (
            not isinstance(vectors, qmodels.VectorParams)
            or vectors.size != dimension
            or vectors.distance != qmodels.Distance.COSINE
        ):
            raise RuntimeError("document index collection configuration mismatch")

    async def _fail(self, job_id: int, failure: DocumentIndexFailure) -> None:
        async with self._session_factory() as session, session.begin():
            job = await session.get(DocumentIndexJob, job_id, with_for_update=True)
            if job is None or job.status != "RUNNING":
                return
            job.status = "FAILED"
            job.error_code = failure.code
            job.error_message = failure.message
            job.finished_at = utc_now()


class DocumentIndexDispatcher:
    """One application-local serial dispatcher over durable document index jobs."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        runtime: DocumentIndexRuntime,
        configuration: DocumentIndexConfiguration,
    ) -> None:
        self._session_factory = session_factory
        self._runtime = runtime
        self.configuration = configuration
        self._wake = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._task is not None:
            raise RuntimeError("document index dispatcher already started")
        self._wake.set()
        self._task = asyncio.create_task(self._run())

    def wake(self) -> None:
        self._wake.set()

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def _run(self) -> None:
        while True:
            try:
                claim = await claim_next_document_index_job(
                    self._session_factory, self.configuration
                )
                if claim is not None:
                    await self._runtime.execute(claim)
                    continue
                self._wake.clear()
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=2.0)
                except TimeoutError:
                    pass
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("knowledge.document_index.dispatch_failed")
                await asyncio.sleep(2.0)


def _safe_error_message(code: str) -> str:
    return {
        "SOURCE_CHUNKS_CHANGED": "文档分块已变化，请重新建立索引。",
        "INDEX_CONFIGURATION_CHANGED": "索引配置已变化，请重新建立索引。",
        "EMBEDDING_FAILED": "文档嵌入失败。",
        "INVALID_EMBEDDING": "文档嵌入结果无效。",
        "INDEX_PUBLICATION_FAILED": "文档索引发布失败。",
        "INDEX_VERIFICATION_FAILED": "文档索引验证失败。",
        "INDEX_INTERRUPTED": "文档索引任务因应用重启而中断。",
    }[code]


def _job_matches_claim(
    job: DocumentIndexJob | None, claim: DocumentIndexClaim, *, stage: str
) -> bool:
    return (
        job is not None
        and job.status == "RUNNING"
        and job.stage == stage
        and job.document_version_id == claim.document_version_id
        and job.attempt_no == claim.attempt_no
        and job.target_chunk_revision == claim.target_chunk_revision
        and job.embedding_model == claim.model
        and job.embedding_revision == claim.revision
        and job.embedding_dimension == claim.dimension
        and job.embedding_representation == claim.representation
    )


def _knowledge_base_uses_configuration(
    knowledge_base: KnowledgeBase, configuration: DocumentIndexConfiguration
) -> bool:
    return (
        knowledge_base.active_embedding_model == configuration.model
        and knowledge_base.active_embedding_revision == configuration.revision
        and knowledge_base.active_embedding_dimension == configuration.dimension
        and knowledge_base.active_embedding_representation
        == configuration.representation
    )


async def _initialize_fresh_knowledge_base_configuration(
    session: AsyncSession,
    *,
    knowledge_base: KnowledgeBase,
    configuration: DocumentIndexConfiguration,
) -> bool:
    if (
        knowledge_base.index_status != "CHUNKED"
        or configuration.representation != DOCUMENT_INDEX_REPRESENTATION
        or any(
            value is not None
            for value in (
                knowledge_base.active_embedding_model,
                knowledge_base.active_embedding_revision,
                knowledge_base.active_embedding_dimension,
                knowledge_base.active_embedding_representation,
            )
        )
        or await _ready_document_version_exists(
            session, knowledge_base_id=knowledge_base.id
        )
    ):
        return False
    knowledge_base.active_embedding_model = configuration.model
    knowledge_base.active_embedding_revision = configuration.revision
    knowledge_base.active_embedding_dimension = configuration.dimension
    knowledge_base.active_embedding_representation = configuration.representation
    return True


async def _ready_document_version_exists(
    session: AsyncSession, *, knowledge_base_id: int
) -> bool:
    return (
        await session.scalar(
            select(DocumentVersion.id)
            .join(Document, Document.id == DocumentVersion.document_id)
            .where(
                Document.knowledge_base_id == knowledge_base_id,
                DocumentVersion.chunk_revision > 0,
                DocumentVersion.indexed_chunk_revision
                == DocumentVersion.chunk_revision,
            )
            .limit(1)
        )
    ) is not None
