"""Real-MySQL integration coverage for Task 3B document indexing."""

import asyncio
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import func, select

from langley.business_time import utc_now
from langley.infrastructure.database import (
    create_database_engine,
    create_session_factory,
    dispose_database_engine,
)
from langley.infrastructure.local_file_storage import LocalFileStorage
from langley.infrastructure.models import (
    DocumentIndexJob,
    DocumentProcessingJob,
    DocumentVersion,
    KnowledgeBase,
    KnowledgeChunk,
    KnowledgeIndexJob,
    User,
)
from langley.knowledge.chunking import CandidateChunk, ChunkingConfig
from langley.knowledge.commands import (
    create_initial_document,
    create_initial_pdf_document,
    create_knowledge_base,
    rebuild_document_version_chunks,
)
from langley.knowledge.contracts import DocumentSourceRef, PdfPageRegion
from langley.knowledge.document_indexing import (
    DOCUMENT_INDEX_COLLECTION,
    DOCUMENT_INDEX_REPRESENTATION,
    DocumentIndexConfiguration,
    DocumentIndexRuntime,
    claim_next_document_index_job,
    reconcile_interrupted_document_index_jobs,
    seed_document_index_backlog,
)
from langley.knowledge.embedding_runtime import KnowledgeEmbeddingRuntime
from langley.knowledge.index_build import IndexBuildAdmissionError, admit_index_build
from langley.knowledge.pdf_processing import publish_pdf_processing_result
from langley.knowledge.pdf_processing_result import PdfProcessingResult
from langley.settings import Settings


@pytest.fixture
def migrated_database(test_database_url: str, reset_database) -> str:
    reset_database()
    config = Config("alembic.ini")
    config.cmd_opts = Namespace(x=["use_test_database=true"])
    command.upgrade(config, "head")
    return test_database_url


def _configuration() -> DocumentIndexConfiguration:
    return DocumentIndexConfiguration(
        model="controlled-model",
        revision="controlled-revision",
        dimension=2,
        representation=DOCUMENT_INDEX_REPRESENTATION,
        qdrant_url="http://qdrant.invalid",
    )


async def _create_factory(database_url: str):
    engine = create_database_engine(database_url)
    return engine, create_session_factory(engine)


async def _seed_user_and_fresh_base(factory) -> KnowledgeBase:
    async with factory() as session, session.begin():
        session.add(User(id=1, created_at=utc_now()))
    async with factory() as session:
        base = await create_knowledge_base(session, user_id=1, name="Task 3C")
    async with factory() as session:
        stored = await session.get(KnowledgeBase, base.id)
        assert stored is not None
        assert stored.index_status == "CHUNKED"
        assert (
            stored.active_embedding_model,
            stored.active_embedding_revision,
            stored.active_embedding_dimension,
            stored.active_embedding_representation,
        ) == (None, None, None, None)
    return base


async def _seed_user_and_base(factory) -> KnowledgeBase:
    base = await _seed_user_and_fresh_base(factory)
    async with factory() as session, session.begin():
        stored = await session.get(KnowledgeBase, base.id)
        assert stored is not None
        configuration = _configuration()
        stored.active_embedding_model = configuration.model
        stored.active_embedding_revision = configuration.revision
        stored.active_embedding_dimension = configuration.dimension
        stored.active_embedding_representation = configuration.representation
    return base


def test_markdown_publication_is_incremental_and_admission_is_revision_scoped(
    migrated_database: str, tmp_path: Path
) -> None:
    async def scenario() -> None:
        engine, factory = await _create_factory(migrated_database)
        try:
            base = await _seed_user_and_base(factory)
            storage = LocalFileStorage(tmp_path / "markdown")
            version = await create_initial_document(
                factory,
                storage,
                user_id=1,
                knowledge_base_id=base.id,
                name="Markdown",
                source_filename="doc.md",
                source_media_type="text/markdown",
                source_bytes=(
                    b"# Root\none two three four five six seven eight nine ten\n"
                ),
            )
            first = await rebuild_document_version_chunks(
                session_factory=factory,
                file_storage=storage,
                user_id=1,
                document_version_id=version.id,
                config=ChunkingConfig(max_chunk_chars=64),
                index_configuration=_configuration(),
            )
            assert first.index_job_created is True
            async with factory() as session, session.begin():
                stored = await session.get(DocumentVersion, version.id)
                knowledge_base = await session.get(KnowledgeBase, base.id)
                assert stored is not None and knowledge_base is not None
                original_revision = stored.chunk_revision
                original_hash = stored.chunk_set_sha256
                original_ids = tuple(
                    (
                        await session.scalars(
                            select(KnowledgeChunk.id)
                            .where(KnowledgeChunk.document_version_id == version.id)
                            .order_by(KnowledgeChunk.ordinal)
                        )
                    ).all()
                )

            same = await rebuild_document_version_chunks(
                session_factory=factory,
                file_storage=storage,
                user_id=1,
                document_version_id=version.id,
                config=ChunkingConfig(max_chunk_chars=64),
                index_configuration=_configuration(),
            )
            assert same.index_job_created is False
            async with factory() as session:
                stored = await session.get(DocumentVersion, version.id)
                knowledge_base = await session.get(KnowledgeBase, base.id)
                jobs = tuple(
                    (
                        await session.scalars(
                            select(DocumentIndexJob)
                            .where(DocumentIndexJob.document_version_id == version.id)
                            .order_by(DocumentIndexJob.attempt_no)
                        )
                    ).all()
                )
                ids = tuple(
                    (
                        await session.scalars(
                            select(KnowledgeChunk.id)
                            .where(KnowledgeChunk.document_version_id == version.id)
                            .order_by(KnowledgeChunk.ordinal)
                        )
                    ).all()
                )
                assert stored is not None and knowledge_base is not None
                assert (stored.chunk_revision, stored.chunk_set_sha256) == (
                    original_revision,
                    original_hash,
                )
                assert ids == original_ids
                assert knowledge_base.index_status == "CHUNKED"
                assert [(job.target_chunk_revision, job.status) for job in jobs] == [
                    (1, "PENDING")
                ]

            async with factory() as session, session.begin():
                job = await session.scalar(select(DocumentIndexJob))
                assert job is not None
                now = utc_now()
                job.status = "SUCCEEDED"
                job.stage = "VERIFYING"
                job.started_at = now
                job.finished_at = now
            behind = await rebuild_document_version_chunks(
                session_factory=factory,
                file_storage=storage,
                user_id=1,
                document_version_id=version.id,
                config=ChunkingConfig(max_chunk_chars=64),
                index_configuration=_configuration(),
            )
            assert behind.index_job_created is True

            changed = await rebuild_document_version_chunks(
                session_factory=factory,
                file_storage=storage,
                user_id=1,
                document_version_id=version.id,
                config=ChunkingConfig(max_chunk_chars=20),
                index_configuration=_configuration(),
            )
            assert changed.index_job_created is True
            async with factory() as session:
                stored = await session.get(DocumentVersion, version.id)
                jobs = tuple(
                    (
                        await session.scalars(
                            select(DocumentIndexJob)
                            .where(DocumentIndexJob.document_version_id == version.id)
                            .order_by(DocumentIndexJob.attempt_no)
                        )
                    ).all()
                )
                assert stored is not None
                assert stored.chunk_revision == 2
                assert stored.chunk_set_sha256 != original_hash
                assert [(job.target_chunk_revision, job.status) for job in jobs] == [
                    (1, "SUCCEEDED"),
                    (1, "PENDING"),
                    (2, "PENDING"),
                ]
        finally:
            await dispose_database_engine(engine)

    asyncio.run(scenario())


def test_pdf_publication_has_the_same_fingerprint_noop_semantics(
    migrated_database: str, tmp_path: Path
) -> None:
    async def scenario() -> None:
        engine, factory = await _create_factory(migrated_database)
        try:
            base = await _seed_user_and_base(factory)
            storage = LocalFileStorage(tmp_path / "pdf")
            admission = await create_initial_pdf_document(
                factory,
                storage,
                user_id=1,
                knowledge_base_id=base.id,
                name="PDF",
                source_filename="doc.pdf",
                source_media_type="application/pdf",
                source_bytes=b"%PDF controlled",
            )
            source_ref = DocumentSourceRef(
                document_version_id=admission.version.id,
                storage_key=admission.version.storage_key,
                source_media_type=admission.version.source_media_type,
                source_sha256=admission.version.source_sha256,
                source_size_bytes=admission.version.source_size_bytes,
            )
            async with factory() as session, session.begin():
                processing = await session.get(
                    DocumentProcessingJob, admission.processing.job_id
                )
                assert processing is not None
                processing.status = "RUNNING"
                processing.stage = "PUBLISHING"
                processing.started_at = utc_now()
            claim = SimpleNamespace(
                job_id=admission.processing.job_id,
                document_version_id=admission.version.id,
                knowledge_base_id=base.id,
                user_id=1,
                attempt_no=1,
                recipe_id="pdf_docling_hybrid512_v1",
                source_ref=source_ref,
            )
            result = PdfProcessingResult(
                page_count=1,
                candidates=(
                    CandidateChunk(
                        1,
                        "PDF body",
                        ("Heading",),
                        (PdfPageRegion(page_start=1, page_end=1),),
                    ),
                ),
            )
            assert await publish_pdf_processing_result(
                factory,
                claim=claim,
                result=result,
                index_configuration=_configuration(),
            )
            async with factory() as session, session.begin():
                version = await session.get(DocumentVersion, admission.version.id)
                knowledge_base = await session.get(KnowledgeBase, base.id)
                assert version is not None and knowledge_base is not None
                original_hash = version.chunk_set_sha256
                original_ids = tuple(
                    (
                        await session.scalars(
                            select(KnowledgeChunk.id).where(
                                KnowledgeChunk.document_version_id == version.id
                            )
                        )
                    ).all()
                )
                second = DocumentProcessingJob(
                    document_version_id=version.id,
                    attempt_no=2,
                    status="RUNNING",
                    stage="PUBLISHING",
                    recipe_id="pdf_docling_hybrid512_v1",
                    error_code=None,
                    error_message=None,
                    created_at=utc_now(),
                    started_at=utc_now(),
                    finished_at=None,
                )
                session.add(second)
                await session.flush()
                second_job_id = second.id
            second_claim = SimpleNamespace(
                **{**claim.__dict__, "job_id": second_job_id, "attempt_no": 2}
            )
            assert not await publish_pdf_processing_result(
                factory,
                claim=second_claim,
                result=result,
                index_configuration=_configuration(),
            )
            async with factory() as session:
                version = await session.get(DocumentVersion, admission.version.id)
                knowledge_base = await session.get(KnowledgeBase, base.id)
                ids = tuple(
                    (
                        await session.scalars(
                            select(KnowledgeChunk.id).where(
                                KnowledgeChunk.document_version_id
                                == admission.version.id
                            )
                        )
                    ).all()
                )
                assert version is not None and knowledge_base is not None
                assert version.chunk_revision == 1
                assert version.chunk_set_sha256 == original_hash
                assert ids == original_ids
                assert knowledge_base.index_status == "CHUNKED"
        finally:
            await dispose_database_engine(engine)

    asyncio.run(scenario())


class _Embedding:
    def encode_documents(self, contents: list[str], **_: object) -> list[list[float]]:
        assert contents
        return [[1.0, 0.0] for _ in contents]


class _FakeQdrant:
    def __init__(self) -> None:
        self.exists = False
        self.points: list[Any] = []
        self.upsert_count = 0

    async def collection_exists(self, collection_name: str) -> bool:
        assert collection_name == DOCUMENT_INDEX_COLLECTION
        return self.exists

    async def create_collection(self, collection_name: str, **_: object) -> None:
        assert collection_name == DOCUMENT_INDEX_COLLECTION
        self.exists = True

    async def delete(self, collection_name: str, **_: object) -> None:
        assert collection_name == DOCUMENT_INDEX_COLLECTION
        self.points.clear()

    async def upsert(
        self, collection_name: str, *, points: list[Any], **_: object
    ) -> None:
        assert collection_name == DOCUMENT_INDEX_COLLECTION
        self.points.extend(points)
        self.upsert_count += 1

    async def count(self, collection_name: str, **_: object) -> SimpleNamespace:
        assert collection_name == DOCUMENT_INDEX_COLLECTION
        return SimpleNamespace(count=len(self.points))

    async def close(self) -> None:
        return None


class _Runtime(DocumentIndexRuntime):
    def __init__(self, factory, qdrant: _FakeQdrant) -> None:
        super().__init__(
            factory,
            _configuration(),
            cast(KnowledgeEmbeddingRuntime, _Embedding()),
        )
        self.qdrant = qdrant

    async def _qdrant_client(self):
        return self.qdrant


def test_fresh_kb_first_index_initializes_contract_and_handles_revision_mismatch(
    migrated_database: str, tmp_path: Path
) -> None:
    async def scenario() -> None:
        engine, factory = await _create_factory(migrated_database)
        try:
            base = await _seed_user_and_fresh_base(factory)
            storage = LocalFileStorage(tmp_path / "runtime")
            version = await create_initial_document(
                factory,
                storage,
                user_id=1,
                knowledge_base_id=base.id,
                name="Runtime",
                source_filename="runtime.md",
                source_media_type="text/markdown",
                source_bytes=b"# Root\none two three four five six seven\n",
            )
            await rebuild_document_version_chunks(
                session_factory=factory,
                file_storage=storage,
                user_id=1,
                document_version_id=version.id,
                config=ChunkingConfig(max_chunk_chars=64),
                index_configuration=_configuration(),
            )
            claim = await claim_next_document_index_job(factory, _configuration())
            assert claim is not None
            qdrant = _FakeQdrant()
            runtime = _Runtime(factory, qdrant)
            await runtime.execute(claim)
            async with factory() as session:
                stored = await session.get(DocumentVersion, version.id)
                job = await session.get(DocumentIndexJob, claim.job_id)
                assert stored is not None and job is not None
                assert stored.indexed_chunk_revision == 1
                assert (job.status, job.stage) == ("SUCCEEDED", "VERIFYING")
                knowledge_base = await session.get(KnowledgeBase, base.id)
                assert knowledge_base is not None
                assert knowledge_base.index_status == "READY"
                configuration = _configuration()
                assert (
                    knowledge_base.active_embedding_model,
                    knowledge_base.active_embedding_revision,
                    knowledge_base.active_embedding_dimension,
                    knowledge_base.active_embedding_representation,
                ) == (
                    configuration.model,
                    configuration.revision,
                    configuration.dimension,
                    configuration.representation,
                )
            assert qdrant.upsert_count == 1

            already_indexed = await rebuild_document_version_chunks(
                session_factory=factory,
                file_storage=storage,
                user_id=1,
                document_version_id=version.id,
                config=ChunkingConfig(max_chunk_chars=64),
                index_configuration=_configuration(),
            )
            assert already_indexed.index_job_created is False
            async with factory() as session:
                assert (
                    await session.scalar(
                        select(func.count(DocumentIndexJob.id)).where(
                            DocumentIndexJob.document_version_id == version.id
                        )
                    )
                ) == 1

            await rebuild_document_version_chunks(
                session_factory=factory,
                file_storage=storage,
                user_id=1,
                document_version_id=version.id,
                config=ChunkingConfig(max_chunk_chars=20),
                index_configuration=_configuration(),
            )
            stale_claim = await claim_next_document_index_job(factory, _configuration())
            assert stale_claim is not None
            await rebuild_document_version_chunks(
                session_factory=factory,
                file_storage=storage,
                user_id=1,
                document_version_id=version.id,
                config=ChunkingConfig(max_chunk_chars=64),
                index_configuration=_configuration(),
            )
            upserts_before = qdrant.upsert_count
            await runtime.execute(stale_claim)
            async with factory() as session:
                stale_job = await session.get(DocumentIndexJob, stale_claim.job_id)
                pending = tuple(
                    (
                        await session.scalars(
                            select(DocumentIndexJob).where(
                                DocumentIndexJob.document_version_id == version.id,
                                DocumentIndexJob.status == "PENDING",
                            )
                        )
                    ).all()
                )
                assert stale_job is not None
                assert (stale_job.status, stale_job.error_code) == (
                    "FAILED",
                    "SOURCE_CHUNKS_CHANGED",
                )
                assert any(job.target_chunk_revision == 3 for job in pending)
            assert qdrant.upsert_count == upserts_before

            interrupted_claim = await claim_next_document_index_job(
                factory, _configuration()
            )
            assert interrupted_claim is not None
            repaired = await reconcile_interrupted_document_index_jobs(factory)
            assert interrupted_claim.job_id in repaired
            async with factory() as session:
                interrupted = await session.get(
                    DocumentIndexJob, interrupted_claim.job_id
                )
                assert interrupted is not None
                assert (interrupted.status, interrupted.error_code) == (
                    "INTERRUPTED",
                    "INDEX_INTERRUPTED",
                )
        finally:
            await dispose_database_engine(engine)

    asyncio.run(scenario())


def test_startup_backlog_stale_claim_and_full_build_admission_are_serialized(
    migrated_database: str, tmp_path: Path
) -> None:
    async def scenario() -> None:
        engine, factory = await _create_factory(migrated_database)
        try:
            base = await _seed_user_and_base(factory)
            storage = LocalFileStorage(tmp_path / "backlog")
            version = await create_initial_document(
                factory,
                storage,
                user_id=1,
                knowledge_base_id=base.id,
                name="Backlog",
                source_filename="backlog.md",
                source_media_type="text/markdown",
                source_bytes=b"# Backlog\ncurrent chunks\n",
            )
            await rebuild_document_version_chunks(
                session_factory=factory,
                file_storage=storage,
                user_id=1,
                document_version_id=version.id,
                config=ChunkingConfig(max_chunk_chars=64),
                index_configuration=_configuration(),
            )
            async with factory() as session, session.begin():
                job = await session.scalar(select(DocumentIndexJob))
                assert job is not None
                job.embedding_model = "old-incompatible-model"

            created = await seed_document_index_backlog(factory, _configuration())
            assert len(created) == 1

            async with factory() as session, session.begin():
                knowledge_base = await session.get(KnowledgeBase, base.id)
                assert knowledge_base is not None
                knowledge_base.index_status = "STALE"
            assert (
                await claim_next_document_index_job(factory, _configuration()) is None
            )

            async with factory() as session, session.begin():
                knowledge_base = await session.get(KnowledgeBase, base.id)
                assert knowledge_base is not None
                knowledge_base.index_status = "CHUNKED"
            settings = Settings(
                knowledge_embedding_model=_configuration().model,
                knowledge_embedding_revision=_configuration().revision,
                knowledge_embedding_dimension=_configuration().dimension,
                qdrant_url=_configuration().qdrant_url,
            )

            async def admit() -> tuple[str, int | None]:
                try:
                    admitted = await admit_index_build(
                        factory,
                        user_id=1,
                        knowledge_base_id=base.id,
                        settings=settings,
                    )
                except IndexBuildAdmissionError as error:
                    return error.code, None
                return "ADMITTED", admitted.job_id

            admission, claim = await asyncio.gather(
                admit(), claim_next_document_index_job(factory, _configuration())
            )
            async with factory() as session:
                document_jobs = tuple(
                    (
                        await session.scalars(
                            select(DocumentIndexJob).order_by(DocumentIndexJob.id)
                        )
                    ).all()
                )
                full_jobs = tuple(
                    (await session.scalars(select(KnowledgeIndexJob))).all()
                )
                knowledge_base = await session.get(KnowledgeBase, base.id)
                assert knowledge_base is not None
            if admission[0] == "ADMITTED":
                assert admission[1] is not None
                assert claim is None
                assert all(job.status == "INTERRUPTED" for job in document_jobs)
                assert all(
                    job.error_code == "INDEX_INTERRUPTED" for job in document_jobs
                )
                assert len(full_jobs) == 1
                assert full_jobs[0].status == "PENDING"
                assert knowledge_base.index_status == "INDEXING"
            else:
                assert admission == ("KNOWLEDGE_BASE_DOCUMENTS_INDEXING", None)
                assert claim is not None
                assert any(
                    job.id == claim.job_id and job.status == "RUNNING"
                    for job in document_jobs
                )
                assert full_jobs == ()
                assert knowledge_base.index_status == "CHUNKED"
            assert not (
                any(job.status == "RUNNING" for job in document_jobs)
                and bool(full_jobs)
            )
        finally:
            await dispose_database_engine(engine)

    asyncio.run(scenario())
