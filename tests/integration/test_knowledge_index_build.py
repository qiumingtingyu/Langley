"""Serial real-MySQL lifecycle coverage for the Task 5.1 manual index build."""

import asyncio
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from qdrant_client.http import models as qmodels
from sqlalchemy import select

from langley.bootstrap import bootstrap_local_user
from langley.business_time import utc_now
from langley.infrastructure.database import (
    create_database_engine,
    create_session_factory,
    dispose_database_engine,
)
from langley.infrastructure.models import (
    Document,
    DocumentVersion,
    KnowledgeBase,
    KnowledgeChunk,
    KnowledgeIndexJob,
    User,
)
from langley.knowledge.index_build import (
    KnowledgeIndexBuildRuntime,
    admit_index_build,
    reconcile_interrupted_index_builds,
    reconcile_stale_ready_index_configurations,
)
from langley.main import create_app
from langley.settings import Settings


@pytest.fixture
def migrated_database(test_database_url: str, reset_database) -> str:
    reset_database()
    config = Config("alembic.ini")
    config.cmd_opts = Namespace(x=["use_test_database=true"])
    command.upgrade(config, "head")
    return test_database_url


def _settings(database_url: str, storage_root: Path, user_id: int = 1) -> Settings:
    return Settings(
        environment="test",
        database_url=database_url,
        knowledge_storage_root=storage_root,
        local_user_id=user_id,
    )


async def _seed_chunked_knowledge_base(
    database_url: str, *, user_id: int = 1
) -> tuple[int, int]:
    engine = create_database_engine(database_url)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory() as session, session.begin():
            now = utc_now()
            for value in (1, 2):
                if await session.get(User, value) is None:
                    session.add(User(id=value, created_at=now))
            await session.flush()
            knowledge_base = KnowledgeBase(
                user_id=user_id,
                name="Index fixture",
                created_at=now,
            )
            session.add(knowledge_base)
            await session.flush()
            document = Document(
                knowledge_base_id=knowledge_base.id, name="fixture", created_at=now
            )
            session.add(document)
            await session.flush()
            version = DocumentVersion(
                document_id=document.id,
                source_filename="fixture.md",
                source_media_type="text/markdown",
                source_sha256="a" * 64,
                source_size_bytes=8,
                storage_key=f"fixture/{knowledge_base.id}.md",
                created_at=now,
            )
            session.add(version)
            await session.flush()
            session.add(
                KnowledgeChunk(
                    document_version_id=version.id,
                    ordinal=1,
                    content="Task 5.1 controlled chunk",
                    heading_path=[],
                    source_regions=[{"kind": "text", "start": 0, "end": 27}],
                    created_at=now,
                )
            )
            await session.flush()
            return knowledge_base.id, version.id
    finally:
        await dispose_database_engine(engine)


async def _read_build_state(
    database_url: str, job_id: int
) -> tuple[KnowledgeIndexJob, KnowledgeBase]:
    engine = create_database_engine(database_url)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory() as session:
            job = await session.get(KnowledgeIndexJob, job_id)
            assert job is not None
            knowledge_base = await session.get(KnowledgeBase, job.knowledge_base_id)
            assert knowledge_base is not None
            return job, knowledge_base
    finally:
        await dispose_database_engine(engine)


class _DeferredRuntime:
    """Records the post-202 schedule call without executing external work."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.scheduled: list[int] = []

    def schedule(self, job_id: int) -> None:
        self.scheduled.append(job_id)


class _FakeQdrant:
    def __init__(
        self,
        *,
        fail_upload: bool = False,
        mismatch: bool = False,
        existing_dimension: int | None = None,
        existing_distance: qmodels.Distance = qmodels.Distance.COSINE,
    ) -> None:
        self.fail_upload = fail_upload
        self.mismatch = mismatch
        self.collections: set[str] = (
            {"langley_knowledge_dense_v1"} if existing_dimension is not None else set()
        )
        self.vector_config = (
            None
            if existing_dimension is None
            else qmodels.VectorParams(
                size=existing_dimension, distance=existing_distance
            )
        )
        self.created_vectors_config: qmodels.VectorParams | None = None
        self.points: list[object] = []
        self.cleanup_failed = False
        self.close_count = 0

    async def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self.collections

    async def create_collection(self, collection_name: str, **kwargs: object) -> None:
        self.created_vectors_config = cast(
            qmodels.VectorParams, kwargs["vectors_config"]
        )
        self.vector_config = self.created_vectors_config
        self.collections.add(collection_name)

    async def get_collection(self, collection_name: str) -> SimpleNamespace:
        assert collection_name in self.collections
        assert self.vector_config is not None
        return SimpleNamespace(
            config=SimpleNamespace(params=SimpleNamespace(vectors=self.vector_config))
        )

    async def upsert(
        self, collection_name: str, *, points: list[Any], **kwargs: object
    ) -> None:
        del collection_name, kwargs
        if self.fail_upload:
            raise RuntimeError("controlled Qdrant write failure")
        self.points.extend(points)

    async def count(self, collection_name: str, **kwargs: object) -> SimpleNamespace:
        del collection_name
        generation_id = cast(Any, kwargs["count_filter"]).must[0].match.value
        count = sum(
            point.payload["generation_id"] == generation_id for point in self.points
        )
        return SimpleNamespace(count=count - 1 if self.mismatch else count)

    async def delete(self, collection_name: str, **kwargs: object) -> None:
        del collection_name, kwargs
        self.cleanup_failed = True
        raise RuntimeError("controlled cleanup failure")

    async def close(self) -> None:
        self.close_count += 1


class _FakeRuntime(KnowledgeIndexBuildRuntime):
    def __init__(
        self,
        *args: object,
        qdrant: _FakeQdrant,
        mutate_before_activation: bool = False,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._fake_qdrant = qdrant
        self._mutate_before_activation = mutate_before_activation

    def _encode_documents(
        self, contents: list[str], job: KnowledgeIndexJob
    ) -> list[list[float]]:
        return [[1.0] + [0.0] * (job.embedding_dimension - 1) for _ in contents]

    async def _qdrant_client(self) -> _FakeQdrant:
        return self._fake_qdrant

    async def _verify(self, job: KnowledgeIndexJob) -> None:
        await super()._verify(job)
        if self._mutate_before_activation:
            async with self._session_factory() as session, session.begin():
                chunk = await session.scalar(
                    select(KnowledgeChunk)
                    .join(DocumentVersion)
                    .join(Document)
                    .where(Document.knowledge_base_id == job.knowledge_base_id)
                )
                assert chunk is not None
                chunk.content = "changed after Qdrant verification"


def test_index_build_http_admission_is_owned_async_and_conflict_safe(
    migrated_database: str, tmp_path: Path
) -> None:
    settings = _settings(migrated_database, tmp_path / "knowledge")
    knowledge_base_id, _ = asyncio.run(_seed_chunked_knowledge_base(migrated_database))
    runtime = _DeferredRuntime(settings)
    app = create_app(settings, knowledge_index_runtime=runtime)  # type: ignore[arg-type]
    with TestClient(app) as client:
        accepted = client.post(f"/api/knowledge-bases/{knowledge_base_id}/index-build")
        assert accepted.status_code == 202
        assert runtime.scheduled == [accepted.json()["job_id"]]
        assert (
            client.get(f"/api/knowledge-bases/{knowledge_base_id}/index-status").json()[
                "index_status"
            ]
            == "INDEXING"
        )
        duplicate = client.post(f"/api/knowledge-bases/{knowledge_base_id}/index-build")
        assert duplicate.status_code == 409
        assert duplicate.json() == {"detail": {"code": "INDEX_BUILD_IN_PROGRESS"}}
        assert client.post("/api/knowledge-bases/999999/index-build").status_code == 404
    with TestClient(
        create_app(_settings(migrated_database, tmp_path / "other", 2))
    ) as other:
        assert (
            other.post(
                f"/api/knowledge-bases/{knowledge_base_id}/index-build"
            ).status_code
            == 404
        )


def test_index_build_rejects_a_chunkless_owned_knowledge_base(
    migrated_database: str, tmp_path: Path
) -> None:
    settings = _settings(migrated_database, tmp_path / "knowledge")
    assert asyncio.run(bootstrap_local_user(settings))
    runtime = _DeferredRuntime(settings)
    with TestClient(create_app(settings, knowledge_index_runtime=runtime)) as client:  # type: ignore[arg-type]
        knowledge_base_id = client.post(
            "/api/knowledge-bases", json={"name": "Empty"}
        ).json()["id"]
        response = client.post(f"/api/knowledge-bases/{knowledge_base_id}/index-build")
    assert response.status_code == 409
    assert response.json() == {"detail": {"code": "KNOWLEDGE_BASE_NOT_CHUNKED"}}
    assert runtime.scheduled == []


@pytest.mark.parametrize(
    ("qdrant", "mutate_before_activation", "error_code", "index_status"),
    [
        (_FakeQdrant(fail_upload=True), False, "INDEX_BUILD_FAILED", "FAILED"),
        (_FakeQdrant(mismatch=True), False, "INDEX_VERIFICATION_MISMATCH", "FAILED"),
        (
            _FakeQdrant(existing_dimension=384),
            False,
            "INDEX_COLLECTION_CONFIGURATION_MISMATCH",
            "FAILED",
        ),
        (
            _FakeQdrant(
                existing_dimension=1024,
                existing_distance=qmodels.Distance.EUCLID,
            ),
            False,
            "INDEX_COLLECTION_CONFIGURATION_MISMATCH",
            "FAILED",
        ),
        (_FakeQdrant(), True, "SOURCE_CHUNKS_CHANGED", "STALE"),
    ],
)
def test_index_build_failure_paths_never_activate(
    migrated_database: str,
    tmp_path: Path,
    qdrant: _FakeQdrant,
    mutate_before_activation: bool,
    error_code: str,
    index_status: str,
) -> None:
    settings = _settings(migrated_database, tmp_path / "knowledge")
    knowledge_base_id, _ = asyncio.run(_seed_chunked_knowledge_base(migrated_database))

    async def run() -> int:
        engine = create_database_engine(migrated_database)
        session_factory = create_session_factory(engine)
        try:
            admitted = await admit_index_build(
                session_factory,
                user_id=1,
                knowledge_base_id=knowledge_base_id,
                settings=settings,
            )
            await _FakeRuntime(
                session_factory,
                settings,
                qdrant=qdrant,
                mutate_before_activation=mutate_before_activation,
            ).run(admitted.job_id)
            return admitted.job_id
        finally:
            await dispose_database_engine(engine)

    job, knowledge_base = asyncio.run(
        _read_build_state(migrated_database, asyncio.run(run()))
    )
    assert job.status == "FAILED"
    assert job.error_code == error_code
    assert job.started_at is not None
    assert job.finished_at is not None
    assert knowledge_base.index_status == index_status
    assert knowledge_base.active_generation_id is None
    assert knowledge_base.building_generation_id is None
    assert qdrant.close_count >= 1


def test_success_activates_after_verify_and_cleanup_failure_is_best_effort(
    migrated_database: str, tmp_path: Path
) -> None:
    settings = _settings(migrated_database, tmp_path / "knowledge")
    knowledge_base_id, _ = asyncio.run(_seed_chunked_knowledge_base(migrated_database))
    qdrant = _FakeQdrant()

    async def run() -> int:
        engine = create_database_engine(migrated_database)
        session_factory = create_session_factory(engine)
        try:
            async with session_factory() as session, session.begin():
                knowledge_base = await session.get(KnowledgeBase, knowledge_base_id)
                assert knowledge_base is not None
                knowledge_base.active_generation_id = "old-generation"
                knowledge_base.index_status = "READY"
            admitted = await admit_index_build(
                session_factory,
                user_id=1,
                knowledge_base_id=knowledge_base_id,
                settings=settings,
            )
            await _FakeRuntime(session_factory, settings, qdrant=qdrant).run(
                admitted.job_id
            )
            return admitted.job_id
        finally:
            await dispose_database_engine(engine)

    job, knowledge_base = asyncio.run(
        _read_build_state(migrated_database, asyncio.run(run()))
    )
    assert job.status == "SUCCEEDED"
    assert job.stage == "ACTIVATING"
    assert job.processed_chunk_count == job.total_chunk_count == 1
    assert knowledge_base.index_status == "READY"
    assert knowledge_base.active_generation_id == job.generation_id
    assert knowledge_base.building_generation_id is None
    assert job.embedding_dimension == settings.knowledge_embedding_dimension
    assert job.embedding_representation == "content_only"
    assert (
        knowledge_base.active_embedding_dimension
        == settings.knowledge_embedding_dimension
    )
    assert knowledge_base.active_embedding_representation == "content_only"
    assert qdrant.cleanup_failed
    assert qdrant.close_count == 3
    assert qdrant.created_vectors_config is not None
    assert qdrant.created_vectors_config.size == settings.knowledge_embedding_dimension
    assert qdrant.created_vectors_config.distance == qmodels.Distance.COSINE
    assert qdrant.points[0].payload.keys() == {
        "knowledge_chunk_id",
        "knowledge_base_id",
        "document_version_id",
        "user_id",
        "generation_id",
    }


def test_unsupported_embedding_representation_fails_before_qdrant_write(
    migrated_database: str, tmp_path: Path
) -> None:
    settings = _settings(migrated_database, tmp_path / "knowledge").model_copy(
        update={"knowledge_embedding_representation": "heading_and_content"}
    )
    knowledge_base_id, _ = asyncio.run(_seed_chunked_knowledge_base(migrated_database))
    qdrant = _FakeQdrant()

    async def run() -> int:
        engine = create_database_engine(migrated_database)
        session_factory = create_session_factory(engine)
        try:
            admitted = await admit_index_build(
                session_factory,
                user_id=1,
                knowledge_base_id=knowledge_base_id,
                settings=settings,
            )
            await _FakeRuntime(session_factory, settings, qdrant=qdrant).run(
                admitted.job_id
            )
            return admitted.job_id
        finally:
            await dispose_database_engine(engine)

    job, knowledge_base = asyncio.run(
        _read_build_state(migrated_database, asyncio.run(run()))
    )
    assert job.status == "FAILED"
    assert job.error_code == "EMBEDDING_REPRESENTATION_UNSUPPORTED"
    assert knowledge_base.index_status == "FAILED"
    assert qdrant.points == []


def test_compatible_existing_collection_is_reused_and_closed(
    migrated_database: str, tmp_path: Path
) -> None:
    settings = _settings(migrated_database, tmp_path / "knowledge")
    knowledge_base_id, _ = asyncio.run(_seed_chunked_knowledge_base(migrated_database))
    qdrant = _FakeQdrant(existing_dimension=settings.knowledge_embedding_dimension)

    async def run() -> int:
        engine = create_database_engine(migrated_database)
        session_factory = create_session_factory(engine)
        try:
            admitted = await admit_index_build(
                session_factory,
                user_id=1,
                knowledge_base_id=knowledge_base_id,
                settings=settings,
            )
            await _FakeRuntime(session_factory, settings, qdrant=qdrant).run(
                admitted.job_id
            )
            return admitted.job_id
        finally:
            await dispose_database_engine(engine)

    job, knowledge_base = asyncio.run(
        _read_build_state(migrated_database, asyncio.run(run()))
    )
    assert job.status == "SUCCEEDED"
    assert knowledge_base.index_status == "READY"
    assert qdrant.created_vectors_config is None
    assert qdrant.close_count == 2


def test_restart_interrupts_orphaned_job_without_retry_or_activation(
    migrated_database: str, tmp_path: Path
) -> None:
    settings = _settings(migrated_database, tmp_path / "knowledge")
    knowledge_base_id, _ = asyncio.run(_seed_chunked_knowledge_base(migrated_database))

    async def interrupt() -> tuple[int, tuple[int, ...]]:
        engine = create_database_engine(migrated_database)
        session_factory = create_session_factory(engine)
        try:
            admitted = await admit_index_build(
                session_factory,
                user_id=1,
                knowledge_base_id=knowledge_base_id,
                settings=settings,
            )
            return admitted.job_id, await reconcile_interrupted_index_builds(
                session_factory
            )
        finally:
            await dispose_database_engine(engine)

    job_id, interrupted = asyncio.run(interrupt())
    job, knowledge_base = asyncio.run(_read_build_state(migrated_database, job_id))
    assert interrupted == (job_id,)
    assert job.status == "INTERRUPTED"
    assert job.error_code == "INDEX_BUILD_INTERRUPTED"
    assert knowledge_base.index_status == "FAILED"
    assert knowledge_base.active_generation_id is None
    assert knowledge_base.building_generation_id is None


@pytest.mark.parametrize(
    "update",
    [
        {"knowledge_embedding_model": "different-model"},
        {"knowledge_embedding_revision": "changed-revision"},
        {"knowledge_embedding_dimension": 768},
        {"knowledge_embedding_representation": "heading_and_content"},
    ],
)
def test_restart_stales_ready_index_when_embedding_configuration_changed(
    migrated_database: str, tmp_path: Path, update: dict[str, object]
) -> None:
    settings = _settings(migrated_database, tmp_path / "knowledge")
    knowledge_base_id, _ = asyncio.run(_seed_chunked_knowledge_base(migrated_database))
    qdrant = _FakeQdrant()

    async def run_and_reconcile() -> tuple[int, tuple[int, ...]]:
        engine = create_database_engine(migrated_database)
        session_factory = create_session_factory(engine)
        try:
            admitted = await admit_index_build(
                session_factory,
                user_id=1,
                knowledge_base_id=knowledge_base_id,
                settings=settings,
            )
            await _FakeRuntime(session_factory, settings, qdrant=qdrant).run(
                admitted.job_id
            )
            stale_ids = await reconcile_stale_ready_index_configurations(
                session_factory,
                settings=settings.model_copy(update=update),
            )
            return admitted.job_id, stale_ids
        finally:
            await dispose_database_engine(engine)

    job_id, stale_ids = asyncio.run(run_and_reconcile())
    job, knowledge_base = asyncio.run(_read_build_state(migrated_database, job_id))
    assert job.status == "SUCCEEDED"
    assert stale_ids == (knowledge_base_id,)
    assert knowledge_base.index_status == "STALE"
    assert knowledge_base.active_generation_id == job.generation_id
