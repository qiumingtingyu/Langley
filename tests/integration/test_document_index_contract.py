"""Real-MySQL migration and schema coverage for Task 3A."""

import asyncio
import json
from argparse import Namespace
from datetime import datetime

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, select, text
from sqlalchemy.exc import DBAPIError, IntegrityError

from langley.infrastructure.database import (
    create_database_engine,
    create_session_factory,
    dispose_database_engine,
)
from langley.infrastructure.models import DocumentIndexJob, DocumentVersion
from langley.knowledge.contracts import PdfPageRegion
from langley.knowledge.document_index_contract import (
    SOURCE_CONTEXT_V1,
    ChunkSetFingerprintMember,
    chunk_set_sha256,
)

_TIMESTAMP = datetime(2026, 8, 30, 1, 0)


class _ChunkFacts:
    def __init__(
        self,
        *,
        ordinal: int,
        content: str,
        heading_path: tuple[str, ...],
        source_regions: tuple[PdfPageRegion, ...],
    ) -> None:
        self.ordinal = ordinal
        self.content = content
        self.heading_path = heading_path
        self.source_regions = source_regions


def _alembic_config() -> Config:
    config = Config("alembic.ini")
    config.cmd_opts = Namespace(x=["use_test_database=true"])
    return config


@pytest.fixture
def migrated_contract_database(test_database_url: str, reset_database) -> str:
    """Seed the prior head, then exercise the real data-bearing upgrade."""

    reset_database()
    config = _alembic_config()
    command.upgrade(config, "0012_pdf_worker_errors")

    async def seed_prior_schema() -> None:
        engine = create_database_engine(test_database_url)
        try:
            async with engine.begin() as connection:
                await connection.execute(
                    text("INSERT INTO users (id, created_at) VALUES (1, :created_at)"),
                    {"created_at": _TIMESTAMP},
                )
                await connection.execute(
                    text(
                        "INSERT INTO knowledge_bases "
                        "(id, user_id, name, index_status, created_at) VALUES "
                        "(1, 1, 'Backfill', 'READY', :created_at)"
                    ),
                    {"created_at": _TIMESTAMP},
                )
                await connection.execute(
                    text(
                        "INSERT INTO documents "
                        "(id, knowledge_base_id, name, created_at) VALUES "
                        "(1, 1, 'Empty', :created_at), "
                        "(2, 1, 'Existing', :created_at)"
                    ),
                    {"created_at": _TIMESTAMP},
                )
                await connection.execute(
                    text(
                        "INSERT INTO document_versions "
                        "(id, document_id, source_filename, source_media_type, "
                        "source_sha256, source_size_bytes, storage_key, created_at) "
                        "VALUES "
                        "(1, 1, 'empty.pdf', 'application/pdf', :sha256, 1, "
                        "'users/1/sources/empty/source', :created_at), "
                        "(2, 2, 'existing.pdf', 'application/pdf', :sha256, 1, "
                        "'users/1/sources/existing/source', :created_at)"
                    ),
                    {"sha256": "a" * 64, "created_at": _TIMESTAMP},
                )
                await connection.execute(
                    text(
                        "INSERT INTO knowledge_chunks "
                        "(document_version_id, ordinal, content, heading_path, "
                        "source_regions, created_at) VALUES "
                        "(2, 2, :content_two, :heading_two, :regions_two, "
                        ":created_at), "
                        "(2, 1, :content_one, :heading_one, :regions_one, "
                        ":created_at)"
                    ),
                    {
                        "content_one": "  第一段\n",
                        "heading_one": json.dumps(["网络", "TCP"], ensure_ascii=False),
                        "regions_one": json.dumps(
                            [{"kind": "pdf_page", "page_start": 2, "page_end": 2}]
                        ),
                        "content_two": "TIME-WAIT ...",
                        "heading_two": json.dumps(
                            ["网络", "TCP", "连接关闭"], ensure_ascii=False
                        ),
                        "regions_two": json.dumps(
                            [
                                {
                                    "kind": "pdf_page",
                                    "page_start": 3,
                                    "page_end": 4,
                                },
                                {
                                    "kind": "pdf_page",
                                    "page_start": 6,
                                    "page_end": 6,
                                },
                            ]
                        ),
                        "created_at": _TIMESTAMP,
                    },
                )
        finally:
            await dispose_database_engine(engine)

    asyncio.run(seed_prior_schema())
    command.upgrade(config, "0013_document_index_contract")

    async def seed_pre_cutover_readiness() -> None:
        engine = create_database_engine(test_database_url)
        try:
            async with engine.begin() as connection:
                await connection.execute(
                    text(
                        "UPDATE document_versions SET indexed_chunk_revision = 1 "
                        "WHERE id = 2"
                    )
                )
                await connection.execute(
                    text(
                        "UPDATE knowledge_bases SET "
                        "active_generation_id = "
                        "'11111111-1111-4111-8111-111111111111', "
                        "active_embedding_model = 'BAAI/bge-m3', "
                        "active_embedding_revision = :revision, "
                        "active_embedding_dimension = 1024, "
                        "active_embedding_representation = 'content_only', "
                        "active_chunk_snapshot_sha256 = :snapshot WHERE id = 1"
                    ),
                    {"revision": "a" * 40, "snapshot": "b" * 64},
                )
        finally:
            await dispose_database_engine(engine)

    asyncio.run(seed_pre_cutover_readiness())
    command.upgrade(config, "head")
    return test_database_url


def test_migration_backfill_matches_production_fingerprint(
    migrated_contract_database: str,
) -> None:
    async def verify() -> None:
        engine = create_database_engine(migrated_contract_database)
        session_factory = create_session_factory(engine)
        try:
            async with session_factory() as session:
                versions = tuple(
                    (
                        await session.scalars(
                            select(DocumentVersion).order_by(DocumentVersion.id)
                        )
                    ).all()
                )
            assert versions[0].chunk_revision == 0
            assert versions[0].chunk_set_sha256 is None
            assert versions[0].indexed_chunk_revision is None

            expected_chunks: tuple[ChunkSetFingerprintMember, ...] = (
                _ChunkFacts(
                    ordinal=1,
                    content="  第一段\n",
                    heading_path=("网络", "TCP"),
                    source_regions=(PdfPageRegion(page_start=2, page_end=2),),
                ),
                _ChunkFacts(
                    ordinal=2,
                    content="TIME-WAIT ...",
                    heading_path=("网络", "TCP", "连接关闭"),
                    source_regions=(
                        PdfPageRegion(page_start=3, page_end=4),
                        PdfPageRegion(page_start=6, page_end=6),
                    ),
                ),
            )
            assert versions[1].chunk_revision == 1
            assert versions[1].chunk_set_sha256 == chunk_set_sha256(expected_chunks)
            assert versions[1].indexed_chunk_revision is None
        finally:
            await dispose_database_engine(engine)

    asyncio.run(verify())


def test_document_index_contract_migration_round_trip(
    migrated_contract_database: str,
) -> None:
    config = _alembic_config()

    command.downgrade(config, "0012_pdf_worker_errors")
    command.upgrade(config, "head")


def test_retrieval_cutover_removes_generation_schema(
    migrated_contract_database: str,
) -> None:
    async def verify() -> None:
        engine = create_database_engine(migrated_contract_database)
        try:
            async with engine.connect() as connection:
                base_columns = await connection.run_sync(
                    lambda sync: {
                        column["name"]
                        for column in inspect(sync).get_columns("knowledge_bases")
                    }
                )
                job_columns = await connection.run_sync(
                    lambda sync: {
                        column["name"]
                        for column in inspect(sync).get_columns("knowledge_index_jobs")
                    }
                )
            assert {
                "active_generation_id",
                "building_generation_id",
                "active_chunk_snapshot_sha256",
            }.isdisjoint(base_columns)
            assert "generation_id" not in job_columns
        finally:
            await dispose_database_engine(engine)

    asyncio.run(verify())


def test_positive_chunk_revision_requires_fingerprint(
    migrated_contract_database: str,
) -> None:
    async def verify() -> None:
        engine = create_database_engine(migrated_contract_database)
        try:
            async with engine.begin() as connection:
                with pytest.raises(DBAPIError):
                    await connection.execute(
                        text(
                            "UPDATE document_versions SET chunk_revision = 1, "
                            "chunk_set_sha256 = NULL WHERE id = 1"
                        )
                    )
        finally:
            await dispose_database_engine(engine)

    asyncio.run(verify())


def test_document_index_job_schema_accepts_one_valid_attempt_and_rejects_risks(
    migrated_contract_database: str,
) -> None:
    async def verify() -> None:
        engine = create_database_engine(migrated_contract_database)
        session_factory = create_session_factory(engine)
        try:
            async with session_factory() as session, session.begin():
                session.add(
                    DocumentIndexJob(
                        document_version_id=2,
                        attempt_no=1,
                        target_chunk_revision=1,
                        status="PENDING",
                        stage=None,
                        embedding_model="BAAI/bge-m3",
                        embedding_revision="5" * 40,
                        embedding_dimension=1024,
                        embedding_representation=SOURCE_CONTEXT_V1,
                        error_code=None,
                        error_message=None,
                        created_at=_TIMESTAMP,
                        started_at=None,
                        finished_at=None,
                    )
                )

            async with session_factory() as session, session.begin():
                session.add(
                    DocumentIndexJob(
                        document_version_id=2,
                        attempt_no=1,
                        target_chunk_revision=1,
                        status="PENDING",
                        stage=None,
                        embedding_model="BAAI/bge-m3",
                        embedding_revision="5" * 40,
                        embedding_dimension=1024,
                        embedding_representation=SOURCE_CONTEXT_V1,
                        error_code=None,
                        error_message=None,
                        created_at=_TIMESTAMP,
                        started_at=None,
                        finished_at=None,
                    )
                )
                with pytest.raises(IntegrityError):
                    await session.flush()

            async with session_factory() as session, session.begin():
                session.add(
                    DocumentIndexJob(
                        document_version_id=2,
                        attempt_no=2,
                        target_chunk_revision=1,
                        status="SUCCEEDED",
                        stage="PUBLISHING",
                        embedding_model="BAAI/bge-m3",
                        embedding_revision="5" * 40,
                        embedding_dimension=1024,
                        embedding_representation=SOURCE_CONTEXT_V1,
                        error_code=None,
                        error_message=None,
                        created_at=_TIMESTAMP,
                        started_at=_TIMESTAMP,
                        finished_at=_TIMESTAMP,
                    )
                )
                with pytest.raises(DBAPIError):
                    await session.flush()
        finally:
            await dispose_database_engine(engine)

    asyncio.run(verify())
