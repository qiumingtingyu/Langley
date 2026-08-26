"""Real-MySQL preflight coverage for the private Formal B0 Observation runner."""

import asyncio
import json
from argparse import Namespace
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config

from eval.slice6 import run_formal_casebook_baseline as runner
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
    Memory,
    User,
)
from langley.settings import Settings

_MANIFEST = Path("eval/slice6/casebook/corpus_manifest_v1.json")
_GENERATION_ID = "11111111-1111-4111-8111-111111111111"


@pytest.fixture
def migrated_database(test_database_url: str, reset_database) -> str:
    reset_database()
    config = Config("alembic.ini")
    config.cmd_opts = Namespace(x=["use_test_database=true"])
    command.upgrade(config, "head")
    return test_database_url


def test_real_mysql_preflight_binds_exact_corpus_and_blocks_active_memory(
    migrated_database: str,
) -> None:
    async def verify() -> None:
        manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
        documents = manifest["documents"]
        assert isinstance(documents, list) and len(documents) == 40
        engine = create_database_engine(migrated_database)
        factory = create_session_factory(engine)
        now = utc_now()
        try:
            async with factory() as session, session.begin():
                session.add(User(id=77, created_at=now))
                await session.flush()
                knowledge_base = KnowledgeBase(
                    user_id=77,
                    name="Formal B0 Eval",
                    index_status="READY",
                    active_generation_id=_GENERATION_ID,
                    active_embedding_model="BAAI/bge-m3",
                    active_embedding_revision=(
                        "5617a9f61b028005a4858fdac845db406aefb181"
                    ),
                    active_embedding_dimension=1024,
                    active_embedding_representation="content_only",
                    active_chunk_snapshot_sha256="b" * 64,
                    created_at=now,
                )
                session.add(knowledge_base)
                await session.flush()
                for index, raw_document in enumerate(documents):
                    assert isinstance(raw_document, dict)
                    document = Document(
                        knowledge_base_id=knowledge_base.id,
                        name=f"runtime-{index}",
                        created_at=now,
                    )
                    session.add(document)
                    await session.flush()
                    version = DocumentVersion(
                        document_id=document.id,
                        source_filename=f"untrusted-name-{index}.md",
                        source_media_type="text/markdown",
                        source_sha256=str(raw_document["sha256"]),
                        source_size_bytes=1,
                        storage_key=f"eval/{index}/source",
                        chunk_max_chars=1200,
                        created_at=now,
                    )
                    session.add(version)
                    await session.flush()
                    session.add(
                        KnowledgeChunk(
                            document_version_id=version.id,
                            ordinal=1,
                            content=f"chunk-{index}",
                            heading_path=[],
                            source_regions=[
                                {"kind": "text_span", "start_byte": 0, "end_byte": 1}
                            ],
                            created_at=now,
                        )
                    )
                session.add(
                    KnowledgeIndexJob(
                        knowledge_base_id=knowledge_base.id,
                        generation_id=_GENERATION_ID,
                        status="SUCCEEDED",
                        stage="ACTIVATING",
                        processed_chunk_count=40,
                        total_chunk_count=40,
                        embedding_model="BAAI/bge-m3",
                        embedding_revision=("5617a9f61b028005a4858fdac845db406aefb181"),
                        embedding_dimension=1024,
                        embedding_representation="content_only",
                        chunk_snapshot_sha256="b" * 64,
                        error_code=None,
                        error_message=None,
                        created_at=now,
                        started_at=now,
                        finished_at=now,
                    )
                )
                await session.flush()
                knowledge_base_id = knowledge_base.id

            settings = Settings(
                environment="test",
                database_url=migrated_database,
                local_user_id=77,
                knowledge_embedding_model="BAAI/bge-m3",
                knowledge_embedding_revision=(
                    "5617a9f61b028005a4858fdac845db406aefb181"
                ),
                knowledge_embedding_dimension=1024,
                knowledge_embedding_representation="content_only",
            )
            result = await runner._load_runtime_preflight(
                factory,
                eval_user_id=77,
                knowledge_base_id=knowledge_base_id,
                manifest_documents=documents,
                settings=settings,
            )
            binding = result["runtime_corpus_binding"]
            assert binding["status"] == "VERIFIED"
            assert binding["eligible_runtime_document_count"] == 40
            assert len(binding["document_version_id_to_document_key"]) == 40
            assert binding["chunking_configuration"] == {
                "max_chunk_chars": 1200,
                "overlap": 0,
            }
            runtime_by_key = binding["document_key_to_runtime"]
            assert all(
                item["chunk_max_chars"] == 1200 for item in runtime_by_key.values()
            )
            first_version_id = next(iter(runtime_by_key.values()))[
                "document_version_id"
            ]

            async with factory() as session, session.begin():
                version = await session.get(DocumentVersion, first_version_id)
                assert version is not None
                version.chunk_max_chars = 1199
            with pytest.raises(runner.BaselineInputError, match="chunk_max_chars"):
                await runner._load_runtime_preflight(
                    factory,
                    eval_user_id=77,
                    knowledge_base_id=knowledge_base_id,
                    manifest_documents=documents,
                    settings=settings,
                )
            async with factory() as session, session.begin():
                version = await session.get(DocumentVersion, first_version_id)
                assert version is not None
                version.chunk_max_chars = 1200

            async with factory() as session, session.begin():
                session.add(
                    Memory(
                        user_id=77,
                        content="must block B0 without deletion",
                        source_message_id=None,
                        valid_until=None,
                        created_at=now,
                        updated_at=now,
                    )
                )
            with pytest.raises(runner.BaselineInputError, match="found 1"):
                await runner._load_runtime_preflight(
                    factory,
                    eval_user_id=77,
                    knowledge_base_id=knowledge_base_id,
                    manifest_documents=documents,
                    settings=settings,
                )
        finally:
            await dispose_database_engine(engine)

    asyncio.run(verify())
