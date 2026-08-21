"""Real MySQL persistence coverage for current KnowledgeChunk facts."""

import asyncio
from argparse import Namespace
from datetime import datetime

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import select
from sqlalchemy.exc import DBAPIError, IntegrityError

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
    User,
)

_TIME = datetime(2026, 8, 21)


@pytest.fixture
def migrated_database(test_database_url: str, reset_database) -> str:
    reset_database()
    config = Config("alembic.ini")
    config.cmd_opts = Namespace(x=["use_test_database=true"])
    command.upgrade(config, "head")
    return test_database_url


def test_knowledge_chunk_constraints_ordering_and_restrict(
    migrated_database: str,
) -> None:
    async def verify() -> None:
        engine = create_database_engine(migrated_database)
        try:
            factory = create_session_factory(engine)
            async with factory() as session:
                async with session.begin():
                    session.add(User(id=1, created_at=_TIME))
                    await session.flush()
                    base = KnowledgeBase(user_id=1, name="Base", created_at=_TIME)
                    session.add(base)
                    await session.flush()
                    document = Document(
                        knowledge_base_id=base.id, name="Doc", created_at=_TIME
                    )
                    session.add(document)
                    await session.flush()
                    version = DocumentVersion(
                        document_id=document.id,
                        source_filename="a.md",
                        source_media_type="text/markdown",
                        source_sha256="a" * 64,
                        source_size_bytes=100,
                        storage_key="users/1/sources/00000000000000000000000000000000/source",
                        created_at=_TIME,
                    )
                    session.add(version)
                    await session.flush()
                    version_id = version.id
                    other_version = DocumentVersion(
                        document_id=document.id,
                        source_filename="b.md",
                        source_media_type="text/markdown",
                        source_sha256="b" * 64,
                        source_size_bytes=1,
                        storage_key="users/1/sources/11111111111111111111111111111111/source",
                        created_at=_TIME,
                    )
                    session.add(other_version)
                    await session.flush()
                    other_version_id = other_version.id
            async with factory() as session:
                async with session.begin():
                    for ordinal in (3, 1, 2):
                        content = "exact content" if ordinal == 1 else str(ordinal)
                        heading_path = ["根", "叶"] if ordinal == 1 else []
                        source_regions = (
                            [
                                {
                                    "kind": "text_span",
                                    "start_byte": 12,
                                    "end_byte": 25,
                                }
                            ]
                            if ordinal == 1
                            else [
                                {
                                    "kind": "text_span",
                                    "start_byte": 0,
                                    "end_byte": 1,
                                }
                            ]
                        )
                        session.add(
                            KnowledgeChunk(
                                document_version_id=version_id,
                                ordinal=ordinal,
                                content=content,
                                heading_path=heading_path,
                                source_regions=source_regions,
                                created_at=_TIME,
                            )
                        )
                    session.add(
                        KnowledgeChunk(
                            document_version_id=other_version_id,
                            ordinal=1,
                            content="other version",
                            heading_path=[],
                            source_regions=[
                                {"kind": "text_span", "start_byte": 0, "end_byte": 1}
                            ],
                            created_at=_TIME,
                        )
                    )
            async with factory() as session:
                rows = (
                    await session.scalars(
                        select(KnowledgeChunk)
                        .where(KnowledgeChunk.document_version_id == version_id)
                        .order_by(KnowledgeChunk.ordinal)
                    )
                ).all()
                assert [row.ordinal for row in rows] == [1, 2, 3]
                exact_row = rows[0]
                assert exact_row.document_version_id == version_id
                assert exact_row.ordinal == 1
                assert exact_row.content == "exact content"
                assert exact_row.heading_path == ["根", "叶"]
                assert exact_row.source_regions == [
                    {"kind": "text_span", "start_byte": 12, "end_byte": 25}
                ]
                assert exact_row.created_at == _TIME
                other_row = await session.scalar(
                    select(KnowledgeChunk).where(
                        KnowledgeChunk.document_version_id == other_version_id,
                        KnowledgeChunk.ordinal == 1,
                    )
                )
                assert other_row is not None
                assert other_row.document_version_id == other_version_id
                assert other_row.ordinal == 1
            async with factory() as session:
                async with session.begin():
                    session.add(
                        KnowledgeChunk(
                            document_version_id=version_id,
                            ordinal=1,
                            content="duplicate",
                            heading_path=[],
                            source_regions=[
                                {"kind": "text_span", "start_byte": 0, "end_byte": 1}
                            ],
                            created_at=_TIME,
                        )
                    )
                    with pytest.raises(IntegrityError):
                        await session.flush()
        finally:
            await dispose_database_engine(engine)

    asyncio.run(verify())


def test_knowledge_chunk_database_constraints(migrated_database: str) -> None:
    async def verify() -> None:
        engine = create_database_engine(migrated_database)
        try:
            factory = create_session_factory(engine)
            async with factory() as session:
                async with session.begin():
                    session.add(User(id=1, created_at=_TIME))
                    await session.flush()
                    base = KnowledgeBase(user_id=1, name="Base", created_at=_TIME)
                    session.add(base)
                    await session.flush()
                    document = Document(
                        knowledge_base_id=base.id, name="Doc", created_at=_TIME
                    )
                    session.add(document)
                    await session.flush()
                    version = DocumentVersion(
                        document_id=document.id,
                        source_filename="a.md",
                        source_media_type="text/markdown",
                        source_sha256="a" * 64,
                        source_size_bytes=1,
                        storage_key="users/1/sources/00000000000000000000000000000000/source",
                        created_at=_TIME,
                    )
                    session.add(version)
                    await session.flush()
                    version_id = version.id
            valid = {
                "document_version_id": version_id,
                "ordinal": 1,
                "content": "x",
                "heading_path": [],
                "source_regions": [
                    {"kind": "text_span", "start_byte": 0, "end_byte": 1}
                ],
                "created_at": _TIME,
            }
            for invalid in (
                {**valid, "ordinal": 0},
                {**valid, "content": ""},
                {**valid, "heading_path": {}},
                {**valid, "source_regions": {}},
                {**valid, "source_regions": []},
                {**valid, "document_version_id": 999999},
            ):
                async with factory() as session:
                    async with session.begin():
                        session.add(KnowledgeChunk(**invalid))
                        with pytest.raises(DBAPIError):
                            await session.flush()
            async with factory() as session:
                async with session.begin():
                    session.add(KnowledgeChunk(**valid))
            async with factory() as session:
                async with session.begin():
                    version = await session.get(DocumentVersion, version_id)
                    assert version is not None
                    await session.delete(version)
                    with pytest.raises(IntegrityError):
                        await session.flush()
        finally:
            await dispose_database_engine(engine)

    asyncio.run(verify())
