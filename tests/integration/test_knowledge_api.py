"""Real MySQL/filesystem HTTP coverage for the Task 1B Knowledge slice."""

import asyncio
from argparse import Namespace
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from langley.bootstrap import bootstrap_local_user
from langley.infrastructure.database import (
    create_database_engine,
    create_session_factory,
    dispose_database_engine,
)
from langley.infrastructure.models import Document, DocumentVersion
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


def _bootstrap(database_url: str, storage_root: Path, user_id: int) -> None:
    assert asyncio.run(
        bootstrap_local_user(_settings(database_url, storage_root, user_id))
    )


def test_knowledge_api_vertical_slice(migrated_database: str, tmp_path: Path) -> None:
    """The frozen endpoints preserve ownership, persistence, and integrity facts."""

    storage_root = tmp_path / "knowledge"
    _bootstrap(migrated_database, storage_root, 1)
    _bootstrap(migrated_database, storage_root, 2)
    app = create_app(_settings(migrated_database, storage_root))
    with TestClient(app) as client:
        invalid = client.post("/api/knowledge-bases", json={"name": "  "})
        extra = client.post("/api/knowledge-bases", json={"name": "No", "user_id": 2})
        first = client.post("/api/knowledge-bases", json={"name": "First"})
        second = client.post("/api/knowledge-bases", json={"name": "Second"})
        assert invalid.json() == {"detail": {"code": "VALIDATION_ERROR"}}
        assert extra.json() == {"detail": {"code": "VALIDATION_ERROR"}}
        assert first.status_code == second.status_code == 201
        first_id = first.json()["id"]
        listed = client.get("/api/knowledge-bases")
        assert [item["id"] for item in listed.json()] == [first_id, second.json()["id"]]

        empty = client.post(
            f"/api/knowledge-bases/{first_id}/documents",
            files={"file": ("empty.md", b"", "application/pdf")},
        )
        invalid_encoding = client.post(
            f"/api/knowledge-bases/{first_id}/documents",
            files={"file": ("bad.md", b"\xff", "text/plain")},
        )
        oversized = client.post(
            f"/api/knowledge-bases/{first_id}/documents",
            files={"file": ("large.md", b"x" * (5 * 1024 * 1024 + 1), "text/markdown")},
        )
        assert empty.json() == {"detail": {"code": "EMPTY_SOURCE"}}
        assert invalid_encoding.json() == {
            "detail": {"code": "INVALID_SOURCE_ENCODING"}
        }
        assert oversized.json() == {"detail": {"code": "UPLOAD_TOO_LARGE"}}
        too_long_name = client.post(
            f"/api/knowledge-bases/{first_id}/documents",
            data={"document_name": "n" * 256},
            files={"file": ("valid.md", b"# Valid\n", "text/markdown")},
        )
        too_long_filename = client.post(
            f"/api/knowledge-bases/{first_id}/documents",
            files={"file": ("f" * 253 + ".md", b"# Valid\n", "text/markdown")},
        )
        boundary = client.post(
            f"/api/knowledge-bases/{first_id}/documents",
            data={"document_name": "b" * 255},
            files={"file": ("b.md", b"# Boundary\n", "text/markdown")},
        )
        assert too_long_name.json() == {"detail": {"code": "VALIDATION_ERROR"}}
        assert too_long_filename.json() == {"detail": {"code": "VALIDATION_ERROR"}}
        assert boundary.status_code == 201

        uploaded = client.post(
            f"/api/knowledge-bases/{first_id}/documents",
            data={
                "document_name": "Readable title",
                "source_media_type": "application/pdf",
            },
            files={"file": ("same.md", b"# One\n", "application/pdf")},
        )
        assert uploaded.status_code == 201
        document = uploaded.json()
        assert document["name"] == "Readable title"
        assert document["source"]["media_type"] == "text/markdown"
        assert document["source"]["document_version_id"] > 0
        fallback = client.post(
            f"/api/knowledge-bases/{first_id}/documents",
            files={"file": ("same.md", b"# Two\n", "text/plain")},
        )
        assert fallback.status_code == 201
        assert fallback.json()["name"] == "same"
        documents = client.get(f"/api/knowledge-bases/{first_id}/documents")
        assert [item["id"] for item in documents.json()] == [
            fallback.json()["id"],
            document["id"],
            boundary.json()["id"],
        ]
        assert documents.json()[1]["source"]["sha256"] == document["source"]["sha256"]

        verified = client.post(
            f"/api/document-versions/{document['source']['document_version_id']}/verify-source"
        )
        assert verified.status_code == 200
        assert verified.json() | {"verified_at": "present"} == {
            "document_version_id": document["source"]["document_version_id"],
            "verified": True,
            "verified_at": "present",
        }

    async def inspect_and_tamper() -> None:
        engine = create_database_engine(migrated_database)
        try:
            factory = create_session_factory(engine)
            async with factory() as session:
                assert (
                    await session.scalar(select(func.count()).select_from(Document))
                ) == 3
                assert (
                    await session.scalar(
                        select(func.count()).select_from(DocumentVersion)
                    )
                ) == 3
                version = await session.get(
                    DocumentVersion, document["source"]["document_version_id"]
                )
                assert version is not None
                source_path = storage_root / version.storage_key
            source_path.write_bytes(b"# Tampered\n")
        finally:
            await dispose_database_engine(engine)

    asyncio.run(inspect_and_tamper())
    with TestClient(create_app(_settings(migrated_database, storage_root))) as client:
        tampered = client.post(
            f"/api/document-versions/{document['source']['document_version_id']}/verify-source"
        )
        assert tampered.status_code == 500
        assert tampered.json() == {"detail": {"code": "SOURCE_INTEGRITY_MISMATCH"}}
        with TestClient(
            create_app(_settings(migrated_database, storage_root, 2))
        ) as other:
            assert other.get(f"/api/knowledge-bases/{first_id}/documents").json() == {
                "detail": {"code": "KNOWLEDGE_BASE_NOT_FOUND"}
            }
            assert other.post(
                f"/api/document-versions/{document['source']['document_version_id']}/verify-source"
            ).json() == {"detail": {"code": "DOCUMENT_VERSION_NOT_FOUND"}}


def test_upload_size_and_filename_boundaries(
    migrated_database: str, tmp_path: Path
) -> None:
    storage_root = tmp_path / "knowledge"
    _bootstrap(migrated_database, storage_root, 1)
    with TestClient(create_app(_settings(migrated_database, storage_root))) as client:
        knowledge_base = client.post("/api/knowledge-bases", json={"name": "Bounds"})
        knowledge_base_id = knowledge_base.json()["id"]
        exact_limit = client.post(
            f"/api/knowledge-bases/{knowledge_base_id}/documents",
            files={"file": ("exact.md", b"x" * (5 * 1024 * 1024), "text/markdown")},
        )
        filename_255 = client.post(
            f"/api/knowledge-bases/{knowledge_base_id}/documents",
            files={"file": ("f" * 252 + ".md", b"# Filename\n", "text/markdown")},
        )
    assert exact_limit.status_code == 201
    assert filename_255.status_code == 201
