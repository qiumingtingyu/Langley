"""Deterministic tests for exact local Knowledge source persistence."""

import asyncio
from hashlib import sha256
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from langley.infrastructure.local_file_storage import (
    InvalidStorageKeyError,
    LocalFileStorage,
)
from langley.main import create_app
from langley.settings import Settings


def test_store_and_read_round_trip_exact_bytes(tmp_path: Path) -> None:
    storage = LocalFileStorage(tmp_path / "knowledge")
    source_bytes = b"# Heading\n\nExact bytes\n"

    stored = asyncio.run(storage.store_source(17, source_bytes))

    assert stored.sha256 == sha256(source_bytes).hexdigest()
    assert stored.size_bytes == len(source_bytes)
    assert stored.storage_key.startswith("users/17/sources/")
    assert asyncio.run(storage.read_source(stored.storage_key)) == source_bytes


def test_repeated_stores_use_distinct_generated_storage_keys(tmp_path: Path) -> None:
    storage = LocalFileStorage(tmp_path / "knowledge")

    first = asyncio.run(storage.store_source(1, b"same"))
    second = asyncio.run(storage.store_source(1, b"same"))

    assert first.storage_key != second.storage_key
    assert first.sha256 == second.sha256


@pytest.mark.parametrize(
    "storage_key",
    (
        "/etc/passwd",
        "../outside/source",
        "users/1/sources/not-a-uuid/source",
        "users/1/sources/00000000000000000000000000000000/not-source",
        "users\\1\\sources\\00000000000000000000000000000000\\source",
        "users/01/sources/00000000000000000000000000000000/source",
    ),
)
def test_read_rejects_malformed_or_escaping_storage_keys(
    tmp_path: Path, storage_key: str
) -> None:
    storage = LocalFileStorage(tmp_path / "knowledge")

    with pytest.raises(InvalidStorageKeyError):
        asyncio.run(storage.read_source(storage_key))


def test_write_failure_does_not_return_finalized_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = LocalFileStorage(tmp_path / "knowledge")

    def fail_replace(source: Path, destination: Path) -> None:
        del source, destination
        raise OSError("rename failed")

    monkeypatch.setattr(
        "langley.infrastructure.local_file_storage.os.replace", fail_replace
    )

    with pytest.raises(OSError, match="rename failed"):
        asyncio.run(storage.store_source(1, b"source"))

    assert list((tmp_path / "knowledge").rglob("source")) == []
    assert list((tmp_path / "knowledge").rglob("source.part")) == []


def test_partial_cleanup_removes_only_incomplete_generated_source(
    tmp_path: Path,
) -> None:
    root = tmp_path / "knowledge"
    incomplete_directory = root / "users" / "1" / "sources" / ("a" * 32)
    finalized_directory = root / "users" / "1" / "sources" / ("b" * 32)
    incomplete_directory.mkdir(parents=True)
    finalized_directory.mkdir(parents=True)
    (incomplete_directory / "source.part").write_bytes(b"partial")
    (finalized_directory / "source").write_bytes(b"final")
    (finalized_directory / "source.part").write_bytes(b"leftover")

    asyncio.run(LocalFileStorage(root).cleanup_partial_sources())
    asyncio.run(LocalFileStorage(root).cleanup_partial_sources())

    assert not incomplete_directory.exists()
    assert (finalized_directory / "source").read_bytes() == b"final"
    assert not (finalized_directory / "source.part").exists()


def test_finalized_source_scan_ignores_unexpected_paths(tmp_path: Path) -> None:
    root = tmp_path / "knowledge"
    valid_directory = root / "users" / "1" / "sources" / ("c" * 32)
    invalid_directory = root / "users" / "not-a-user" / "sources" / ("d" * 32)
    valid_directory.mkdir(parents=True)
    invalid_directory.mkdir(parents=True)
    (valid_directory / "source").write_bytes(b"final")
    (invalid_directory / "source").write_bytes(b"ignore")

    finalized_keys = asyncio.run(LocalFileStorage(root).list_finalized_source_keys())

    assert finalized_keys == frozenset({f"users/1/sources/{'c' * 32}/source"})


def test_maintenance_skips_generated_looking_source_directory_redirected_outside_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "knowledge"
    outside_directory = tmp_path / "outside"
    outside_directory.mkdir()
    outside_partial = outside_directory / "source.part"
    outside_final = outside_directory / "source"
    outside_partial.write_bytes(b"outside partial")
    outside_final.write_bytes(b"outside final")
    redirected_directory = root / "users" / "1" / "sources" / ("f" * 32)
    redirected_directory.parent.mkdir(parents=True)
    try:
        redirected_directory.symlink_to(outside_directory, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlink creation is unavailable: {error}")

    storage = LocalFileStorage(root)
    asyncio.run(storage.cleanup_partial_sources())
    finalized_keys = asyncio.run(storage.list_finalized_source_keys())

    assert outside_partial.read_bytes() == b"outside partial"
    assert outside_final.read_bytes() == b"outside final"
    assert finalized_keys == frozenset()


def test_finalized_source_scan_skips_redirected_source_file_outside_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "knowledge"
    source_directory = root / "users" / "1" / "sources" / ("0" * 32)
    source_directory.mkdir(parents=True)
    outside_source = tmp_path / "outside-source"
    outside_source.write_bytes(b"outside final")
    redirected_source = source_directory / "source"
    try:
        redirected_source.symlink_to(outside_source)
    except OSError as error:
        pytest.skip(f"symlink creation is unavailable: {error}")

    finalized_keys = asyncio.run(LocalFileStorage(root).list_finalized_source_keys())

    assert outside_source.read_bytes() == b"outside final"
    assert finalized_keys == frozenset()


def test_application_startup_cleans_incomplete_generated_sources(
    tmp_path: Path,
) -> None:
    root = tmp_path / "knowledge"
    partial_path = root / "users" / "1" / "sources" / ("e" * 32) / "source.part"
    partial_path.parent.mkdir(parents=True)
    partial_path.write_bytes(b"partial")

    with TestClient(create_app(Settings(knowledge_storage_root=root))):
        assert not partial_path.exists()


def test_application_startup_preserves_cleanup_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fail_cleanup(self: LocalFileStorage) -> None:
        del self
        raise OSError("cleanup failed")

    monkeypatch.setattr(LocalFileStorage, "cleanup_partial_sources", fail_cleanup)

    with pytest.raises(OSError, match="cleanup failed"):
        with TestClient(create_app(Settings(knowledge_storage_root=tmp_path))):
            pass
