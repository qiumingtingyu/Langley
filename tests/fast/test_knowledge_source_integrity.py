"""Deterministic integrity checks for detached Knowledge source references."""

import asyncio
from hashlib import sha256

import pytest

from langley.knowledge.commands import SourceIntegrityError, read_verified_source
from langley.knowledge.contracts import DocumentSourceRef, StoredSource


class _InMemoryFileStorage:
    def __init__(self, source_bytes: bytes | None) -> None:
        self._source_bytes = source_bytes

    async def store_source(self, user_id: int, source_bytes: bytes) -> StoredSource:
        del user_id, source_bytes
        raise AssertionError("store_source is not used by verified read")

    async def read_source(self, storage_key: str) -> bytes:
        del storage_key
        if self._source_bytes is None:
            raise FileNotFoundError
        return self._source_bytes


def _source_ref(
    source_bytes: bytes, *, size_bytes: int | None = None
) -> DocumentSourceRef:
    return DocumentSourceRef(
        document_version_id=1,
        storage_key="users/1/sources/00000000000000000000000000000000/source",
        source_media_type="text/markdown",
        source_sha256=sha256(source_bytes).hexdigest(),
        source_size_bytes=len(source_bytes) if size_bytes is None else size_bytes,
    )


def test_verified_read_returns_exact_matching_bytes() -> None:
    source_bytes = b"# Exact\n"

    assert (
        asyncio.run(
            read_verified_source(
                _InMemoryFileStorage(source_bytes), _source_ref(source_bytes)
            )
        )
        == source_bytes
    )


@pytest.mark.parametrize(
    ("stored_bytes", "source_ref"),
    (
        (None, _source_ref(b"# Exact\n")),
        (b"# truncated", _source_ref(b"# Exact\n")),
        (b"# changed!\n", _source_ref(b"# Exact\n")),
        (b"# Exact\n", _source_ref(b"# Exact\n", size_bytes=99)),
    ),
)
def test_verified_read_rejects_missing_or_corrupted_source(
    stored_bytes: bytes | None, source_ref: DocumentSourceRef
) -> None:
    with pytest.raises(SourceIntegrityError):
        asyncio.run(
            read_verified_source(_InMemoryFileStorage(stored_bytes), source_ref)
        )
