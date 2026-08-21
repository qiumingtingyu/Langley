"""Minimal typed boundaries for Knowledge source persistence."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class StoredSource:
    """Immutable integrity facts returned after source finalization."""

    storage_key: str
    sha256: str
    size_bytes: int


class FileStorage(Protocol):
    """Store and read exact source bytes for the current Knowledge consumer."""

    async def store_source(self, user_id: int, source_bytes: bytes) -> StoredSource:
        """Finalize exact source bytes and return their detached storage facts."""

    async def read_source(self, storage_key: str) -> bytes:
        """Read the exact bytes stored at an opaque relative storage key."""


@dataclass(frozen=True)
class DocumentSourceRef:
    """Detached persisted source facts for a user-authorized DocumentVersion."""

    document_version_id: int
    storage_key: str
    source_media_type: str
    source_sha256: str
    source_size_bytes: int
