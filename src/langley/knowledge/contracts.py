"""Minimal typed boundaries for Knowledge source persistence."""

from dataclasses import dataclass
from typing import Protocol, TypeAlias


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


@dataclass(frozen=True)
class TextSpanRegion:
    start_byte: int
    end_byte: int
    kind: str = "text_span"

    def __post_init__(self) -> None:
        if (
            self.kind != "text_span"
            or self.start_byte < 0
            or self.end_byte <= self.start_byte
        ):
            raise ValueError("invalid text span region")


SourceRegion: TypeAlias = TextSpanRegion


def encode_source_region(region: SourceRegion) -> dict[str, object]:
    return {
        "kind": region.kind,
        "start_byte": region.start_byte,
        "end_byte": region.end_byte,
    }


def decode_source_region(value: object) -> SourceRegion:
    if not isinstance(value, dict) or set(value) != {"kind", "start_byte", "end_byte"}:
        raise ValueError("invalid source region")
    kind, start, end = value["kind"], value["start_byte"], value["end_byte"]
    if type(start) is not int or type(end) is not int or not isinstance(kind, str):
        raise ValueError("invalid source region")
    return TextSpanRegion(start_byte=start, end_byte=end, kind=kind)


def validate_heading_path(value: object) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError("invalid heading path")
    return value


def validate_source_regions(value: object) -> list[SourceRegion]:
    if not isinstance(value, list) or not value:
        raise ValueError("invalid source regions")
    regions = [decode_source_region(item) for item in value]
    if any(
        current.start_byte < previous.start_byte
        for previous, current in zip(regions, regions[1:])
    ):
        raise ValueError("source regions must be source-ordered")
    return regions
