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


@dataclass(frozen=True)
class PdfPageRegion:
    page_start: int
    page_end: int
    kind: str = "pdf_page"

    def __post_init__(self) -> None:
        if (
            self.kind != "pdf_page"
            or type(self.page_start) is not int
            or type(self.page_end) is not int
            or self.page_start < 1
            or self.page_end < self.page_start
        ):
            raise ValueError("invalid PDF page region")


SourceRegion: TypeAlias = TextSpanRegion | PdfPageRegion


def encode_source_region(region: SourceRegion) -> dict[str, object]:
    if isinstance(region, TextSpanRegion):
        return {
            "kind": region.kind,
            "start_byte": region.start_byte,
            "end_byte": region.end_byte,
        }
    return {
        "kind": region.kind,
        "page_start": region.page_start,
        "page_end": region.page_end,
    }


def decode_source_region(value: object) -> SourceRegion:
    if not isinstance(value, dict):
        raise ValueError("invalid source region")
    kind = value.get("kind")
    if kind == "text_span" and set(value) == {"kind", "start_byte", "end_byte"}:
        start, end = value["start_byte"], value["end_byte"]
        if type(start) is int and type(end) is int:
            return TextSpanRegion(start_byte=start, end_byte=end)
    if kind == "pdf_page" and set(value) == {"kind", "page_start", "page_end"}:
        start, end = value["page_start"], value["page_end"]
        if type(start) is int and type(end) is int:
            return PdfPageRegion(page_start=start, page_end=end)
    raise ValueError("invalid source region")


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
    region_type = type(regions[0])
    if any(type(region) is not region_type for region in regions[1:]):
        raise ValueError("source regions must be homogeneous")
    if region_type is TextSpanRegion:
        text_regions = [
            region for region in regions if isinstance(region, TextSpanRegion)
        ]
        if any(
            current.start_byte < previous.start_byte
            for previous, current in zip(text_regions, text_regions[1:])
        ):
            raise ValueError("source regions must be source-ordered")
    else:
        pdf_regions = [
            region for region in regions if isinstance(region, PdfPageRegion)
        ]
        if any(
            current.page_start <= previous.page_end
            for previous, current in zip(pdf_regions, pdf_regions[1:])
        ):
            raise ValueError("PDF page regions must be ordered and non-overlapping")
    return regions
