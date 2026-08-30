"""Deterministic document-index facts shared by processing and indexing."""

import json
from hashlib import sha256
from typing import Protocol, Sequence

from langley.knowledge.contracts import (
    PdfPageRegion,
    TextSpanRegion,
    encode_source_region,
    validate_heading_path,
    validate_source_regions,
)

SOURCE_CONTEXT_V1 = "source_context_v1"


class ChunkSetFingerprintMember(Protocol):
    """Only semantic chunk facts participate in the chunk-set fingerprint."""

    @property
    def ordinal(self) -> int: ...

    @property
    def content(self) -> str: ...

    @property
    def heading_path(self) -> object: ...

    @property
    def source_regions(self) -> object: ...


def build_source_context_v1(content: str, heading_path: Sequence[str]) -> str:
    """Build the exact heading-aware retrieval text without normalizing content."""

    if not isinstance(content, str):
        raise ValueError("invalid chunk content")
    headings = _validated_heading_path(heading_path)
    if not headings:
        return content
    return "\n".join(headings) + "\n\n" + content


def chunk_set_sha256(chunks: Sequence[ChunkSetFingerprintMember]) -> str:
    """Hash ordered evidence, structure, and provenance as canonical UTF-8 JSON."""

    if not chunks:
        raise ValueError("chunk set must not be empty")

    members: list[dict[str, object]] = []
    ordinals: set[int] = set()
    for chunk in chunks:
        if (
            type(chunk.ordinal) is not int
            or chunk.ordinal <= 0
            or chunk.ordinal in ordinals
        ):
            raise ValueError("invalid chunk ordinal")
        if not isinstance(chunk.content, str) or not chunk.content:
            raise ValueError("invalid chunk content")
        ordinals.add(chunk.ordinal)
        headings = _validated_heading_path(chunk.heading_path)
        regions = _validated_source_regions(chunk.source_regions)
        members.append(
            {
                "ordinal": chunk.ordinal,
                "content": chunk.content,
                "heading_path": headings,
                "source_regions": regions,
            }
        )

    serialized = json.dumps(
        sorted(members, key=lambda member: int(member["ordinal"])),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(serialized).hexdigest()


def _validated_heading_path(value: object) -> list[str]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("invalid heading path")
    return validate_heading_path(list(value))


def _validated_source_regions(value: object) -> list[dict[str, object]]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("invalid source regions")
    encoded: list[dict[str, object]] = []
    for region in value:
        if isinstance(region, dict):
            encoded.append(region)
        elif isinstance(region, (TextSpanRegion, PdfPageRegion)):
            encoded.append(encode_source_region(region))
        else:
            raise ValueError("invalid source regions")
    validated = validate_source_regions(encoded)
    return [encode_source_region(region) for region in validated]
