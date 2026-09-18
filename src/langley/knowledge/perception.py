"""Scoped structural discovery and bounded direct Knowledge observation."""

import asyncio
from collections import Counter
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from langley.infrastructure.local_file_storage import InvalidStorageKeyError
from langley.knowledge.commands import SourceIntegrityError, read_verified_source
from langley.knowledge.contracts import (
    FileStorage,
    KnowledgeLocator,
    KnowledgePerceptionError,
    KnowledgeReadResult,
    PdfPageRegion,
    TextSpanRegion,
)
from langley.knowledge.markdown import ParsedDocument, parse_markdown
from langley.knowledge.reads import (
    PerceptionDocumentRead,
    list_documents_for_knowledge_base,
    read_perception_document,
)


def validate_locator(locator: KnowledgeLocator) -> None:
    """Reject semantic combinations without normalizing or guessing a location."""
    heading = locator.heading_path
    start, end = locator.page_start, locator.page_end
    valid = (
        (
            locator.kind == "document"
            and heading is None
            and start is None
            and end is None
        )
        or (
            locator.kind == "heading"
            and heading is not None
            and bool(heading)
            and all(bool(part.strip()) for part in heading)
            and start is None
            and end is None
        )
        or (
            locator.kind == "pages"
            and heading is None
            and start is not None
            and end is not None
            and 1 <= start <= end
        )
    )
    if not valid:
        raise KnowledgePerceptionError("INVALID_LOCATOR", "Invalid location fields.")


@dataclass(frozen=True)
class _HeadingLocation:
    label: str
    path: tuple[str, ...]
    start: int
    end: int
    addressable: bool


def _heading_locations(parsed: ParsedDocument) -> tuple[_HeadingLocation, ...]:
    stack: list[tuple[int, str]] = []
    locations: list[_HeadingLocation] = []
    # Close subtrees in source order, including nested headings and heading bytes.
    ends = [len(parsed.source_bytes)] * len(parsed.headings)
    open_headings: list[int] = []
    for index, heading in enumerate(parsed.headings):
        while (
            open_headings and parsed.headings[open_headings[-1]].level >= heading.level
        ):
            ends[open_headings.pop()] = heading.source_region.start_byte
        open_headings.append(index)
    for index, heading in enumerate(parsed.headings):
        while stack and stack[-1][0] >= heading.level:
            stack.pop()
        stack.append((heading.level, heading.text))
        path = tuple(label for _, label in stack)
        locations.append(
            _HeadingLocation(
                heading.text,
                path,
                heading.source_region.start_byte,
                ends[index],
                all(bool(label.strip()) for label in path),
            )
        )
    return tuple(locations)


async def _markdown_source(
    storage: FileStorage, document: PerceptionDocumentRead
) -> ParsedDocument:
    try:
        source = await read_verified_source(storage, document.source)
        return await asyncio.to_thread(parse_markdown, source)
    except (
        SourceIntegrityError,
        InvalidStorageKeyError,
        OSError,
        UnicodeDecodeError,
    ) as error:
        raise KnowledgePerceptionError(
            "KNOWLEDGE_READ_UNAVAILABLE", "Verified Markdown source is unavailable."
        ) from error


def _public_locator(locator: KnowledgeLocator) -> dict[str, object]:
    return locator.model_dump(exclude_none=True)


async def inspect_knowledge(
    session_factory: async_sessionmaker[AsyncSession],
    storage: FileStorage,
    *,
    user_id: int,
    knowledge_base_id: int,
    document_id: int | None,
) -> dict[str, object]:
    if document_id is None:
        async with session_factory() as session, session.begin():
            documents = await list_documents_for_knowledge_base(
                session, user_id=user_id, knowledge_base_id=knowledge_base_id
            )
        if documents is None:
            raise KnowledgePerceptionError(
                "DOCUMENT_NOT_AVAILABLE", "Knowledge base is unavailable."
            )
        counts = Counter(doc.id for doc in documents)
        entries = []
        seen: set[int] = set()
        for doc in documents:
            if doc.id in seen:
                continue
            seen.add(doc.id)
            # Temporary ambiguity policy, not a new current-version selector.
            if counts[doc.id] != 1:
                entries.append(
                    {
                        "label": doc.name,
                        "available": False,
                        "reason": "ambiguous_version",
                    }
                )
                continue
            media = doc.source.media_type
            available = media in {"text/markdown", "application/pdf"}
            entries.append(
                {
                    "label": doc.name,
                    "format": {
                        "text/markdown": "markdown",
                        "application/pdf": "pdf",
                    }.get(media, "unsupported"),
                    "available": available,
                    **(
                        {
                            "locator": _public_locator(
                                KnowledgeLocator(document_id=doc.id, kind="document")
                            )
                        }
                        if available
                        else {}
                    ),
                }
            )
        return {"documents": entries}
    document = await read_perception_document(
        session_factory,
        user_id=user_id,
        knowledge_base_id=knowledge_base_id,
        document_id=document_id,
        max_content_bytes=0,
    )
    base: dict[str, object] = {
        "label": document.name,
        "locator": _public_locator(
            KnowledgeLocator(document_id=document_id, kind="document")
        ),
    }
    if document.source.source_media_type == "text/markdown":
        parsed = await _markdown_source(storage, document)
        locations = _heading_locations(parsed)
        path_counts = Counter(location.path for location in locations)
        headings = []
        for location in locations:
            readable = location.addressable and path_counts[location.path] == 1
            headings.append(
                {
                    "label": location.label,
                    "heading_path": list(location.path),
                    "readable": readable,
                    **(
                        {
                            "locator": _public_locator(
                                KnowledgeLocator(
                                    document_id=document_id,
                                    kind="heading",
                                    heading_path=list(location.path),
                                )
                            )
                        }
                        if readable
                        else {"reason": "empty_or_ambiguous_heading"}
                    ),
                }
            )
        return {**base, "format": "markdown", "headings": headings}
    if document.source.source_media_type == "application/pdf":
        # Never combine consecutive chunks with the same display heading.
        return {
            **base,
            "format": "pdf",
            "detected_regions": [
                {
                    "detected_heading_labels": list(chunk.heading_path),
                    "pages": {"start": region.page_start, "end": region.page_end},
                    "locator": _public_locator(
                        KnowledgeLocator(
                            document_id=document_id,
                            kind="pages",
                            page_start=region.page_start,
                            page_end=region.page_end,
                        )
                    ),
                }
                for chunk in document.pdf_chunks
                for region in chunk.source_regions
            ],
        }
    raise KnowledgePerceptionError(
        "KNOWLEDGE_READ_UNAVAILABLE", "Unsupported document format."
    )


def _pdf_coverage(document: PerceptionDocumentRead) -> tuple[PdfPageRegion, ...]:
    ordered = sorted(
        (region for chunk in document.pdf_chunks for region in chunk.source_regions),
        key=lambda region: (region.page_start, region.page_end),
    )
    merged: list[PdfPageRegion] = []
    for region in ordered:
        if merged and region.page_start <= merged[-1].page_end + 1:
            previous = merged.pop()
            merged.append(
                PdfPageRegion(
                    previous.page_start, max(previous.page_end, region.page_end)
                )
            )
        else:
            merged.append(region)
    return tuple(merged)


async def read_knowledge(
    session_factory: async_sessionmaker[AsyncSession],
    storage: FileStorage,
    *,
    user_id: int,
    knowledge_base_id: int,
    locator: KnowledgeLocator,
    max_content_bytes: int,
) -> KnowledgeReadResult:
    validate_locator(locator)
    document = await read_perception_document(
        session_factory,
        user_id=user_id,
        knowledge_base_id=knowledge_base_id,
        document_id=locator.document_id,
        locator=locator,
        max_content_bytes=max_content_bytes,
    )
    heading_path: tuple[str, ...] = ()
    regions: tuple[TextSpanRegion | PdfPageRegion, ...]
    if document.source.source_media_type == "text/markdown":
        if locator.kind == "pages":
            raise KnowledgePerceptionError(
                "INVALID_LOCATOR", "Markdown does not support pages."
            )
        parsed = await _markdown_source(storage, document)
        start, end = 0, len(parsed.source_bytes)
        if locator.kind == "heading":
            matches = [
                item
                for item in _heading_locations(parsed)
                if item.addressable and list(item.path) == locator.heading_path
            ]
            if len(matches) != 1:
                raise KnowledgePerceptionError(
                    "LOCATION_NOT_FOUND",
                    "Heading path is missing or ambiguous; inspect again.",
                )
            location = matches[0]
            start, end, heading_path = location.start, location.end, location.path
        content = parsed.source_bytes[start:end].decode("utf-8")
        regions = (TextSpanRegion(start, end),) if end > start else ()
    elif document.source.source_media_type == "application/pdf":
        if locator.kind == "heading":
            raise KnowledgePerceptionError(
                "INVALID_LOCATOR", "PDF v0 does not support headings."
            )
        if any(chunk.content is None for chunk in document.pdf_chunks):
            raise RuntimeError("PDF read is missing selected content")
        content = "\n\n".join(
            chunk.content for chunk in document.pdf_chunks if chunk.content is not None
        )
        regions = _pdf_coverage(document)
        # Detected labels are never PDF heading locator or durable citation authority.
        heading_path = ()
    else:
        raise KnowledgePerceptionError(
            "KNOWLEDGE_READ_UNAVAILABLE", "Unsupported document format."
        )
    if len(content.encode("utf-8")) > max_content_bytes:
        raise KnowledgePerceptionError(
            "KNOWLEDGE_READ_SCOPE_TOO_LARGE",
            "Use inspect_knowledge for a narrower location or search_knowledge.",
        )
    if not content.strip() or not regions:
        raise KnowledgePerceptionError(
            "KNOWLEDGE_READ_UNAVAILABLE", "No readable content is available."
        )
    return KnowledgeReadResult(
        locator=locator,
        document_version_id=document.source.document_version_id,
        content=content,
        source_display_name=document.name,
        source_sha256=document.source.source_sha256,
        heading_path=heading_path,
        source_regions=regions,
    )
