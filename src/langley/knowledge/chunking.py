"""Deterministic heading-aware chunk construction over parsed Markdown."""

from dataclasses import dataclass

from semantic_text_splitter import TextSplitter

from langley.knowledge.contracts import SourceRegion, TextSpanRegion
from langley.knowledge.markdown import ParsedDocument, ParsedHeading


@dataclass(frozen=True)
class ChunkingConfig:
    max_chunk_chars: int = 1200

    def __post_init__(self) -> None:
        if self.max_chunk_chars <= 0:
            raise ValueError("max_chunk_chars must be positive")


@dataclass(frozen=True)
class CandidateChunk:
    ordinal: int
    content: str
    heading_path: tuple[str, ...]
    source_regions: tuple[SourceRegion, ...]


def _validate_headings(headings: tuple[ParsedHeading, ...], source_length: int) -> None:
    previous_end = 0
    for heading in headings:
        region = heading.source_region
        if not 1 <= heading.level <= 6:
            raise ValueError("heading level must be between 1 and 6")
        if not (0 <= region.start_byte < region.end_byte <= source_length):
            raise ValueError("heading source region is out of bounds")
        if region.start_byte < previous_end:
            raise ValueError(
                "heading source regions must be ordered and non-overlapping"
            )
        previous_end = region.end_byte


def _sections(parsed: ParsedDocument):
    headings = parsed.headings
    source_length = len(parsed.source_bytes)
    if not headings:
        yield 0, source_length, ()
        return

    yield 0, headings[0].source_region.start_byte, ()
    slots: list[str | None] = [None] * 6
    for index, heading in enumerate(headings):
        level_index = heading.level - 1
        for slot_index in range(level_index, 6):
            slots[slot_index] = None
        if heading.text != "":
            slots[level_index] = heading.text
        section_end = (
            headings[index + 1].source_region.start_byte
            if index + 1 < len(headings)
            else source_length
        )
        yield (
            heading.source_region.end_byte,
            section_end,
            tuple(label for label in slots[: heading.level] if label is not None),
        )


def _direct_chunk(
    ordinal: int,
    content: str,
    heading_path: tuple[str, ...],
    start_byte: int,
    end_byte: int,
) -> CandidateChunk:
    return CandidateChunk(
        ordinal=ordinal,
        content=content,
        heading_path=heading_path,
        source_regions=(TextSpanRegion(start_byte, end_byte),),
    )


def _split_section(
    source_bytes: bytes,
    section_text: str,
    section_start_byte: int,
    section_end_byte: int,
    heading_path: tuple[str, ...],
    config: ChunkingConfig,
    first_ordinal: int,
) -> list[CandidateChunk]:
    splitter = TextSplitter(capacity=config.max_chunk_chars, overlap=0, trim=False)
    char_cursor = 0
    byte_cursor = section_start_byte
    candidates: list[CandidateChunk] = []
    for char_start, content in splitter.chunk_indices(section_text):
        if (
            char_start != char_cursor
            or content == ""
            or len(content) > config.max_chunk_chars
            or section_text[char_start : char_start + len(content)] != content
        ):
            raise RuntimeError("text splitter returned invalid chunks")
        content_bytes = content.encode("utf-8")
        end_byte = byte_cursor + len(content_bytes)
        if not (0 <= byte_cursor < end_byte <= len(source_bytes)) or (
            source_bytes[byte_cursor:end_byte] != content_bytes
        ):
            raise RuntimeError("text splitter chunk does not match source bytes")
        candidates.append(
            _direct_chunk(
                first_ordinal + len(candidates),
                content,
                heading_path,
                byte_cursor,
                end_byte,
            )
        )
        char_cursor += len(content)
        byte_cursor = end_byte
    if char_cursor != len(section_text) or byte_cursor != section_end_byte:
        raise RuntimeError("text splitter did not cover the accepted section")
    return candidates


def build_candidate_chunks(
    parsed: ParsedDocument, config: ChunkingConfig
) -> tuple[CandidateChunk, ...]:
    """Build lossless current candidates from parsed heading boundaries."""

    source_bytes = parsed.source_bytes
    _validate_headings(parsed.headings, len(source_bytes))
    candidates: list[CandidateChunk] = []
    for start_byte, end_byte, heading_path in _sections(parsed):
        section_text = source_bytes[start_byte:end_byte].decode(
            "utf-8", errors="strict"
        )
        if section_text == "" or section_text.isspace():
            continue
        ordinal = len(candidates) + 1
        if len(section_text) <= config.max_chunk_chars:
            candidates.append(
                _direct_chunk(ordinal, section_text, heading_path, start_byte, end_byte)
            )
        else:
            candidates.extend(
                _split_section(
                    source_bytes,
                    section_text,
                    start_byte,
                    end_byte,
                    heading_path,
                    config,
                    ordinal,
                )
            )
    return tuple(candidates)
