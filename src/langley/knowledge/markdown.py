"""Minimal CommonMark heading extraction over exact source bytes."""

from dataclasses import dataclass

from markdown_it import MarkdownIt

from langley.knowledge.contracts import TextSpanRegion


@dataclass(frozen=True)
class ParsedHeading:
    level: int
    text: str
    source_region: TextSpanRegion


@dataclass(frozen=True)
class ParsedDocument:
    source_bytes: bytes
    headings: tuple[ParsedHeading, ...]


def _line_start_byte_offsets(source_bytes: bytes) -> list[int]:
    """Return original-byte offsets for logical lines plus the EOF sentinel."""

    offsets = [0]
    index = 0
    source_length = len(source_bytes)
    while index < source_length:
        byte = source_bytes[index]
        if byte == ord("\r"):
            index += (
                2
                if index + 1 < source_length and source_bytes[index + 1] == ord("\n")
                else 1
            )
            offsets.append(index)
        elif byte == ord("\n"):
            index += 1
            offsets.append(index)
        else:
            index += 1
    if offsets[-1] != source_length:
        offsets.append(source_length)
    return offsets


def parse_markdown(source_bytes: bytes) -> ParsedDocument:
    """Extract document-level CommonMark headings with original UTF-8 byte spans."""

    source_text = source_bytes.decode("utf-8")
    line_offsets = _line_start_byte_offsets(source_bytes)
    tokens = MarkdownIt("commonmark").parse(source_text)
    headings: list[ParsedHeading] = []

    for index, token in enumerate(tokens):
        if token.type != "heading_open" or token.level != 0:
            continue
        if token.tag not in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            raise RuntimeError("top-level heading has an invalid tag")
        if token.map is None:
            raise RuntimeError("top-level heading has no source map")
        if index + 1 >= len(tokens) or tokens[index + 1].type != "inline":
            raise RuntimeError("top-level heading is not followed by inline content")

        start_line, end_line = token.map
        if not (0 <= start_line < end_line < len(line_offsets)):
            raise RuntimeError("top-level heading has an invalid source map")
        headings.append(
            ParsedHeading(
                level=int(token.tag[1:]),
                text=tokens[index + 1].content,
                source_region=TextSpanRegion(
                    start_byte=line_offsets[start_line],
                    end_byte=line_offsets[end_line],
                ),
            )
        )

    return ParsedDocument(source_bytes=source_bytes, headings=tuple(headings))
