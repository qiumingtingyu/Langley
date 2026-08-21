"""Focused Task 2.2 CommonMark heading extraction contracts."""

import pytest

from langley.knowledge.markdown import ParsedHeading, parse_markdown


def _span(source_bytes: bytes, heading: ParsedHeading) -> bytes:
    return source_bytes[
        heading.source_region.start_byte : heading.source_region.end_byte
    ]


def test_atx_levels_source_order_and_skipped_levels() -> None:
    source = b"# A\n### C\n## B\n#### D\n##### E\n###### F\n"

    document = parse_markdown(source)

    assert [(heading.level, heading.text) for heading in document.headings] == [
        (1, "A"),
        (3, "C"),
        (2, "B"),
        (4, "D"),
        (5, "E"),
        (6, "F"),
    ]
    assert [
        _span(source, heading) for heading in document.headings
    ] == source.splitlines(keepends=True)


def test_setext_headings_cover_complete_blocks() -> None:
    source = b"Process\n=======\nThread\n------\n"

    document = parse_markdown(source)

    assert [(heading.level, heading.text) for heading in document.headings] == [
        (1, "Process"),
        (2, "Thread"),
    ]
    assert [_span(source, heading) for heading in document.headings] == [
        b"Process\n=======\n",
        b"Thread\n------\n",
    ]


def test_fenced_and_blockquote_headings_are_not_document_level() -> None:
    source = b"# Real\n```python\n# fake\n```\n> ## quoted\n"

    document = parse_markdown(source)

    assert [(heading.level, heading.text) for heading in document.headings] == [
        (1, "Real")
    ]


@pytest.mark.parametrize("source", [b"", b"plain text\n\n- list item\n"])
def test_empty_and_heading_free_documents_are_allowed(source: bytes) -> None:
    assert parse_markdown(source).headings == ()


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (b"## Process\n", b"## Process\n"),
        (b"## Process\r\n", b"## Process\r\n"),
        (b"## Process\r", b"## Process\r"),
        (b"## Process", b"## Process"),
    ],
)
def test_heading_spans_preserve_original_line_terminators(
    source: bytes, expected: bytes
) -> None:
    heading = parse_markdown(source).headings[0]

    assert _span(source, heading) == expected
    assert heading.source_region.end_byte == len(source)


def test_setext_crlf_span_preserves_both_physical_lines() -> None:
    source = b"Process\r\n=======\r\n"

    heading = parse_markdown(source).headings[0]

    assert _span(source, heading) == source


@pytest.mark.parametrize("text", ["进程", "プロセス", "emoji 🙂"])
def test_multibyte_heading_text_and_utf8_byte_spans(text: str) -> None:
    source = f"## {text}\n".encode()

    heading = parse_markdown(source).headings[0]

    assert heading.text == text
    assert _span(source, heading) == source


def test_inline_markdown_text_is_parser_provided_inline_content() -> None:
    heading = parse_markdown(b"## **A** + `B`\n").headings[0]

    assert heading.text == "**A** + `B`"


def test_parse_is_deterministic() -> None:
    source = b"# A\r\nProcess\r\n=======\r\n"

    assert parse_markdown(source) == parse_markdown(source)


def test_empty_atx_heading_is_preserved_as_a_structural_fact() -> None:
    source = b"# A\n##\nbody\n### C\n"

    document = parse_markdown(source)

    assert [(heading.level, heading.text) for heading in document.headings] == [
        (1, "A"),
        (2, ""),
        (3, "C"),
    ]
    assert _span(source, document.headings[1]) == b"##\n"


def test_invalid_utf8_propagates_strict_decode_error() -> None:
    with pytest.raises(UnicodeDecodeError):
        parse_markdown(b"# bad\xff")
