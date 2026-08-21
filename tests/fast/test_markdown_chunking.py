"""Focused Task 2.3 heading-aware chunking contracts."""

import pytest

from langley.knowledge.chunking import ChunkingConfig, build_candidate_chunks
from langley.knowledge.contracts import TextSpanRegion
from langley.knowledge.markdown import ParsedDocument, ParsedHeading, parse_markdown


def _build(source: bytes, maximum: int = 1200):
    return build_candidate_chunks(parse_markdown(source), ChunkingConfig(maximum))


def _facts(chunks):
    return [(item.content, item.heading_path) for item in chunks]


def _assert_provenance(source: bytes, chunks) -> None:
    assert [item.ordinal for item in chunks] == list(range(1, len(chunks) + 1))
    for item in chunks:
        region = item.source_regions[0]
        assert source[region.start_byte : region.end_byte] == item.content.encode()


def test_no_headings_and_preamble_preserve_exact_content() -> None:
    source = b"preamble\n# A\nbody\n"
    chunks = _build(source)
    assert _facts(chunks) == [("preamble\n", ()), ("body\n", ("A",))]
    _assert_provenance(source, chunks)


def test_nested_consecutive_skipped_and_branch_reset_headings() -> None:
    source = b"# A\na\n### C\nc\n## B\nb\n# D\nd\n"
    chunks = _build(source)
    assert _facts(chunks) == [
        ("a\n", ("A",)),
        ("c\n", ("A", "C")),
        ("b\n", ("A", "B")),
        ("d\n", ("D",)),
    ]


def test_empty_heading_keeps_boundary_and_resets_old_branch() -> None:
    source = b"# A\na\n## B\nb\n##\nempty\n### C\nc\n"
    assert _facts(_build(source)) == [
        ("a\n", ("A",)),
        ("b\n", ("A", "B")),
        ("empty\n", ("A",)),
        ("c\n", ("A", "C")),
    ]


def test_whole_whitespace_sections_are_skipped_but_accepted_whitespace_is_exact() -> (
    None
):
    source = b"# A\n \t\n# B\n  keep  \n\n"
    chunks = _build(source)
    assert _facts(chunks) == [("  keep  \n\n", ("B",))]
    _assert_provenance(source, chunks)


def test_exact_maximum_is_direct_and_maximum_plus_one_uses_splitter(
    monkeypatch,
) -> None:
    source = b"abcde"
    assert _facts(_build(source, 5)) == [("abcde", ())]

    class FakeSplitter:
        def __init__(self, **_kwargs) -> None:
            pass

        def chunk_indices(self, text: str):
            assert text == "abcde"
            return [(0, "abc"), (3, "de")]

    monkeypatch.setattr("langley.knowledge.chunking.TextSplitter", FakeSplitter)
    assert _facts(_build(source, 4)) == [("abc", ()), ("de", ())]


@pytest.mark.parametrize(
    "source",
    [
        b"x" * 40,
        "中文".encode() * 20,
        "プロセス".encode() * 12,
        "🙂e\u0301".encode() * 18,
        b"repeat repeat repeat repeat repeat ",
        b"# A\r\nbody\r\n",
    ],
)
def test_oversize_and_unicode_provenance_is_lossless(source: bytes) -> None:
    chunks = _build(source, 12)
    assert (
        "".join(item.content for item in chunks)
        == (source if not source.startswith(b"#") else b"body\r\n").decode()
    )
    _assert_provenance(source, chunks)


def test_splitter_whitespace_piece_is_retained(monkeypatch) -> None:
    class FakeSplitter:
        def __init__(self, **_kwargs) -> None:
            pass

        def chunk_indices(self, _text: str):
            return [(0, "abc"), (3, "   "), (6, "def")]

    monkeypatch.setattr("langley.knowledge.chunking.TextSplitter", FakeSplitter)
    source = b"abc   def"
    chunks = _build(source, 4)
    assert [item.content for item in chunks] == ["abc", "   ", "def"]
    _assert_provenance(source, chunks)


def test_deterministic_and_only_headings_produce_no_candidates() -> None:
    source = b"# A\n## B\n \n"
    assert _build(source) == _build(source) == ()


@pytest.mark.parametrize(
    "heading",
    [
        ParsedHeading(0, "bad", TextSpanRegion(0, 1)),
        ParsedHeading(7, "bad", TextSpanRegion(0, 1)),
        ParsedHeading(1, "bad", TextSpanRegion(0, 2)),
    ],
)
def test_malformed_headings_fail_closed(heading: ParsedHeading) -> None:
    parsed = ParsedDocument(b"x", (heading,))
    with pytest.raises(ValueError):
        build_candidate_chunks(parsed, ChunkingConfig())


def test_overlapping_headings_fail_closed() -> None:
    parsed = ParsedDocument(
        b"abc",
        (
            ParsedHeading(1, "A", TextSpanRegion(0, 2)),
            ParsedHeading(2, "B", TextSpanRegion(1, 3)),
        ),
    )
    with pytest.raises(ValueError):
        build_candidate_chunks(parsed, ChunkingConfig())


def test_reversed_heading_region_fails_closed() -> None:
    region = TextSpanRegion(0, 1)
    object.__setattr__(region, "start_byte", 1)
    object.__setattr__(region, "end_byte", 0)
    parsed = ParsedDocument(b"x", (ParsedHeading(1, "A", region),))
    with pytest.raises(ValueError):
        build_candidate_chunks(parsed, ChunkingConfig())


@pytest.mark.parametrize("maximum", [0, -1])
def test_chunking_config_requires_positive_capacity(maximum: int) -> None:
    with pytest.raises(ValueError):
        ChunkingConfig(maximum)
