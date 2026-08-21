"""Task 2.0 frozen third-party splitter and Markdown parser preflight."""

from markdown_it import MarkdownIt
from semantic_text_splitter import TextSplitter


def _split(text: str, capacity: int = 24):
    return TextSplitter(capacity=capacity, overlap=0, trim=False).chunk_indices(text)


def test_text_splitter_is_lossless_deterministic_and_unicode_safe() -> None:
    text = "  前导\r\n\r\n中文🙂é🚀\r\nプロセスは資源管理の単位です。\r\n尾部  "
    first = _split(text)
    second = _split(text)
    assert first == second
    assert "".join(chunk for _, chunk in first) == text
    assert all(len(chunk) <= 24 for _, chunk in first)
    assert all(0 <= start <= start + len(chunk) <= len(text) for start, chunk in first)
    source_bytes = text.encode("utf-8")
    for start, chunk in first:
        assert text[start : start + len(chunk)] == chunk
        start_byte = len(text[:start].encode("utf-8"))
        end_byte = start_byte + len(chunk.encode("utf-8"))
        assert source_bytes[start_byte:end_byte].decode("utf-8") == chunk


def test_text_splitter_handles_repeated_and_long_unspaced_text() -> None:
    text = "abc" * 30 + "中文" * 30
    chunks = _split(text)
    assert "".join(chunk for _, chunk in chunks) == text
    assert all(len(chunk) <= 24 for _, chunk in chunks)
    cursor = 0
    for start, chunk in chunks:
        assert start == cursor
        assert text[start : start + len(chunk)] == chunk
        cursor += len(chunk)
    assert cursor == len(text)


def test_markdown_it_commonmark_smoke_distinguishes_heading_contexts() -> None:
    tokens = MarkdownIt("commonmark").parse("""# ATX

Setext
===

```
# not heading
```

> # quoted
""")
    heading_opens = [token for token in tokens if token.type == "heading_open"]
    assert [(token.tag, token.level) for token in heading_opens] == [
        ("h1", 0),
        ("h1", 0),
        ("h1", 1),
    ]
    assert "# not heading\n" in [
        token.content for token in tokens if token.type == "fence"
    ]
    assert [
        tokens[index + 1].content
        for index, token in enumerate(tokens)
        if token.type == "heading_open" and token.level == 0
    ] == ["ATX", "Setext"]
