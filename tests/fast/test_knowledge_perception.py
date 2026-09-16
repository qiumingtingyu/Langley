"""Source-native perception without retrieval, models or database emulation."""

import asyncio
from dataclasses import replace
from hashlib import sha256

import pytest

from langley.knowledge import perception
from langley.knowledge.contracts import (
    DocumentSourceRef,
    KnowledgeLocator,
    KnowledgePerceptionError,
    PdfPageRegion,
)
from langley.knowledge.reads import PerceptionDocumentRead, PublishedPdfChunkRead


class SourceStorage:
    def __init__(self, data):
        self.data = data
        self.reads = 0

    async def read_source(self, key):
        self.reads += 1
        if self.data is None:
            raise FileNotFoundError
        return self.data


def markdown_document(source):
    return PerceptionDocumentRead(
        12,
        "source.md",
        DocumentSourceRef(
            23, "internal", "text/markdown", sha256(source).hexdigest(), len(source)
        ),
    )


def supply_document(monkeypatch, document):
    async def read(*args, **kwargs):
        assert kwargs["user_id"] == 1 and kwargs["knowledge_base_id"] == 7
        assert kwargs["document_id"] == document.document_id
        return document

    monkeypatch.setattr(perception, "read_perception_document", read)


def inspect(storage):
    return asyncio.run(
        perception.inspect_knowledge(
            None, storage, user_id=1, knowledge_base_id=7, document_id=12
        )
    )


def read(storage, locator, bound=16384):
    return asyncio.run(
        perception.read_knowledge(
            None,
            storage,
            user_id=1,
            knowledge_base_id=7,
            locator=KnowledgeLocator.model_validate(locator),
            max_content_bytes=bound,
        )
    )


def test_markdown_inspect_locator_passes_through_to_exact_subtree(monkeypatch):
    source = (
        "前言\r\n# MySQL\r\n总览\r\n\r\n事务\r\n----\r\n正文\r\n"
        "### **MVCC**\r\n版本链\r\n```md\r\n# fake\r\n```\r\n"
        "> # quoted\r\n## Locking\r\n锁\r\n"
    ).encode()
    supply_document(monkeypatch, markdown_document(source))
    storage = SourceStorage(source)
    outline = inspect(storage)
    entries = outline["headings"]
    assert [entry["label"] for entry in entries] == [
        "MySQL",
        "事务",
        "**MVCC**",
        "Locking",
    ]
    assert "版本链" not in str(outline)
    for entry in entries:
        result = read(storage, entry["locator"])
        (region,) = result.source_regions
        assert source[region.start_byte : region.end_byte].decode() == result.content
        assert result.heading_path == tuple(entry["locator"]["heading_path"])
    transaction = read(storage, entries[1]["locator"])
    assert transaction.content.startswith("事务\r\n----\r\n")
    assert (
        "### **MVCC**" in transaction.content
        and "## Locking" not in transaction.content
    )
    assert read(storage, outline["locator"]).content.encode() == source
    assert "internal" not in str(outline) and "start_byte" not in str(outline)


def test_duplicate_and_empty_headings_are_not_guessed(monkeypatch):
    source = b"# A\n## B\none\n## B\ntwo\n##\n### C\nthree\n"
    supply_document(monkeypatch, markdown_document(source))
    storage = SourceStorage(source)
    outline = inspect(storage)
    assert all("locator" not in entry for entry in outline["headings"][1:])
    for path in (["A", "B"], ["A", "missing"], ["a"]):
        with pytest.raises(KnowledgePerceptionError) as error:
            read(storage, {"document_id": 12, "kind": "heading", "heading_path": path})
        assert error.value.code == "LOCATION_NOT_FOUND"


@pytest.mark.parametrize(
    "fields",
    [
        {"kind": "document", "heading_path": ["x"]},
        {"kind": "heading", "heading_path": []},
        {"kind": "heading", "heading_path": [" "]},
        {"kind": "pages", "page_start": 3, "page_end": 2},
        {"kind": "pages", "page_start": 0, "page_end": 1},
        {"kind": "pages", "page_start": 1},
    ],
)
def test_invalid_locator_combinations_fail_before_storage(fields):
    with pytest.raises(KnowledgePerceptionError) as error:
        read(None, {"document_id": 12, **fields})
    assert error.value.code == "INVALID_LOCATOR"


def test_read_bounds_and_integrity_fail_without_partial_content(monkeypatch):
    source = "# 标题\n内容\n".encode()
    supply_document(monkeypatch, markdown_document(source))
    storage = SourceStorage(source)
    locator = {"document_id": 12, "kind": "document"}
    assert read(storage, locator, len(source)).content.encode() == source
    assert read(storage, locator, len(source) + 1).content.encode() == source
    with pytest.raises(KnowledgePerceptionError) as error:
        read(storage, locator, len(source) - 1)
    assert error.value.code == "KNOWLEDGE_READ_SCOPE_TOO_LARGE"
    for bad in (None, b"corrupted"):
        with pytest.raises(KnowledgePerceptionError) as error:
            read(SourceStorage(bad), locator)
        assert error.value.code == "KNOWLEDGE_READ_UNAVAILABLE"
    with pytest.raises(KnowledgePerceptionError) as error:
        read(
            storage,
            {"document_id": 12, "kind": "pages", "page_start": 1, "page_end": 2},
        )
    assert error.value.code == "INVALID_LOCATOR"


def test_pdf_outline_keeps_regions_and_read_has_no_heading_authority(monkeypatch):
    document = replace(
        markdown_document(b"pdf"),
        source=DocumentSourceRef(23, "internal", "application/pdf", "a" * 64, 3),
        pdf_chunks=(
            PublishedPdfChunkRead(
                1, ("Transactions", "MVCC"), (PdfPageRegion(45, 48),), "first"
            ),
            PublishedPdfChunkRead(
                2, ("Transactions", "MVCC"), (PdfPageRegion(49, 52),), "second"
            ),
            PublishedPdfChunkRead(
                3, (), (PdfPageRegion(52, 53), PdfPageRegion(57, 57)), "third"
            ),
        ),
    )
    supply_document(monkeypatch, document)
    outline = inspect(None)
    assert len(outline["detected_regions"]) == 4
    assert [e["locator"]["page_start"] for e in outline["detected_regions"]] == [
        45,
        49,
        52,
        57,
    ]
    assert "page_count" not in outline
    for locator in (outline["locator"], outline["detected_regions"][0]["locator"]):
        result = read(None, locator)
        assert result.heading_path == ()
        assert result.content == "first\n\nsecond\n\nthird"
        assert result.source_regions == (PdfPageRegion(45, 53), PdfPageRegion(57, 57))
    with pytest.raises(KnowledgePerceptionError) as error:
        read(
            None,
            {"document_id": 12, "kind": "heading", "heading_path": ["Transactions"]},
        )
    assert error.value.code == "INVALID_LOCATOR"
