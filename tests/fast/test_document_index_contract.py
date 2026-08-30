"""Focused deterministic tests for the Task 3A document-index contract."""

from dataclasses import dataclass, replace
from datetime import datetime

import pytest

from langley.knowledge.contracts import PdfPageRegion, SourceRegion, TextSpanRegion
from langley.knowledge.document_index_contract import (
    build_source_context_v1,
    chunk_set_sha256,
)


@dataclass(frozen=True)
class _StoredChunk:
    id: int
    created_at: datetime
    ordinal: int
    content: str
    heading_path: tuple[str, ...]
    source_regions: tuple[SourceRegion, ...]


def _chunks() -> tuple[_StoredChunk, ...]:
    return (
        _StoredChunk(
            id=11,
            created_at=datetime(2026, 8, 30, 1, 0),
            ordinal=1,
            content="  第一段\n",
            heading_path=("网络", "TCP"),
            source_regions=(PdfPageRegion(page_start=2, page_end=2),),
        ),
        _StoredChunk(
            id=12,
            created_at=datetime(2026, 8, 30, 1, 1),
            ordinal=2,
            content="TIME-WAIT ...",
            heading_path=("网络", "TCP", "连接关闭"),
            source_regions=(
                PdfPageRegion(page_start=3, page_end=4),
                PdfPageRegion(page_start=6, page_end=6),
            ),
        ),
    )


def test_source_context_v1_without_heading_is_exact_content() -> None:
    content = "  exact content\n\n"

    assert build_source_context_v1(content, ()) == content


def test_source_context_v1_joins_root_to_leaf_then_exact_content() -> None:
    content = "TIME-WAIT ..."

    assert (
        build_source_context_v1(content, ("Network", "TCP", "Connection Close"))
        == "Network\nTCP\nConnection Close\n\nTIME-WAIT ..."
    )


def test_source_context_v1_preserves_heading_and_content_whitespace() -> None:
    assert build_source_context_v1("\n body  \n", (" Root ",)) == (
        " Root \n\n\n body  \n"
    )
    with pytest.raises(ValueError, match="invalid heading path"):
        build_source_context_v1("body", ("",))


def test_chunk_set_fingerprint_ignores_row_identity_and_input_order() -> None:
    chunks = _chunks()
    same_facts_new_rows = (
        replace(chunks[1], id=102, created_at=datetime(2030, 1, 1)),
        replace(chunks[0], id=101, created_at=datetime(2030, 1, 1)),
    )

    assert chunk_set_sha256(chunks) == chunk_set_sha256(same_facts_new_rows)


def test_chunk_set_fingerprint_changes_for_each_semantic_or_provenance_fact() -> None:
    chunks = _chunks()
    baseline = chunk_set_sha256(chunks)

    assert (
        chunk_set_sha256((replace(chunks[0], content="changed"), chunks[1])) != baseline
    )
    assert (
        chunk_set_sha256((replace(chunks[0], heading_path=("网络", "UDP")), chunks[1]))
        != baseline
    )
    assert (
        chunk_set_sha256(
            (
                replace(
                    chunks[0],
                    source_regions=(TextSpanRegion(start_byte=0, end_byte=5),),
                ),
                chunks[1],
            )
        )
        != baseline
    )
    assert chunk_set_sha256((chunks[0], replace(chunks[1], ordinal=3))) != baseline
