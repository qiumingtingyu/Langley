"""Focused contracts for Run-local cumulative Knowledge evidence."""

from langley.answering.knowledge_evidence import KnowledgeEvidenceSession
from langley.answering.knowledge_qa import validated_answer_completion
from langley.knowledge.contracts import (
    KnowledgeLocator,
    KnowledgeReadResult,
    PdfPageRegion,
)
from langley.knowledge.reads import AdjacentKnowledgeChunkRead
from langley.knowledge.retrieval import RetrievalHit


def _hit(chunk_id: int, *, content: str) -> RetrievalHit:
    return RetrievalHit(
        knowledge_chunk_id=chunk_id,
        rank=1,
        retrieval_rank=1,
        score=0.9,
        rerank_score=None,
        chunk_ordinal=4,
        content=content,
        heading_path=("TCP",),
        source_regions=({"kind": "text_span", "start_byte": 0, "end_byte": 4},),
        document_id=12,
        document_version_id=13,
        source_display_name="tcp.md",
        source_sha256="a" * 64,
    )


def _adjacent_chunk(chunk_id: int, *, content: str) -> AdjacentKnowledgeChunkRead:
    return AdjacentKnowledgeChunkRead(
        position="next",
        knowledge_chunk_id=chunk_id,
        chunk_ordinal=5,
        document_id=12,
        document_version_id=13,
        content=content,
        heading_path=("TCP", "Continuation"),
        source_regions=({"kind": "text_span", "start_byte": 4, "end_byte": 8},),
        source_display_name="tcp.md",
        source_sha256="a" * 64,
    )


def test_registration_assigns_stable_handles_and_resolves_authoritative_facts() -> None:
    session = KnowledgeEvidenceSession()

    first, second = session.register_hits(
        (_hit(11, content="first"), _hit(22, content="second"))
    )

    assert (first.evidence_handle, second.evidence_handle) == (1, 2)
    assert session.resolve(1) == first
    assert session.resolve(2) == second
    assert first.knowledge_chunk_id == 11
    assert first.chunk_ordinal == 4
    assert first.document_id == 12
    assert first.document_version_id == 13
    assert first.content == "first"
    assert first.heading_path == ("TCP",)
    assert first.source_regions == (
        {"kind": "text_span", "start_byte": 0, "end_byte": 4},
    )
    assert first.source_display_name == "tcp.md"
    assert first.source_sha256 == "a" * 64


def test_duplicate_chunk_reuses_existing_handle_without_new_evidence() -> None:
    session = KnowledgeEvidenceSession()
    original = session.register_hit(_hit(11, content="authoritative first read"))

    repeated = session.register_hit(_hit(11, content="later duplicate"))

    assert repeated is original
    assert repeated.evidence_handle == 1
    assert repeated.content == "authoritative first read"
    assert session.evidence == (original,)
    assert session.resolve(2) is None

    adjacent = session.register_chunk(_adjacent_chunk(22, content="next chunk"))
    repeated_adjacent = session.register_chunk(
        _adjacent_chunk(22, content="changed duplicate")
    )

    assert adjacent.evidence_handle == 2
    assert adjacent.content == "next chunk"
    assert repeated_adjacent is adjacent
    assert session.resolve_chunk(22) is adjacent


def test_mixed_search_read_handles_reuse_citation_snapshot_contract():
    session = KnowledgeEvidenceSession()
    first = session.register_hit(_hit(11, content="search evidence"))
    read = session.register_read(
        KnowledgeReadResult(
            locator=KnowledgeLocator(
                document_id=12, kind="pages", page_start=2, page_end=3
            ),
            document_version_id=13,
            content="entire read",
            source_display_name="book.pdf",
            source_sha256="b" * 64,
            heading_path=(),
            source_regions=(PdfPageRegion(2, 4),),
        )
    )
    assert read.evidence_handle == 2
    assert read.knowledge_chunk_id is None and read.chunk_ordinal is None
    assert session.register_hit(_hit(11, content="duplicate")) is first
    answer = validated_answer_completion(
        "answer [K2] and [K1] [K2]",
        session,
        requires_citation=True,
        abstention_control_token=None,
    )
    second, first_draft = answer.citations
    assert second.evidence_handle == 2 and first_draft.evidence_handle == 1
    assert second.heading_path_snapshot == ()
    assert second.source_regions_snapshot == (
        {"kind": "pdf_page", "page_start": 2, "page_end": 4},
    )
    assert second.evidence_text == "entire read" and second.document_version_id == 13
