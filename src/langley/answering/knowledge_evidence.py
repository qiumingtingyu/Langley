"""Run-local Knowledge evidence identities for one answer execution."""

from dataclasses import dataclass

from langley.knowledge.contracts import KnowledgeReadResult, encode_source_region
from langley.knowledge.reads import AdjacentKnowledgeChunkRead
from langley.knowledge.retrieval import RetrievalHit


@dataclass(frozen=True)
class KnowledgeEvidence:
    """Authoritative evidence facts exposed under one stable model-visible K#."""

    evidence_handle: int
    knowledge_chunk_id: int | None
    chunk_ordinal: int | None
    document_id: int
    document_version_id: int
    content: str
    heading_path: tuple[str, ...]
    source_regions: tuple[object, ...]
    source_display_name: str
    source_sha256: str

    def __post_init__(self) -> None:
        if (self.knowledge_chunk_id is None) != (self.chunk_ordinal is None):
            raise ValueError("retrieval identity fields must be present together")


class KnowledgeEvidenceSession:
    """Assign and resolve cumulative Knowledge evidence handles within one Run."""

    def __init__(self) -> None:
        self._evidence_by_handle: dict[int, KnowledgeEvidence] = {}
        self._handle_by_chunk_id: dict[int, int] = {}

    @property
    def evidence(self) -> tuple[KnowledgeEvidence, ...]:
        return tuple(self._evidence_by_handle.values())

    @property
    def has_chunk_evidence(self) -> bool:
        return bool(self._handle_by_chunk_id)

    def register_read(self, result: KnowledgeReadResult) -> KnowledgeEvidence:
        """Register one complete bounded read without inventing retrieval identity."""
        return self._store(
            KnowledgeEvidence(
                evidence_handle=len(self._evidence_by_handle) + 1,
                knowledge_chunk_id=None,
                chunk_ordinal=None,
                document_id=result.locator.document_id,
                document_version_id=result.document_version_id,
                content=result.content,
                heading_path=result.heading_path,
                source_regions=tuple(
                    encode_source_region(region) for region in result.source_regions
                ),
                source_display_name=result.source_display_name,
                source_sha256=result.source_sha256,
            )
        )

    def register_hits(
        self, hits: tuple[RetrievalHit, ...]
    ) -> tuple[KnowledgeEvidence, ...]:
        return tuple(self.register_hit(hit) for hit in hits)

    def register_hit(self, hit: RetrievalHit) -> KnowledgeEvidence:
        existing = self.resolve_chunk(hit.knowledge_chunk_id)
        if existing is not None:
            return existing

        evidence = KnowledgeEvidence(
            evidence_handle=len(self._evidence_by_handle) + 1,
            knowledge_chunk_id=hit.knowledge_chunk_id,
            chunk_ordinal=hit.chunk_ordinal,
            document_id=hit.document_id,
            document_version_id=hit.document_version_id,
            content=hit.content,
            heading_path=hit.heading_path,
            source_regions=hit.source_regions,
            source_display_name=hit.source_display_name,
            source_sha256=hit.source_sha256,
        )
        return self._store(evidence)

    def register_chunk(self, chunk: AdjacentKnowledgeChunkRead) -> KnowledgeEvidence:
        existing = self.resolve_chunk(chunk.knowledge_chunk_id)
        if existing is not None:
            return existing

        evidence = KnowledgeEvidence(
            evidence_handle=len(self._evidence_by_handle) + 1,
            knowledge_chunk_id=chunk.knowledge_chunk_id,
            chunk_ordinal=chunk.chunk_ordinal,
            document_id=chunk.document_id,
            document_version_id=chunk.document_version_id,
            content=chunk.content,
            heading_path=chunk.heading_path,
            source_regions=chunk.source_regions,
            source_display_name=chunk.source_display_name,
            source_sha256=chunk.source_sha256,
        )
        return self._store(evidence)

    def resolve_chunk(self, knowledge_chunk_id: int) -> KnowledgeEvidence | None:
        evidence_handle = self._handle_by_chunk_id.get(knowledge_chunk_id)
        if evidence_handle is None:
            return None
        return self._evidence_by_handle[evidence_handle]

    def _store(self, evidence: KnowledgeEvidence) -> KnowledgeEvidence:
        self._evidence_by_handle[evidence.evidence_handle] = evidence
        if evidence.knowledge_chunk_id is not None:
            self._handle_by_chunk_id[evidence.knowledge_chunk_id] = (
                evidence.evidence_handle
            )
        return evidence

    def resolve(self, evidence_handle: int) -> KnowledgeEvidence | None:
        return self._evidence_by_handle.get(evidence_handle)
