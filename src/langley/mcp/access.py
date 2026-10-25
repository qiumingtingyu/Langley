"""Short authoritative ownership/version checks; no parser or retrieval logic."""

from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from langley.infrastructure.models import (
    Document,
    DocumentVersion,
    KnowledgeBase,
    KnowledgeChunk,
    User,
)
from langley.knowledge.retrieval import RetrievalHit
from langley.mcp.config import Principal
from langley.mcp.contracts import DocumentLocation, GateError, Locator, encoded_regions


@dataclass(frozen=True)
class DocumentFacts:
    version: int
    revision: int
    media: str
    source_sha256: str
    source_bytes: int


class Access:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]):
        self.sessions = sessions

    async def user_exists(self, user_id: int) -> bool:
        async with self.sessions() as session:
            return (
                await session.scalar(select(User.id).where(User.id == user_id))
                is not None
            )

    @staticmethod
    def grant(principal: Principal, kb_id: int) -> None:
        if kb_id not in principal.allowed_kb_ids:
            raise GateError("RESOURCE_UNAVAILABLE")

    async def kb(self, principal: Principal, kb_id: int) -> None:
        self.grant(principal, kb_id)
        async with self.sessions() as session:
            found = await session.scalar(
                select(KnowledgeBase.id)
                .join(User)
                .where(KnowledgeBase.id == kb_id, User.id == principal.user_id)
            )
        if found is None:
            raise GateError("RESOURCE_UNAVAILABLE")

    async def document(self, principal: Principal, locator: Locator) -> DocumentFacts:
        self.grant(principal, locator.kb_id)
        async with self.sessions() as session, session.begin():
            rows = (
                await session.scalars(
                    select(DocumentVersion)
                    .join(Document)
                    .join(KnowledgeBase)
                    .join(User)
                    .where(
                        User.id == principal.user_id,
                        KnowledgeBase.id == locator.kb_id,
                        Document.id == locator.document_id,
                    )
                    .limit(2)
                )
            ).all()
            if not rows:
                raise GateError("RESOURCE_UNAVAILABLE")
            if len(rows) != 1:
                raise GateError("RESOURCE_VERSION_AMBIGUOUS")
            version = rows[0]
            if (
                version.id != locator.document_version_id
                or version.chunk_revision != locator.chunk_revision
            ):
                raise GateError("RESOURCE_CHANGED")
            if version.source_media_type == "text/markdown":
                if version.source_size_bytes > 5 * 1024 * 1024:
                    raise GateError("RESOURCE_TOO_LARGE")
            elif version.source_media_type == "application/pdf":
                count, metadata, regions = (
                    await session.execute(
                        select(
                            func.count(KnowledgeChunk.id),
                            func.sum(
                                func.octet_length(KnowledgeChunk.heading_path)
                                + func.octet_length(KnowledgeChunk.source_regions)
                            ),
                            func.sum(func.json_length(KnowledgeChunk.source_regions)),
                        ).where(KnowledgeChunk.document_version_id == version.id)
                    )
                ).one()
                if count > 1000 or (metadata or 0) > 262144 or (regions or 0) > 1000:
                    raise GateError("RESOURCE_TOO_LARGE")
            else:
                raise GateError("READ_UNAVAILABLE")
            return DocumentFacts(
                version.id,
                version.chunk_revision,
                version.source_media_type,
                version.source_sha256,
                version.source_size_bytes,
            )

    async def hits(
        self, principal: Principal, kb_id: int, hits: tuple[RetrievalHit, ...]
    ) -> list[Locator]:
        self.grant(principal, kb_id)
        async with self.sessions() as session, session.begin():
            owner = await session.scalar(
                select(KnowledgeBase.id)
                .join(User)
                .where(KnowledgeBase.id == kb_id, User.id == principal.user_id)
            )
            if owner is None:
                raise GateError("RESOURCE_UNAVAILABLE")
            rows = (
                await session.execute(
                    select(KnowledgeChunk, DocumentVersion, Document)
                    .join(
                        DocumentVersion,
                        DocumentVersion.id == KnowledgeChunk.document_version_id,
                    )
                    .join(Document, Document.id == DocumentVersion.document_id)
                    .where(
                        Document.knowledge_base_id == kb_id,
                        KnowledgeChunk.id.in_([h.knowledge_chunk_id for h in hits]),
                    )
                )
            ).all()
            by_id = {chunk.id: (chunk, version, doc) for chunk, version, doc in rows}
            locators = []
            for hit in hits:
                current = by_id.get(hit.knowledge_chunk_id)
                if current is None:
                    raise GateError("RESOURCE_CHANGED")
                chunk, version, doc = current
                if (
                    doc.id != hit.document_id
                    or version.id != hit.document_version_id
                    or version.source_sha256 != hit.source_sha256
                    or version.chunk_revision < 1
                    or version.indexed_chunk_revision != version.chunk_revision
                    or chunk.content != hit.content
                    or chunk.ordinal != hit.chunk_ordinal
                    or chunk.heading_path != list(hit.heading_path)
                    or chunk.source_regions != encoded_regions(hit.source_regions)
                ):
                    raise GateError("RESOURCE_CHANGED")
                versions = await session.scalar(
                    select(func.count(DocumentVersion.id)).where(
                        DocumentVersion.document_id == doc.id
                    )
                )
                if versions != 1:
                    raise GateError("RESOURCE_VERSION_AMBIGUOUS")
                locators.append(
                    Locator(
                        v=1,
                        kb_id=kb_id,
                        document_id=doc.id,
                        document_version_id=version.id,
                        chunk_revision=version.chunk_revision,
                        location=DocumentLocation(kind="document"),
                    )
                )
            return locators
