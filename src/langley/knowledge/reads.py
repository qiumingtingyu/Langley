"""Detached, owned read models for the minimal Knowledge browser."""

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from langley.infrastructure.models import (
    Document,
    DocumentVersion,
    KnowledgeBase,
    KnowledgeChunk,
)
from langley.knowledge.chunking import ChunkingConfig


@dataclass(frozen=True)
class KnowledgeBaseRead:
    id: int
    name: str
    created_at: datetime


@dataclass(frozen=True)
class DocumentSourceRead:
    document_version_id: int
    filename: str
    media_type: str
    size_bytes: int
    sha256: str
    created_at: datetime


@dataclass(frozen=True)
class DocumentRead:
    id: int
    name: str
    created_at: datetime
    source: DocumentSourceRead


@dataclass(frozen=True)
class ChunkRead:
    ordinal: int
    content: str
    heading_path: list[str]
    source_regions: list[object]


@dataclass(frozen=True)
class DocumentVersionChunksRead:
    document_version_id: int
    successful_chunk_max_chars: int | None
    suggested_chunk_max_chars: int
    chunk_count: int
    chunks: tuple[ChunkRead, ...]


async def list_knowledge_bases(
    session: AsyncSession, *, user_id: int
) -> tuple[KnowledgeBaseRead, ...]:
    """List only one user's KnowledgeBases in stable creation order."""

    rows = (
        await session.execute(
            select(KnowledgeBase.id, KnowledgeBase.name, KnowledgeBase.created_at)
            .where(KnowledgeBase.user_id == user_id)
            .order_by(KnowledgeBase.created_at.asc(), KnowledgeBase.id.asc())
        )
    ).all()
    return tuple(
        KnowledgeBaseRead(id=row.id, name=row.name, created_at=row.created_at)
        for row in rows
    )


async def list_documents_for_knowledge_base(
    session: AsyncSession, *, user_id: int, knowledge_base_id: int
) -> tuple[DocumentRead, ...] | None:
    """List detached initial-source facts, or None when the base is not owned."""

    owned = await session.scalar(
        select(KnowledgeBase.id).where(
            KnowledgeBase.id == knowledge_base_id,
            KnowledgeBase.user_id == user_id,
        )
    )
    if owned is None:
        return None

    rows = (
        await session.execute(
            select(
                Document.id,
                Document.name,
                Document.created_at,
                DocumentVersion.id.label("document_version_id"),
                DocumentVersion.source_filename,
                DocumentVersion.source_media_type,
                DocumentVersion.source_size_bytes,
                DocumentVersion.source_sha256,
                DocumentVersion.created_at.label("source_created_at"),
            )
            .join(DocumentVersion, DocumentVersion.document_id == Document.id)
            .where(Document.knowledge_base_id == knowledge_base_id)
            .order_by(Document.created_at.desc(), Document.id.desc())
        )
    ).all()
    return tuple(
        DocumentRead(
            id=row.id,
            name=row.name,
            created_at=row.created_at,
            source=DocumentSourceRead(
                document_version_id=row.document_version_id,
                filename=row.source_filename,
                media_type=row.source_media_type,
                size_bytes=row.source_size_bytes,
                sha256=row.source_sha256,
                created_at=row.source_created_at,
            ),
        )
        for row in rows
    )


async def read_document_version_chunks(
    session: AsyncSession,
    *,
    user_id: int,
    document_version_id: int,
    offset: int,
    limit: int,
) -> DocumentVersionChunksRead | None:
    """Read one owned Version's current chunk set and its successful config fact."""

    version = (
        await session.execute(
            select(DocumentVersion.id, DocumentVersion.chunk_max_chars)
            .join(Document, Document.id == DocumentVersion.document_id)
            .join(KnowledgeBase, KnowledgeBase.id == Document.knowledge_base_id)
            .where(
                DocumentVersion.id == document_version_id,
                KnowledgeBase.user_id == user_id,
            )
        )
    ).one_or_none()
    if version is None:
        return None
    chunk_count = await session.scalar(
        select(func.count())
        .select_from(KnowledgeChunk)
        .where(KnowledgeChunk.document_version_id == document_version_id)
    )
    rows = (
        await session.execute(
            select(
                KnowledgeChunk.ordinal,
                KnowledgeChunk.content,
                KnowledgeChunk.heading_path,
                KnowledgeChunk.source_regions,
            )
            .where(KnowledgeChunk.document_version_id == document_version_id)
            .order_by(KnowledgeChunk.ordinal.asc())
            .offset(offset)
            .limit(limit)
        )
    ).all()
    return DocumentVersionChunksRead(
        document_version_id=version.id,
        successful_chunk_max_chars=version.chunk_max_chars,
        suggested_chunk_max_chars=ChunkingConfig().max_chunk_chars,
        chunk_count=chunk_count or 0,
        chunks=tuple(
            ChunkRead(
                ordinal=row.ordinal,
                content=row.content,
                heading_path=list(row.heading_path),
                source_regions=list(row.source_regions),
            )
            for row in rows
        ),
    )
