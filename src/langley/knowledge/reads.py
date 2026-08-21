"""Detached, owned read models for the minimal Knowledge browser."""

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from langley.infrastructure.models import Document, DocumentVersion, KnowledgeBase


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
