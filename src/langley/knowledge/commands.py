"""Focused commands for authoritative Knowledge source admission."""

from hashlib import sha256

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from langley.business_time import utc_now
from langley.infrastructure.local_file_storage import LocalFileStorage
from langley.infrastructure.models import (
    Document,
    DocumentVersion,
    KnowledgeBase,
    KnowledgeChunk,
    User,
)
from langley.knowledge.chunking import (
    CandidateChunk,
    ChunkingConfig,
    build_candidate_chunks,
)
from langley.knowledge.contracts import (
    DocumentSourceRef,
    FileStorage,
    encode_source_region,
)
from langley.knowledge.markdown import parse_markdown

_ASCII_WHITESPACE = " \t\n\r\f\v"
_SUPPORTED_SOURCE_MEDIA_TYPE = "text/markdown"


class KnowledgeBaseNotFoundError(Exception):
    """Raised when a KnowledgeBase is missing or outside the requested user scope."""


class DocumentVersionNotFoundError(Exception):
    """Raised when a DocumentVersion is missing or outside the requested user scope."""


class SourceIntegrityError(Exception):
    """Raised when an admitted source no longer matches its persisted facts."""


def canonicalize_source_media_type(source_media_type: str) -> str:
    """Canonicalize one unparameterized source media type before support checks."""
    canonical = source_media_type.strip(_ASCII_WHITESPACE).lower()
    if canonical != _SUPPORTED_SOURCE_MEDIA_TYPE:
        raise ValueError("unsupported source_media_type")
    return canonical


async def create_knowledge_base(
    session: AsyncSession, *, user_id: int, name: str
) -> KnowledgeBase:
    """Create one user-owned KnowledgeBase in a short transaction."""
    _require_nonblank(name, "name")
    async with session.begin():
        if await session.get(User, user_id) is None:
            raise KnowledgeBaseNotFoundError
        knowledge_base = KnowledgeBase(
            user_id=user_id,
            name=name,
            created_at=utc_now(),
        )
        session.add(knowledge_base)
        await session.flush()
    return knowledge_base


async def create_initial_document(
    session_factory: async_sessionmaker[AsyncSession],
    file_storage: FileStorage,
    *,
    user_id: int,
    knowledge_base_id: int,
    name: str,
    source_filename: str,
    source_media_type: str,
    source_bytes: bytes,
) -> DocumentVersion:
    """Finalize an initial source before authoritatively admitting its DB facts."""
    _require_nonblank(name, "name")
    _require_nonblank(source_filename, "source_filename")

    async with session_factory() as precheck_session:
        if not await _knowledge_base_belongs_to_user(
            precheck_session, knowledge_base_id, user_id
        ):
            raise KnowledgeBaseNotFoundError

    canonical_media_type = canonicalize_source_media_type(source_media_type)
    _validate_markdown_source(source_bytes)
    stored_source = await file_storage.store_source(user_id, source_bytes)

    async with session_factory() as admission_session:
        async with admission_session.begin():
            if not await _knowledge_base_belongs_to_user(
                admission_session, knowledge_base_id, user_id
            ):
                raise KnowledgeBaseNotFoundError
            now = utc_now()
            document = Document(
                knowledge_base_id=knowledge_base_id,
                name=name,
                created_at=now,
            )
            admission_session.add(document)
            await admission_session.flush()
            version = DocumentVersion(
                document_id=document.id,
                source_filename=source_filename,
                source_media_type=canonical_media_type,
                source_sha256=stored_source.sha256,
                source_size_bytes=stored_source.size_bytes,
                storage_key=stored_source.storage_key,
                created_at=now,
            )
            admission_session.add(version)
            await admission_session.flush()
    return version


async def load_document_source_ref(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    user_id: int,
    document_version_id: int,
) -> DocumentSourceRef:
    """Return detached source facts only after transitive ownership validation."""
    async with session_factory() as session:
        row = (
            await session.execute(
                select(
                    DocumentVersion.id,
                    DocumentVersion.storage_key,
                    DocumentVersion.source_media_type,
                    DocumentVersion.source_sha256,
                    DocumentVersion.source_size_bytes,
                )
                .join(Document, Document.id == DocumentVersion.document_id)
                .join(KnowledgeBase, KnowledgeBase.id == Document.knowledge_base_id)
                .where(
                    DocumentVersion.id == document_version_id,
                    KnowledgeBase.user_id == user_id,
                )
            )
        ).one_or_none()
    if row is None:
        raise DocumentVersionNotFoundError
    return DocumentSourceRef(
        document_version_id=row.id,
        storage_key=row.storage_key,
        source_media_type=row.source_media_type,
        source_sha256=row.source_sha256,
        source_size_bytes=row.source_size_bytes,
    )


async def read_verified_source(
    file_storage: FileStorage, source_ref: DocumentSourceRef
) -> bytes:
    """Read detached source bytes and verify their stored size and SHA-256 facts."""
    try:
        source_bytes = await file_storage.read_source(source_ref.storage_key)
    except FileNotFoundError as error:
        raise SourceIntegrityError("stored source is missing") from error
    if len(source_bytes) != source_ref.source_size_bytes:
        raise SourceIntegrityError("stored source size does not match persisted facts")
    if sha256(source_bytes).hexdigest() != source_ref.source_sha256:
        raise SourceIntegrityError("stored source hash does not match persisted facts")
    return source_bytes


async def rebuild_document_version_chunks(
    *,
    session_factory: async_sessionmaker[AsyncSession],
    file_storage: FileStorage,
    user_id: int,
    document_version_id: int,
    config: ChunkingConfig,
) -> None:
    """Atomically replace current chunks from one verified immutable source."""

    source_ref = await load_document_source_ref(
        session_factory, user_id=user_id, document_version_id=document_version_id
    )
    source_bytes = await read_verified_source(file_storage, source_ref)
    candidates = build_candidate_chunks(parse_markdown(source_bytes), config)
    prepared_rows = _materialize_chunk_rows(document_version_id, candidates)
    await _replace_document_version_chunks(
        session_factory=session_factory,
        user_id=user_id,
        source_ref=source_ref,
        prepared_rows=prepared_rows,
    )


def _materialize_chunk_rows(
    document_version_id: int, candidates: tuple[CandidateChunk, ...]
) -> list[KnowledgeChunk]:
    now = utc_now()
    return [
        KnowledgeChunk(
            document_version_id=document_version_id,
            ordinal=candidate.ordinal,
            content=candidate.content,
            heading_path=list(candidate.heading_path),
            source_regions=[
                encode_source_region(region) for region in candidate.source_regions
            ],
            created_at=now,
        )
        for candidate in candidates
    ]


async def _replace_document_version_chunks(
    *,
    session_factory: async_sessionmaker[AsyncSession],
    user_id: int,
    source_ref: DocumentSourceRef,
    prepared_rows: list[KnowledgeChunk],
) -> None:
    owned_document_ids = (
        select(Document.id).join(KnowledgeBase).where(KnowledgeBase.user_id == user_id)
    )
    async with session_factory() as session:
        async with session.begin():
            version = await session.scalar(
                select(DocumentVersion)
                .where(
                    DocumentVersion.id == source_ref.document_version_id,
                    DocumentVersion.document_id.in_(owned_document_ids),
                )
                .with_for_update()
            )
            if version is None:
                raise DocumentVersionNotFoundError
            if not _document_version_matches_source_ref(version, source_ref):
                raise RuntimeError("document source identity changed during rebuild")
            await session.execute(
                delete(KnowledgeChunk).where(
                    KnowledgeChunk.document_version_id == version.id
                )
            )
            session.add_all(prepared_rows)
            knowledge_base = await session.scalar(
                select(KnowledgeBase)
                .join(Document, Document.knowledge_base_id == KnowledgeBase.id)
                .where(Document.id == version.document_id)
                .with_for_update()
            )
            if knowledge_base is None:
                raise RuntimeError("document knowledge base disappeared during rebuild")
            if knowledge_base.index_status != "INDEXING":
                knowledge_base.index_status = (
                    "STALE"
                    if knowledge_base.active_generation_id is not None
                    else "CHUNKED"
                )
            await session.flush()


def _document_version_matches_source_ref(
    version: DocumentVersion, source_ref: DocumentSourceRef
) -> bool:
    """Compare every immutable source fact captured before slow rebuild work."""
    return (
        version.storage_key == source_ref.storage_key
        and version.source_media_type == source_ref.source_media_type
        and version.source_sha256 == source_ref.source_sha256
        and version.source_size_bytes == source_ref.source_size_bytes
    )


async def find_unreferenced_local_sources(
    session_factory: async_sessionmaker[AsyncSession],
    file_storage: LocalFileStorage,
) -> tuple[str, ...]:
    """Identify finalized local sources absent from detached authoritative DB keys."""
    async with session_factory() as session:
        referenced_storage_keys = set(
            (await session.scalars(select(DocumentVersion.storage_key))).all()
        )
    finalized_storage_keys = await file_storage.list_finalized_source_keys()
    return tuple(sorted(finalized_storage_keys - referenced_storage_keys))


async def _knowledge_base_belongs_to_user(
    session: AsyncSession, knowledge_base_id: int, user_id: int
) -> bool:
    return (
        await session.scalar(
            select(KnowledgeBase.id).where(
                KnowledgeBase.id == knowledge_base_id,
                KnowledgeBase.user_id == user_id,
            )
        )
    ) is not None


def _require_nonblank(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")


def _validate_markdown_source(source_bytes: bytes) -> None:
    if not source_bytes:
        raise ValueError("source_bytes must not be empty")
    source_bytes.decode("utf-8", errors="strict")
