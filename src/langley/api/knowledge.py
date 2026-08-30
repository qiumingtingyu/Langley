"""Minimal owned Knowledge management HTTP resources."""

from pathlib import Path, PurePosixPath

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Response,
    UploadFile,
    status,
)
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from langley.api.dependencies import (
    get_current_user_id,
    get_document_index_dispatcher,
    get_document_processing_dispatcher,
    get_knowledge_index_runtime,
    get_local_file_storage,
    get_session,
    get_session_factory,
)
from langley.api.responses import as_utc
from langley.business_time import utc_now
from langley.infrastructure.local_file_storage import LocalFileStorage
from langley.knowledge.chunking import ChunkingConfig
from langley.knowledge.commands import (
    DocumentAdmissionConflictError,
    DocumentRebuildConflictError,
    DocumentVersionNotFoundError,
    KnowledgeBaseNotFoundError,
    SourceIntegrityError,
    create_initial_document,
    create_initial_pdf_document,
    create_knowledge_base,
    load_document_source_ref,
    read_verified_source,
    rebuild_document_version_chunks,
)
from langley.knowledge.document_indexing import DocumentIndexDispatcher
from langley.knowledge.document_processing import PDF_PROCESSING_RECIPE_ID
from langley.knowledge.index_build import (
    IndexBuildAdmissionError,
    IndexJobRead,
    IndexStatusRead,
    KnowledgeIndexBuildRuntime,
    admit_index_build,
    read_index_status,
)
from langley.knowledge.pdf_processing import DocumentProcessingDispatcher
from langley.knowledge.reads import (
    DocumentProcessingJobRead,
    DocumentProcessingRead,
    DocumentRead,
    DocumentSourceRead,
    DocumentVersionChunksRead,
    KnowledgeBaseRead,
    list_documents_for_knowledge_base,
    list_knowledge_bases,
    read_document_processing_status,
    read_document_version_chunks,
)
from langley.knowledge.retrieval import (
    IndexNotReadyError,
    KnowledgeBaseRetrievalNotFoundError,
    RetrievalEmbeddingInvalidError,
    RetrievalEmbeddingUnavailableError,
    RetrievalError,
    RetrievalHit,
    RetrievalIndexChangedError,
    RetrievalIndexInconsistentError,
    RetrievalQdrantUnavailableError,
    RetrievalResult,
    retrieve_dense,
)

router = APIRouter(tags=["knowledge"])

MAX_MARKDOWN_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_PDF_UPLOAD_BYTES = 64 * 1024 * 1024
_UPLOAD_READ_CHUNK_BYTES = 64 * 1024
_MAX_DOCUMENT_TEXT_LENGTH = 255
_MAX_CHUNKS_PAGE_SIZE = 100
_MAX_RETRIEVAL_TOP_K = 50


class KnowledgeBaseCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=255)


class KnowledgeBaseResponse(BaseModel):
    id: int
    name: str
    created_at: str


class DocumentSourceResponse(BaseModel):
    document_version_id: int
    filename: str
    media_type: str
    size_bytes: int
    sha256: str
    created_at: str


class DocumentResponse(BaseModel):
    id: int
    name: str
    created_at: str
    source: DocumentSourceResponse


class DocumentProcessingAcceptedResponse(BaseModel):
    job_id: int
    attempt_no: int
    status: str
    recipe_id: str


class PdfDocumentAcceptedResponse(DocumentResponse):
    processing_job: DocumentProcessingAcceptedResponse


class DocumentProcessingJobResponse(BaseModel):
    id: int
    attempt_no: int
    status: str
    stage: str | None
    recipe_id: str
    error_code: str | None
    error_message: str | None
    created_at: str
    started_at: str | None
    finished_at: str | None


class DocumentProcessingStatusResponse(BaseModel):
    document_version_id: int
    latest_attempt: DocumentProcessingJobResponse | None
    published_chunks_exist: bool


class VerifySourceResponse(BaseModel):
    document_version_id: int
    verified: bool
    verified_at: str


class IndexJobResponse(BaseModel):
    id: int
    status: str
    stage: str | None
    processed_chunk_count: int
    total_chunk_count: int
    error_code: str | None
    error_message: str | None
    created_at: str
    started_at: str | None
    finished_at: str | None


class IndexStatusResponse(BaseModel):
    index_status: str
    latest_job: IndexJobResponse | None


class IndexBuildAcceptedResponse(BaseModel):
    job_id: int


class ChunkResponse(BaseModel):
    ordinal: int
    content: str
    heading_path: list[str]
    source_regions: list[object]


class DocumentVersionChunksResponse(BaseModel):
    document_version_id: int
    successful_chunk_max_chars: int | None
    suggested_chunk_max_chars: int
    chunk_count: int
    offset: int
    limit: int
    chunks: list[ChunkResponse]


class ChunkRebuildRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_chunk_chars: int = Field(gt=0)


class ChunkRebuildResponse(BaseModel):
    document_version_id: int
    successful_chunk_max_chars: int
    chunk_count: int
    resulting_index_status: str


class RetrievalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    top_k: int = Field(ge=1, le=_MAX_RETRIEVAL_TOP_K)


class RetrievalHitResponse(BaseModel):
    knowledge_chunk_id: int
    rank: int
    score: float
    chunk_ordinal: int
    content: str
    heading_path: list[str]
    source_regions: list[object]
    document_id: int
    document_version_id: int
    source_display_name: str
    source_sha256: str


class RetrievalResponse(BaseModel):
    knowledge_base_id: int
    hits: list[RetrievalHitResponse]


def _validation_error() -> HTTPException:
    return HTTPException(status_code=422, detail={"code": "VALIDATION_ERROR"})


def _knowledge_base_response(value: KnowledgeBaseRead) -> KnowledgeBaseResponse:
    return KnowledgeBaseResponse(
        id=value.id,
        name=value.name,
        created_at=as_utc(value.created_at),
    )


def _source_response(value: DocumentSourceRead) -> DocumentSourceResponse:
    return DocumentSourceResponse(
        document_version_id=value.document_version_id,
        filename=value.filename,
        media_type=value.media_type,
        size_bytes=value.size_bytes,
        sha256=value.sha256,
        created_at=as_utc(value.created_at),
    )


def _document_response(value: DocumentRead) -> DocumentResponse:
    return DocumentResponse(
        id=value.id,
        name=value.name,
        created_at=as_utc(value.created_at),
        source=_source_response(value.source),
    )


def _document_processing_job_response(
    value: DocumentProcessingJobRead,
) -> DocumentProcessingJobResponse:
    return DocumentProcessingJobResponse(
        id=value.id,
        attempt_no=value.attempt_no,
        status=value.status,
        stage=value.stage,
        recipe_id=value.recipe_id,
        error_code=value.error_code,
        error_message=value.error_message,
        created_at=as_utc(value.created_at),
        started_at=None if value.started_at is None else as_utc(value.started_at),
        finished_at=None if value.finished_at is None else as_utc(value.finished_at),
    )


def _document_processing_status_response(
    value: DocumentProcessingRead,
) -> DocumentProcessingStatusResponse:
    return DocumentProcessingStatusResponse(
        document_version_id=value.document_version_id,
        latest_attempt=None
        if value.latest_attempt is None
        else _document_processing_job_response(value.latest_attempt),
        published_chunks_exist=value.published_chunks_exist,
    )


def _index_job_response(value: IndexJobRead) -> IndexJobResponse:
    return IndexJobResponse(
        id=value.id,
        status=value.status,
        stage=value.stage,
        processed_chunk_count=value.processed_chunk_count,
        total_chunk_count=value.total_chunk_count,
        error_code=value.error_code,
        error_message=value.error_message,
        created_at=as_utc(value.created_at),
        started_at=None if value.started_at is None else as_utc(value.started_at),
        finished_at=None if value.finished_at is None else as_utc(value.finished_at),
    )


def _index_status_response(value: IndexStatusRead) -> IndexStatusResponse:
    return IndexStatusResponse(
        index_status=value.index_status,
        latest_job=None
        if value.latest_job is None
        else _index_job_response(value.latest_job),
    )


def _document_version_chunks_response(
    value: DocumentVersionChunksRead, *, offset: int, limit: int
) -> DocumentVersionChunksResponse:
    return DocumentVersionChunksResponse(
        document_version_id=value.document_version_id,
        successful_chunk_max_chars=value.successful_chunk_max_chars,
        suggested_chunk_max_chars=value.suggested_chunk_max_chars,
        chunk_count=value.chunk_count,
        offset=offset,
        limit=limit,
        chunks=[
            ChunkResponse(
                ordinal=chunk.ordinal,
                content=chunk.content,
                heading_path=chunk.heading_path,
                source_regions=chunk.source_regions,
            )
            for chunk in value.chunks
        ],
    )


def _retrieval_hit_response(value: RetrievalHit) -> RetrievalHitResponse:
    return RetrievalHitResponse(
        knowledge_chunk_id=value.knowledge_chunk_id,
        rank=value.rank,
        score=value.score,
        chunk_ordinal=value.chunk_ordinal,
        content=value.content,
        heading_path=list(value.heading_path),
        source_regions=list(value.source_regions),
        document_id=value.document_id,
        document_version_id=value.document_version_id,
        source_display_name=value.source_display_name,
        source_sha256=value.source_sha256,
    )


def _raise_retrieval_error(error: RetrievalError) -> None:
    if isinstance(error, KnowledgeBaseRetrievalNotFoundError):
        raise HTTPException(status_code=404, detail={"code": error.code}) from error
    if isinstance(error, (IndexNotReadyError, RetrievalIndexChangedError)):
        raise HTTPException(status_code=409, detail={"code": error.code}) from error
    if isinstance(
        error,
        (
            RetrievalIndexInconsistentError,
            RetrievalEmbeddingInvalidError,
        ),
    ):
        raise HTTPException(status_code=500, detail={"code": error.code}) from error
    if isinstance(
        error,
        (RetrievalEmbeddingUnavailableError, RetrievalQdrantUnavailableError),
    ):
        raise HTTPException(status_code=503, detail={"code": error.code}) from error
    raise error


def _upload_document_name(document_name: str | None, filename: str) -> str:
    if document_name is not None and document_name.strip():
        return document_name
    leaf = PurePosixPath(filename.replace("\\", "/")).name
    return Path(leaf).stem


async def _read_bounded_upload(file: UploadFile, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = await file.read(_UPLOAD_READ_CHUNK_BYTES)
        if not chunk:
            return b"".join(chunks)
        size += len(chunk)
        if size > max_bytes:
            raise HTTPException(status_code=413, detail={"code": "UPLOAD_TOO_LARGE"})
        chunks.append(chunk)


async def _read_markdown_upload(file: UploadFile) -> bytes:
    return await _read_bounded_upload(file, MAX_MARKDOWN_UPLOAD_BYTES)


def _raise_upload_error(error: Exception) -> None:
    if isinstance(error, KnowledgeBaseNotFoundError):
        raise HTTPException(
            status_code=404, detail={"code": "KNOWLEDGE_BASE_NOT_FOUND"}
        ) from error
    if isinstance(error, DocumentAdmissionConflictError):
        raise HTTPException(status_code=409, detail={"code": str(error)}) from error
    if isinstance(error, UnicodeDecodeError):
        raise HTTPException(
            status_code=422, detail={"code": "INVALID_SOURCE_ENCODING"}
        ) from error
    if isinstance(error, ValueError):
        if str(error) == "source_bytes must not be empty":
            raise HTTPException(
                status_code=422, detail={"code": "EMPTY_SOURCE"}
            ) from error
        if str(error) == "unsupported source_media_type":
            raise HTTPException(
                status_code=415, detail={"code": "UNSUPPORTED_MEDIA_TYPE"}
            ) from error
        raise _validation_error() from error
    if isinstance(error, OSError):
        raise HTTPException(
            status_code=500, detail={"code": "SOURCE_STORAGE_FAILED"}
        ) from error
    raise HTTPException(
        status_code=500, detail={"code": "KNOWLEDGE_ADMISSION_FAILED"}
    ) from error


def _raise_verify_error(error: Exception) -> None:
    if isinstance(error, DocumentVersionNotFoundError):
        raise HTTPException(
            status_code=404, detail={"code": "DOCUMENT_VERSION_NOT_FOUND"}
        ) from error
    if isinstance(error, SourceIntegrityError):
        code = (
            "SOURCE_MISSING"
            if isinstance(error.__cause__, FileNotFoundError)
            else "SOURCE_INTEGRITY_MISMATCH"
        )
        raise HTTPException(status_code=500, detail={"code": code}) from error
    if isinstance(error, OSError):
        raise HTTPException(
            status_code=500, detail={"code": "SOURCE_STORAGE_FAILED"}
        ) from error
    raise error


@router.get("/api/knowledge-bases", response_model=list[KnowledgeBaseResponse])
async def get_knowledge_bases(
    session: AsyncSession = Depends(get_session),
    current_user_id: int = Depends(get_current_user_id),
) -> list[KnowledgeBaseResponse]:
    values = await list_knowledge_bases(session, user_id=current_user_id)
    return [_knowledge_base_response(value) for value in values]


@router.post(
    "/api/knowledge-bases",
    response_model=KnowledgeBaseResponse,
    status_code=status.HTTP_201_CREATED,
)
async def post_knowledge_base(
    body: KnowledgeBaseCreateRequest,
    session: AsyncSession = Depends(get_session),
    current_user_id: int = Depends(get_current_user_id),
) -> KnowledgeBaseResponse:
    try:
        value = await create_knowledge_base(
            session, user_id=current_user_id, name=body.name
        )
    except ValueError as error:
        raise _validation_error() from error
    return KnowledgeBaseResponse(
        id=value.id, name=value.name, created_at=as_utc(value.created_at)
    )


@router.get(
    "/api/knowledge-bases/{knowledge_base_id}/documents",
    response_model=list[DocumentResponse],
)
async def get_documents(
    knowledge_base_id: int,
    session: AsyncSession = Depends(get_session),
    current_user_id: int = Depends(get_current_user_id),
) -> list[DocumentResponse]:
    values = await list_documents_for_knowledge_base(
        session, user_id=current_user_id, knowledge_base_id=knowledge_base_id
    )
    if values is None:
        raise HTTPException(
            status_code=404, detail={"code": "KNOWLEDGE_BASE_NOT_FOUND"}
        )
    return [_document_response(value) for value in values]


@router.post(
    "/api/knowledge-bases/{knowledge_base_id}/documents",
    response_model=DocumentResponse | PdfDocumentAcceptedResponse,
    status_code=status.HTTP_201_CREATED,
)
async def post_document(
    knowledge_base_id: int,
    response: Response,
    file: UploadFile = File(...),
    document_name: str | None = Form(default=None),
    session_factory: async_sessionmaker[AsyncSession] = Depends(get_session_factory),
    file_storage: LocalFileStorage = Depends(get_local_file_storage),
    document_processing_dispatcher: DocumentProcessingDispatcher = Depends(
        get_document_processing_dispatcher
    ),
    current_user_id: int = Depends(get_current_user_id),
) -> DocumentResponse | PdfDocumentAcceptedResponse:
    filename = PurePosixPath((file.filename or "").replace("\\", "/")).name
    pdf_upload = Path(filename).suffix.lower() == ".pdf"
    if pdf_upload and file.content_type != "application/pdf":
        raise HTTPException(status_code=415, detail={"code": "UNSUPPORTED_MEDIA_TYPE"})
    source_bytes = await _read_bounded_upload(
        file, MAX_PDF_UPLOAD_BYTES if pdf_upload else MAX_MARKDOWN_UPLOAD_BYTES
    )
    name = _upload_document_name(document_name, filename)
    if (
        not name.strip()
        or not filename.strip()
        or len(name) > _MAX_DOCUMENT_TEXT_LENGTH
        or len(filename) > _MAX_DOCUMENT_TEXT_LENGTH
    ):
        raise _validation_error()
    try:
        if pdf_upload:
            admission = await create_initial_pdf_document(
                session_factory,
                file_storage,
                user_id=current_user_id,
                knowledge_base_id=knowledge_base_id,
                name=name,
                source_filename=filename,
                source_media_type="application/pdf",
                source_bytes=source_bytes,
            )
            document_processing_dispatcher.wake()
            response.status_code = status.HTTP_202_ACCEPTED
            version = admission.version
            return PdfDocumentAcceptedResponse(
                id=version.document_id,
                name=name,
                created_at=as_utc(version.created_at),
                source=DocumentSourceResponse(
                    document_version_id=version.id,
                    filename=version.source_filename,
                    media_type=version.source_media_type,
                    size_bytes=version.source_size_bytes,
                    sha256=version.source_sha256,
                    created_at=as_utc(version.created_at),
                ),
                processing_job=DocumentProcessingAcceptedResponse(
                    job_id=admission.processing.job_id,
                    attempt_no=admission.processing.attempt_no,
                    status="PENDING",
                    recipe_id=PDF_PROCESSING_RECIPE_ID,
                ),
            )
        version = await create_initial_document(
            session_factory,
            file_storage,
            user_id=current_user_id,
            knowledge_base_id=knowledge_base_id,
            name=name,
            source_filename=filename,
            source_media_type="text/markdown",
            source_bytes=source_bytes,
        )
    except Exception as error:
        _raise_upload_error(error)
    return DocumentResponse(
        id=version.document_id,
        name=name,
        created_at=as_utc(version.created_at),
        source=DocumentSourceResponse(
            document_version_id=version.id,
            filename=version.source_filename,
            media_type=version.source_media_type,
            size_bytes=version.source_size_bytes,
            sha256=version.source_sha256,
            created_at=as_utc(version.created_at),
        ),
    )


@router.post(
    "/api/document-versions/{document_version_id}/verify-source",
    response_model=VerifySourceResponse,
)
async def verify_source(
    document_version_id: int,
    session_factory: async_sessionmaker[AsyncSession] = Depends(get_session_factory),
    file_storage: LocalFileStorage = Depends(get_local_file_storage),
    current_user_id: int = Depends(get_current_user_id),
) -> VerifySourceResponse:
    try:
        source_ref = await load_document_source_ref(
            session_factory,
            user_id=current_user_id,
            document_version_id=document_version_id,
        )
        await read_verified_source(file_storage, source_ref)
    except Exception as error:
        _raise_verify_error(error)
    return VerifySourceResponse(
        document_version_id=document_version_id,
        verified=True,
        verified_at=as_utc(utc_now()),
    )


@router.get(
    "/api/document-versions/{document_version_id}/processing-status",
    response_model=DocumentProcessingStatusResponse,
)
async def get_document_processing_status(
    document_version_id: int,
    session: AsyncSession = Depends(get_session),
    current_user_id: int = Depends(get_current_user_id),
) -> DocumentProcessingStatusResponse:
    value = await read_document_processing_status(
        session,
        user_id=current_user_id,
        document_version_id=document_version_id,
    )
    if value is None:
        raise HTTPException(
            status_code=404, detail={"code": "DOCUMENT_VERSION_NOT_FOUND"}
        )
    return _document_processing_status_response(value)


@router.get(
    "/api/document-versions/{document_version_id}/chunks",
    response_model=DocumentVersionChunksResponse,
)
async def get_document_version_chunks(
    document_version_id: int,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, gt=0, le=_MAX_CHUNKS_PAGE_SIZE),
    session: AsyncSession = Depends(get_session),
    current_user_id: int = Depends(get_current_user_id),
) -> DocumentVersionChunksResponse:
    value = await read_document_version_chunks(
        session,
        user_id=current_user_id,
        document_version_id=document_version_id,
        offset=offset,
        limit=limit,
    )
    if value is None:
        raise HTTPException(
            status_code=404, detail={"code": "DOCUMENT_VERSION_NOT_FOUND"}
        )
    return _document_version_chunks_response(value, offset=offset, limit=limit)


@router.post(
    "/api/document-versions/{document_version_id}/chunks/rebuild",
    response_model=ChunkRebuildResponse,
)
async def post_document_version_chunks_rebuild(
    document_version_id: int,
    body: ChunkRebuildRequest,
    session_factory: async_sessionmaker[AsyncSession] = Depends(get_session_factory),
    file_storage: LocalFileStorage = Depends(get_local_file_storage),
    document_index_dispatcher: DocumentIndexDispatcher = Depends(
        get_document_index_dispatcher
    ),
    current_user_id: int = Depends(get_current_user_id),
) -> ChunkRebuildResponse:
    try:
        result = await rebuild_document_version_chunks(
            session_factory=session_factory,
            file_storage=file_storage,
            user_id=current_user_id,
            document_version_id=document_version_id,
            config=ChunkingConfig(max_chunk_chars=body.max_chunk_chars),
            index_configuration=document_index_dispatcher.configuration,
        )
    except DocumentVersionNotFoundError as error:
        raise HTTPException(
            status_code=404, detail={"code": "DOCUMENT_VERSION_NOT_FOUND"}
        ) from error
    except DocumentRebuildConflictError as error:
        raise HTTPException(status_code=409, detail={"code": str(error)}) from error
    except (SourceIntegrityError, OSError) as error:
        _raise_verify_error(error)
    if result.index_job_created:
        document_index_dispatcher.wake()
    return ChunkRebuildResponse(
        document_version_id=result.document_version_id,
        successful_chunk_max_chars=body.max_chunk_chars,
        chunk_count=result.chunk_count,
        resulting_index_status=result.index_status,
    )


@router.post(
    "/api/knowledge-bases/{knowledge_base_id}/index-build",
    response_model=IndexBuildAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def post_index_build(
    knowledge_base_id: int,
    session_factory: async_sessionmaker[AsyncSession] = Depends(get_session_factory),
    current_user_id: int = Depends(get_current_user_id),
    runtime: KnowledgeIndexBuildRuntime = Depends(get_knowledge_index_runtime),
) -> IndexBuildAcceptedResponse:
    try:
        admitted = await admit_index_build(
            session_factory,
            user_id=current_user_id,
            knowledge_base_id=knowledge_base_id,
            settings=runtime.settings,
        )
    except IndexBuildAdmissionError as error:
        if error.code == "KNOWLEDGE_BASE_NOT_FOUND":
            raise HTTPException(status_code=404, detail={"code": error.code}) from error
        if error.code == "INDEX_BUILD_IN_PROGRESS":
            raise HTTPException(status_code=409, detail={"code": error.code}) from error
        raise HTTPException(status_code=409, detail={"code": error.code}) from error
    runtime.schedule(admitted.job_id)
    return IndexBuildAcceptedResponse(job_id=admitted.job_id)


@router.post(
    "/api/knowledge-bases/{knowledge_base_id}/retrieval",
    response_model=RetrievalResponse,
)
async def post_knowledge_base_retrieval(
    knowledge_base_id: int,
    body: RetrievalRequest,
    session_factory: async_sessionmaker[AsyncSession] = Depends(get_session_factory),
    current_user_id: int = Depends(get_current_user_id),
    runtime: KnowledgeIndexBuildRuntime = Depends(get_knowledge_index_runtime),
) -> RetrievalResponse:
    if not body.query.strip():
        raise _validation_error()
    try:
        result: RetrievalResult = await retrieve_dense(
            session_factory,
            runtime,
            user_id=current_user_id,
            knowledge_base_id=knowledge_base_id,
            query=body.query,
            top_k=body.top_k,
        )
    except RetrievalError as error:
        _raise_retrieval_error(error)
    return RetrievalResponse(
        knowledge_base_id=result.knowledge_base_id,
        hits=[_retrieval_hit_response(hit) for hit in result.hits],
    )


@router.get(
    "/api/knowledge-bases/{knowledge_base_id}/index-status",
    response_model=IndexStatusResponse,
)
async def get_index_status(
    knowledge_base_id: int,
    session: AsyncSession = Depends(get_session),
    current_user_id: int = Depends(get_current_user_id),
) -> IndexStatusResponse:
    value = await read_index_status(
        session, user_id=current_user_id, knowledge_base_id=knowledge_base_id
    )
    if value is None:
        raise HTTPException(
            status_code=404, detail={"code": "KNOWLEDGE_BASE_NOT_FOUND"}
        )
    return _index_status_response(value)
