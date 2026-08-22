"""Minimal owned Knowledge management HTTP resources."""

from pathlib import Path, PurePosixPath

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from langley.api.dependencies import (
    get_current_user_id,
    get_knowledge_index_runtime,
    get_local_file_storage,
    get_session,
    get_session_factory,
)
from langley.api.responses import as_utc
from langley.business_time import utc_now
from langley.infrastructure.local_file_storage import LocalFileStorage
from langley.knowledge.commands import (
    DocumentVersionNotFoundError,
    KnowledgeBaseNotFoundError,
    SourceIntegrityError,
    create_initial_document,
    create_knowledge_base,
    load_document_source_ref,
    read_verified_source,
)
from langley.knowledge.index_build import (
    IndexBuildAdmissionError,
    IndexJobRead,
    IndexStatusRead,
    KnowledgeIndexBuildRuntime,
    admit_index_build,
    read_index_status,
)
from langley.knowledge.reads import (
    DocumentRead,
    DocumentSourceRead,
    KnowledgeBaseRead,
    list_documents_for_knowledge_base,
    list_knowledge_bases,
)

router = APIRouter(tags=["knowledge"])

MAX_MARKDOWN_UPLOAD_BYTES = 5 * 1024 * 1024
_UPLOAD_READ_CHUNK_BYTES = 64 * 1024
_MAX_DOCUMENT_TEXT_LENGTH = 255


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


def _upload_document_name(document_name: str | None, filename: str) -> str:
    if document_name is not None and document_name.strip():
        return document_name
    leaf = PurePosixPath(filename.replace("\\", "/")).name
    return Path(leaf).stem


async def _read_markdown_upload(file: UploadFile) -> bytes:
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = await file.read(_UPLOAD_READ_CHUNK_BYTES)
        if not chunk:
            return b"".join(chunks)
        size += len(chunk)
        if size > MAX_MARKDOWN_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail={"code": "UPLOAD_TOO_LARGE"})
        chunks.append(chunk)


def _raise_upload_error(error: Exception) -> None:
    if isinstance(error, KnowledgeBaseNotFoundError):
        raise HTTPException(
            status_code=404, detail={"code": "KNOWLEDGE_BASE_NOT_FOUND"}
        ) from error
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
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def post_document(
    knowledge_base_id: int,
    file: UploadFile = File(...),
    document_name: str | None = Form(default=None),
    session_factory: async_sessionmaker[AsyncSession] = Depends(get_session_factory),
    file_storage: LocalFileStorage = Depends(get_local_file_storage),
    current_user_id: int = Depends(get_current_user_id),
) -> DocumentResponse:
    source_bytes = await _read_markdown_upload(file)
    filename = PurePosixPath((file.filename or "").replace("\\", "/")).name
    name = _upload_document_name(document_name, filename)
    if (
        not name.strip()
        or not filename.strip()
        or len(name) > _MAX_DOCUMENT_TEXT_LENGTH
        or len(filename) > _MAX_DOCUMENT_TEXT_LENGTH
    ):
        raise _validation_error()
    try:
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
