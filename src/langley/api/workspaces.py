"""Managed-copy creation and small owned Workspace read APIs."""

import os
import shutil
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from starlette.datastructures import UploadFile

from langley.api.dependencies import (
    get_current_user_id,
    get_session_factory,
    get_settings,
    get_workspace_storage,
)
from langley.api.responses import as_utc
from langley.business_time import utc_now
from langley.infrastructure.models import Conversation, Workspace
from langley.settings import Settings
from langley.workspace_storage import (
    WorkspaceFileError,
    WorkspaceStorage,
    import_exclusion,
)

router = APIRouter(prefix="/api/workspaces", tags=["workspaces"])


class CreateWorkspaceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=255)


def _response(workspace: Workspace) -> dict:
    return {
        "id": workspace.id,
        "name": workspace.name,
        "created_at": as_utc(workspace.created_at),
    }


async def _publish(
    factory: async_sessionmaker[AsyncSession],
    storage: WorkspaceStorage,
    staging_key: str,
    user_id: int,
    name: str,
    skipped: dict,
) -> dict:
    if not name.strip() or len(name) > 255:
        raise WorkspaceFileError("INVALID_NAME")
    key = uuid4().hex
    root = storage.workspace_root(key)
    # Files are fully staged before opening a DB transaction. Rename is local/short.
    os.rename(storage.workspace_root(staging_key), root)
    commit_started = False
    try:
        async with factory() as session, session.begin():
            now = utc_now()
            workspace = Workspace(
                user_id=user_id, name=name.strip(), storage_key=key, created_at=now
            )
            session.add(workspace)
            await session.flush()
            conversation = Conversation(
                user_id=user_id,
                workspace_id=workspace.id,
                title=None,
                created_at=now,
                updated_at=now,
            )
            session.add(conversation)
            await session.flush()
            result = {
                **_response(workspace),
                "conversation_id": conversation.id,
                "skipped": skipped,
            }
            # The transaction context commits on exit. Once it starts, a lost
            # acknowledgement cannot establish that the rows were not published.
            commit_started = True
        return result
    except BaseException:
        if not commit_started:
            shutil.rmtree(root)
        raise


@router.post("", status_code=201)
async def create_workspace(
    body: CreateWorkspaceRequest,
    user_id: int = Depends(get_current_user_id),
    factory=Depends(get_session_factory),
    storage: WorkspaceStorage = Depends(get_workspace_storage),
) -> dict:
    key = uuid4().hex
    staging = storage.workspace_root(key)
    staging.mkdir(parents=True)
    try:
        return await _publish(factory, storage, key, user_id, body.name, {})
    except WorkspaceFileError as error:
        raise HTTPException(422, detail={"code": error.code}) from error
    finally:
        if staging.exists():
            shutil.rmtree(staging)


@router.post("/import", status_code=201)
async def import_workspace(
    request: Request,
    user_id: int = Depends(get_current_user_id),
    factory=Depends(get_session_factory),
    settings: Settings = Depends(get_settings),
    storage: WorkspaceStorage = Depends(get_workspace_storage),
) -> dict:
    key = uuid4().hex
    staging = storage.workspace_root(key)
    staging.mkdir(parents=True)
    total, count = 0, 0
    skipped: dict[str, int] = {}
    seen: set[str] = set()
    try:
        # Limit streamed multipart bytes BEFORE the parser can spool arbitrary uploads.
        # Starlette's max_part_size bounds fields, not uploaded file bytes.
        original_receive = request._receive
        received = 0

        async def bounded_receive():
            nonlocal received
            message = await original_receive()
            received += len(message.get("body", b""))
            if (
                received
                > settings.workspace_max_import_total_bytes
                + settings.workspace_max_import_files * 4096
            ):
                raise WorkspaceFileError("IMPORT_TOO_LARGE")
            return message

        request._receive = bounded_receive
        async with request.form(
            max_files=settings.workspace_max_import_files, max_fields=1
        ) as form:
            name = str(form.get("name", "Imported Workspace"))
            for field, upload in form.multi_items():
                if field == "name":
                    continue
                # Starlette UploadFile is the actual multipart parser type.
                if not isinstance(upload, UploadFile) or field != "files":
                    raise WorkspaceFileError("INVALID_IMPORT")
                path = upload.filename or ""
                target = storage.resolve(key, path)
                if not path or path.casefold() in seen:
                    raise WorkspaceFileError("INVALID_IMPORT")
                seen.add(path.casefold())
                count += 1
                if count > settings.workspace_max_import_files:
                    raise WorkspaceFileError("IMPORT_TOO_LARGE")
                reason = import_exclusion(path)
                if reason:
                    skipped[reason] = skipped.get(reason, 0) + 1
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                size = 0
                with target.open("xb") as stream:
                    while chunk := await upload.read(65536):
                        size += len(chunk)
                        total += len(chunk)
                        if (
                            size > settings.workspace_max_import_file_bytes
                            or total > settings.workspace_max_import_total_bytes
                        ):
                            raise WorkspaceFileError("IMPORT_TOO_LARGE")
                        stream.write(chunk)
            return await _publish(factory, storage, key, user_id, name, skipped)
    except (WorkspaceFileError, FileExistsError, NotADirectoryError) as error:
        code = error.code if isinstance(error, WorkspaceFileError) else "INVALID_IMPORT"
        raise HTTPException(422, detail={"code": code}) from error
    finally:
        if staging.exists():
            shutil.rmtree(staging)


@router.get("")
async def list_workspaces(
    user_id: int = Depends(get_current_user_id), factory=Depends(get_session_factory)
) -> list[dict]:
    async with factory() as session:
        rows = (
            await session.scalars(
                select(Workspace)
                .where(Workspace.user_id == user_id)
                .order_by(Workspace.id.desc())
            )
        ).all()
        return [_response(row) for row in rows]


@router.get("/{workspace_id}/files")
async def workspace_files(
    workspace_id: int,
    path: str = "",
    preview: bool = False,
    offset: int = Query(default=1, ge=1),
    limit: int = Query(default=200, ge=1),
    user_id: int = Depends(get_current_user_id),
    factory=Depends(get_session_factory),
    storage: WorkspaceStorage = Depends(get_workspace_storage),
) -> dict:
    async with factory() as session:
        workspace = await session.get(Workspace, workspace_id)
        if workspace is None or workspace.user_id != user_id:
            raise HTTPException(404, detail={"code": "WORKSPACE_NOT_FOUND"})
        key = workspace.storage_key
    try:
        async with storage.execution_lock(key):
            return (
                storage.read_file(key, path, offset, limit)
                if preview
                else storage.list_files(key, path)
            )
    except WorkspaceFileError as error:
        raise HTTPException(422, detail={"code": error.code}) from error
