"""Minimal owned Personal Memory HTTP resources and live outcome SSE."""

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from langley.api.dependencies import (
    get_current_user_id,
    get_memory_lane,
    get_memory_policy,
    get_memory_policy_status,
    get_memory_subscribers,
    get_session,
    get_session_factory,
    get_settings,
)
from langley.api.responses import as_optional_utc, as_utc, message_response
from langley.business_time import normalize_aware_datetime_to_utc_naive, utc_now
from langley.infrastructure.models import Memory, User
from langley.memory.events import MemoryEventSubscribers, MemoryOutcome
from langley.memory.policy import MemoryPolicy, MemoryPolicyStatus
from langley.memory.processing import (
    MemoryCapacityReachedError,
    MemoryProcessingStopReason,
    MemorySynchronizationUnavailableError,
    add_memory_direct,
    correct_memory_direct,
    forget_memory_direct,
    get_pending_memory_evidence_status,
    set_auto_memory_enabled,
    synchronize_memory,
)
from langley.memory.reads import (
    get_memory_source_context,
    get_owned_current_memory,
    list_current_memories,
)
from langley.settings import Settings

router = APIRouter(tags=["memories"])


class MemoryWriteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1, max_length=1000)
    valid_until: datetime | None = None


class MemorySettingsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    auto_memory_enabled: bool


class MemoryResponse(BaseModel):
    id: int
    content: str
    valid_until: str | None
    source_message_id: int | None
    created_at: str
    updated_at: str


class MemorySettingsResponse(BaseModel):
    auto_memory_enabled: bool


class MemoryOperationalStatusResponse(BaseModel):
    auto_memory_enabled: bool
    policy_status: MemoryPolicyStatus
    pending_evidence_count: int
    oldest_pending_message_id: int | None
    oldest_pending_created_at: str | None


class MemorySyncResponse(BaseModel):
    processed_count: int
    remaining_count: int
    complete: bool
    stop_reason: MemoryProcessingStopReason
    oldest_pending_message_id: int | None
    oldest_pending_created_at: str | None


class DirectMemorySourceResponse(BaseModel):
    kind: Literal["direct"] = "direct"


class ConversationMemorySourceResponse(BaseModel):
    kind: Literal["conversation"] = "conversation"
    conversation_id: int
    conversation_title: str | None
    conversation_deleted: bool
    context_messages: list[dict[str, object]]


def _memory_response(memory: Memory) -> MemoryResponse:
    return MemoryResponse(
        id=memory.id,
        content=memory.content,
        valid_until=as_optional_utc(memory.valid_until),
        source_message_id=memory.source_message_id,
        created_at=as_utc(memory.created_at),
        updated_at=as_utc(memory.updated_at),
    )


def _normalized_valid_until(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    try:
        normalized = normalize_aware_datetime_to_utc_naive(value)
    except ValueError as error:
        raise _validation_error() from error
    if normalized <= utc_now():
        raise _validation_error()
    return normalized


def _validation_error() -> HTTPException:
    return HTTPException(status_code=422, detail={"code": "VALIDATION_ERROR"})


def _raise_memory_error(error: Exception) -> None:
    if isinstance(error, MemoryCapacityReachedError):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "MEMORY_CAPACITY_REACHED"},
        ) from error
    if isinstance(error, MemorySynchronizationUnavailableError):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "MEMORY_SYNC_UNAVAILABLE"},
        ) from error
    if isinstance(error, KeyError):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "MEMORY_NOT_FOUND"},
        ) from error
    if isinstance(error, ValueError):
        raise _validation_error() from error
    raise error


@router.get("/api/memories", response_model=list[MemoryResponse])
async def get_memories(
    session: AsyncSession = Depends(get_session),
    current_user_id: int = Depends(get_current_user_id),
) -> list[MemoryResponse]:
    memories = await list_current_memories(
        session, user_id=current_user_id, now=utc_now()
    )
    return [_memory_response(memory) for memory in memories]


@router.post("/api/memories", response_model=MemoryResponse, status_code=201)
async def post_memory(
    body: MemoryWriteRequest,
    settings: Settings = Depends(get_settings),
    session_factory=Depends(get_session_factory),
    policy: MemoryPolicy | None = Depends(get_memory_policy),
    lane: asyncio.Lock = Depends(get_memory_lane),
    current_user_id: int = Depends(get_current_user_id),
) -> MemoryResponse:
    if policy is None:
        raise HTTPException(status_code=503, detail={"code": "MEMORY_SYNC_UNAVAILABLE"})
    try:
        memory = await add_memory_direct(
            session_factory,
            user_id=current_user_id,
            content=body.content,
            valid_until=_normalized_valid_until(body.valid_until),
            policy=policy,
            local_timezone=settings.local_timezone,
            lane=lane,
        )
    except Exception as error:
        _raise_memory_error(error)
    return _memory_response(memory)


@router.put("/api/memories/{memory_id}", response_model=MemoryResponse)
async def put_memory(
    memory_id: int,
    body: MemoryWriteRequest,
    settings: Settings = Depends(get_settings),
    session_factory=Depends(get_session_factory),
    policy: MemoryPolicy | None = Depends(get_memory_policy),
    lane: asyncio.Lock = Depends(get_memory_lane),
    current_user_id: int = Depends(get_current_user_id),
) -> MemoryResponse:
    if policy is None:
        raise HTTPException(status_code=503, detail={"code": "MEMORY_SYNC_UNAVAILABLE"})
    try:
        memory = await correct_memory_direct(
            session_factory,
            user_id=current_user_id,
            memory_id=memory_id,
            content=body.content,
            valid_until=_normalized_valid_until(body.valid_until),
            policy=policy,
            local_timezone=settings.local_timezone,
            lane=lane,
        )
    except Exception as error:
        _raise_memory_error(error)
    return _memory_response(memory)


@router.delete("/api/memories/{memory_id}", status_code=204)
async def delete_memory(
    memory_id: int,
    settings: Settings = Depends(get_settings),
    session_factory=Depends(get_session_factory),
    policy: MemoryPolicy | None = Depends(get_memory_policy),
    lane: asyncio.Lock = Depends(get_memory_lane),
    current_user_id: int = Depends(get_current_user_id),
) -> Response:
    async with session_factory() as session:
        existing = await get_owned_current_memory(
            session, user_id=current_user_id, memory_id=memory_id, now=utc_now()
        )
    if existing is None:
        return Response(status_code=204)
    if policy is None:
        raise HTTPException(status_code=503, detail={"code": "MEMORY_SYNC_UNAVAILABLE"})
    try:
        await forget_memory_direct(
            session_factory,
            user_id=current_user_id,
            memory_id=memory_id,
            policy=policy,
            local_timezone=settings.local_timezone,
            lane=lane,
        )
    except KeyError:
        return Response(status_code=204)
    except Exception as error:
        _raise_memory_error(error)
    return Response(status_code=204)


@router.get("/api/memories/{memory_id}/source")
async def get_memory_source(
    memory_id: int,
    session: AsyncSession = Depends(get_session),
    current_user_id: int = Depends(get_current_user_id),
) -> DirectMemorySourceResponse | ConversationMemorySourceResponse:
    memory = await get_owned_current_memory(
        session, user_id=current_user_id, memory_id=memory_id, now=utc_now()
    )
    if memory is None:
        raise HTTPException(status_code=404, detail={"code": "MEMORY_NOT_FOUND"})
    if memory.source_message_id is None:
        return DirectMemorySourceResponse()
    source = await get_memory_source_context(
        session, user_id=current_user_id, memory=memory
    )
    if source is None:
        raise HTTPException(status_code=404, detail={"code": "MEMORY_NOT_FOUND"})
    return ConversationMemorySourceResponse(
        conversation_id=source.conversation.id,
        conversation_title=source.conversation.title,
        conversation_deleted=source.conversation.deleted_at is not None,
        context_messages=[
            message_response(message).model_dump()
            for message in source.context_messages
        ],
    )


@router.get("/api/memory-settings", response_model=MemorySettingsResponse)
async def get_memory_settings(
    session: AsyncSession = Depends(get_session),
    current_user_id: int = Depends(get_current_user_id),
) -> MemorySettingsResponse:
    user = await session.get(User, current_user_id)
    assert user is not None
    return MemorySettingsResponse(auto_memory_enabled=user.auto_memory_enabled)


@router.get("/api/memory-status", response_model=MemoryOperationalStatusResponse)
async def get_memory_status(
    session: AsyncSession = Depends(get_session),
    policy_status: MemoryPolicyStatus = Depends(get_memory_policy_status),
    current_user_id: int = Depends(get_current_user_id),
) -> MemoryOperationalStatusResponse:
    user = await session.get(User, current_user_id)
    assert user is not None
    pending = await get_pending_memory_evidence_status(session, user_id=current_user_id)
    return MemoryOperationalStatusResponse(
        auto_memory_enabled=user.auto_memory_enabled,
        policy_status=policy_status,
        pending_evidence_count=pending.pending_evidence_count,
        oldest_pending_message_id=pending.oldest_pending_message_id,
        oldest_pending_created_at=as_optional_utc(pending.oldest_pending_created_at),
    )


@router.post("/api/memory-sync", response_model=MemorySyncResponse)
async def post_memory_sync(
    settings: Settings = Depends(get_settings),
    session_factory=Depends(get_session_factory),
    policy: MemoryPolicy | None = Depends(get_memory_policy),
    policy_status: MemoryPolicyStatus = Depends(get_memory_policy_status),
    lane: asyncio.Lock = Depends(get_memory_lane),
    subscribers: MemoryEventSubscribers = Depends(get_memory_subscribers),
    current_user_id: int = Depends(get_current_user_id),
) -> MemorySyncResponse:
    sync = await synchronize_memory(
        session_factory,
        user_id=current_user_id,
        policy=policy if policy_status is MemoryPolicyStatus.READY else None,
        local_timezone=settings.local_timezone,
        lane=lane,
        outcome_callback=subscribers.publish,
    )
    response = MemorySyncResponse(
        processed_count=sync.processing.processed_count,
        remaining_count=sync.pending.pending_evidence_count,
        complete=sync.processing.complete,
        stop_reason=sync.processing.stop_reason,
        oldest_pending_message_id=sync.pending.oldest_pending_message_id,
        oldest_pending_created_at=as_optional_utc(
            sync.pending.oldest_pending_created_at
        ),
    )
    if response.stop_reason in {
        MemoryProcessingStopReason.COMPLETE,
        MemoryProcessingStopReason.LIMIT_REACHED,
    }:
        return response
    detail = {
        "stop_reason": response.stop_reason.value,
        "processed_count": response.processed_count,
        "remaining_count": response.remaining_count,
    }
    if response.stop_reason is MemoryProcessingStopReason.CONTEXT_INFEASIBLE:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "MEMORY_SYNC_BLOCKED", **detail},
        )
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={"code": "MEMORY_SYNC_UNAVAILABLE", **detail},
    )


@router.patch("/api/memory-settings", response_model=MemorySettingsResponse)
async def patch_memory_settings(
    body: MemorySettingsRequest,
    settings: Settings = Depends(get_settings),
    session_factory=Depends(get_session_factory),
    policy: MemoryPolicy | None = Depends(get_memory_policy),
    lane: asyncio.Lock = Depends(get_memory_lane),
    current_user_id: int = Depends(get_current_user_id),
) -> MemorySettingsResponse:
    try:
        await set_auto_memory_enabled(
            session_factory,
            user_id=current_user_id,
            enabled=body.auto_memory_enabled,
            policy=policy,
            local_timezone=settings.local_timezone,
            lane=lane,
        )
    except Exception as error:
        _raise_memory_error(error)
    return MemorySettingsResponse(auto_memory_enabled=body.auto_memory_enabled)


@router.get("/api/memory-events")
async def get_memory_events(
    subscribers: MemoryEventSubscribers = Depends(get_memory_subscribers),
    current_user_id: int = Depends(get_current_user_id),
) -> StreamingResponse:
    queue = subscribers.subscribe(current_user_id)

    async def stream() -> AsyncIterator[str]:
        try:
            while True:
                outcome = await queue.get()
                yield _encode_outcome(outcome)
        finally:
            subscribers.unsubscribe(current_user_id, queue)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _encode_outcome(outcome: MemoryOutcome) -> str:
    body: dict[str, object] = {
        "conversation_id": outcome.conversation_id,
        "source_message_id": outcome.source_message_id,
    }
    if outcome.kind == "updated":
        body |= {
            "user_requested_memory_action": outcome.user_requested_memory_action,
            "created_count": outcome.created_count,
            "changed_count": outcome.changed_count,
            "forgotten_count": outcome.forgotten_count,
        }
    name = f"memory.{outcome.kind}"
    return f"event: {name}\ndata: {json.dumps(body, separators=(',', ':'))}\n\n"
