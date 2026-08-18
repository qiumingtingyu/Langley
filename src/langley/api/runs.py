"""HTTP recovery reads and explicit cancellation for owned answer Runs."""

import asyncio
import json
from collections.abc import AsyncIterator

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from langley.answer_execution import AnswerExecutionManager
from langley.answer_runtime import ActiveAnswer, StreamItem
from langley.api.dependencies import (
    get_current_user_id,
    get_execution_manager,
    get_session,
)
from langley.api.responses import (
    MessageResponse,
    RunResponse,
    message_response,
    run_response,
)
from langley.runs import (
    RunNotCancellableError,
    RunNotFoundError,
    cancel_owned_run,
    get_owned_run,
)

router = APIRouter(prefix="/api/runs", tags=["runs"])


class RunQueryResponse(BaseModel):
    """An authoritative Run snapshot and only its durable assistant result."""

    run: RunResponse
    assistant_message: MessageResponse | None


def _raise_run_http_error(error: Exception) -> None:
    if isinstance(error, RunNotFoundError):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "RUN_NOT_FOUND"},
        ) from error
    if isinstance(error, RunNotCancellableError):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "RUN_NOT_CANCELLABLE"},
        ) from error
    raise error


@router.get("/{run_id}", response_model=RunQueryResponse)
async def get_run(
    run_id: int,
    session: AsyncSession = Depends(get_session),
    current_user_id: int = Depends(get_current_user_id),
) -> RunQueryResponse:
    """Return a pure authoritative recovery read for one owned Run."""

    try:
        result = await get_owned_run(session, user_id=current_user_id, run_id=run_id)
    except Exception as error:
        _raise_run_http_error(error)
    return RunQueryResponse(
        run=run_response(result.run),
        assistant_message=(
            message_response(result.assistant_message)
            if result.assistant_message is not None
            else None
        ),
    )


@router.get("/{run_id}/events")
async def get_run_events(
    run_id: int,
    session: AsyncSession = Depends(get_session),
    execution_manager: AnswerExecutionManager = Depends(get_execution_manager),
    current_user_id: int = Depends(get_current_user_id),
) -> StreamingResponse:
    """Observe one owned Run without scheduling, claiming, or cancelling it."""

    try:
        result = await get_owned_run(session, user_id=current_user_id, run_id=run_id)
        terminal_event = _terminal_event(result.run)
    except Exception as error:
        _raise_run_http_error(error)
    finally:
        await session.rollback()
    if terminal_event is not None:
        return _terminal_response(terminal_event)

    attached = execution_manager.subscribe(run_id)
    if attached is not None:
        answer, prefix, queue = attached
        return _active_response(run_id, answer, prefix, queue, execution_manager)

    # A live answer can become terminal between the first authoritative read and
    # the attach handoff.  Re-read rather than leaving an observer hung.
    missing_runtime_fields: tuple[int, int, str] | None = None
    try:
        refreshed = await get_owned_run(session, user_id=current_user_id, run_id=run_id)
        terminal_event = _terminal_event(refreshed.run)
        if terminal_event is None:
            missing_runtime_fields = (
                refreshed.run.id,
                refreshed.run.conversation_id,
                refreshed.run.status,
            )
    except Exception as error:
        _raise_run_http_error(error)
    finally:
        await session.rollback()
    if terminal_event is not None:
        return _terminal_response(terminal_event)
    assert missing_runtime_fields is not None
    missing_run_id, conversation_id, run_status = missing_runtime_fields
    structlog.get_logger(__name__).warning(
        "answer.run.active_answer_missing",
        run_id=missing_run_id,
        conversation_id=conversation_id,
        run_status=run_status,
    )
    return _empty_response()


@router.post("/{run_id}/cancel", response_model=RunResponse)
async def post_cancel_run(
    run_id: int,
    session: AsyncSession = Depends(get_session),
    execution_manager: AnswerExecutionManager = Depends(get_execution_manager),
    current_user_id: int = Depends(get_current_user_id),
) -> RunResponse:
    """Cancel one owned active Run with MySQL as the business authority."""

    try:
        run = await cancel_owned_run(session, user_id=current_user_id, run_id=run_id)
    except Exception as error:
        _raise_run_http_error(error)
    try:
        await execution_manager.stop_cancelled_run(run.id)
    except Exception:
        structlog.get_logger(__name__).exception(
            "answer.run.cancel_cleanup_failed",
            run_id=run.id,
            conversation_id=run.conversation_id,
        )
    return run_response(run)


def _terminal_event(run) -> StreamItem | None:
    """Map only an authoritative terminal Run snapshot to one SSE event."""

    if run.status == "SUCCEEDED":
        return "run.succeeded", {"run_id": run.id}
    if run.status == "FAILED":
        return "run.failed", {"run_id": run.id, "error_code": run.error_code}
    if run.status == "CANCELLED":
        return "run.cancelled", {"run_id": run.id}
    return None


def _terminal_response(event: StreamItem) -> StreamingResponse:
    """Send one terminal notification for an already-finished Run and close."""

    async def stream() -> AsyncIterator[str]:
        yield _encode_event(event)

    return _sse_response(stream())


def _active_response(
    run_id: int,
    answer: ActiveAnswer,
    prefix: str,
    queue: asyncio.Queue[StreamItem | None],
    execution_manager: AnswerExecutionManager,
) -> StreamingResponse:
    """Render a gap-free prefix followed by subscriber-local live delivery."""

    async def stream() -> AsyncIterator[str]:
        try:
            if prefix:
                yield _encode_event(
                    ("message.delta", {"run_id": run_id, "delta": prefix})
                )
            while (event := await queue.get()) is not None:
                yield _encode_event(event)
        finally:
            execution_manager.unsubscribe(answer, queue)

    return _sse_response(stream())


def _empty_response() -> StreamingResponse:
    """Close an observer with no local live runtime after an active re-read."""

    async def stream() -> AsyncIterator[str]:
        return
        yield ""

    return _sse_response(stream())


def _sse_response(stream: AsyncIterator[str]) -> StreamingResponse:
    """Build the minimal transient SSE transport without replay directives."""

    return StreamingResponse(
        stream,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _encode_event(event: StreamItem) -> str:
    """Encode one event with standard SSE framing and a single JSON payload."""

    name, body = event
    payload = json.dumps(body, ensure_ascii=False, separators=(",", ":"))
    return f"event: {name}\ndata: {payload}\n\n"
