"""Owned Run reads and terminal transitions backed solely by MySQL."""

from dataclasses import dataclass

import structlog
from sqlalchemy import select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from langley.business_time import utc_now
from langley.infrastructure.models import Conversation, Message, Run


class RunNotFoundError(Exception):
    """Raised when a Run is missing or is not owned by the current user."""


class RunNotCancellableError(Exception):
    """Raised when an existing Run has already reached another terminal state."""


class RunResultInvariantError(RuntimeError):
    """Raised when a successful Run lacks its required persisted ASSISTANT."""


@dataclass(frozen=True)
class OwnedRunResult:
    """One authoritative Run and its persisted terminal result when present."""

    run: Run
    assistant_message: Message | None


async def get_owned_run(
    session: AsyncSession, *, user_id: int, run_id: int
) -> OwnedRunResult:
    """Read an owned Run without claiming, repairing, or scheduling execution."""

    run = await _get_owned_run(session, user_id=user_id, run_id=run_id)
    if run.status != "SUCCEEDED":
        return OwnedRunResult(run=run, assistant_message=None)
    assistant_message = await session.scalar(
        select(Message)
        .where(Message.run_id == run.id, Message.role == "ASSISTANT")
        .limit(1)
    )
    if assistant_message is None:
        raise RunResultInvariantError("successful run has no persisted assistant")
    return OwnedRunResult(run=run, assistant_message=assistant_message)


async def cancel_owned_run(session: AsyncSession, *, user_id: int, run_id: int) -> Run:
    """Commit an owned active Run's cancellation before any local Task stop."""

    changed = False
    async with session.begin():
        run = await _get_owned_run(session, user_id=user_id, run_id=run_id)
        if run.status == "CANCELLED":
            return run
        if run.status in {"SUCCEEDED", "FAILED"}:
            raise RunNotCancellableError("run already has a non-cancellable outcome")

        now = utc_now()
        result = await session.execute(
            update(Run)
            # A losing conditional UPDATE must not mutate the loaded Run into a
            # terminal state that MySQL did not commit.
            .execution_options(synchronize_session=False)
            .where(
                Run.id == run.id,
                Run.conversation_id == run.conversation_id,
                Run.status.in_(("PENDING", "RUNNING")),
            )
            .values(
                status="CANCELLED",
                finished_at=now,
                error_code=None,
                updated_at=now,
            )
        )
        if not isinstance(result, CursorResult):
            raise RuntimeError("unexpected cancellation update result")
        changed = result.rowcount == 1
        if changed:
            await session.refresh(run)
            cancelled_run = run

    if changed:
        structlog.get_logger(__name__).info(
            "answer.run.cancelled",
            run_id=cancelled_run.id,
            conversation_id=cancelled_run.conversation_id,
        )
        return cancelled_run

    # A concurrent legal terminal commit won the conditional update. Refresh the
    # originally ownership-scoped Run after the transaction, rather than using a
    # potentially stale identity-map value from before the race.
    await session.refresh(run)
    if run.status == "CANCELLED":
        return run
    raise RunNotCancellableError("run already has a non-cancellable outcome")


async def _get_owned_run(session: AsyncSession, *, user_id: int, run_id: int) -> Run:
    """Find one non-deleted user-owned Run through its Conversation."""

    run = await session.scalar(
        select(Run)
        .join(Conversation, Run.conversation_id == Conversation.id)
        .where(
            Run.id == run_id,
            Conversation.user_id == user_id,
            Conversation.deleted_at.is_(None),
        )
    )
    if run is None:
        raise RunNotFoundError("run is missing or not owned by current user")
    return run
