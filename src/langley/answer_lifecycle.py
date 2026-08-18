"""Single-worker answer admission and process-interruption lifecycle handling."""

import structlog
from sqlalchemy import select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from langley.business_time import utc_now
from langley.infrastructure.models import Run


async def interrupt_active_runs(
    session_factory: async_sessionmaker[AsyncSession],
) -> list[int]:
    """Conditionally terminalize every authoritative active Run after interruption."""

    repaired_runs: list[tuple[int, int]] = []
    now = utc_now()
    async with session_factory() as session:
        async with session.begin():
            active_runs = list(
                (
                    await session.execute(
                        select(Run.id, Run.conversation_id).where(
                            Run.status.in_(("PENDING", "RUNNING"))
                        )
                    )
                ).all()
            )
            for run_id, conversation_id in active_runs:
                result = await session.execute(
                    update(Run)
                    .where(
                        Run.id == run_id,
                        Run.status.in_(("PENDING", "RUNNING")),
                    )
                    .values(
                        status="FAILED",
                        finished_at=now,
                        error_code="PROCESS_INTERRUPTED",
                        updated_at=now,
                    )
                )
                if not isinstance(result, CursorResult):
                    raise RuntimeError("unexpected interruption repair update result")
                if result.rowcount == 1:
                    repaired_runs.append((run_id, conversation_id))

    logger = structlog.get_logger(__name__)
    for run_id, conversation_id in repaired_runs:
        logger.warning(
            "answer.run.interrupted",
            run_id=run_id,
            conversation_id=conversation_id,
            error_code="PROCESS_INTERRUPTED",
        )
    return [run_id for run_id, _ in repaired_runs]
