"""Short-transaction execution of an already-admitted answer Run."""

import asyncio
from typing import Any, Callable, Coroutine

import structlog
from sqlalchemy import select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from langley.answer_runtime import ActiveAnswer, StreamItem
from langley.answering.errors import WorkflowFailure
from langley.answering.workflow import LearningAssistantWorkflow
from langley.business_time import utc_now
from langley.conversation_commands import AnswerCommandResult
from langley.infrastructure.models import Conversation, Message, Run


class RunExecutionStateError(RuntimeError):
    """Raised when a newly accepted Run can no longer make its expected transition."""


TaskScheduler = Callable[[Coroutine[Any, Any, None]], asyncio.Task[None]]


class AnswerExecutionManager:
    """Own process-local Tasks while MySQL remains the Run authority."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        workflow_factory: Callable[[], LearningAssistantWorkflow],
        task_scheduler: TaskScheduler = asyncio.create_task,
    ) -> None:
        self._session_factory = session_factory
        self._workflow_factory = workflow_factory
        self._active_answers: dict[int, ActiveAnswer] = {}
        self._task_scheduler = task_scheduler

    async def schedule(self, command: AnswerCommandResult) -> None:
        """Schedule exactly one newly accepted Run without making HTTP its owner."""

        if command.is_replay:
            return

        answer = ActiveAnswer()
        execution: Coroutine[Any, Any, None] | None = None
        try:
            if command.run.id in self._active_answers:
                return
            self._active_answers[command.run.id] = answer
            workflow = self._workflow_factory()
            execution = self._execute(answer, command, workflow)
            task = self._task_scheduler(execution)
            answer.task = task
        except asyncio.CancelledError:
            self._active_answers.pop(command.run.id, None)
            raise
        except Exception:
            if execution is not None:
                execution.close()
            structlog.get_logger(__name__).exception(
                "answer.run.scheduling_failed",
                run_id=command.run.id,
                conversation_id=command.run.conversation_id,
            )
            self._close(
                answer,
                (
                    "run.failed",
                    {"run_id": command.run.id, "error_code": "ANSWER_EXECUTION_FAILED"},
                ),
            )
            self._active_answers.pop(command.run.id, None)
            await _mark_active_execution_failed(
                self._session_factory,
                conversation_id=command.run.conversation_id,
                run_id=command.run.id,
            )

    async def stop_cancelled_run(self, run_id: int) -> None:
        """Best-effort local stop only after MySQL has committed CANCELLED."""

        answer = self._active_answers.get(run_id)
        if answer is None:
            return
        self._close(answer, ("run.cancelled", {"run_id": run_id}))
        task = answer.task
        if task is not None and not task.done():
            task.cancel()

    async def stop_interrupted_runs(self, run_ids: list[int]) -> None:
        """Best-effort local task stop after committed PROCESS_INTERRUPTED facts."""

        for run_id in run_ids:
            answer = self._active_answers.get(run_id)
            if answer is None:
                continue
            self._close(
                answer,
                ("run.failed", {"run_id": run_id, "error_code": "PROCESS_INTERRUPTED"}),
            )
            task = answer.task
            if task is not None and not task.done():
                task.cancel()

    def subscribe(
        self, run_id: int
    ) -> tuple[ActiveAnswer, str, asyncio.Queue[StreamItem | None]] | None:
        """Attach one SSE queue and snapshot the live prefix without a gap."""

        answer = self._active_answers.get(run_id)
        if answer is None or answer.closed:
            return None
        queue: asyncio.Queue[StreamItem | None] = asyncio.Queue()
        answer.streams.add(queue)
        return answer, answer.partial_text, queue

    @staticmethod
    def unsubscribe(
        answer: ActiveAnswer, queue: asyncio.Queue[StreamItem | None]
    ) -> None:
        """Forget an SSE observer without touching answer execution."""

        answer.streams.discard(queue)

    async def _execute(
        self,
        answer: ActiveAnswer,
        command: AnswerCommandResult,
        workflow: LearningAssistantWorkflow,
    ) -> None:
        """Run database phases and slow streaming without holding a DB lock."""

        try:
            await _start_running(
                self._session_factory,
                conversation_id=command.run.conversation_id,
                run_id=command.run.id,
            )
            self._publish(answer, ("run.started", {"run_id": command.run.id}))
            success = await workflow.execute(
                self._session_factory,
                run_id=command.run.id,
                conversation_id=command.run.conversation_id,
                input_message_id=command.user_message.id,
                on_assistant_delta=lambda delta: self._publish_delta(
                    answer, command.run.id, delta
                ),
            )
            await _commit_success(
                self._session_factory,
                conversation_id=command.run.conversation_id,
                run_id=command.run.id,
                content=success,
            )
            self._close(answer, ("run.succeeded", {"run_id": command.run.id}))
        except WorkflowFailure as failure:
            failed = await mark_run_failed_if_running(
                self._session_factory,
                conversation_id=command.run.conversation_id,
                run_id=command.run.id,
                error_code=failure.error_code.value,
            )
            if failed:
                self._close(
                    answer,
                    (
                        "run.failed",
                        {
                            "run_id": command.run.id,
                            "error_code": failure.error_code.value,
                        },
                    ),
                )
        except asyncio.CancelledError:
            raise
        except RunExecutionStateError:
            return
        except Exception:
            structlog.get_logger(__name__).exception(
                "answer.run.execution_failed",
                run_id=command.run.id,
                conversation_id=command.run.conversation_id,
            )
            changed = await _mark_active_execution_failed(
                self._session_factory,
                conversation_id=command.run.conversation_id,
                run_id=command.run.id,
            )
            if changed:
                self._close(
                    answer,
                    (
                        "run.failed",
                        {
                            "run_id": command.run.id,
                            "error_code": "ANSWER_EXECUTION_FAILED",
                        },
                    ),
                )
        finally:
            self._close(answer, None)
            if self._active_answers.get(command.run.id) is answer:
                del self._active_answers[command.run.id]

    async def _publish_delta(
        self, answer: ActiveAnswer, run_id: int, delta: str
    ) -> None:
        if answer.closed:
            return
        answer.partial_text += delta
        self._publish(answer, ("message.delta", {"run_id": run_id, "delta": delta}))

    @staticmethod
    def _publish(answer: ActiveAnswer, item: StreamItem) -> None:
        if not answer.closed:
            for queue in tuple(answer.streams):
                queue.put_nowait(item)

    @staticmethod
    def _close(answer: ActiveAnswer, item: StreamItem | None) -> None:
        if answer.closed:
            return
        answer.closed = True
        for queue in tuple(answer.streams):
            if item is not None:
                queue.put_nowait(item)
            queue.put_nowait(None)


async def _start_running(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    conversation_id: int,
    run_id: int,
) -> None:
    """Commit the short conditional PENDING to RUNNING transition."""

    now = utc_now()
    async with session_factory() as session:
        async with session.begin():
            result = await session.execute(
                update(Run)
                .where(
                    Run.id == run_id,
                    Run.conversation_id == conversation_id,
                    Run.status == "PENDING",
                )
                .values(status="RUNNING", started_at=now, updated_at=now)
            )
            if not isinstance(result, CursorResult) or result.rowcount != 1:
                raise RunExecutionStateError(
                    "run was not pending when execution started"
                )

    structlog.get_logger(__name__).info(
        "answer.run.started",
        run_id=run_id,
        conversation_id=conversation_id,
    )


async def _commit_success(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    conversation_id: int,
    run_id: int,
    content: str,
) -> Message:
    """Atomically append ASSISTANT and conditionally mark the Run successful."""

    now = utc_now()
    async with session_factory() as session:
        async with session.begin():
            conversation = await session.scalar(
                select(Conversation)
                .where(Conversation.id == conversation_id)
                .with_for_update()
            )
            if conversation is None:
                raise RunExecutionStateError("run conversation is missing")

            result = await session.execute(
                update(Run)
                .where(
                    Run.id == run_id,
                    Run.conversation_id == conversation_id,
                    Run.status == "RUNNING",
                )
                .values(status="SUCCEEDED", finished_at=now, updated_at=now)
            )
            if not isinstance(result, CursorResult) or result.rowcount != 1:
                raise RunExecutionStateError(
                    "run was not running when success committed"
                )

            last_sequence_no = await session.scalar(
                select(Message.sequence_no)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.sequence_no.desc())
                .limit(1)
                .with_for_update()
            )
            assistant_message = Message(
                conversation_id=conversation_id,
                sequence_no=(last_sequence_no or 0) + 1,
                role="ASSISTANT",
                content=content,
                run_id=run_id,
                regenerated_from_message_id=None,
                created_at=now,
            )
            session.add(assistant_message)
            conversation.last_message_at = now
            conversation.updated_at = now
            await session.flush()

    structlog.get_logger(__name__).info(
        "answer.run.succeeded",
        run_id=run_id,
        conversation_id=conversation_id,
    )
    return assistant_message


async def mark_run_failed_if_running(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    conversation_id: int,
    run_id: int,
    error_code: str,
) -> bool:
    """Mark a running Run failed and report whether this task won the transition."""

    now = utc_now()
    async with session_factory() as session:
        async with session.begin():
            result = await session.execute(
                update(Run)
                .where(
                    Run.id == run_id,
                    Run.conversation_id == conversation_id,
                    Run.status == "RUNNING",
                )
                .values(
                    status="FAILED",
                    finished_at=now,
                    error_code=error_code,
                    updated_at=now,
                )
            )
            if not isinstance(result, CursorResult):
                raise RuntimeError("unexpected handled failure update result")
            changed = result.rowcount == 1

    if changed:
        structlog.get_logger(__name__).info(
            "answer.run.failed",
            run_id=run_id,
            conversation_id=conversation_id,
            error_code=error_code,
        )
    return changed


async def _mark_active_execution_failed(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    conversation_id: int,
    run_id: int,
) -> bool:
    """Best-effort failure transition for an active Run after an internal error."""

    now = utc_now()
    async with session_factory() as session:
        async with session.begin():
            result = await session.execute(
                update(Run)
                .where(
                    Run.id == run_id,
                    Run.conversation_id == conversation_id,
                    Run.status.in_(("PENDING", "RUNNING")),
                )
                .values(
                    status="FAILED",
                    finished_at=now,
                    error_code="ANSWER_EXECUTION_FAILED",
                    updated_at=now,
                )
            )
            if not isinstance(result, CursorResult):
                raise RuntimeError("unexpected execution failure update result")
            changed = result.rowcount == 1

    if changed:
        structlog.get_logger(__name__).error(
            "answer.run.failed",
            run_id=run_id,
            conversation_id=conversation_id,
            error_code="ANSWER_EXECUTION_FAILED",
        )
    return changed
