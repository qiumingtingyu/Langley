"""Deterministic completed-turn reconstruction tests for AnswerContextBuilder."""

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from langley.answering.context_builder import AnswerContextBuilder
from langley.infrastructure.models import Memory, Message, Run

NOW = datetime(2026, 8, 14, tzinfo=UTC)


def _user(
    message_id: int,
    sequence_no: int,
    content: str,
    *,
    regenerated_from_message_id: int | None = None,
) -> Message:
    return Message(
        id=message_id,
        conversation_id=1,
        sequence_no=sequence_no,
        role="USER",
        content=content,
        run_id=None,
        regenerated_from_message_id=regenerated_from_message_id,
        created_at=NOW,
    )


def _assistant(message_id: int, sequence_no: int, run_id: int, content: str) -> Message:
    return Message(
        id=message_id,
        conversation_id=1,
        sequence_no=sequence_no,
        role="ASSISTANT",
        content=content,
        run_id=run_id,
        regenerated_from_message_id=None,
        created_at=NOW,
    )


def _run(run_id: int, input_message_id: int, status: str) -> Run:
    started_at = NOW if status in {"RUNNING", "SUCCEEDED"} else None
    finished_at = NOW if status in {"SUCCEEDED", "FAILED", "CANCELLED"} else None
    return Run(
        id=run_id,
        conversation_id=1,
        input_message_id=input_message_id,
        client_request_id=f"request-{run_id}",
        attempt_no=1,
        status=status,
        started_at=started_at,
        finished_at=finished_at,
        error_code="ANSWER_EXECUTION_FAILED" if status == "FAILED" else None,
        created_at=NOW,
        updated_at=NOW,
    )


def _memory(
    memory_id: int,
    content: str,
    *,
    source_message_id: int | None = None,
) -> Memory:
    return Memory(
        id=memory_id,
        user_id=1,
        content=content,
        source_message_id=source_message_id,
        valid_until=None,
        created_at=NOW,
        updated_at=NOW,
    )


def _build(
    messages: tuple[Message, ...],
    runs: tuple[Run, ...],
    current_user_message_id: int,
    budget: int = 100,
    memory_budget: int = 8_192,
    memories: tuple[Memory, ...] = (),
):
    return AnswerContextBuilder(
        history_estimated_token_budget=budget,
        memory_estimated_token_budget=memory_budget,
    )._assemble(
        messages=messages,
        runs=runs,
        memories=memories,
        conversation_id=1,
        current_user_message_id=current_user_message_id,
    )


def test_builder_includes_only_completed_turns_and_current_user() -> None:
    messages = (
        _user(1, 1, "completed user"),
        _assistant(2, 2, 10, "completed answer"),
        _user(3, 3, "failed"),
        _user(4, 4, "cancelled"),
        _user(5, 5, "active"),
        _user(6, 6, "orphan"),
        _user(7, 7, "current"),
    )
    context = _build(
        messages,
        (
            _run(10, 1, "SUCCEEDED"),
            _run(11, 3, "FAILED"),
            _run(12, 4, "CANCELLED"),
            _run(13, 5, "RUNNING"),
            _run(14, 7, "PENDING"),
        ),
        7,
    )

    assert context.current_user_content == "current"
    assert [
        (turn.user_content, turn.assistant_content) for turn in context.completed_turns
    ] == [("completed user", "completed answer")]


def test_builder_preserves_retry_and_regenerate_linear_context() -> None:
    retry_context = _build(
        (
            _user(1, 1, "first"),
            _assistant(2, 2, 10, "first answer"),
            _user(3, 3, "retry"),
        ),
        (_run(10, 1, "SUCCEEDED"), _run(11, 3, "FAILED"), _run(12, 3, "PENDING")),
        3,
    )
    regenerate_context = _build(
        (
            _user(1, 1, "first"),
            _assistant(2, 2, 10, "first answer"),
            _user(3, 3, "first", regenerated_from_message_id=1),
        ),
        (_run(10, 1, "SUCCEEDED"), _run(11, 3, "PENDING")),
        3,
    )

    assert retry_context.current_user_content == "retry"
    assert regenerate_context.current_user_content == "first"
    assert [turn.user_content for turn in retry_context.completed_turns] == ["first"]
    assert [turn.assistant_content for turn in regenerate_context.completed_turns] == [
        "first answer"
    ]


def test_builder_stops_at_the_first_newest_turn_that_exceeds_budget() -> None:
    messages = (
        _user(1, 1, "old"),
        _assistant(2, 2, 10, "old answer"),
        _user(3, 3, "recent"),
        _assistant(4, 4, 11, "recent answer is too large"),
        _user(5, 5, "current"),
    )
    context = _build(
        messages,
        (_run(10, 1, "SUCCEEDED"), _run(11, 3, "SUCCEEDED"), _run(12, 5, "PENDING")),
        5,
        budget=10,
    )

    assert context.completed_turns == ()
    assert context.current_user_content == "current"


def test_builder_does_not_skip_an_oversized_turn_for_an_older_one() -> None:
    messages = (
        _user(1, 1, "old"),
        _assistant(2, 2, 10, "old"),
        _user(3, 3, "middle"),
        _assistant(4, 4, 11, "middle answer"),
        _user(5, 5, "newer"),
        _assistant(6, 6, 12, "newer"),
        _user(7, 7, "newest"),
        _assistant(8, 8, 13, "newest"),
        _user(9, 9, "current"),
    )
    context = _build(
        messages,
        (
            _run(10, 1, "SUCCEEDED"),
            _run(11, 3, "SUCCEEDED"),
            _run(12, 5, "SUCCEEDED"),
            _run(13, 7, "SUCCEEDED"),
            _run(14, 9, "PENDING"),
        ),
        9,
        budget=24,
    )

    assert [turn.user_content for turn in context.completed_turns] == [
        "newer",
        "newest",
    ]


def test_builder_loads_all_personal_context_or_marks_it_unavailable() -> None:
    messages = (_user(1, 1, "current"),)
    memory = _memory(10, "preference")

    exact_fit = _build(
        messages,
        (_run(1, 1, "PENDING"),),
        1,
        memory_budget=len(memory.content) + 32,
        memories=(memory,),
    )
    over_budget = _build(
        messages,
        (_run(1, 1, "PENDING"),),
        1,
        memory_budget=len(memory.content) + 31,
        memories=(memory,),
    )

    assert [
        (item.memory_id, item.content) for item in exact_fit.personal_context or ()
    ] == [(10, "preference")]
    assert over_budget.personal_context is None


def test_builder_suppresses_only_exposed_canonical_source_identity() -> None:
    messages = (
        _user(1, 1, "history user"),
        _assistant(2, 2, 10, "history answer"),
        _user(3, 3, "current"),
    )
    context = _build(
        messages,
        (_run(10, 1, "SUCCEEDED"), _run(11, 3, "PENDING")),
        3,
        memories=(
            _memory(10, "from exposed original", source_message_id=1),
            _memory(11, "same meaning is not identity", source_message_id=99),
        ),
    )

    assert [item.content for item in context.personal_context or ()] == [
        "same meaning is not identity"
    ]


def test_builder_maps_regenerated_current_user_to_canonical_source_identity() -> None:
    messages = (
        _user(1, 1, "original"),
        _user(2, 2, "original", regenerated_from_message_id=1),
    )
    context = _build(
        messages,
        (_run(10, 2, "PENDING"),),
        2,
        memories=(_memory(10, "from original", source_message_id=1),),
    )

    assert context.personal_context == ()


def test_builder_releases_its_database_scope_before_returning_context() -> None:
    class ScalarResult:
        def __init__(self, values: tuple[object, ...]) -> None:
            self._values = values

        def all(self) -> tuple[object, ...]:
            return self._values

    class RecordingSessionFactory:
        def __init__(self) -> None:
            self.exited = False
            self._calls = 0

        def __call__(self) -> "RecordingSessionFactory":
            return self

        async def __aenter__(self) -> "RecordingSessionFactory":
            return self

        async def __aexit__(self, *args: object) -> None:
            self.exited = True

        @asynccontextmanager
        async def begin(self):
            yield

        async def scalars(self, statement: object) -> ScalarResult:
            del statement
            self._calls += 1
            if self._calls == 1:
                return ScalarResult((_user(1, 1, "current"),))
            if self._calls == 2:
                return ScalarResult((_run(10, 1, "PENDING"),))
            return ScalarResult(())

    factory = RecordingSessionFactory()

    async def build_context() -> object:
        return await AnswerContextBuilder(history_estimated_token_budget=10).build(
            factory,
            conversation_id=1,
            current_user_message_id=1,
        )

    context = asyncio.run(build_context())

    assert factory.exited is True
    assert context.current_user_content == "current"
