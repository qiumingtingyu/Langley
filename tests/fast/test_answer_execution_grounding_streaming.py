"""Focused SSE publication order for AUTO and REQUIRED grounding."""

from types import SimpleNamespace
from typing import cast

import pytest

import langley.answer_execution as execution_module
from langley.answer_execution import AnswerExecutionManager
from langley.answer_runtime import ActiveAnswer
from langley.answering.errors import (
    InvalidResponseSubtype,
    RunErrorCode,
    WorkflowFailure,
)
from langley.answering.knowledge_qa import (
    INSUFFICIENT_EVIDENCE_ANSWER,
    AnswerCompletion,
)


class _Workflow:
    def __init__(
        self,
        completion: AnswerCompletion | None = None,
        *,
        provisional: str | None = None,
        failure: WorkflowFailure | None = None,
    ) -> None:
        self._completion = completion
        self._provisional = provisional
        self._failure = failure

    async def execute(self, *args: object, **kwargs: object) -> AnswerCompletion:
        del args
        if self._provisional is not None:
            await kwargs["on_assistant_delta"](self._provisional)  # type: ignore[operator]
        if self._failure is not None:
            raise self._failure
        assert self._completion is not None
        return self._completion


def _command(policy: str) -> object:
    return SimpleNamespace(
        user_id=1,
        user_message=SimpleNamespace(id=12),
        run=SimpleNamespace(
            id=13,
            conversation_id=14,
            knowledge_base_id=15 if policy == "REQUIRED" else None,
            grounding_policy=policy,
        ),
    )


async def _exercise(
    monkeypatch: pytest.MonkeyPatch,
    workflow: _Workflow,
    *,
    policy: str,
) -> tuple[ActiveAnswer, list[object], list[str]]:
    answer = ActiveAnswer()
    queue = answer_queue = __import__("asyncio").Queue()
    answer.streams.add(answer_queue)
    events: list[str] = []

    async def start(*args: object, **kwargs: object) -> None:
        del args, kwargs
        events.append("running")

    async def commit(*args: object, **kwargs: object) -> None:
        del args, kwargs
        if policy == "REQUIRED":
            assert answer.partial_text == ""
        else:
            assert answer.partial_text == "token"
        events.append("durable_commit")

    async def fail(*args: object, **kwargs: object) -> bool:
        del args, kwargs
        events.append("durable_fail")
        return True

    monkeypatch.setattr(execution_module, "_start_running", start)
    monkeypatch.setattr(execution_module, "_commit_success", commit)
    monkeypatch.setattr(execution_module, "mark_run_failed_if_running", fail)
    manager = AnswerExecutionManager(cast(object, None), lambda: cast(object, workflow))
    manager._active_answers[13] = answer
    original_publish_delta = manager._publish_delta

    async def publish_delta(active: ActiveAnswer, run_id: int, delta: str) -> None:
        if policy == "REQUIRED":
            assert events[-1] == "durable_commit"
        events.append("message_delta")
        await original_publish_delta(active, run_id, delta)

    monkeypatch.setattr(manager, "_publish_delta", publish_delta)
    await manager._execute(
        answer,
        cast(object, _command(policy)),
        cast(object, workflow),
    )
    items: list[object] = []
    while not queue.empty():
        items.append(queue.get_nowait())
    return answer, items, events


@pytest.mark.anyio
@pytest.mark.parametrize(
    "content", ["canonical grounded [K1]", INSUFFICIENT_EVIDENCE_ANSWER]
)
async def test_required_publishes_only_canonical_content_after_commit(
    monkeypatch: pytest.MonkeyPatch, content: str
) -> None:
    completion = AnswerCompletion(
        content=content,
        citations=(),
        abstained=content == INSUFFICIENT_EVIDENCE_ANSWER,
    )

    answer, items, events = await _exercise(
        monkeypatch, _Workflow(completion), policy="REQUIRED"
    )

    assert answer.partial_text == content
    assert events == ["running", "durable_commit", "message_delta"]
    assert items == [
        ("run.started", {"run_id": 13}),
        ("message.delta", {"run_id": 13, "delta": content}),
        ("run.succeeded", {"run_id": 13}),
        None,
    ]


@pytest.mark.anyio
async def test_required_invalid_has_zero_delta_and_zero_partial_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = WorkflowFailure(
        RunErrorCode.LLM_RESPONSE_INVALID,
        invalid_response_subtype=InvalidResponseSubtype.INVALID_ABSTENTION_FORMAT,
    )

    answer, items, events = await _exercise(
        monkeypatch, _Workflow(failure=failure), policy="REQUIRED"
    )

    assert answer.partial_text == ""
    assert events == ["running", "durable_fail"]
    assert items == [
        ("run.started", {"run_id": 13}),
        (
            "run.failed",
            {"run_id": 13, "error_code": "LLM_RESPONSE_INVALID"},
        ),
        None,
    ]


@pytest.mark.anyio
async def test_auto_keeps_existing_token_streaming_without_canonical_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completion = AnswerCompletion(content="canonical", citations=(), abstained=False)

    answer, items, events = await _exercise(
        monkeypatch,
        _Workflow(completion, provisional="token"),
        policy="AUTO",
    )

    assert answer.partial_text == "token"
    assert events == ["running", "message_delta", "durable_commit"]
    assert items == [
        ("run.started", {"run_id": 13}),
        ("message.delta", {"run_id": 13, "delta": "token"}),
        ("run.succeeded", {"run_id": 13}),
        None,
    ]
