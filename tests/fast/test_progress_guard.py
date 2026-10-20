"""Exact detection, complete-batch association and request-only integration."""

import asyncio

import pytest

from langley.answering.contracts import (
    AssistantRuntimeMessage,
    ToolCall,
    ToolResult,
    ToolResultKind,
)
from langley.answering.progress_guard import ProgressState, observe_batch


def pair(i, args='{"a":1}', content="same", error=False, name="run_command"):
    call = ToolCall(str(i), name, args)
    result = ToolResult(
        str(i),
        name,
        ToolResultKind.TOOL_ERROR if error else ToolResultKind.SUCCESS,
        content,
    )
    return call, result


def feed(samples):
    state, alerts = ProgressState(), []
    for i, (call, result) in enumerate(samples, 1):
        state, warnings = observe_batch(state, (call,), (result,))
        alerts.extend((i, w.rule) for w in warnings)
    return alerts


@pytest.mark.parametrize(
    ("content", "error", "position", "rule"),
    [
        ("ok", False, 4, "ACTION_OBSERVATION_4"),
        ("bad", True, 3, "ACTION_ERROR_3"),
        ('{"exit_code":1,"timed_out":false}', False, 3, "ACTION_ERROR_3"),
        ('{"exit_code":null,"timed_out":true}', False, 3, "ACTION_ERROR_3"),
    ],
)
def test_first_warning_and_unchanged_streak_suppression(content, error, position, rule):
    assert feed([pair(i, content=content, error=error) for i in range(12)]) == [
        (position, rule)
    ]


def test_alternating_suppression_and_break_rearms():
    samples = [pair(i, args=str(i % 2)) for i in range(12)]
    assert feed(samples) == [(6, "ALTERNATING_AB_3")]
    samples += [pair(20, content="changed")]
    samples += [pair(i, args=str(i % 2)) for i in range(21, 27)]
    assert feed(samples) == [(6, "ALTERNATING_AB_3"), (19, "ALTERNATING_AB_3")]


def test_json_order_only_normalization_and_business_id_preservation():
    samples = [
        pair(i, args='{"b":2,"a":1}' if i % 2 else '{"a":1,"b":2}') for i in range(4)
    ]
    assert feed(samples) == [(4, "ACTION_OBSERVATION_4")]
    assert feed([pair(i, args='{"trace_id":%d}' % i) for i in range(10)]) == []
    assert feed([pair(i, content=str(i)) for i in range(10)]) == []
    assert feed([pair(i, args=str(i)) for i in range(10)]) == []


def test_complete_multi_call_order_is_by_id_not_result_arrival():
    samples = [pair(i, args=str(i % 2)) for i in range(6)]
    calls, results = tuple(zip(*samples, strict=True))
    state, warnings = observe_batch(ProgressState(), calls, tuple(reversed(results)))
    assert [w.rule for w in warnings] == ["ALTERNATING_AB_3"]
    assert len(state.recent) == 6
    for bad in (results[:-1], results + (results[0],), results[1:] + (pair(99)[1],)):
        reset, warnings = observe_batch(state, calls, bad)
        assert reset == ProgressState() and not warnings
    assert observe_batch(state, (calls[0], calls[0]), results[:2])[0] == ProgressState()
    wrong = ToolResult(results[0].call_id, "wrong", results[0].kind, "same")
    assert observe_batch(state, calls, (wrong,) + results[1:])[0] == ProgressState()


@pytest.mark.parametrize("enabled", [False, True])
def test_workflow_delivers_once_without_mutating_history_or_user_index(enabled):
    from test_run_context_workflow import Context, LargeTool, discard, round_

    from langley.answering.context_builder import AnswerContext
    from langley.answering.fake_provider import FakeProvider
    from langley.answering.tools import ToolContext, ToolExecutor
    from langley.answering.tracing import _NoopTrace
    from langley.answering.workflow import LearningAssistantWorkflow

    provider = FakeProvider(
        [
            *(round_(ToolCall(str(i), "get_current_time", "{}")) for i in range(7)),
            round_(text="done"),
        ]
    )
    tool = LargeTool()
    flow = LearningAssistantWorkflow(
        context_builder=Context(),
        provider=provider,
        tool_executor=ToolExecutor(tools=(tool,)),
        max_llm_rounds=10,
        max_tool_calls=10,
        overall_deadline_seconds=5,
        provider_name="fake",
        model="fake",
        progress_guard_enabled=enabled,
    )
    state = asyncio.run(
        flow._run_graph(
            AnswerContext((), "request"),
            ToolContext(run_id=1, user_id=1, knowledge_base_id=None),
            discard,
            _NoopTrace(),
        )
    )
    delivered = [
        i + 1
        for i, r in enumerate(provider.requests)
        if isinstance(r.transcript[-1], AssistantRuntimeMessage)
        and "[Progress Guard:" in r.transcript[-1].content
    ]
    assert delivered == ([5] if enabled else [])
    assert tool.calls == 7 and len(state["transcript"]) == 16
    assert all(r.current_user_message_index == 0 for r in provider.requests)
    assert all(
        "[Progress Guard:" not in getattr(x, "content", "") for x in state["transcript"]
    )
    # The following request contains only the canonical history again.
    assert provider.requests[5].transcript == state["transcript"][:11]
