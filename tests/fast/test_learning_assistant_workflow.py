"""Deterministic production Workflow and LangGraph regression tests."""

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast

import pytest
from langsmith import get_tracing_context

from langley.answering.context_builder import (
    AnswerContext,
    AnswerContextBuilder,
    PersonalContextItem,
)
from langley.answering.contracts import (
    AssistantContentDelta,
    LLMFinishReason,
    LLMResponseCompleted,
    ToolCall,
    ToolResult,
    ToolResultKind,
    UserRuntimeMessage,
)
from langley.answering.errors import RunErrorCode, WorkflowFailure
from langley.answering.fake_provider import FakeProvider, ScriptedProviderRound
from langley.answering.tools import CurrentTimeTool, ToolExecutor
from langley.answering.workflow import LearningAssistantWorkflow


@dataclass
class _StaticContextBuilder:
    context: AnswerContext

    async def build(self, *args: object, **kwargs: object) -> AnswerContext:
        return self.context


@dataclass
class _BlockingContextBuilder:
    started: asyncio.Event
    blocked_until: asyncio.Event

    async def build(self, *args: object, **kwargs: object) -> AnswerContext:
        self.started.set()
        await self.blocked_until.wait()
        return _context()


class _TracingContextCheckingProvider:
    def __init__(self, provider: FakeProvider) -> None:
        self._provider = provider
        self.enabled_values: list[object] = []

    def stream(self, request: object):
        self.enabled_values.append(get_tracing_context()["enabled"])
        return self._provider.stream(cast(object, request))


def _context() -> AnswerContext:
    return AnswerContext(completed_turns=(), current_user_content="现在几点？")


def _completion(
    *,
    content: str,
    tool_calls: tuple[ToolCall, ...] = (),
    finish_reason: LLMFinishReason = LLMFinishReason.STOP,
) -> LLMResponseCompleted:
    return LLMResponseCompleted(
        assistant_content=content,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        usage=None,
    )


def _workflow(
    provider: FakeProvider,
    *,
    context_builder: AnswerContextBuilder | None = None,
    tool_executor: ToolExecutor | None = None,
    trace_content_enabled: bool = False,
    max_llm_rounds: int = 4,
    max_tool_calls: int = 3,
    overall_deadline_seconds: float = 1.0,
) -> LearningAssistantWorkflow:
    return LearningAssistantWorkflow(
        context_builder=context_builder
        or cast(AnswerContextBuilder, _StaticContextBuilder(_context())),
        provider=provider,
        tool_executor=tool_executor
        or ToolExecutor(
            CurrentTimeTool(clock=lambda: datetime(2026, 8, 14, 9, 0, tzinfo=UTC))
        ),
        max_llm_rounds=max_llm_rounds,
        max_tool_calls=max_tool_calls,
        overall_deadline_seconds=overall_deadline_seconds,
        provider_name="fake",
        model="fake-script",
        trace_content_enabled=trace_content_enabled,
    )


async def _discard_delta(content: str) -> None:
    del content


@pytest.mark.anyio
async def test_direct_answer_uses_canonical_completion_not_stream_deltas() -> None:
    provider = FakeProvider(
        [
            ScriptedProviderRound(
                events=(
                    AssistantContentDelta(content="temporary presentation"),
                    _completion(content="canonical final"),
                )
            )
        ]
    )
    deltas: list[str] = []

    async def capture_delta(content: str) -> None:
        deltas.append(content)

    success = await _workflow(provider).execute(
        cast(object, None),
        conversation_id=11,
        input_message_id=22,
        on_assistant_delta=capture_delta,
    )

    assert success == "canonical final"
    assert deltas == ["temporary presentation"]
    assert provider.requests[0].transcript == (
        UserRuntimeMessage(content="现在几点？"),
    )


@pytest.mark.anyio
async def test_tool_loop_round_trips_tool_observation() -> None:
    tool_call = ToolCall(
        call_id="time-1", name="get_current_time", raw_arguments='{"timezone":"UTC"}'
    )
    provider = FakeProvider(
        [
            ScriptedProviderRound(
                events=(
                    AssistantContentDelta(content="我来查询。"),
                    _completion(
                        content="我来查询。",
                        tool_calls=(tool_call,),
                        finish_reason=LLMFinishReason.TOOL_CALLS,
                    ),
                )
            ),
            ScriptedProviderRound(events=(_completion(content="现在是 09:00 UTC。"),)),
        ]
    )

    success = await _workflow(provider).execute(
        cast(object, None),
        conversation_id=11,
        input_message_id=22,
        on_assistant_delta=_discard_delta,
    )

    assert success == "我来查询。\n\n现在是 09:00 UTC。"
    second_request = provider.requests[1]
    assert second_request.transcript[1].tool_calls == (tool_call,)
    tool_result = cast(ToolResult, second_request.transcript[2])
    assert tool_result.call_id == "time-1"
    assert tool_result.kind is ToolResultKind.SUCCESS


@pytest.mark.anyio
async def test_personal_context_is_not_transcript_and_stays_stable_across_rounds() -> (
    None
):
    tool_call = ToolCall(
        call_id="time-1", name="get_current_time", raw_arguments='{"timezone":"UTC"}'
    )
    provider = FakeProvider(
        [
            ScriptedProviderRound(
                events=(
                    _completion(
                        content="",
                        tool_calls=(tool_call,),
                        finish_reason=LLMFinishReason.TOOL_CALLS,
                    ),
                )
            ),
            ScriptedProviderRound(events=(_completion(content="done"),)),
        ]
    )
    context = AnswerContext(
        completed_turns=(),
        current_user_content="current request",
        personal_context=(
            PersonalContextItem(memory_id=99, content="prefers short examples"),
        ),
    )

    await _workflow(
        provider,
        context_builder=cast(AnswerContextBuilder, _StaticContextBuilder(context)),
    ).execute(
        cast(object, None),
        conversation_id=11,
        input_message_id=22,
        on_assistant_delta=_discard_delta,
    )

    assert provider.requests[0].personal_context == ("prefers short examples",)
    assert provider.requests[1].personal_context == ("prefers short examples",)
    assert all(
        "prefers short examples" not in item.content
        for request in provider.requests
        for item in request.transcript
        if isinstance(item, UserRuntimeMessage)
    )
    assert provider.requests[0].current_user_message_index == 0
    assert provider.requests[1].current_user_message_index == 0


@pytest.mark.anyio
@pytest.mark.parametrize(
    "tool_round_finish",
    [LLMFinishReason.LENGTH, LLMFinishReason.FILTERED, LLMFinishReason.UNKNOWN],
)
async def test_nonempty_tool_calls_route_to_the_tool_node_regardless_of_finish_label(
    tool_round_finish: LLMFinishReason,
) -> None:
    tool_call = ToolCall(
        call_id="time-1", name="get_current_time", raw_arguments='{"timezone":"UTC"}'
    )
    provider = FakeProvider(
        [
            ScriptedProviderRound(
                events=(
                    _completion(
                        content="",
                        tool_calls=(tool_call,),
                        finish_reason=tool_round_finish,
                    ),
                )
            ),
            ScriptedProviderRound(events=(_completion(content="工具路径已完成。"),)),
        ]
    )

    success = await _workflow(provider).execute(
        cast(object, None),
        conversation_id=11,
        input_message_id=22,
        on_assistant_delta=_discard_delta,
    )

    assert success == "工具路径已完成。"
    assert isinstance(provider.requests[1].transcript[2], ToolResult)


@pytest.mark.anyio
async def test_malformed_tool_arguments_become_a_recoverable_runtime_observation() -> (
    None
):
    malformed_call = ToolCall(
        call_id="time-1", name="get_current_time", raw_arguments="{"
    )
    provider = FakeProvider(
        [
            ScriptedProviderRound(
                events=(
                    _completion(
                        content="",
                        tool_calls=(malformed_call,),
                        finish_reason=LLMFinishReason.TOOL_CALLS,
                    ),
                )
            ),
            ScriptedProviderRound(events=(_completion(content="请提供有效时区。"),)),
        ]
    )

    success = await _workflow(provider).execute(
        cast(object, None),
        conversation_id=11,
        input_message_id=22,
        on_assistant_delta=_discard_delta,
    )

    assert success == "请提供有效时区。"
    tool_result = cast(ToolResult, provider.requests[1].transcript[2])
    assert tool_result.kind is ToolResultKind.INVALID_ARGUMENTS


class _TrackingTimeTool(CurrentTimeTool):
    def __init__(self) -> None:
        self.executed: list[str] = []

    def validate_arguments(self, arguments: dict[str, object]) -> bool:
        return set(arguments) == {"timezone"} and isinstance(arguments["timezone"], str)

    async def execute(self, arguments: dict[str, object]) -> str:
        self.executed.append("get_current_time")
        return '{"timezone":"UTC","datetime":"2026-08-14T09:00:00+00:00"}'


@pytest.mark.anyio
async def test_whole_tool_batch_budget_preflight_executes_none() -> None:
    tracking_tool = _TrackingTimeTool()
    provider = FakeProvider(
        [
            ScriptedProviderRound(
                events=(
                    _completion(
                        content="",
                        tool_calls=(
                            ToolCall("one", "get_current_time", '{"timezone":"UTC"}'),
                            ToolCall("two", "get_current_time", '{"timezone":"UTC"}'),
                        ),
                        finish_reason=LLMFinishReason.TOOL_CALLS,
                    ),
                )
            )
        ]
    )

    with pytest.raises(WorkflowFailure) as raised:
        await _workflow(
            provider,
            tool_executor=ToolExecutor(tracking_tool),
            max_tool_calls=1,
        ).execute(
            cast(object, None),
            conversation_id=11,
            input_message_id=22,
            on_assistant_delta=_discard_delta,
        )

    assert raised.value.error_code is RunErrorCode.AGENT_EXECUTION_LIMIT
    assert tracking_tool.executed == []


@pytest.mark.anyio
async def test_invalid_tool_call_still_counts_against_the_agent_tool_budget() -> None:
    provider = FakeProvider(
        [
            ScriptedProviderRound(
                events=(
                    _completion(
                        content="",
                        tool_calls=(ToolCall("one", "get_current_time", "{"),),
                        finish_reason=LLMFinishReason.TOOL_CALLS,
                    ),
                )
            )
        ]
    )

    with pytest.raises(WorkflowFailure) as raised:
        await _workflow(provider, max_tool_calls=0).execute(
            cast(object, None),
            conversation_id=11,
            input_message_id=22,
            on_assistant_delta=_discard_delta,
        )

    assert raised.value.error_code is RunErrorCode.AGENT_EXECUTION_LIMIT


@pytest.mark.anyio
async def test_round_limit_and_deadline_are_typed_workflow_failures() -> None:
    tool_call = ToolCall("one", "get_current_time", '{"timezone":"UTC"}')
    round_limited = FakeProvider(
        [
            ScriptedProviderRound(
                events=(
                    _completion(
                        content="",
                        tool_calls=(tool_call,),
                        finish_reason=LLMFinishReason.TOOL_CALLS,
                    ),
                )
            )
        ]
    )
    with pytest.raises(WorkflowFailure) as round_raised:
        await _workflow(round_limited, max_llm_rounds=1).execute(
            cast(object, None),
            conversation_id=11,
            input_message_id=22,
            on_assistant_delta=_discard_delta,
        )
    assert round_raised.value.error_code is RunErrorCode.AGENT_EXECUTION_LIMIT

    started = asyncio.Event()
    blocked_until = asyncio.Event()
    deadline_limited = FakeProvider(
        [ScriptedProviderRound(events=(), started=started, blocked_until=blocked_until)]
    )
    task = asyncio.create_task(
        _workflow(deadline_limited, overall_deadline_seconds=0.01).execute(
            cast(object, None),
            conversation_id=11,
            input_message_id=22,
            on_assistant_delta=_discard_delta,
        )
    )
    await started.wait()
    with pytest.raises(WorkflowFailure) as deadline_raised:
        await task
    assert deadline_raised.value.error_code is RunErrorCode.AGENT_EXECUTION_TIMEOUT

    context_started = asyncio.Event()
    context_deadline_limited = _BlockingContextBuilder(
        started=context_started, blocked_until=asyncio.Event()
    )
    context_task = asyncio.create_task(
        _workflow(
            FakeProvider([]),
            context_builder=cast(AnswerContextBuilder, context_deadline_limited),
            overall_deadline_seconds=0.01,
        ).execute(
            cast(object, None),
            conversation_id=11,
            input_message_id=22,
            on_assistant_delta=_discard_delta,
        )
    )
    await context_started.wait()
    with pytest.raises(WorkflowFailure) as context_deadline_raised:
        await context_task
    assert (
        context_deadline_raised.value.error_code is RunErrorCode.AGENT_EXECUTION_TIMEOUT
    )


@pytest.mark.anyio
@pytest.mark.parametrize(
    "completion",
    [
        _completion(content=""),
        _completion(content="truncated", finish_reason=LLMFinishReason.LENGTH),
        _completion(content="blocked", finish_reason=LLMFinishReason.FILTERED),
    ],
)
async def test_invalid_final_candidates_never_become_workflow_success(
    completion: LLMResponseCompleted,
) -> None:
    provider = FakeProvider([ScriptedProviderRound(events=(completion,))])

    with pytest.raises(WorkflowFailure) as raised:
        await _workflow(provider).execute(
            cast(object, None),
            conversation_id=11,
            input_message_id=22,
            on_assistant_delta=_discard_delta,
        )

    assert raised.value.error_code is RunErrorCode.LLM_RESPONSE_INVALID


@pytest.mark.anyio
async def test_workflow_cancellation_propagates_through_the_blocked_provider() -> None:
    started = asyncio.Event()
    provider = FakeProvider(
        [
            ScriptedProviderRound(
                events=(), started=started, blocked_until=asyncio.Event()
            )
        ]
    )
    task = asyncio.create_task(
        _workflow(provider, overall_deadline_seconds=10).execute(
            cast(object, None),
            conversation_id=11,
            input_message_id=22,
            on_assistant_delta=_discard_delta,
        )
    )
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.anyio
async def test_graph_scope_disables_automatic_langsmith_tracing_despite_global_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    provider = _TracingContextCheckingProvider(
        FakeProvider([ScriptedProviderRound(events=(_completion(content="正常回答"),))])
    )

    success = await _workflow(cast(FakeProvider, provider)).execute(
        cast(object, None),
        run_id=919,
        conversation_id=11,
        input_message_id=22,
        on_assistant_delta=_discard_delta,
    )

    assert success == "正常回答"
    assert provider.enabled_values == [False]
