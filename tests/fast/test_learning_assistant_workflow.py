"""Deterministic production Workflow and LangGraph regression tests."""

import asyncio
from dataclasses import dataclass, field
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
    LLMProvider,
    LLMRequest,
    LLMResponseCompleted,
    ToolCall,
    ToolResult,
    ToolResultKind,
    UserRuntimeMessage,
)
from langley.answering.errors import RunErrorCode, WorkflowFailure
from langley.answering.fake_provider import FakeProvider, ScriptedProviderRound
from langley.answering.knowledge_qa import INSUFFICIENT_EVIDENCE_ANSWER
from langley.answering.tools import (
    CurrentTimeTool,
    SearchKnowledgeTool,
    ToolContext,
    ToolExecutionOutput,
    ToolExecutor,
)
from langley.answering.tracing import Tracer
from langley.answering.workflow import LearningAssistantWorkflow
from langley.knowledge.retrieval import RetrievalHit, RetrievalResult
from langley.knowledge.retrieval_service import KnowledgeSearchError


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
    provider: LLMProvider,
    *,
    context_builder: AnswerContextBuilder | None = None,
    tool_executor: ToolExecutor | None = None,
    trace_content_enabled: bool = False,
    max_llm_rounds: int = 4,
    max_tool_calls: int = 3,
    overall_deadline_seconds: float = 1.0,
    tracer: Tracer | None = None,
) -> LearningAssistantWorkflow:
    return LearningAssistantWorkflow(
        context_builder=context_builder
        or cast(AnswerContextBuilder, _StaticContextBuilder(_context())),
        provider=provider,
        tool_executor=tool_executor
        or ToolExecutor(
            tools=(
                CurrentTimeTool(
                    clock=lambda: datetime(2026, 8, 14, 9, 0, tzinfo=UTC),
                ),
            ),
        ),
        max_llm_rounds=max_llm_rounds,
        max_tool_calls=max_tool_calls,
        overall_deadline_seconds=overall_deadline_seconds,
        provider_name="fake",
        model="fake-script",
        trace_content_enabled=trace_content_enabled,
        tracer=tracer,
    )


async def _discard_delta(content: str) -> None:
    del content


async def _execute_workflow(
    workflow: LearningAssistantWorkflow,
    on_assistant_delta=_discard_delta,
    knowledge_base_id: int | None = None,
):
    return await workflow.execute(
        cast(object, None),
        run_id=101,
        user_id=1,
        conversation_id=11,
        input_message_id=22,
        knowledge_base_id=knowledge_base_id,
        on_assistant_delta=on_assistant_delta,
    )


class _LifecycleLLMTrace:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    def content_delta(self, delta: AssistantContentDelta) -> None:
        assert delta.content == "answer"
        self._events.append("delta")

    def finish(self, response: LLMResponseCompleted) -> None:
        assert response.assistant_content == "answer"
        self._events.append("finish")

    def failure(self, error_code: str) -> None:
        self._events.append(f"failure:{error_code}")


class _LifecycleExecutionTrace:
    def __init__(self, events: list[str], *, fail_begin: bool = False) -> None:
        self._events = events
        self._fail_begin = fail_begin

    def begin_llm(self, request: LLMRequest, round_: int) -> _LifecycleLLMTrace:
        del request
        if self._fail_begin:
            raise RuntimeError("trace unavailable")
        assert round_ == 1
        self._events.append("begin")
        return _LifecycleLLMTrace(self._events)

    def begin_tool(self, call: ToolCall, tool_calls_used: int):
        del call, tool_calls_used
        raise AssertionError("no tool call expected")

    def citation_validate(self, **kwargs: object) -> None:
        del kwargs

    def success(self, answer: str, stop_reason: str = "FINAL_ANSWER") -> None:
        del answer, stop_reason
        self._events.append("success")

    def failure(self, error_code: str) -> None:
        self._events.append(f"root_failure:{error_code}")


class _LifecycleTracer:
    def __init__(self, events: list[str], *, fail_begin: bool = False) -> None:
        self._trace = _LifecycleExecutionTrace(events, fail_begin=fail_begin)

    def start(
        self, run_id: int, provider: str, model: str, include_content: bool
    ) -> _LifecycleExecutionTrace:
        del run_id, provider, model, include_content
        return self._trace


class _BoundaryCheckingProvider:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    def stream(self, request: LLMRequest):
        del request

        async def events():
            assert self._events == ["begin"]
            self._events.append("provider_started")
            yield AssistantContentDelta("answer")
            yield _completion(content="answer")

        return events()


def _hit() -> RetrievalHit:
    return RetrievalHit(
        knowledge_chunk_id=11,
        rank=1,
        score=0.9,
        chunk_ordinal=4,
        content="Authoritative TCP evidence.",
        heading_path=("TCP",),
        source_regions=(),
        document_id=12,
        document_version_id=13,
        source_display_name="tcp.md",
        source_sha256="a" * 64,
    )


@dataclass
class _FakeRetrievalService:
    calls: list[tuple[int, int, str, int]] = field(default_factory=list)
    error: KnowledgeSearchError | None = None

    async def search(
        self, *, user_id: int, knowledge_base_id: int, query: str, top_k: int
    ) -> RetrievalResult:
        self.calls.append((user_id, knowledge_base_id, query, top_k))
        if self.error is not None:
            raise self.error
        return RetrievalResult(
            knowledge_base_id=knowledge_base_id,
            generation_id="generation",
            hits=(_hit(),),
        )


def _rag_executor(service: _FakeRetrievalService) -> ToolExecutor:
    return ToolExecutor(
        tools=(
            CurrentTimeTool(),
            SearchKnowledgeTool(service),  # type: ignore[arg-type]
        )
    )


@pytest.mark.anyio
async def test_llm_trace_begins_before_provider_stream_and_finishes_on_completion() -> (
    None
):
    events: list[str] = []
    provider = _BoundaryCheckingProvider(events)

    result = await _execute_workflow(
        _workflow(provider, tracer=_LifecycleTracer(events))
    )

    assert result.content == "answer"
    assert events[:4] == ["begin", "provider_started", "delta", "finish"]


@pytest.mark.anyio
async def test_llm_trace_begin_failure_is_fail_open() -> None:
    events: list[str] = []
    provider = FakeProvider(
        [ScriptedProviderRound(events=(_completion(content="answer"),))]
    )

    result = await _execute_workflow(
        _workflow(provider, tracer=_LifecycleTracer(events, fail_begin=True))
    )

    assert result.content == "answer"
    assert events == ["success"]


@pytest.mark.anyio
async def test_provider_failure_closes_the_open_llm_trace_with_error() -> None:
    events: list[str] = []
    provider = FakeProvider(
        [
            ScriptedProviderRound(
                events=(),
                failure=WorkflowFailure(RunErrorCode.LLM_PROVIDER_FAILED),
            )
        ]
    )

    with pytest.raises(WorkflowFailure) as raised:
        await _execute_workflow(_workflow(provider, tracer=_LifecycleTracer(events)))

    assert raised.value.error_code is RunErrorCode.LLM_PROVIDER_FAILED
    assert events == [
        "begin",
        "failure:LLM_PROVIDER_FAILED",
        "root_failure:LLM_PROVIDER_FAILED",
    ]


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

    success = await _execute_workflow(_workflow(provider), capture_delta)

    assert success.content == "canonical final"
    assert deltas == ["temporary presentation"]
    assert provider.requests[0].transcript == (
        UserRuntimeMessage(content="现在几点？"),
    )


@pytest.mark.anyio
async def test_knowledge_scope_controls_tools_and_grounded_citations() -> None:
    service = _FakeRetrievalService()
    direct_provider = FakeProvider(
        [ScriptedProviderRound(events=(_completion(content="direct answer"),))]
    )
    await _execute_workflow(
        _workflow(direct_provider, tool_executor=_rag_executor(service))
    )
    assert tuple(tool.name for tool in direct_provider.requests[0].allowed_tools) == (
        "get_current_time",
    )

    search_call = ToolCall("search-1", "search_knowledge", '{"query":"TCP"}')
    grounded_provider = FakeProvider(
        [
            ScriptedProviderRound(
                events=(
                    _completion(
                        content="I will search first [K999].",
                        tool_calls=(search_call,),
                        finish_reason=LLMFinishReason.TOOL_CALLS,
                    ),
                )
            ),
            ScriptedProviderRound(events=(_completion(content="TCP uses [K1]."),)),
        ]
    )
    completion = await _execute_workflow(
        _workflow(grounded_provider, tool_executor=_rag_executor(service)),
        knowledge_base_id=44,
    )

    assert tuple(tool.name for tool in grounded_provider.requests[0].allowed_tools) == (
        "get_current_time",
        "search_knowledge",
    )
    assert service.calls == [(1, 44, "TCP", 5)]
    assert completion.content == "TCP uses [K1]."
    assert [
        (draft.evidence_handle, draft.document_version_id)
        for draft in completion.citations
    ] == [(1, 13)]
    observation = cast(ToolResult, grounded_provider.requests[1].transcript[2])
    assert "knowledge_base_id" not in observation.content
    assert "evidence_handle" in observation.content


@pytest.mark.anyio
async def test_retrieval_evidence_survives_a_later_time_tool_batch() -> None:
    service = _FakeRetrievalService()
    provider = FakeProvider(
        [
            ScriptedProviderRound(
                events=(
                    _completion(
                        content="",
                        tool_calls=(
                            ToolCall("search-1", "search_knowledge", '{"query":"TCP"}'),
                        ),
                        finish_reason=LLMFinishReason.TOOL_CALLS,
                    ),
                )
            ),
            ScriptedProviderRound(
                events=(
                    _completion(
                        content="",
                        tool_calls=(
                            ToolCall(
                                "time-1", "get_current_time", '{"timezone":"UTC"}'
                            ),
                        ),
                        finish_reason=LLMFinishReason.TOOL_CALLS,
                    ),
                )
            ),
            ScriptedProviderRound(events=(_completion(content="TCP uses [K1]."),)),
        ]
    )

    completion = await _execute_workflow(
        _workflow(provider, tool_executor=_rag_executor(service)),
        knowledge_base_id=44,
    )

    assert service.calls == [(1, 44, "TCP", 5)]
    assert [citation.evidence_handle for citation in completion.citations] == [1]


@pytest.mark.anyio
async def test_grounded_citation_validation_and_second_search_stop_before_service() -> (
    None
):
    service = _FakeRetrievalService()
    search_call = ToolCall("search-1", "search_knowledge", '{"query":"TCP"}')
    missing_citation = FakeProvider(
        [
            ScriptedProviderRound(
                events=(
                    _completion(
                        content="",
                        tool_calls=(search_call,),
                        finish_reason=LLMFinishReason.TOOL_CALLS,
                    ),
                )
            ),
            ScriptedProviderRound(events=(_completion(content="uncited"),)),
        ]
    )
    with pytest.raises(WorkflowFailure) as raised:
        await _execute_workflow(
            _workflow(missing_citation, tool_executor=_rag_executor(service)),
            knowledge_base_id=44,
        )
    assert raised.value.error_code is RunErrorCode.LLM_RESPONSE_INVALID

    insufficient = FakeProvider(
        [
            ScriptedProviderRound(
                events=(
                    _completion(
                        content="",
                        tool_calls=(search_call,),
                        finish_reason=LLMFinishReason.TOOL_CALLS,
                    ),
                )
            ),
            ScriptedProviderRound(
                events=(_completion(content="[[INSUFFICIENT_EVIDENCE]]"),)
            ),
        ]
    )
    abstained = await _execute_workflow(
        _workflow(insufficient, tool_executor=_rag_executor(service)),
        knowledge_base_id=44,
    )
    assert abstained.abstained is True
    assert abstained.citations == ()

    repeated_search = FakeProvider(
        [
            ScriptedProviderRound(
                events=(
                    _completion(
                        content="",
                        tool_calls=(search_call,),
                        finish_reason=LLMFinishReason.TOOL_CALLS,
                    ),
                )
            ),
            ScriptedProviderRound(
                events=(
                    _completion(
                        content="",
                        tool_calls=(
                            ToolCall("search-2", "search_knowledge", '{"query":"TCP"}'),
                        ),
                        finish_reason=LLMFinishReason.TOOL_CALLS,
                    ),
                )
            ),
        ]
    )
    with pytest.raises(WorkflowFailure) as raised:
        await _execute_workflow(
            _workflow(repeated_search, tool_executor=_rag_executor(service)),
            knowledge_base_id=44,
        )
    assert raised.value.error_code is RunErrorCode.AGENT_EXECUTION_LIMIT
    assert service.calls == [(1, 44, "TCP", 5)] * 3


@pytest.mark.anyio
async def test_failed_knowledge_search_returns_safe_unavailable_completion() -> None:
    service = _FakeRetrievalService(
        error=KnowledgeSearchError("KNOWLEDGE_INDEX_NOT_READY", retryable=False)
    )
    search_call = ToolCall("search-1", "search_knowledge", '{"query":"TCP"}')
    provider = FakeProvider(
        [
            ScriptedProviderRound(
                events=(
                    _completion(
                        content="",
                        tool_calls=(search_call,),
                        finish_reason=LLMFinishReason.TOOL_CALLS,
                    ),
                )
            ),
            ScriptedProviderRound(
                events=(_completion(content="[[INSUFFICIENT_EVIDENCE]]"),)
            ),
        ]
    )

    completion = await _execute_workflow(
        _workflow(provider, tool_executor=_rag_executor(service)),
        knowledge_base_id=44,
    )

    assert completion.content == "知识库检索未成功，暂时无法基于资料可靠回答。"
    assert completion.content != INSUFFICIENT_EVIDENCE_ANSWER
    assert completion.abstained is False
    assert completion.citations == ()
    assert service.calls == [(1, 44, "TCP", 5)]


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

    success = await _execute_workflow(_workflow(provider))

    assert success.content == "现在是 09:00 UTC。"
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

    await _execute_workflow(
        _workflow(
            provider,
            context_builder=cast(AnswerContextBuilder, _StaticContextBuilder(context)),
        )
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

    success = await _execute_workflow(_workflow(provider))

    assert success.content == "工具路径已完成。"
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

    success = await _execute_workflow(_workflow(provider))

    assert success.content == "请提供有效时区。"
    tool_result = cast(ToolResult, provider.requests[1].transcript[2])
    assert tool_result.kind is ToolResultKind.INVALID_ARGUMENTS


class _TrackingTimeTool(CurrentTimeTool):
    def __init__(self) -> None:
        self.executed: list[str] = []

    def validate_arguments(self, arguments: dict[str, object]) -> bool:
        return set(arguments) == {"timezone"} and isinstance(arguments["timezone"], str)

    async def execute(
        self, arguments: dict[str, object], context: ToolContext | None
    ) -> ToolExecutionOutput:
        del context
        self.executed.append("get_current_time")
        return ToolExecutionOutput(
            observation='{"timezone":"UTC","datetime":"2026-08-14T09:00:00+00:00"}'
        )


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
        await _execute_workflow(
            _workflow(
                provider,
                tool_executor=ToolExecutor(tools=(tracking_tool,)),
                max_tool_calls=1,
            )
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
        await _execute_workflow(_workflow(provider, max_tool_calls=0))

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
        await _execute_workflow(_workflow(round_limited, max_llm_rounds=1))
    assert round_raised.value.error_code is RunErrorCode.AGENT_EXECUTION_LIMIT

    started = asyncio.Event()
    blocked_until = asyncio.Event()
    deadline_limited = FakeProvider(
        [ScriptedProviderRound(events=(), started=started, blocked_until=blocked_until)]
    )
    task = asyncio.create_task(
        _execute_workflow(_workflow(deadline_limited, overall_deadline_seconds=0.01))
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
        _execute_workflow(
            _workflow(
                FakeProvider([]),
                context_builder=cast(AnswerContextBuilder, context_deadline_limited),
                overall_deadline_seconds=0.01,
            )
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
        await _execute_workflow(_workflow(provider))

    assert raised.value.error_code is RunErrorCode.LLM_RESPONSE_INVALID


@pytest.mark.anyio
async def test_workflow_cancellation_propagates_through_the_blocked_provider() -> None:
    started = asyncio.Event()
    events: list[str] = []
    provider = FakeProvider(
        [
            ScriptedProviderRound(
                events=(), started=started, blocked_until=asyncio.Event()
            )
        ]
    )
    task = asyncio.create_task(
        _execute_workflow(
            _workflow(
                provider,
                overall_deadline_seconds=10,
                tracer=_LifecycleTracer(events),
            )
        )
    )
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert events == ["begin", "failure:CANCELLED", "root_failure:CANCELLED"]


@pytest.mark.anyio
async def test_graph_scope_disables_automatic_langsmith_tracing_despite_global_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    provider = _TracingContextCheckingProvider(
        FakeProvider([ScriptedProviderRound(events=(_completion(content="正常回答"),))])
    )

    success = await _execute_workflow(_workflow(cast(FakeProvider, provider)))

    assert success.content == "正常回答"
    assert provider.enabled_values == [False]
