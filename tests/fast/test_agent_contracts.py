"""Regression tests for Slice 4 provider-neutral contract boundaries."""

from langley.answering.contracts import (
    AssistantContentDelta,
    AssistantRuntimeMessage,
    LLMFinishReason,
    LLMRequest,
    LLMResponseCompleted,
    LLMUsage,
    ToolCall,
    ToolResult,
    ToolResultKind,
    ToolSpec,
    UserRuntimeMessage,
)
from langley.answering.errors import RunErrorCode, WorkflowFailure


def test_normalized_request_expresses_the_full_transient_agent_transcript() -> None:
    tool_call = ToolCall(
        call_id="call-1", name="get_current_time", raw_arguments='{"timezone":"UTC"}'
    )
    tool_result = ToolResult(
        call_id="call-1",
        name="get_current_time",
        kind=ToolResultKind.SUCCESS,
        content='{"timezone":"UTC","datetime":"2026-08-14T00:00:00+00:00"}',
    )
    request = LLMRequest(
        system_input="You are a learning assistant.",
        transcript=(
            UserRuntimeMessage(content="What time is it?"),
            AssistantRuntimeMessage(content="", tool_calls=(tool_call,)),
            tool_result,
        ),
        allowed_tools=(
            ToolSpec(
                name="get_current_time",
                description="Get the current time in an IANA timezone.",
                arguments_schema={"type": "object"},
            ),
        ),
    )

    assert request.transcript[1] == AssistantRuntimeMessage(
        content="", tool_calls=(tool_call,)
    )
    assert request.transcript[2] == tool_result


def test_canonical_completion_is_separate_from_transient_delta() -> None:
    delta = AssistantContentDelta(content="transient")
    completion = LLMResponseCompleted(
        assistant_content="canonical",
        tool_calls=(),
        finish_reason=LLMFinishReason.STOP,
        usage=LLMUsage(input_tokens=10, output_tokens=2),
    )

    assert delta.content == "transient"
    assert completion.assistant_content == "canonical"
    assert completion.assistant_content == "canonical"


def test_workflow_failure_exposes_only_a_stable_run_error_code() -> None:
    failure = WorkflowFailure(RunErrorCode.LLM_PROVIDER_FAILED)

    assert failure.error_code is RunErrorCode.LLM_PROVIDER_FAILED
    assert str(failure) == "LLM_PROVIDER_FAILED"


def test_run_error_codes_match_the_complete_frozen_slice_4_contract() -> None:
    assert {error_code.value for error_code in RunErrorCode} == {
        "LLM_PROVIDER_FAILED",
        "LLM_RESPONSE_INVALID",
        "ANSWER_CONTEXT_TOO_LARGE",
        "AGENT_EXECUTION_LIMIT",
        "AGENT_EXECUTION_TIMEOUT",
        "TOOL_EXECUTION_FAILED",
        "ANSWER_EXECUTION_FAILED",
        "PROCESS_INTERRUPTED",
    }
