"""Production Workflow fixtures for Slice 3 execution-shell integration tests."""

from datetime import UTC, datetime

from langley.answering.context_builder import AnswerContextBuilder
from langley.answering.contracts import LLMProvider
from langley.answering.tools import CurrentTimeTool, ToolExecutor
from langley.answering.workflow import LearningAssistantWorkflow


def workflow_for(
    provider: LLMProvider,
    *,
    max_llm_rounds: int = 4,
    max_tool_calls: int = 3,
    overall_deadline_seconds: float = 10.0,
) -> LearningAssistantWorkflow:
    """Use the real Workflow/Graph with a deterministic Provider boundary."""

    def fixed_clock() -> datetime:
        return datetime(2026, 8, 14, 9, 0, tzinfo=UTC)

    return LearningAssistantWorkflow(
        context_builder=AnswerContextBuilder(history_estimated_token_budget=16_000),
        provider=provider,
        tool_executor=ToolExecutor(CurrentTimeTool(clock=fixed_clock)),
        max_llm_rounds=max_llm_rounds,
        max_tool_calls=max_tool_calls,
        overall_deadline_seconds=overall_deadline_seconds,
        provider_name="fake",
        model="fake-script",
    )
