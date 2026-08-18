"""Stable Learning Assistant failure categories.

Only the execution shell turns these typed failures into authoritative Run
state.  Provider, Tool, and Graph layers must not mutate Runs themselves.
"""

from enum import StrEnum


class RunErrorCode(StrEnum):
    """Stable Slice 4 and retained execution-shell Run error categories."""

    LLM_PROVIDER_FAILED = "LLM_PROVIDER_FAILED"
    LLM_RESPONSE_INVALID = "LLM_RESPONSE_INVALID"
    ANSWER_CONTEXT_TOO_LARGE = "ANSWER_CONTEXT_TOO_LARGE"
    AGENT_EXECUTION_LIMIT = "AGENT_EXECUTION_LIMIT"
    AGENT_EXECUTION_TIMEOUT = "AGENT_EXECUTION_TIMEOUT"
    TOOL_EXECUTION_FAILED = "TOOL_EXECUTION_FAILED"
    ANSWER_EXECUTION_FAILED = "ANSWER_EXECUTION_FAILED"
    PROCESS_INTERRUPTED = "PROCESS_INTERRUPTED"


class WorkflowFailure(Exception):
    """A typed, safe-to-persist failure from Learning Assistant execution."""

    def __init__(self, error_code: RunErrorCode) -> None:
        super().__init__(error_code.value)
        self.error_code = error_code
