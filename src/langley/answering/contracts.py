"""Provider-neutral contracts for one Learning Assistant LLM round.

These types deliberately describe transient runtime protocol only.  They do
not map to MySQL Message records, vendor SDK objects, or LangGraph state.
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, TypeAlias

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


class ToolResultKind(StrEnum):
    """System-observed outcomes that can be returned to an Agent."""

    SUCCESS = "SUCCESS"
    NOT_ALLOWED = "NOT_ALLOWED"
    INVALID_ARGUMENTS = "INVALID_ARGUMENTS"
    TOOL_ERROR = "TOOL_ERROR"


class LLMFinishReason(StrEnum):
    """Normalized terminal semantics for one canonical provider completion."""

    STOP = "STOP"
    TOOL_CALLS = "TOOL_CALLS"
    LENGTH = "LENGTH"
    FILTERED = "FILTERED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class ToolSpec:
    """A tool the provider may call."""

    name: str
    description: str
    arguments_schema: dict[str, JSONValue]


@dataclass(frozen=True)
class ToolCall:
    """One canonical, model-generated Tool invocation intent."""

    call_id: str
    name: str
    raw_arguments: str


@dataclass(frozen=True)
class ToolResult:
    """One system-observed Tool outcome correlated by its stable call identity."""

    call_id: str
    name: str
    kind: ToolResultKind
    content: str


@dataclass(frozen=True)
class UserRuntimeMessage:
    """A USER input carried only in the transient provider runtime transcript."""

    content: str


@dataclass(frozen=True)
class AssistantRuntimeMessage:
    """A canonical Assistant round retained for a later provider request."""

    content: str
    tool_calls: tuple[ToolCall, ...]


RuntimeTranscriptItem: TypeAlias = (
    UserRuntimeMessage | AssistantRuntimeMessage | ToolResult
)


@dataclass(frozen=True)
class LLMRequest:
    """A provider-neutral request for exactly one LLM round."""

    system_input: str
    transcript: tuple[RuntimeTranscriptItem, ...]
    allowed_tools: tuple[ToolSpec, ...]


@dataclass(frozen=True)
class LLMUsage:
    """Optional normalized usage reported for one completed provider round."""

    input_tokens: int | None
    output_tokens: int | None


@dataclass(frozen=True)
class LLMResponseCompleted:
    """The canonical and authoritative transient result of one LLM round."""

    assistant_content: str
    tool_calls: tuple[ToolCall, ...]
    finish_reason: LLMFinishReason
    usage: LLMUsage | None


@dataclass(frozen=True)
class AssistantContentDelta:
    """A transient presentation fragment, never a canonical completion."""

    content: str


LLMStreamEvent: TypeAlias = AssistantContentDelta | LLMResponseCompleted


class LLMProvider(Protocol):
    """Perform one normalized streaming LLM round without owning an Agent loop."""

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        """Yield presentation fragments followed by one canonical completion."""
