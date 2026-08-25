"""System-owned Tool capability envelope and execution boundary for Slice 4."""

import asyncio
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Annotated, Protocol, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
)

from langley.answering.contracts import (
    JSONValue,
    ToolCall,
    ToolResult,
    ToolResultKind,
    ToolSpec,
)
from langley.answering.errors import RunErrorCode, WorkflowFailure
from langley.answering.tracing import tool_trace_context
from langley.knowledge.retrieval import RetrievalHit
from langley.knowledge.retrieval_service import (
    KnowledgeRetrievalService,
    KnowledgeSearchError,
)

if TYPE_CHECKING:
    from langley.answering.tracing import ExecutionTrace, ToolTrace


@dataclass(frozen=True)
class ToolContext:
    """Server-owned scope facts shared by every call in one tool batch."""

    run_id: int
    user_id: int
    knowledge_base_id: int | None


@dataclass(frozen=True)
class ToolExecutionOutput:
    """One transient tool outcome split between model and workflow channels."""

    observation: str
    retrieval_hits: tuple[RetrievalHit, ...] = ()


class AgentTool(Protocol):
    """Minimal tool contract used by the executor registry."""

    spec: ToolSpec

    def validate_arguments(self, arguments: dict[str, JSONValue]) -> bool:
        """Return whether model arguments satisfy this tool's schema."""

    async def execute(
        self,
        arguments: dict[str, JSONValue],
        context: ToolContext | None,
    ) -> ToolExecutionOutput:
        """Execute using only validated arguments and server-owned context."""


class ToolExecutionError(Exception):
    """A safe, expected tool observation rather than a workflow failure."""

    def __init__(self, code: str, *, retryable: bool) -> None:
        super().__init__(code)
        self.code = code
        self.retryable = retryable


class CurrentTimeArguments(BaseModel):
    """The one structural contract for the model-visible time Tool arguments."""

    model_config = ConfigDict(extra="forbid")

    timezone: str

    @field_validator("timezone")
    @classmethod
    def timezone_must_be_an_iana_identifier(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as error:
            raise ValueError("timezone must be a valid IANA identifier") from error
        return value


class CurrentTimeTool:
    """Read-only implementation of the Slice 4 time lookup capability."""

    spec = ToolSpec(
        name="get_current_time",
        description="Get the current local time for a valid IANA timezone identifier.",
        arguments_schema=cast(
            dict[str, JSONValue], CurrentTimeArguments.model_json_schema()
        ),
    )

    def __init__(self, clock: Callable[[], datetime] | None = None) -> None:
        self._clock = clock or (lambda: datetime.now(UTC))

    def validate_arguments(self, arguments: dict[str, JSONValue]) -> bool:
        try:
            CurrentTimeArguments.model_validate(arguments)
        except ValidationError:
            return False
        return True

    async def execute(
        self,
        arguments: dict[str, JSONValue],
        context: ToolContext | None = None,
    ) -> ToolExecutionOutput:
        del context
        validated = CurrentTimeArguments.model_validate(arguments)
        instant = self._clock()
        if instant.tzinfo is None or instant.utcoffset() is None:
            raise ValueError("clock must return a timezone-aware datetime")

        local_time = instant.astimezone(ZoneInfo(validated.timezone))
        return ToolExecutionOutput(
            observation=json.dumps(
                {"timezone": validated.timezone, "datetime": local_time.isoformat()},
                separators=(",", ":"),
            )
        )


class SearchKnowledgeArguments(BaseModel):
    """The complete model-visible schema for the scoped knowledge search tool."""

    model_config = ConfigDict(extra="forbid")

    query: Annotated[
        str,
        StringConstraints(strict=True, strip_whitespace=True, min_length=1),
    ]
    top_k: Annotated[int, Field(strict=True, ge=1, le=5)] = 5


class SearchKnowledgeTool:
    """Expose one server-scoped dense retrieval capability to an Agent."""

    spec = ToolSpec(
        name="search_knowledge",
        description=(
            "Search the knowledge base already bound to this run and accessible "
            "to its user for facts or sources. Do not use for casual chat, pure "
            "calculation, or general knowledge. This tool cannot expand scope."
        ),
        arguments_schema=cast(
            dict[str, JSONValue], SearchKnowledgeArguments.model_json_schema()
        ),
    )

    def __init__(self, retrieval_service: KnowledgeRetrievalService) -> None:
        self._retrieval_service = retrieval_service

    def validate_arguments(self, arguments: dict[str, JSONValue]) -> bool:
        try:
            SearchKnowledgeArguments.model_validate(arguments)
        except ValidationError:
            return False
        return True

    async def execute(
        self,
        arguments: dict[str, JSONValue],
        context: ToolContext | None,
    ) -> ToolExecutionOutput:
        validated = SearchKnowledgeArguments.model_validate(arguments)
        if context is None or context.knowledge_base_id is None:
            raise ToolExecutionError("KNOWLEDGE_SCOPE_UNAVAILABLE", retryable=False)

        try:
            result = await self._retrieval_service.search(
                user_id=context.user_id,
                knowledge_base_id=context.knowledge_base_id,
                query=validated.query,
                top_k=validated.top_k,
            )
        except KnowledgeSearchError as error:
            raise ToolExecutionError(error.code, retryable=error.retryable) from error

        return ToolExecutionOutput(
            observation=json.dumps(
                {
                    "evidence": [
                        {
                            "evidence_handle": evidence_handle,
                            "content": hit.content,
                            "source_display_name": hit.source_display_name,
                            "heading_path": list(hit.heading_path),
                        }
                        for evidence_handle, hit in enumerate(result.hits, start=1)
                    ]
                },
                separators=(",", ":"),
            ),
            retrieval_hits=result.hits,
        )


class ToolExecutor:
    """Validate and execute tools available to the agent."""

    def __init__(self, *, tools: Iterable[AgentTool] | None = None) -> None:
        resolved_tools = (CurrentTimeTool(),) if tools is None else tuple(tools)
        self._tools_by_name = self._build_registry(resolved_tools)

    @property
    def allowed_tools(self) -> tuple[ToolSpec, ...]:
        """Return the model-visible capability envelope in canonical order."""

        return tuple(tool.spec for tool in self._tools_by_name.values())

    async def execute_batch(
        self,
        calls: tuple[ToolCall, ...],
        *,
        context: ToolContext | None = None,
        on_tool_execution: (
            Callable[[ToolCall, ToolExecutionOutput], None] | None
        ) = None,
        trace: "ExecutionTrace | None" = None,
        tool_calls_used_start: int = 0,
    ) -> tuple[ToolResult, ...]:
        """Execute independent read-only calls serially in canonical order."""

        self._validate_call_identities(calls)
        results: list[ToolResult] = []
        for call_index, call in enumerate(calls, start=1):
            tool_trace = self._start_tool_trace(
                trace,
                call,
                tool_calls_used_start + call_index,
            )
            result: ToolResult | None = None
            error_code: str | None = None
            try:
                with tool_trace_context(tool_trace):
                    result = await self._execute_call(
                        call,
                        context,
                        on_tool_execution,
                    )
            except asyncio.CancelledError:
                error_code = "CANCELLED"
                raise
            except WorkflowFailure as error:
                error_code = error.error_code.value
                raise
            except Exception:
                error_code = RunErrorCode.TOOL_EXECUTION_FAILED.value
                raise
            finally:
                self._finish_tool_trace(tool_trace, result, error_code)
            assert result is not None
            results.append(result)
        return tuple(results)

    async def _execute_call(
        self,
        call: ToolCall,
        context: ToolContext | None,
        on_tool_execution: Callable[[ToolCall, ToolExecutionOutput], None] | None,
    ) -> ToolResult:
        tool = self._tools_by_name.get(call.name)
        if tool is None:
            return self._result(call, ToolResultKind.NOT_ALLOWED)

        arguments = self._parse_arguments(call.raw_arguments)
        if arguments is None or not tool.validate_arguments(arguments):
            return self._result(call, ToolResultKind.INVALID_ARGUMENTS)

        try:
            output = await tool.execute(arguments, context)
        except asyncio.CancelledError:
            raise
        except ToolExecutionError as error:
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                kind=ToolResultKind.TOOL_ERROR,
                content=json.dumps(
                    {
                        "error": {
                            "code": error.code,
                            "retryable": error.retryable,
                        }
                    },
                    separators=(",", ":"),
                ),
            )
        except Exception as error:
            raise WorkflowFailure(RunErrorCode.TOOL_EXECUTION_FAILED) from error

        if on_tool_execution is not None:
            on_tool_execution(call, output)

        return ToolResult(
            call_id=call.call_id,
            name=call.name,
            kind=ToolResultKind.SUCCESS,
            content=output.observation,
        )

    @staticmethod
    def _parse_arguments(raw_arguments: str) -> dict[str, JSONValue] | None:
        try:
            parsed = json.loads(raw_arguments)
        except (json.JSONDecodeError, TypeError):
            return None
        return parsed if isinstance(parsed, dict) else None

    @staticmethod
    def _result(call: ToolCall, kind: ToolResultKind) -> ToolResult:
        return ToolResult(
            call_id=call.call_id,
            name=call.name,
            kind=kind,
            content=json.dumps({"error": kind.value}, separators=(",", ":")),
        )

    @staticmethod
    def _validate_call_identities(calls: tuple[ToolCall, ...]) -> None:
        call_ids = [call.call_id for call in calls]
        if any(not call_id.strip() for call_id in call_ids) or len(
            set(call_ids)
        ) != len(call_ids):
            raise WorkflowFailure(RunErrorCode.LLM_RESPONSE_INVALID)

    @staticmethod
    def _build_registry(tools: tuple[AgentTool, ...]) -> dict[str, AgentTool]:
        registry: dict[str, AgentTool] = {}
        for tool in tools:
            if tool.spec.name in registry:
                raise ValueError(f"duplicate tool registration: {tool.spec.name}")
            registry[tool.spec.name] = tool
        return registry

    @staticmethod
    def _start_tool_trace(
        trace: "ExecutionTrace | None",
        call: ToolCall,
        tool_calls_used: int,
    ) -> "ToolTrace | None":
        if trace is None:
            return None
        try:
            return trace.begin_tool(call, tool_calls_used)
        except Exception:
            return None

    @staticmethod
    def _finish_tool_trace(
        trace: "ToolTrace | None",
        result: ToolResult | None,
        error_code: str | None,
    ) -> None:
        if trace is None:
            return
        try:
            trace.finish(result, error_code)
        except Exception:
            return
