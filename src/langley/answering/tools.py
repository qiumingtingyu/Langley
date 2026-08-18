"""System-owned Tool capability envelope and execution boundary for Slice 4."""

import asyncio
import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from langley.answering.contracts import (
    JSONValue,
    ToolCall,
    ToolResult,
    ToolResultKind,
    ToolSpec,
)
from langley.answering.errors import RunErrorCode, WorkflowFailure


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

    async def execute(self, arguments: dict[str, JSONValue]) -> str:
        validated = CurrentTimeArguments.model_validate(arguments)
        instant = self._clock()
        if instant.tzinfo is None or instant.utcoffset() is None:
            raise ValueError("clock must return a timezone-aware datetime")

        local_time = instant.astimezone(ZoneInfo(validated.timezone))
        return json.dumps(
            {"timezone": validated.timezone, "datetime": local_time.isoformat()},
            separators=(",", ":"),
        )


class ToolExecutor:
    """Validate and execute tools available to the agent."""

    def __init__(self, current_time_tool: CurrentTimeTool | None = None) -> None:
        self._current_time_tool = current_time_tool or CurrentTimeTool()

    @property
    def allowed_tools(self) -> tuple[ToolSpec, ...]:
        """Return the model-visible capability envelope in canonical order."""

        return (self._current_time_tool.spec,)

    async def execute_batch(
        self, calls: tuple[ToolCall, ...]
    ) -> tuple[ToolResult, ...]:
        """Execute independent read-only calls serially in canonical order."""

        self._validate_call_identities(calls)
        results: list[ToolResult] = []
        for call in calls:
            results.append(await self._execute_call(call))
        return tuple(results)

    async def _execute_call(self, call: ToolCall) -> ToolResult:
        if call.name != self._current_time_tool.spec.name:
            return self._result(call, ToolResultKind.NOT_ALLOWED)

        arguments = self._parse_arguments(call.raw_arguments)
        if arguments is None or not self._current_time_tool.validate_arguments(
            arguments
        ):
            return self._result(call, ToolResultKind.INVALID_ARGUMENTS)

        try:
            content = await self._current_time_tool.execute(arguments)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            raise WorkflowFailure(RunErrorCode.TOOL_EXECUTION_FAILED) from error

        return ToolResult(
            call_id=call.call_id,
            name=call.name,
            kind=ToolResultKind.SUCCESS,
            content=content,
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
