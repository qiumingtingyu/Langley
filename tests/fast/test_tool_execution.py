"""Regression tests for the Slice 4 Tool execution boundary."""

import asyncio
import json
from datetime import UTC, datetime

import pytest

from langley.answering.contracts import JSONValue, ToolCall, ToolResultKind
from langley.answering.errors import RunErrorCode, WorkflowFailure
from langley.answering.tools import (
    CurrentTimeArguments,
    CurrentTimeTool,
    ToolExecutor,
)


def _call(
    call_id: str = "call-1",
    name: str = "get_current_time",
    raw_arguments: str = '{"timezone":"UTC"}',
) -> ToolCall:
    return ToolCall(call_id=call_id, name=name, raw_arguments=raw_arguments)


@pytest.mark.anyio
async def test_time_tool_contract_drives_exposure_validation_and_dispatch() -> None:
    def fixed_clock() -> datetime:
        return datetime(2026, 8, 14, 9, 0, tzinfo=UTC)

    boundary = ToolExecutor(CurrentTimeTool(clock=fixed_clock))

    assert boundary.allowed_tools == (CurrentTimeTool.spec,)
    assert (
        CurrentTimeTool.spec.arguments_schema
        == CurrentTimeArguments.model_json_schema()
    )
    result = (
        await boundary.execute_batch(
            (_call(raw_arguments='{"timezone":"Asia/Tokyo"}'),)
        )
    )[0]

    assert result.call_id == "call-1"
    assert result.name == "get_current_time"
    assert result.kind is ToolResultKind.SUCCESS
    assert json.loads(result.content) == {
        "timezone": "Asia/Tokyo",
        "datetime": "2026-08-14T18:00:00+09:00",
    }


@pytest.mark.anyio
async def test_naive_injected_clock_is_rejected_as_a_typed_workflow_failure() -> None:
    def naive_clock() -> datetime:
        return datetime(2026, 8, 14, 9, 0)

    with pytest.raises(WorkflowFailure) as raised:
        await ToolExecutor(CurrentTimeTool(clock=naive_clock)).execute_batch((_call(),))

    assert raised.value.error_code is RunErrorCode.TOOL_EXECUTION_FAILED


@pytest.mark.anyio
async def test_unknown_tool_returns_safe_not_allowed_observation() -> None:
    result = (await ToolExecutor().execute_batch((_call(name="hidden_tool"),)))[0]

    assert result.call_id == "call-1"
    assert result.name == "hidden_tool"
    assert result.kind is ToolResultKind.NOT_ALLOWED
    assert json.loads(result.content) == {"error": "NOT_ALLOWED"}


@pytest.mark.anyio
@pytest.mark.parametrize(
    "raw_arguments",
    [
        "{",
        "[]",
        '{"timezone": 3}',
        '{"timezone":"Mars/Olympus"}',
        '{"timezone":"UTC","extra":true}',
    ],
)
async def test_malformed_or_schema_invalid_arguments_are_independent_safe_observations(
    raw_arguments: str,
) -> None:
    result = (
        await ToolExecutor().execute_batch((_call(raw_arguments=raw_arguments),))
    )[0]

    assert result.kind is ToolResultKind.INVALID_ARGUMENTS
    assert json.loads(result.content) == {"error": "INVALID_ARGUMENTS"}


class _UnexpectedFailureTimeTool(CurrentTimeTool):
    async def execute(self, arguments: dict[str, JSONValue]) -> str:
        raise RuntimeError("implementation defect")


class _CancelledTimeTool(CurrentTimeTool):
    async def execute(self, arguments: dict[str, JSONValue]) -> str:
        raise asyncio.CancelledError


@pytest.mark.anyio
async def test_unexpected_tool_failure_becomes_a_typed_workflow_failure() -> None:
    with pytest.raises(WorkflowFailure) as raised:
        await ToolExecutor(_UnexpectedFailureTimeTool()).execute_batch((_call(),))

    assert raised.value.error_code is RunErrorCode.TOOL_EXECUTION_FAILED


@pytest.mark.anyio
async def test_cancellation_propagates_from_a_tool() -> None:
    with pytest.raises(asyncio.CancelledError):
        await ToolExecutor(_CancelledTimeTool()).execute_batch((_call(),))


@pytest.mark.anyio
@pytest.mark.parametrize("calls", [(_call(call_id=""),), (_call(), _call())])
async def test_missing_or_duplicate_call_identity_is_invalid_provider_response(
    calls: tuple[ToolCall, ...],
) -> None:
    with pytest.raises(WorkflowFailure) as raised:
        await ToolExecutor().execute_batch(calls)

    assert raised.value.error_code is RunErrorCode.LLM_RESPONSE_INVALID
