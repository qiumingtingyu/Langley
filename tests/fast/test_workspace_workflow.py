"""Production graph repair loop, runtime cleanup, and standalone envelope."""

import asyncio
import json
from dataclasses import dataclass

import pytest

from langley.answering.context_builder import AnswerContext
from langley.answering.contracts import (
    LLMFinishReason,
    LLMResponseCompleted,
    ToolCall,
    ToolResult,
)
from langley.answering.errors import RunErrorCode, WorkflowFailure
from langley.answering.fake_provider import FakeProvider, ScriptedProviderRound
from langley.answering.tools import ToolContext, ToolExecutor
from langley.answering.tracing import _NoopTrace
from langley.answering.workflow import LearningAssistantWorkflow
from langley.answering.workspace_tools import WORKSPACE_TOOL_NAMES, WorkspaceTool
from langley.settings import Settings
from langley.workspace_storage import WorkspaceStorage


@dataclass
class StaticContext:
    context: AnswerContext

    async def build(self, *args, **kwargs):
        return self.context


def round_(name=None, args=None, calls=None):
    return ScriptedProviderRound(
        events=(
            LLMResponseCompleted(
                assistant_content="done" if name is None and calls is None else "",
                tool_calls=calls
                or (() if name is None else (ToolCall(name, name, json.dumps(args)),)),
                finish_reason=LLMFinishReason.STOP
                if name is None and calls is None
                else LLMFinishReason.TOOL_CALLS,
                usage=None,
            ),
        )
    )


def workflow(storage, provider):
    return LearningAssistantWorkflow(
        context_builder=StaticContext(
            AnswerContext(
                (),
                "repair",
                user_id=1,
                workspace_id=1,
                workspace_name="demo",
                workspace_storage_key="a" * 32,
            )
        ),
        provider=provider,
        tool_executor=ToolExecutor(
            tools=[WorkspaceTool(n, storage) for n in sorted(WORKSPACE_TOOL_NAMES)]
        ),
        workspace_storage=storage,
        max_llm_rounds=4,
        max_tool_calls=3,
        overall_deadline_seconds=10,
        provider_name="fake",
        model="script",
    )


async def execute(flow):
    async def delta(_):
        pass

    return await flow.execute(
        None,
        run_id=1,
        user_id=1,
        conversation_id=1,
        input_message_id=1,
        knowledge_base_id=None,
        on_assistant_delta=delta,
    )


def test_workspace_system_guidance_distinguishes_shell_and_compute():
    prompt = LearningAssistantWorkflow._system_input(
        ToolContext(1, 1, None, workspace_id=1)
    )
    assert all(
        guidance in prompt
        for guidance in (
            "Each command starts a fresh shell (bash -lc);"
            " shell-local state is not inherited.",
            "Commands in the same Run normally reuse the same container and its"
            " container-local filesystem state (such as /tmp).",
            "A command timeout or execution reset destroys that container;"
            " the next run_command then starts fresh compute.",
            "Workspace files persist across compute resets and Runs.",
        )
    )


def test_write_command_failure_repair_observation_loop(tmp_path, monkeypatch):
    storage = WorkspaceStorage(Settings(workspace_storage_root=tmp_path))
    storage.workspace_root("a" * 32).mkdir()
    closed = []

    class CommandBoundary:
        def __init__(self, root, settings):
            self.root = root

        async def execute(self, command, timeout_seconds=None):
            value = (self.root / "answer.txt").read_text()
            return {
                "exit_code": 1 if value == "bad" else 0,
                "output": value,
                "timed_out": False,
                "truncated": False,
            }

        async def close(self):
            closed.append(True)

    monkeypatch.setattr("langley.answering.workflow.SandboxRuntime", CommandBoundary)
    provider = FakeProvider(
        [
            round_(
                calls=(
                    ToolCall("r", "expand_evidence", "{}"),
                    ToolCall("w", "write_file", '{"path":"ignored","content":"bad"}'),
                )
            ),
            round_("write_file", {"path": "answer.txt", "content": "bad"}),
            round_("run_command", {"command": "check"}),
            round_(
                "edit_file",
                {
                    "path": "answer.txt",
                    "edits": [{"old_text": "bad", "new_text": "good"}],
                },
            ),
            round_("run_command", {"command": "check"}),
            round_(),
        ]
    )
    flow = workflow(storage, provider)
    completion = asyncio.run(execute(flow))
    assert completion.workspace_changes["added"] == ["answer.txt"]
    assert closed and len(provider.requests) == 6
    assert "Top-level" in provider.requests[0].system_input
    assert not storage.resolve("a" * 32, "ignored").exists()
    observations = [
        item
        for item in provider.requests[-1].transcript
        if isinstance(item, ToolResult)
    ]
    assert "SIDE_EFFECT_BATCH_UNSUPPORTED" in observations[0].content
    command_results = [
        json.loads(r.content)["exit_code"]
        for r in observations
        if r.name == "run_command"
    ]
    assert command_results == [1, 0]
    standalone = ToolContext(1, 1, None)
    assert not WORKSPACE_TOOL_NAMES.intersection(
        t.name for t in flow._allowed_tools(standalone)
    )
    assert "Workspace overview" not in flow._system_input(standalone)


@pytest.mark.parametrize("end", ["failure", "cancel", "deadline"])
def test_terminal_cleanup_preserves_files(tmp_path, monkeypatch, end):
    storage = WorkspaceStorage(Settings(workspace_storage_root=tmp_path))
    storage.workspace_root("a" * 32).mkdir()
    closed = []

    class Boundary:
        def __init__(self, *args):
            pass

        async def close(self):
            closed.append(True)

    monkeypatch.setattr("langley.answering.workflow.SandboxRuntime", Boundary)

    async def check():
        started = asyncio.Event()
        provider = FakeProvider(
            [
                round_("write_file", {"path": "kept", "content": "yes"}),
                ScriptedProviderRound(
                    events=(),
                    failure=RuntimeError("failed") if end == "failure" else None,
                    started=started,
                    blocked_until=asyncio.Event() if end != "failure" else None,
                ),
            ]
        )
        flow = workflow(storage, provider)
        if end == "deadline":
            flow._overall_deadline_seconds = 0.5
        task = asyncio.create_task(execute(flow))
        await asyncio.wait_for(started.wait(), timeout=2)
        if end == "cancel":
            task.cancel()
        with pytest.raises(
            asyncio.CancelledError if end == "cancel" else WorkflowFailure
        ):
            await task
        assert storage.read_file("a" * 32, "kept")["content"] == "yes"
        assert closed and not storage.execution_lock("a" * 32).locked()

    asyncio.run(check())


@pytest.mark.parametrize("used", [8, 9])
def test_mixed_batch_budget_precedes_rejection_and_traces_it(
    tmp_path, monkeypatch, used
):
    storage = WorkspaceStorage(Settings(workspace_storage_root=tmp_path))
    storage.workspace_root("a" * 32).mkdir()

    def reads(count):
        return tuple(ToolCall(f"r{i}", "list_files", "{}") for i in range(count))

    mixed = (
        ToolCall("read", "list_files", "{}"),
        ToolCall("write", "write_file", '{"path":"must-not-exist","content":"bad"}'),
    )
    provider = FakeProvider(
        [
            round_(calls=reads(3)),
            round_(calls=reads(3)),
            round_(calls=reads(used - 6)),
            round_(calls=mixed),
            round_(),
        ]
    )
    flow = workflow(storage, provider)
    records = []

    class RecordedTool:
        def __init__(self, call, ordinal):
            self.call = call
            self.ordinal = ordinal

        def finish(self, result, error_code=None, metadata=None):
            records.append((self.call, self.ordinal, result, error_code, metadata))

    class Trace(_NoopTrace):
        def begin_tool(self, call, tool_calls_used):
            return RecordedTool(call, tool_calls_used)

    monkeypatch.setattr(flow, "_start_trace", lambda _: Trace())
    if used == 9:
        with pytest.raises(WorkflowFailure) as error:
            asyncio.run(execute(flow))
        assert error.value.error_code is RunErrorCode.AGENT_EXECUTION_LIMIT
        assert len(records) == 9
    else:
        asyncio.run(execute(flow))
        assert [record[1] for record in records] == list(range(1, 11))
        for call, _, result, error_code, metadata in records[-2:]:
            assert result.call_id == call.call_id
            assert (
                json.loads(result.content)["error"]["code"]
                == "SIDE_EFFECT_BATCH_UNSUPPORTED"
            )
            assert error_code is None
            assert metadata == {
                "success": False,
                "executed": False,
                "tool_error_code": "SIDE_EFFECT_BATCH_UNSUPPORTED",
                "retryable": False,
            }
        assert len(provider.requests) == 5
    assert not storage.resolve("a" * 32, "must-not-exist").exists()
    assert all(record[2].kind.value == "SUCCESS" for record in records[:used])
