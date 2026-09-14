"""Local diagnostics, content separation, and strict Tool recovery invariants."""

import asyncio
import json
import subprocess

import pytest
from pydantic import ValidationError

from langley.answering.context_builder import AnswerContext
from langley.answering.contracts import (
    AssistantContentDelta,
    LLMFinishReason,
    LLMRequest,
    LLMResponseCompleted,
    LLMUsage,
    ToolCall,
    ToolResult,
    ToolResultKind,
    UserRuntimeMessage,
)
from langley.answering.errors import RunErrorCode, WorkflowFailure
from langley.answering.fake_provider import FakeProvider, ScriptedProviderRound
from langley.answering.local_tracing import CompositeTracer, LocalJsonTracer
from langley.answering.tools import ToolExecutor
from langley.answering.tracing import (
    LangSmithTracer,
    _LangSmithTrace,
    current_context_compaction_trace_parent,
)
from langley.answering.workspace_tools import (
    WORKSPACE_TOOL_NAMES,
    RunCommandArguments,
    WorkspaceTool,
)
from langley.main import create_app
from langley.sandbox import SandboxRuntime
from langley.settings import Settings
from langley.workspace_storage import WorkspaceStorage

COMPACT = dict(
    estimated_before=12000,
    estimated_after=8000,
    newly_compacted_turn_count=2,
    recent_raw_turn_count=1,
    compactor_model="fake",
    provider_model="fake-version",
    provider_input_tokens=20,
    provider_output_tokens=10,
    duration_ms=1.2,
    success=True,
    outcome="COMPACTED",
)

MODEL_TIMEOUT_SCHEMA = {
    "type": "integer",
    "minimum": 1,
    "description": "Timeout in whole seconds. Omit this field to use the default.",
}


def test_model_schema_changes_only_run_command_timeout(tmp_path):
    storage = WorkspaceStorage(Settings(workspace_storage_root=tmp_path))
    runtime_schema = RunCommandArguments.model_json_schema()
    for name in WORKSPACE_TOOL_NAMES:
        tool = WorkspaceTool(name, storage)
        expected = tool.arguments.model_json_schema()
        if name == "run_command":
            expected["properties"]["timeout_seconds"] = MODEL_TIMEOUT_SCHEMA
            assert tool.spec.arguments_schema["required"] == ["command"]
            assert tool.spec.arguments_schema["additionalProperties"] is False
        assert tool.spec.arguments_schema == expected
    # Schema exposure must not mutate Pydantic's runtime schema or defaults.
    assert RunCommandArguments.model_json_schema() == runtime_schema
    assert runtime_schema["properties"]["timeout_seconds"]["default"] is None
    assert runtime_schema["properties"]["timeout_seconds"]["anyOf"] == [
        {"minimum": 1, "type": "integer"},
        {"type": "null"},
    ]


@pytest.mark.parametrize(
    "timeout,expected",
    [
        ({"timeout_seconds": 1}, True),
        ({}, True),
        ({"timeout_seconds": None}, True),
        ({"timeout_seconds": "1"}, False),
        ({"timeout_seconds": 1.0}, False),
        ({"timeout_seconds": 0}, False),
    ],
)
def test_optional_integer_exposure_preserves_strict_runtime(
    tmp_path, timeout, expected
):
    tool = WorkspaceTool(
        "run_command", WorkspaceStorage(Settings(workspace_storage_root=tmp_path))
    )
    arguments = {"command": "echo ok", **timeout}
    assert tool.validate_arguments(arguments) is expected
    if expected:
        parsed = RunCommandArguments.model_validate(arguments)
        assert parsed.timeout_seconds == timeout.get("timeout_seconds")
    else:
        with pytest.raises(ValidationError):
            RunCommandArguments.model_validate(arguments)


def events(root, run_id=701):
    return [
        json.loads(line)
        for line in (root / f"run-{run_id}.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]


@pytest.mark.parametrize(
    "include_content,fail,broken_writer",
    [
        (True, False, False),
        (False, False, False),
        (True, True, False),
        (False, True, False),
        (True, False, True),
        (True, True, True),
    ],
)
def test_production_workflow_local_trace_and_recovery(
    tmp_path, monkeypatch, include_content, fail, broken_writer
):
    root = tmp_path / "diagnostics"
    if broken_writer:
        root.write_text("not a directory")
    settings = Settings(
        database_url="mysql+asyncmy://unused/langley_test",
        workspace_storage_root=tmp_path / "workspace",
        local_run_diagnostics_enabled=True,
        local_run_diagnostics_include_content=include_content,
        local_run_diagnostics_root=root,
        tracing_enabled=False,
        trace_content_enabled=False,
        workspace_max_llm_rounds=3,
    )
    invalid = ToolCall(
        "invalid",
        "run_command",
        '{"command":"echo private-command","timeout_seconds":"1"}',
    )
    valid = ToolCall(
        "valid", "run_command", '{"command":"echo private-command","timeout_seconds":1}'
    )
    proposed = [invalid, invalid, invalid] if fail else [invalid, valid, None]
    provider = FakeProvider(
        [
            ScriptedProviderRound(
                events=(
                    LLMResponseCompleted(
                        assistant_content="private-answer" if call is None else "",
                        tool_calls=() if call is None else (call,),
                        finish_reason=LLMFinishReason.STOP
                        if call is None
                        else LLMFinishReason.TOOL_CALLS,
                        usage=LLMUsage(10, 5),
                        provider_model="fake-version",
                    ),
                )
            )
            for call in proposed
        ]
    )
    app = create_app(settings, provider=provider)
    storage = app.state.workspace_storage
    storage.workspace_root("a" * 32).mkdir(parents=True)
    flow = app.state.execution_manager._workflow_factory()

    class Context:
        async def build(self, *args, **kwargs):
            current_context_compaction_trace_parent().context_compact(**COMPACT)
            return AnswerContext(
                (),
                "private-user",
                user_id=1,
                workspace_id=1,
                workspace_name="demo",
                workspace_storage_key="a" * 32,
            )

    flow._context_builder = Context()
    executed = []

    async def sandbox_execute(self, command, timeout_seconds=None):
        executed.append((command, timeout_seconds))
        return {
            "exit_code": 0,
            "output": "private-result",
            "timed_out": False,
            "truncated": False,
        }

    monkeypatch.setattr(SandboxRuntime, "execute", sandbox_execute)

    async def run():
        async def delta(_):
            pass

        try:
            return await flow.execute(
                None,
                run_id=701,
                user_id=1,
                conversation_id=1,
                input_message_id=1,
                knowledge_base_id=None,
                on_assistant_delta=delta,
            )
        finally:
            await app.state.database_engine.dispose()

    if fail:
        with pytest.raises(WorkflowFailure) as caught:
            asyncio.run(run())
        assert caught.value.error_code is RunErrorCode.AGENT_EXECUTION_LIMIT
        assert executed == []
    else:
        assert asyncio.run(run()).content == "private-answer"
        assert executed == [("echo private-command", 1)]
        recovery = next(
            item
            for item in provider.requests[1].transcript
            if isinstance(item, ToolResult)
        )
        assert recovery.kind is ToolResultKind.INVALID_ARGUMENTS
        assert json.loads(recovery.content)["error"]["issues"] == [
            {"field": "timeout_seconds", "message": "Input should be a valid integer"}
        ]
    for request in provider.requests:
        exposed = next(
            tool for tool in request.allowed_tools if tool.name == "run_command"
        )
        assert (
            exposed.arguments_schema["properties"]["timeout_seconds"]
            == MODEL_TIMEOUT_SCHEMA
        )
        assert exposed.arguments_schema["required"] == ["command"]
    if broken_writer:
        assert root.read_text() == "not a directory"
        return
    rows = events(root)
    assert rows[0]["kind"] == "run.start"
    assert rows[0]["configured_model"] == settings.llm_model
    assert rows[-1]["kind"] == ("run.failure" if fail else "run.success")
    assert all(row["timestamp"] and row["run_id"] == 701 for row in rows)
    assert (
        next(row for row in rows if row["kind"] == "context.compact")[
            "estimated_before"
        ]
        == 12000
    )
    tools = [row for row in rows if row["kind"] == "tool"]
    assert tools[0]["result_kind"] == "INVALID_ARGUMENTS"
    assert tools[0]["metadata"]["executed"] is False
    assert [row["ordinal"] for row in tools] == list(range(1, len(tools) + 1))
    llms = [row for row in rows if row["kind"] == "llm"]
    assert [row["round"] for row in llms] == [1, 2, 3]
    assert llms[0]["total_tokens"] == 15 and llms[0]["provider_model"] == "fake-version"
    assert all(row["duration_ms"] >= 0 for row in llms + tools)
    if include_content:
        assert llms[0]["tool_calls"][0]["raw_arguments"] == invalid.raw_arguments
        assert tools[0]["raw_arguments"] == invalid.raw_arguments
        assert (
            json.loads(tools[0]["result_content"])["error"]["issues"][0]["field"]
            == "timeout_seconds"
        )
    else:
        assert "private-" not in json.dumps(rows)
        assert all("request" not in row and "tool_calls" not in row for row in llms)
        assert all(
            "raw_arguments" not in row and "result_content" not in row for row in tools
        )


def test_local_content_does_not_enable_langsmith_content_and_append_only(
    tmp_path, monkeypatch
):
    remote = []

    class Client:
        def create_run(self, **kwargs):
            remote.append(kwargs)

        def update_run(self, *args, **kwargs):
            remote.append(kwargs)

    monkeypatch.setattr(
        _LangSmithTrace,
        "_submit",
        lambda self, operation, action, *args, **kwargs: action(*args, **kwargs),
    )
    tracer = CompositeTracer(
        LangSmithTracer(enabled=True, project="test", client_factory=Client),
        LocalJsonTracer(tmp_path, include_content=True),
    )
    trace = tracer.start(1, "fake", "configured", False)
    llm = trace.begin_llm(
        LLMRequest(
            system_input="private-system 中文\nsecond line",
            transcript=(UserRuntimeMessage("private-user"),),
            allowed_tools=(),
            personal_context=("private-memory",),
            conversation_compact_context="private-compact",
            evidence_context="private-evidence",
        ),
        1,
    )
    llm.content_delta(AssistantContentDelta("private-answer"))
    llm.finish(LLMResponseCompleted("private-answer", (), LLMFinishReason.STOP, None))
    call = ToolCall("one", "run_command", '{"command":"private-command"}')
    trace.begin_tool(call, 1).finish(
        ToolResult("one", "run_command", ToolResultKind.SUCCESS, "private-result")
    )
    trace.success("private-answer")
    assert "private-" not in json.dumps(remote, default=str)
    assert (
        events(tmp_path, 1)[1]["request"]["conversation_compact_context"]
        == "private-compact"
    )
    assert events(tmp_path, 1)[1]["ttft_ms"] is not None
    before = (tmp_path / "run-1.jsonl").read_bytes()
    assert "中文".encode("utf-8") in before
    tracer.start(2, "fake", "configured", False).failure("TEST")
    tracer.start(1, "fake", "configured", False).failure("TEST")
    assert (tmp_path / "run-1.jsonl").read_bytes().startswith(before)
    assert events(tmp_path, 2)[-1]["kind"] == "run.failure"


def test_fanout_isolates_failing_sink_at_start_and_events(tmp_path):
    class Broken:
        def start(self, *args):
            return self

        def __getattr__(self, name):
            def fail(*args, **kwargs):
                raise OSError("sink unavailable")

            return fail

    class BrokenStart:
        def start(self, *args):
            raise OSError("start unavailable")

    trace = CompositeTracer(
        BrokenStart(), Broken(), LocalJsonTracer(tmp_path, include_content=False)
    ).start(1, "fake", "fake", False)
    llm = trace.begin_llm(LLMRequest("system", (), ()), 1)
    llm.failure("TEST")
    trace.begin_tool(ToolCall("one", "read_file", "{}"), 1).finish(None, "TEST")
    trace.failure("TEST")
    assert [row["kind"] for row in events(tmp_path, 1)] == [
        "run.start",
        "llm",
        "tool",
        "run.failure",
    ]


@pytest.mark.parametrize(
    "environment,enabled,content,expected",
    [
        ("development", None, None, (True, True)),
        ("production", None, None, (False, False)),
        ("test", None, None, (False, False)),
        ("development", "false", "false", (False, False)),
        ("production", "true", "true", (True, True)),
        ("development", "false", "true", (False, True)),
    ],
)
def test_local_settings_environment_overrides(
    monkeypatch, environment, enabled, content, expected
):
    monkeypatch.setenv("LANGLEY_ENVIRONMENT", environment)
    for suffix, value in [("ENABLED", enabled), ("INCLUDE_CONTENT", content)]:
        name = "LANGLEY_LOCAL_RUN_DIAGNOSTICS_" + suffix
        monkeypatch.delenv(name, raising=False)
        if value is not None:
            monkeypatch.setenv(name, value)
    monkeypatch.setenv("LANGLEY_TRACE_CONTENT_ENABLED", "false")
    settings = Settings()
    assert (
        settings.effective_local_run_diagnostics_enabled,
        settings.effective_local_run_diagnostics_include_content,
    ) == expected
    assert settings.trace_content_enabled is False


def test_runtime_ignored():
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", ".runtime/traces/run-701.jsonl"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == ".runtime/traces/run-701.jsonl"


def test_explicit_local_disable_does_not_create_diagnostics(tmp_path):
    app = create_app(
        Settings(
            database_url="mysql+asyncmy://unused/langley_test",
            local_run_diagnostics_enabled=False,
            local_run_diagnostics_include_content=True,
            local_run_diagnostics_root=tmp_path / "absent",
            tracing_enabled=False,
        ),
        provider=FakeProvider([]),
    )
    flow = app.state.execution_manager._workflow_factory()
    flow._start_trace(1).success("private-answer")
    assert not (tmp_path / "absent").exists()
    asyncio.run(app.state.database_engine.dispose())


@pytest.mark.parametrize(
    "raw,field",
    [
        (
            '{"command":"echo ok","timeout_seconds":"private-invalid"}',
            "timeout_seconds",
        ),
        ('{"command":"echo ok","timeout_seconds":1.0}', "timeout_seconds"),
        ('{"command":"echo ok","private-extra":"private-invalid"}', "$"),
        ("{private-invalid", "$"),
        ('["private-invalid"]', "$"),
    ],
)
def test_invalid_arguments_sanitized_and_traced(tmp_path, raw, field):
    storage = WorkspaceStorage(Settings(workspace_storage_root=tmp_path))
    tool = WorkspaceTool("run_command", storage)
    assert tool.validate_arguments({"command": "echo ok", "timeout_seconds": 1})
    assert (
        "must not be a string"
        in tool.arguments.model_json_schema()["properties"]["timeout_seconds"][
            "description"
        ]
    )
    trace = LocalJsonTracer(tmp_path / "traces", include_content=True).start(
        1, "fake", "fake", False
    )
    result = asyncio.run(
        ToolExecutor(tools=[tool]).execute_batch(
            (ToolCall("bad", "run_command", raw),), trace=trace
        )
    )[0]
    assert result.kind is ToolResultKind.INVALID_ARGUMENTS
    assert "private-" not in result.content
    issue = json.loads(result.content)["error"]["issues"][0]
    assert issue["field"] == field and len(issue["message"]) <= 160
    assert set(issue) == {"field", "message"}
    assert events(tmp_path / "traces", 1)[-1]["result_kind"] == "INVALID_ARGUMENTS"


def test_nested_issue_paths_and_issue_cap(tmp_path):
    tool = WorkspaceTool(
        "edit_file", WorkspaceStorage(Settings(workspace_storage_root=tmp_path))
    )
    issues = tool.explain_invalid_arguments(
        {"path": "file", "edits": [{"old_text": 42, "new_text": False}] * 10}
    )
    assert len(issues) == 5
    assert issues[0]["field"] == "edits[0].old_text"
    assert issues[2]["field"] == "edits[1].old_text"
