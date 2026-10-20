"""Workflow, trace and protocol integration using offline scripted providers."""

import asyncio
import json
from dataclasses import dataclass
from typing import cast

import httpx
import pytest
from pydantic import SecretStr

from langley.answering.context_builder import AnswerContext
from langley.answering.contracts import (
    AssistantRuntimeMessage,
    LLMFinishReason,
    LLMProvider,
    LLMResponseCompleted,
    LLMUsage,
    ToolCall,
    ToolResult,
    ToolResultKind,
    ToolSpec,
)
from langley.answering.errors import RunErrorCode, WorkflowFailure
from langley.answering.fake_provider import FakeProvider, ScriptedProviderRound
from langley.answering.local_tracing import LocalJsonTracer
from langley.answering.run_context import (
    HANDOFF_HEADINGS,
    RunContextCompactor,
    RunContextConfig,
)
from langley.answering.tools import ToolContext, ToolExecutionOutput, ToolExecutor
from langley.answering.tracing import _NoopTrace
from langley.answering.workflow import LearningAssistantWorkflow
from langley.infrastructure.deepseek_provider import DeepSeekProvider
from langley.observatory.store import ObservatoryStore
from langley.skills import SkillRegistry

SUMMARY = "\n".join(f"## {h}\nContinue report." for h in HANDOFF_HEADINGS)
SUMMARY += "\n### Done\nDraft.\n### In Progress\nReport.\n### Blocked\nNone."
CONFIG = RunContextConfig(
    input_budget=30_000, trigger=6000, recent_target=1000, summary_target=800
)


def round_(*calls, text=""):
    return ScriptedProviderRound(
        (
            LLMResponseCompleted(
                text,
                calls,
                LLMFinishReason.TOOL_CALLS if calls else LLMFinishReason.STOP,
                LLMUsage(42, 8),
                "fake",
            ),
        )
    )


@dataclass
class Context:
    async def build(self, *args, **kwargs):
        return AnswerContext((), "Finish the report")


class LargeTool:
    spec = ToolSpec("get_current_time", "offline observation", {"type": "object"})

    def __init__(self):
        self.calls = 0

    def validate_arguments(self, args):
        return args == {}

    async def execute(self, args, context):
        self.calls += 1
        return ToolExecutionOutput("x" * 10_000)


def flow(provider, tool, compactor=None, registry=None, deadline=5):
    return LearningAssistantWorkflow(
        context_builder=Context(),
        provider=provider,
        tool_executor=ToolExecutor(tools=(tool,)),
        max_llm_rounds=12,
        max_tool_calls=12,
        overall_deadline_seconds=deadline,
        provider_name="fake",
        model="fake",
        run_compactor=compactor,
        skill_registry=registry,
    )


async def discard(_):
    pass


def test_whole_run_repeated_compaction_skill_snapshot_and_trace_projection(tmp_path):
    package = tmp_path / "skills" / "office"
    package.mkdir(parents=True)
    (package / "SKILL.md").write_text(
        "---\nname: office\ndescription: Office task\n---\nVerified procedure.\n"
    )
    (package / "references").mkdir()
    (package / "references" / "tables.md").write_text("table supporting data\n" * 100)
    provider = FakeProvider(
        [
            round_(ToolCall("load", "load_skill", '{"name":"office"}')),
            round_(
                ToolCall(
                    "read", "read_skill_resource", '{"path":"references/tables.md"}'
                )
            ),
            *(round_(ToolCall(str(i), "get_current_time", "{}")) for i in range(6)),
            round_(text="final report"),
        ]
    )
    summary_provider = FakeProvider([round_(text=SUMMARY), round_(text=SUMMARY)])
    compactor = RunContextCompactor(summary_provider, "fake", CONFIG)
    trace_root = tmp_path / "traces"
    trace = LocalJsonTracer(trace_root, include_content=False).start(
        1, "fake", "fake", False
    )
    tool = LargeTool()
    workflow = flow(provider, tool, compactor, SkillRegistry(tmp_path / "skills"))
    state = asyncio.run(
        workflow._run_graph(
            AnswerContext((), "Finish the report"),
            ToolContext(run_id=1, user_id=1, knowledge_base_id=None),
            discard,
            trace,
        )
    )
    assert state["llm_rounds"] == 9 and state["tool_calls_used"] == 8
    assert state["run_compact_state"].revision == 2
    assert tool.calls == 6
    assert len(state["transcript"]) == 18  # USER + eight intact batches + final
    assert len([i for i in state["transcript"] if isinstance(i, ToolResult)]) == 8
    assert state["transcript"][1].tool_calls[0].name == "load_skill"
    assert state["skill_resources"].get("references/tables.md") is not None
    assert provider.requests[1].transcript[1].tool_calls[0].name == "load_skill"
    assert all(
        r.active_skill == provider.requests[1].active_skill
        for r in provider.requests[1:]
    )
    assert all(r.current_user_message_index == 0 for r in provider.requests)
    assert all(r.active_skill_resources for r in provider.requests[1:])
    assert any(
        "[Run task handoff]" in i.content
        for i in provider.requests[-1].transcript
        if isinstance(i, AssistantRuntimeMessage)
    )
    rows = [
        json.loads(line)
        for line in (trace_root / "run-1.jsonl").read_text().splitlines()
    ]
    assert len([e for e in rows if e["kind"] == "llm"]) == 9
    compact = [e for e in rows if e["kind"] == "context.compact"]
    assert len(compact) == 2 and all(e["scope"] == "RUN" for e in compact)
    assert compact[1]["estimated_before"] < compact[1]["raw_transcript_estimate"]
    assert compact[1]["projected_byte_count"] < compact[1]["raw_byte_count"]
    from langley.api.observatory import _safe_event_metadata

    metadata = _safe_event_metadata("context.compact", compact[1])
    assert metadata["scope"] == "RUN"
    assert metadata["revision"] == 2
    assert metadata["attempted_through_index"] == compact[1]["through_index"]
    assert "Finish the report" not in json.dumps(rows)
    store = ObservatoryStore(tmp_path / "observatory.sqlite", trace_root)
    store.initialize()
    store.refresh()
    assert (
        len(store.list_rounds(1)) == 9
    )  # Summary usage cannot overwrite generation rounds.
    assert len([e for e in store.list_events(1) if e.kind == "context.compact"]) == 2


def test_disabled_no_summary_and_byte_identical_original_requests():
    def run(compactor=None):
        provider = FakeProvider(
            [round_(ToolCall("one", "get_current_time", "{}")), round_(text="done")]
        )
        tool = LargeTool()
        workflow = flow(provider, tool, compactor)
        state = asyncio.run(
            workflow._run_graph(
                AnswerContext((), "question"),
                ToolContext(run_id=1, user_id=1, knowledge_base_id=None),
                discard,
                _NoopTrace(),
            )
        )
        return provider.requests, state

    original, state = run()
    assert original[1].transcript == state["transcript"][:-1]
    assert "[Run task handoff]" not in original[1].system_input
    assert state["run_compact_state"].attempt_count == 0


def test_overall_deadline_propagates_through_summary_without_more_tools():
    async def run():
        provider = FakeProvider(
            [*(round_(ToolCall(str(i), "get_current_time", "{}")) for i in range(6))]
        )
        started = asyncio.Event()
        summary = FakeProvider(
            [ScriptedProviderRound((), started=started, blocked_until=asyncio.Event())]
        )
        tool = LargeTool()
        workflow = flow(
            provider, tool, RunContextCompactor(summary, "fake", CONFIG), deadline=0.1
        )
        with pytest.raises(WorkflowFailure) as exc:
            await workflow.execute(
                None,
                run_id=1,
                user_id=1,
                conversation_id=1,
                input_message_id=1,
                knowledge_base_id=None,
                on_assistant_delta=discard,
            )
        assert exc.value.error_code is RunErrorCode.AGENT_EXECUTION_TIMEOUT
        assert started.is_set() and tool.calls < 6

    asyncio.run(run())


def test_deepseek_mock_transport_accepts_projected_followup_without_paid_call():
    captured = []

    def handler(req):
        captured.append(json.loads(req.content))
        body = {
            "choices": [{"delta": {"content": "done"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 1},
        }
        return httpx.Response(
            200, content=("data: " + json.dumps(body) + "\n\ndata: [DONE]\n\n").encode()
        )

    provider = DeepSeekProvider(
        api_key=SecretStr("test"),
        base_url="https://test.invalid",
        model="fake",
        client_factory=lambda: httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ),
    )
    from langley.answering.contracts import LLMRequest, UserRuntimeMessage

    request = LLMRequest(
        "system",
        (
            UserRuntimeMessage("task"),
            AssistantRuntimeMessage("[Run task handoff]\n" + SUMMARY, ()),
            AssistantRuntimeMessage("", (ToolCall("one", "get_current_time", "{}"),)),
            ToolResult("one", "get_current_time", ToolResultKind.SUCCESS, "ok"),
        ),
        (),
        current_user_message_index=0,
    )

    async def run():
        return [event async for event in cast(LLMProvider, provider).stream(request)]

    assert asyncio.run(run())[-1].assistant_content == "done"
    assert captured[0]["messages"][2]["role"] == "assistant"
    assert captured[0]["messages"][3]["role"] == "assistant"
