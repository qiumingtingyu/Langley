"""Offline invariants for whole-batch handoff projection and bounded failures."""

import asyncio
import json
from dataclasses import fields, replace

import pytest
from pydantic import SecretStr

from langley.answering.contracts import (
    ActiveSkill,
    AssistantContentDelta,
    AssistantRuntimeMessage,
    LLMFinishReason,
    LLMRequest,
    LLMResponseCompleted,
    LLMUsage,
    SkillResourceSummary,
    ToolCall,
    ToolResult,
    ToolResultKind,
    UserRuntimeMessage,
)
from langley.answering.errors import RunErrorCode, WorkflowFailure
from langley.answering.fake_provider import FakeProvider, ScriptedProviderRound
from langley.answering.local_tracing import LocalJsonTracer
from langley.answering.run_context import (
    HANDOFF_HEADINGS,
    RunCompactState,
    RunContextCompactor,
    RunContextConfig,
    cut_point,
    file_operations,
    project_request,
    summary_request,
    tool_batches,
)
from langley.answering.tracing import _NoopTrace
from langley.infrastructure.deepseek_provider import DeepSeekProvider
from langley.infrastructure.qwen_provider import QwenProvider

SUMMARY = "\n".join(f"## {heading}\nKeep task facts." for heading in HANDOFF_HEADINGS)
SUMMARY += "\n### Done\nWritten.\n### In Progress\nEditing.\n### Blocked\nNone."
CONFIG = RunContextConfig(
    input_budget=20_000,
    trigger=6000,
    recent_target=1000,
    summary_target=800,
    max_calls=2,
)


def batch(i, names=("run_command",), size=6000, reverse=False):
    calls = tuple(
        ToolCall(f"{i}-{n}", name, json.dumps({"path": "report.md"}))
        for n, name in enumerate(names)
    )
    results = tuple(
        ToolResult(c.call_id, c.name, ToolResultKind.SUCCESS, "x" * size) for c in calls
    )
    return (AssistantRuntimeMessage("working", calls),) + (
        results[::-1] if reverse else results
    )


def request(count=6, size=6000):
    prefix = (
        UserRuntimeMessage("same"),
        AssistantRuntimeMessage("old", ()),
        UserRuntimeMessage("same"),
    )
    return LLMRequest(
        "base",
        prefix + sum((batch(i, size=size) for i in range(count)), ()),
        (),
        personal_context=None,
        current_user_message_index=2,
        active_skill=ActiveSkill("office", "verified instructions"),
        active_skill_resources=(SkillResourceSummary("references/tables.md", 50),),
    )


def completed(text=SUMMARY, reason=LLMFinishReason.STOP, calls=()):
    return LLMResponseCompleted(
        text, calls, reason, LLMUsage(123, 45, 20, 103), "served"
    )


def prepare(raw, provider, state=RunCompactState(), config=CONFIG, trace=None):
    return asyncio.run(
        RunContextCompactor(provider, "fake", config).prepare(
            raw, state, trace or _NoopTrace(), 7
        )
    )


def test_unordered_parallel_results_and_all_request_fields_preserved():
    raw = replace(
        request(0),
        transcript=request(0).transcript
        + batch(1, ("load_skill", "read_skill_resource"), reverse=True)
        + batch(2),
    )
    original = raw.transcript
    batches = tool_batches(raw)
    assert [(b.start, b.end) for b in batches] == [(3, 6), (6, 8)]
    state = RunCompactState(SUMMARY, 6, 1)
    projected = project_request(raw, state)
    assert raw.transcript is original
    assert projected.transcript[:3] == raw.transcript[:3]
    assert projected.transcript[4:] == raw.transcript[6:]
    assert projected.transcript[2] is raw.transcript[2]
    for field in fields(raw):
        if field.name != "transcript":
            assert getattr(projected, field.name) == getattr(raw, field.name)


@pytest.mark.parametrize(
    "bad",
    ["missing", "duplicate", "wrong_name", "extra", "duplicate_call", "blank_call"],
)
def test_malformed_batches_never_sent(bad):
    items = list(batch(1, ("read_file", "write_file")))
    if bad == "missing":
        items.pop()
    if bad == "duplicate":
        items[2] = items[1]
    if bad == "wrong_name":
        items[1] = replace(items[1], name="other")
    if bad == "extra":
        items.append(items[-1])
    if bad == "duplicate_call":
        items[0] = replace(items[0], tool_calls=(items[0].tool_calls[0],) * 2)
    if bad == "blank_call":
        items[0] = replace(items[0], tool_calls=(ToolCall("", "read_file", "{}"),))
    raw = replace(request(0), transcript=request(0).transcript + tuple(items))
    provider = FakeProvider([])
    with pytest.raises(WorkflowFailure) as exc:
        prepare(raw, provider)
    assert exc.value.error_code is RunErrorCode.LLM_RESPONSE_INVALID
    assert not provider.requests


@pytest.mark.parametrize(
    "name", ["search_knowledge", "read_knowledge", "search_web", "unknown"]
)
def test_evidence_and_unknown_screen_but_known_skill_tools_compact(name):
    raw = request(0)
    raw = replace(
        raw,
        transcript=raw.transcript
        + batch(1, ("load_skill",))
        + batch(2, ("read_skill_resource",))
        + batch(3, (name,))
        + batch(4),
    )
    assert cut_point(raw, RunCompactState(), CONFIG) == 7
    assert all(b.summarizable for b in tool_batches(raw)[:2])


def test_repeated_compaction_keeps_original_suffix_and_does_not_resummarize():
    raw = request()
    provider = FakeProvider(
        [ScriptedProviderRound((completed(),)), ScriptedProviderRound((completed(),))]
    )
    projected, state = prepare(raw, provider)
    assert state.revision == 1 and len(projected.transcript) < len(raw.transcript)
    assert projected.transcript[-2:] == raw.transcript[-2:]
    assert provider.requests[0].allowed_tools == ()
    assert provider.requests[0].active_skill is None
    assert provider.requests[0].current_user_message_index is None
    later = replace(
        raw, transcript=raw.transcript + sum((batch(i) for i in range(6, 11)), ())
    )
    second, state2 = prepare(later, provider, state)
    assert state2.revision == 2 and state2.through_index > state.through_index
    payload = json.loads(provider.requests[1].transcript[0].content)
    assert payload["previous_summary"] == state.summary
    assert (
        len(payload["newly_aged_history"]) == state2.through_index - state.through_index
    )
    assert second.transcript[-2:] == later.transcript[-2:]
    assert raw.transcript[3:5] == batch(0)


@pytest.mark.parametrize(
    "response",
    [
        completed(""),
        completed("not a handoff"),
        completed(reason=LLMFinishReason.LENGTH),
        completed(calls=(ToolCall("bad", "run_command", "{}"),)),
        completed(SUMMARY + "x" * 5000),
    ],
)
def test_invalid_summary_preserves_complete_fallback_and_known_usage(
    response, tmp_path
):
    provider = FakeProvider([ScriptedProviderRound((response,))])
    trace = LocalJsonTracer(tmp_path, include_content=False).start(
        1, "fake", "fake", False
    )
    projected, state = prepare(request(), provider, trace=trace)
    assert projected.transcript == request().transcript and state.revision == 0
    assert state.attempt_count == 1
    rows = [
        json.loads(line) for line in (tmp_path / "run-1.jsonl").read_text().splitlines()
    ]
    event = rows[-1]
    assert event["scope"] == "RUN" and event["provider_input_tokens"] == 123
    assert "Keep task facts" not in json.dumps(rows)


def test_failure_after_previous_compaction_never_loses_uncompacted_suffix():
    provider = FakeProvider(
        [
            ScriptedProviderRound((completed(),)),
            ScriptedProviderRound((), failure=RuntimeError("private")),
        ]
    )
    raw = request()
    _, state = prepare(raw, provider)
    later = replace(
        raw, transcript=raw.transcript + sum((batch(i) for i in range(6, 10)), ())
    )
    projected, failed = prepare(later, provider, state)
    assert (
        failed.summary == state.summary and failed.through_index == state.through_index
    )
    assert projected.transcript[4:] == later.transcript[state.through_index :]
    prepare(later, provider, failed)
    assert len(provider.requests) == 2  # no automatic retry of the same span


def test_no_room_on_failure_or_huge_recent_batch_and_no_summary_budget():
    raw = request(2, size=30_000)
    provider = FakeProvider([ScriptedProviderRound(())])
    with pytest.raises(WorkflowFailure) as exc:
        prepare(raw, provider, config=replace(CONFIG, input_budget=10_000))
    assert exc.value.error_code is RunErrorCode.ANSWER_CONTEXT_TOO_LARGE
    # A single huge latest batch cannot be cut, and is never silently truncated.
    provider = FakeProvider([])
    with pytest.raises(WorkflowFailure):
        prepare(request(1, size=100_000), provider)
    assert not provider.requests
    # Whole summary input is checked too, even after per-item serialization bounds.
    provider = FakeProvider([])
    with pytest.raises(WorkflowFailure):
        prepare(request(80), provider)
    assert not provider.requests


def test_local_timeout_fallback_and_cancel_propagation():
    async def run():
        provider = FakeProvider(
            [ScriptedProviderRound((), blocked_until=asyncio.Event())]
        )
        config = replace(CONFIG, timeout_seconds=0.01)
        projected, _ = await RunContextCompactor(provider, "fake", config).prepare(
            request(), RunCompactState(), _NoopTrace(), 1
        )
        assert projected.transcript == request().transcript
        started = asyncio.Event()
        provider = FakeProvider(
            [ScriptedProviderRound((), started=started, blocked_until=asyncio.Event())]
        )
        task = asyncio.create_task(
            RunContextCompactor(provider, "fake", CONFIG).prepare(
                request(), RunCompactState(), _NoopTrace(), 1
            )
        )
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())


def test_bounded_serialization_and_deterministic_intents():
    raw = request(0)
    calls = (
        ToolCall(
            "write",
            "write_file",
            json.dumps({"path": "report.docx", "content": "x" * 50_000}),
        ),
        ToolCall("cmd", "run_command", '{"command":"touch secret"}'),
        ToolCall("resource", "read_skill_resource", '{"path":"references/tables.md"}'),
    )
    raw = replace(
        raw,
        transcript=raw.transcript
        + (AssistantRuntimeMessage("", calls),)
        + tuple(
            ToolResult(c.call_id, c.name, ToolResultKind.TOOL_ERROR, "y" * 50_000)
            for c in calls
        ),
    )
    assert file_operations(raw.transcript) == ('write_file: "report.docx"',)
    summary = summary_request(raw, RunCompactState(), len(raw.transcript), CONFIG)
    assert "omitted" in summary.transcript[0].content
    assert len(summary.transcript[0].content) < 10_000


@pytest.mark.parametrize("provider_class", [DeepSeekProvider, QwenProvider])
def test_synthetic_assistant_payload_keeps_user_envelope_and_tools(provider_class):
    raw = request()
    state = RunCompactState(SUMMARY, 7, 1)
    projected = project_request(raw, state)
    provider = provider_class(
        api_key=SecretStr("test"), base_url="https://example.invalid", model="test"
    )
    payload = provider._request_payload(projected)
    assert (
        json.loads(payload["messages"][3]["content"])["current_user_request"] == "same"
    )
    assert payload["messages"][4]["role"] == "assistant"
    assert "tool_calls" not in payload["messages"][4]
    assert payload["messages"][5]["role"] == "assistant"
    assert "verified instructions" in payload["messages"][0]["content"]


def test_configuration_validation():
    from langley.settings import Settings

    assert Settings().run_compaction_enabled is False
    with pytest.raises(ValueError):
        RunContextConfig(trigger=20_000)
    with pytest.raises(ValueError):
        Settings(run_recent_target_estimate=40_000)
    with pytest.raises(ValueError):
        RunContextConfig(timeout_seconds=float("inf"))


@pytest.mark.parametrize(
    "events",
    [
        (),
        (completed(), completed()),
        (completed(), AssistantContentDelta("late")),
        (AssistantContentDelta("x" * 20_000),),
    ],
)
def test_invalid_stream_is_bounded_without_losing_fallback(events):
    provider = FakeProvider([ScriptedProviderRound(events)])
    projected, state = prepare(request(), provider)
    assert projected.transcript == request().transcript
    assert state.revision == 0 and state.attempt_count == 1


def test_trace_failure_is_fail_open_and_call_ids_are_batch_local():
    class BrokenTrace(_NoopTrace):
        def context_compact(self, **fields):
            raise OSError("not writable")

    raw = request(0)
    raw = replace(raw, transcript=raw.transcript + batch(1) * 6)
    assert len(tool_batches(raw)) == 6
    projected, state = prepare(
        raw, FakeProvider([ScriptedProviderRound((completed(),))]), trace=BrokenTrace()
    )
    assert state.revision == 1 and len(projected.transcript) < len(raw.transcript)
