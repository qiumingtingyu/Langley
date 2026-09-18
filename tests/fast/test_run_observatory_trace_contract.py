"""Run Observatory v0 Task 1 local trace and ContextFrame contracts."""

import json
from dataclasses import dataclass
from typing import Any, cast

from langley.answering.context_frame import build_context_frame
from langley.answering.contracts import (
    ActiveSkill,
    AssistantContentDelta,
    AssistantRuntimeMessage,
    LLMFinishReason,
    LLMRequest,
    LLMResponseCompleted,
    LLMUsage,
    SkillResourceSummary,
    SkillSummary,
    ToolCall,
    ToolResult,
    ToolResultKind,
    ToolSpec,
    UserRuntimeMessage,
)
from langley.answering.local_tracing import LocalJsonTracer
from langley.answering.tracing import CitationNamespace, KnowledgeSearchOrigin


def _events(root, run_id=901):
    return [
        json.loads(line)
        for line in (root / f"run-{run_id}.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]


def _completion(*tool_calls: ToolCall) -> LLMResponseCompleted:
    return LLMResponseCompleted(
        assistant_content="private-answer" if not tool_calls else "",
        tool_calls=tool_calls,
        finish_reason=(
            LLMFinishReason.TOOL_CALLS if tool_calls else LLMFinishReason.STOP
        ),
        usage=LLMUsage(input_tokens=21, output_tokens=8),
        provider_model="provider-model",
    )


def _rich_request() -> LLMRequest:
    call = ToolCall("call-1", "search_knowledge", '{"query":"秘密"}')
    return LLMRequest(
        system_input="private-system 中文",
        transcript=(
            UserRuntimeMessage("private-user"),
            AssistantRuntimeMessage("private-assistant", (call,)),
            ToolResult(
                "call-1",
                "search_knowledge",
                ToolResultKind.SUCCESS,
                "private-tool-result",
            ),
        ),
        allowed_tools=(
            ToolSpec(
                "search_knowledge",
                "private-schema-description",
                {"type": "object"},
            ),
        ),
        personal_context=("private-memory",),
        current_user_message_index=0,
        conversation_compact_context="private-compact",
        evidence_context="private-evidence",
        available_skills=(SkillSummary("catalog-skill", "private-catalog"),),
        active_skill=ActiveSkill("active-skill", "private-instructions"),
        active_skill_resources=(SkillResourceSummary("reference.md", 17),),
    )


def test_full_content_trace_records_normalized_request_and_body_free_frame(tmp_path):
    trace = LocalJsonTracer(tmp_path, include_content=True).start(
        901, "fake", "configured", False
    )
    llm = trace.begin_llm(_rich_request(), 1)
    llm.content_delta(AssistantContentDelta("first token"))
    llm.finish(_completion())

    rows = _events(tmp_path)
    assert [row["seq"] for row in rows] == [1, 2]
    assert all(row["schema_version"] == 1 for row in rows)
    assert all(row["capture_mode"] == "FULL_CONTENT" for row in rows)
    event = rows[-1]
    assert event["request"]["system_input"] == "private-system 中文"
    assert event["request"]["transcript"][2]["content"] == "private-tool-result"
    assert event["input_tokens"] == 21
    assert event["output_tokens"] == 8
    assert event["ttft_ms"] is not None

    frame = event["context_frame"]
    kinds = {component["kind"] for component in frame["components"]}
    assert {
        "system",
        "personal_context",
        "conversation_compact",
        "skill.active",
        "skill.catalog",
        "skill.resources",
        "tools.schema",
        "transcript.user",
        "transcript.assistant",
        "transcript.tool_call",
        "transcript.tool_result",
        "evidence_context",
    } <= kinds
    assert all("content_sha256" in item for item in frame["components"])
    assert "private-system" not in json.dumps(frame)
    assert not any("content" in item or "body" in item for item in frame["components"])


def test_metadata_only_omits_content_but_keeps_context_measurements(tmp_path):
    trace = LocalJsonTracer(tmp_path, include_content=False).start(
        901, "fake", "configured", False
    )
    trace.begin_llm(_rich_request(), 1).finish(_completion())

    event = _events(tmp_path)[-1]
    assert event["capture_mode"] == "METADATA_ONLY"
    assert "request" not in event
    assert "assistant_content" not in event
    assert "private-" not in json.dumps(event)
    assert event["context_frame"]["estimated_tokens"] > 0
    assert all(
        "content_sha256" not in item for item in event["context_frame"]["components"]
    )


def test_round_correlation_local_parity_events_and_multiple_tools(tmp_path):
    trace = LocalJsonTracer(tmp_path, include_content=True).start(
        901, "fake", "configured", False
    )
    first = ToolCall("search-1", "search_knowledge", '{"query":"Langley"}')
    second = ToolCall("read-1", "read_file", '{"path":"notes.md"}')
    trace.begin_llm(LLMRequest("system", (), ()), 1).finish(_completion(first, second))

    first_trace = trace.begin_tool(first, 1)
    first_trace.begin_knowledge_search(
        knowledge_base_id=7,
        top_k=5,
        query="Langley",
    ).finish(2)
    first_trace.finish(
        ToolResult(first.call_id, first.name, ToolResultKind.SUCCESS, "search-result")
    )
    trace.begin_tool(second, 2).finish(
        ToolResult(second.call_id, second.name, ToolResultKind.SUCCESS, "file-result")
    )

    trace.rejected_response(_completion(), "FINAL_RESPONSE_EMPTY")
    trace.citation_validate(
        namespace=CitationNamespace.KNOWLEDGE,
        available_evidence_count=2,
        cited_handles=(1,),
        cited_document_version_ids=(11,),
        abstained=False,
        error_code=None,
    )

    rows = _events(tmp_path)
    tools = [row for row in rows if row["kind"] == "tool"]
    assert [row["round"] for row in tools] == [1, 1]
    search = next(row for row in rows if row["kind"] == "knowledge.search")
    assert search["round"] == 1
    assert search["tool_call_id"] == "search-1"
    assert search["tool_ordinal"] == 1
    assert search["origin"] == KnowledgeSearchOrigin.AGENT_TOOL.value
    assert search["query"] == "Langley"
    rejected = next(row for row in rows if row["kind"] == "response.rejected")
    assert rejected["round"] == 1
    assert rejected["invalid_response_subtype"] == "FINAL_RESPONSE_EMPTY"
    citation = next(row for row in rows if row["kind"] == "citation.validate")
    assert citation["round"] == 1
    assert citation["evidence_handles"] == [1]
    assert citation["document_version_ids"] == [11]
    assert [row["seq"] for row in rows] == list(range(1, len(rows) + 1))


def test_root_knowledge_search_before_llm_does_not_invent_round(tmp_path):
    trace = LocalJsonTracer(tmp_path, include_content=False).start(
        901, "fake", "configured", False
    )
    trace.begin_knowledge_search(
        origin=KnowledgeSearchOrigin.HARNESS_REQUIRED,
        knowledge_base_id=7,
        top_k=5,
        query="private-query",
    ).finish(None, "RETRIEVAL_UNAVAILABLE")

    search = _events(tmp_path)[-1]
    assert search["round"] is None
    assert search["tool_call_id"] is None
    assert search["origin"] == "HARNESS_REQUIRED"
    assert search["error_code"] == "RETRIEVAL_UNAVAILABLE"
    assert "query" not in search


def test_reopened_run_keeps_process_local_sequence_monotonic(tmp_path):
    tracer = LocalJsonTracer(tmp_path, include_content=False)
    tracer.start(901, "fake", "configured", False).failure("FIRST")
    assert 901 not in tracer._seq_by_run
    tracer.start(901, "fake", "configured", False).failure("SECOND")
    assert 901 not in tracer._seq_by_run

    rows = _events(tmp_path)
    assert [row["seq"] for row in rows] == [1, 2, 3, 4]


def test_llm_trace_construction_failure_does_not_leave_stale_round(
    tmp_path, monkeypatch
):
    trace = LocalJsonTracer(tmp_path, include_content=False).start(
        901, "fake", "configured", False
    )

    def fail_context_frame(*args, **kwargs):
        raise RuntimeError("diagnostics construction failed")

    monkeypatch.setattr(
        "langley.answering.local_tracing.build_context_frame", fail_context_frame
    )
    try:
        trace.begin_llm(LLMRequest("system", (), ()), 7)
    except RuntimeError as error:
        assert str(error) == "diagnostics construction failed"
    else:
        raise AssertionError("expected diagnostics construction failure")

    trace.begin_tool(ToolCall("tool-1", "read_file", "{}"), 1).finish(
        ToolResult("tool-1", "read_file", ToolResultKind.SUCCESS, "result")
    )
    assert _events(tmp_path)[-1]["round"] is None


def test_trace_v1_reserved_envelope_fields_cannot_be_overridden(tmp_path):
    trace = cast(
        Any,
        LocalJsonTracer(tmp_path, include_content=False).start(
            901, "fake", "configured", False
        ),
    )
    trace.event(
        "expected.kind",
        schema_version=999,
        timestamp="forged",
        run_id=999,
        seq=999,
        kind="forged.kind",
        capture_mode="FORGED",
    )

    event = _events(tmp_path)[-1]
    assert event["schema_version"] == 1
    assert event["timestamp"] != "forged"
    assert event["run_id"] == 901
    assert event["seq"] == 2
    assert event["kind"] == "expected.kind"
    assert event["capture_mode"] == "METADATA_ONLY"


def test_chinese_utf8_measurement_and_request_metadata_classification():
    frame = build_context_frame(
        LLMRequest(
            system_input="中文A",
            transcript=(),
            allowed_tools=(),
            current_user_message_index=99,
        ),
        include_content_hashes=False,
    ).as_json()
    system = frame["components"][0]
    assert system["kind"] == "system"
    assert system["char_count"] == 3
    assert system["byte_count"] == 7
    assert system["estimated_tokens"] == 3
    assert not any(
        item["kind"] == "unclassified.request_field" for item in frame["components"]
    )


def test_unknown_non_empty_request_field_emits_unclassified_component():
    @dataclass(frozen=True)
    class FutureRequest(LLMRequest):
        future_context: str = "未来字段"

    frame = build_context_frame(
        FutureRequest("system", (), ()), include_content_hashes=True
    ).as_json()
    unknown = next(
        item
        for item in frame["components"]
        if item["kind"] == "unclassified.request_field"
    )
    assert unknown["key"] == "unclassified.request_field.future_context"
    assert unknown["source"] == "future_context"
    assert len(unknown["content_sha256"]) == 64


def test_unknown_non_empty_transcript_kind_cannot_silently_disappear():
    @dataclass(frozen=True)
    class FutureTranscriptItem:
        content: str

    request = LLMRequest(
        "system",
        (FutureTranscriptItem("future transcript"),),  # type: ignore[arg-type]
        (),
    )
    frame = build_context_frame(request, include_content_hashes=False).as_json()
    unknown = next(
        item
        for item in frame["components"]
        if item["key"] == "unclassified.request_field.transcript.0"
    )
    assert unknown["source"] == "transcript[0]"


def test_local_writer_failure_remains_fail_open(tmp_path):
    broken_root = tmp_path / "not-a-directory"
    broken_root.write_text("occupied", encoding="utf-8")
    trace = LocalJsonTracer(broken_root, include_content=True).start(
        901, "fake", "configured", False
    )
    trace.begin_llm(_rich_request(), 1).finish(_completion())
    trace.citation_validate(
        namespace=CitationNamespace.WEB,
        available_evidence_count=0,
        cited_handles=(),
        cited_document_version_ids=(),
        abstained=False,
        error_code="INVALID",
    )
    trace.failure("TEST")
    assert broken_root.read_text(encoding="utf-8") == "occupied"
