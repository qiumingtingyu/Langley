"""Opt-in acceptance, bounded repair and evidence integrity; no model/network IO."""

import asyncio
import hashlib
import json
from dataclasses import dataclass, replace
from zipfile import ZipFile

import pytest

from langley.answering.completion_verifier import (
    CompletionVerifier,
    Constraint,
    EvidenceUnavailable,
    Source,
    Verdict,
    extract_docx,
    issue,
)
from langley.answering.context_builder import AnswerContext
from langley.answering.contracts import (
    AssistantContentDelta,
    LLMFinishReason,
    LLMResponseCompleted,
    LLMUsage,
    ToolCall,
    ToolResult,
    ToolResultKind,
    UserRuntimeMessage,
)
from langley.answering.errors import WorkflowFailure
from langley.answering.fake_provider import FakeProvider, ScriptedProviderRound
from langley.answering.tools import ToolExecutor
from langley.answering.workflow import LearningAssistantWorkflow
from langley.answering.workspace_tools import WORKSPACE_TOOL_NAMES, WorkspaceTool
from langley.sandbox import SandboxRuntime
from langley.settings import Settings
from langley.workspace_storage import WorkspaceStorage


def docx(path, body="正文", suffix=""):
    with ZipFile(path, "w") as z:
        z.writestr(
            "word/document.xml",
            (
                '<w:document xmlns:w="http://schemas.openxmlformats.org/'
                'wordprocessingml/2006/main"><w:body><w:p><w:pPr>'
                '<w:pStyle w:val="Title"/></w:pPr><w:r><w:t>标题</w:t></w:r></w:p>'
                '<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr>'
                "<w:r><w:t>章节</w:t></w:r></w:p>"
                f"<w:p><w:r><w:t>{body}</w:t></w:r></w:p>{suffix}"
                "</w:body></w:document>"
            ),
        )
        z.writestr(
            "word/styles.xml",
            (
                '<w:styles xmlns:w="http://schemas.openxmlformats.org/'
                'wordprocessingml/2006/main"><w:style w:styleId="Heading1">'
                '<w:name w:val="heading 1"/></w:style></w:styles>'
            ),
        )


def provider_round(content="已完成", calls=(), reason=LLMFinishReason.STOP):
    return ScriptedProviderRound(
        events=(
            AssistantContentDelta(content),
            LLMResponseCompleted(
                content, calls, reason, LLMUsage(10, 2, 0, 10), "FAKE_NOT_BILLED"
            ),
        )
    )


@dataclass
class Builder:
    context: AnswerContext

    async def build(self, *args, **kwargs):
        return self.context


class StubVerifier(CompletionVerifier):
    def __init__(self, verdicts, events):
        super().__init__(
            FakeProvider([]),
            artifact="report.docx",
            sources=(),
            event_sink=events.append,
        )
        self.verdicts = list(verdicts)
        self.goals = []

    async def judge(self, **kwargs):
        self.goals.append(kwargs["goal"])
        return self.verdicts.pop(0)


def setup(tmp_path, monkeypatch, rounds, verdicts):
    settings = Settings(
        _env_file=None,
        workspace_storage_root=tmp_path,
        workspace_max_llm_rounds=140,
        workspace_max_tool_calls=200,
    )
    storage = WorkspaceStorage(settings)
    root = storage.workspace_root("a" * 32)
    root.mkdir(parents=True)
    docx(root / "report.docx")
    (root / "facts.md").write_text("authoritative facts", encoding="utf8")
    commands = []

    async def execute(self, command, timeout_seconds=None):
        commands.append(command)
        return {
            "exit_code": 0,
            "timed_out": False,
            "truncated": False,
            "output": json.dumps({"success": True, "data": {"count": 0}}),
        }

    monkeypatch.setattr(SandboxRuntime, "execute", execute)
    events = []
    v = StubVerifier(verdicts, events)
    provider = FakeProvider(rounds)
    flow = LearningAssistantWorkflow(
        context_builder=Builder(
            AnswerContext(
                (),
                "原始目标：生成report.docx并validate",
                user_id=1,
                workspace_id=1,
                workspace_name="test",
                workspace_storage_key="a" * 32,
            )
        ),
        provider=provider,
        tool_executor=ToolExecutor(
            tools=[WorkspaceTool(n, storage) for n in sorted(WORKSPACE_TOOL_NAMES)]
        ),
        max_llm_rounds=140,
        max_tool_calls=200,
        overall_deadline_seconds=10,
        provider_name="fake",
        model="fake",
        workspace_storage=storage,
        completion_verifier=v,
    )
    return flow, provider, v, events, commands, root


async def run(flow, seen):
    async def sink(delta):
        seen.append(delta)

    return await flow.execute(
        None,
        run_id=1,
        user_id=1,
        conversation_id=1,
        input_message_id=1,
        knowledge_base_id=None,
        on_assistant_delta=sink,
    )


def needs():
    return Verdict(
        status="NEEDS_REPAIR",
        issues=(
            issue(
                "FACT", "wrong", "correct", "Read source", "USER", "user", "原始目标"
            ),
        ),
    )


def test_hold_candidate_repair_feedback_and_no_tool_replay(tmp_path, monkeypatch):
    read = ToolCall("read-once", "read_file", json.dumps({"path": "facts.md"}))
    f, p, v, events, commands, _ = setup(
        tmp_path,
        monkeypatch,
        [
            provider_round("错误完成声明"),
            provider_round("", (read,), LLMFinishReason.TOOL_CALLS),
            provider_round("最终已核对"),
        ],
        [needs(), Verdict(status="PASS")],
    )

    class Projection:
        async def prepare(self, request, state, trace, round_):
            return replace(
                request, transcript=(UserRuntimeMessage("compressed"),)
            ), state

    f._run_compactor = Projection()
    seen = []
    result = asyncio.run(run(f, seen))
    assert seen == ["最终已核对"] and result.content == "最终已核对"
    assert v.goals == ["原始目标：生成report.docx并validate"] * 2
    assert len(p.requests) == 3 and len(commands) == 2
    assert "trusted_completion_verifier" in p.requests[1].transcript[-1].content
    assert len([e for e in events if not e["early"]]) == 2
    assert events[-1]["tool_calls_used"] == 3  # two schema checks + one actual read


def test_two_repairs_then_reject_without_success_text(tmp_path, monkeypatch):
    f, p, _, events, _, _ = setup(
        tmp_path, monkeypatch, [provider_round() for _ in range(3)], [needs()] * 3
    )
    seen = []
    with pytest.raises(WorkflowFailure):
        asyncio.run(run(f, seen))
    assert seen == [] and len(p.requests) == 3 and len(events) == 3
    assert all(e["verdict"]["status"] == "NEEDS_REPAIR" for e in events)


@pytest.mark.parametrize("status", ["INSUFFICIENT_EVIDENCE", "VERIFIER_FAILED"])
def test_verifier_failure_never_publishes_candidate(tmp_path, monkeypatch, status):
    f, p, _, events, _, _ = setup(
        tmp_path, monkeypatch, [provider_round()], [Verdict(status=status)]
    )
    seen = []
    with pytest.raises(WorkflowFailure):
        asyncio.run(run(f, seen))
    assert (
        seen == [] and len(p.requests) == 1 and events[0]["verdict"]["status"] == status
    )


def test_early_review_once_and_final_gate(tmp_path, monkeypatch):
    def read(n):
        return ToolCall(str(n), "read_file", json.dumps({"path": "facts.md"}))

    f, p, v, events, _, _ = setup(
        tmp_path,
        monkeypatch,
        [
            provider_round("", (read(1),), LLMFinishReason.TOOL_CALLS),
            provider_round("", (read(2),), LLMFinishReason.TOOL_CALLS),
            provider_round("final"),
        ],
        [Verdict(status="PASS")] * 2,
    )
    v.early_round = 90
    original = f._compile_graph

    class StartAt89:
        async def ainvoke(self, state, config):
            return await original_graph.ainvoke(
                {**state, "llm_rounds": 89}, config=config
            )

    def compile_(*args, **kwargs):
        nonlocal original_graph
        original_graph = original(*args, **kwargs)
        return StartAt89()

    original_graph = None
    f._compile_graph = compile_
    seen = []
    asyncio.run(run(f, seen))
    assert seen == ["final"] and [e["early"] for e in events] == [True, False]
    assert events[-1]["round"] == 92 and len(p.requests) == 3


def test_deadline_and_cancellation_do_not_release_candidate(tmp_path, monkeypatch):
    f, _, v, _, _, _ = setup(
        tmp_path, monkeypatch, [provider_round()], [Verdict(status="PASS")]
    )

    async def blocked(**kwargs):
        await asyncio.sleep(60)

    v.judge = blocked
    f._overall_deadline_seconds = 0.03
    seen = []
    with pytest.raises(WorkflowFailure) as exc:
        asyncio.run(run(f, seen))
    assert exc.value.error_code.value == "AGENT_EXECUTION_TIMEOUT" and not seen
    f._overall_deadline_seconds = 10
    f._provider = FakeProvider([provider_round()])

    async def cancel():
        task = asyncio.create_task(run(f, seen))
        await asyncio.sleep(0.03)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel())
    assert not seen


def test_integrity_change_while_judging_is_not_pass(tmp_path, monkeypatch):
    f, _, v, events, _, root = setup(
        tmp_path, monkeypatch, [provider_round()], [Verdict(status="PASS")]
    )

    async def mutate(**kwargs):
        docx(root / "report.docx", "外部改动")
        return Verdict(status="PASS")

    v.judge = mutate
    seen = []
    with pytest.raises(WorkflowFailure):
        asyncio.run(run(f, seen))
    assert not seen and events[-1]["verdict"]["status"] == "VERIFIER_FAILED"


def test_checker_sources_counts_and_identifiers(tmp_path):
    storage = WorkspaceStorage(
        Settings(_env_file=None, workspace_storage_root=tmp_path)
    )
    root = storage.workspace_root("b" * 32)
    root.mkdir(parents=True)
    source = "资料ID：AB01。状态为待评估。"
    (root / "source.md").write_text(source, encoding="utf8")
    docx(root / "report.docx", "SRC-01中待评估 {{unfinished}}")
    goal = "read source.md and write report.docx"
    v = CompletionVerifier(
        FakeProvider([]),
        artifact="report.docx",
        sources=(Source("source.md", hashlib.sha256(source.encode()).hexdigest()),),
        constraints=(Constraint("minimum_tables", 0, "USER", "user", "report.docx"),),
        closed_source_quote="source.md",
    )
    ev = v.inspect(goal, storage, "b" * 32)
    assert ev["artifact"]["body_han"] == 4
    assert {x["code"] for x in ev["checker_issues"]} == {
        "PLACEHOLDER",
        "UNKNOWN_SOURCE_ID",
    }
    (root / "source.md").write_text("changed", encoding="utf8")
    with pytest.raises(EvidenceUnavailable):
        v.inspect(goal, storage, "b" * 32)


def test_judge_cannot_promote_quality_or_invent_provenance(tmp_path):
    schema = ToolResult(
        "s",
        "run_command",
        ToolResultKind.SUCCESS,
        json.dumps(
            {
                "exit_code": 0,
                "timed_out": False,
                "truncated": False,
                "output": json.dumps({"success": True, "data": {"count": 0}}),
            }
        ),
    )
    evidence = {
        "checker_issues": [],
        "artifact": {"full_text": "actual text"},
        "sources": {"facts.md": "AB01 is current"},
    }
    for kind, quote in [("QUALITY", ""), ("USER", "invented requirement")]:
        bad = {
            "status": "NEEDS_REPAIR",
            "issues": [
                {
                    "code": "invented",
                    "severity": "ERROR",
                    "actual": "x",
                    "expected": "y",
                    "repair": "z",
                    "requirement_kind": kind,
                    "requirement_source": "user",
                    "requirement_quote": quote,
                }
            ],
        }
        provider = FakeProvider([provider_round(json.dumps(bad))])
        v = CompletionVerifier(provider, artifact="report.docx", sources=())
        verdict = asyncio.run(
            v.judge(
                goal="original goal",
                candidate="claimed done",
                evidence=evidence,
                schema=schema,
                recent_results=(),
            )
        )
        assert verdict.status == "VERIFIER_FAILED"
        payload = json.loads(provider.requests[0].transcript[0].content)
        assert (
            payload["actual_artifact"]["full_text"] == "actual text"
            and payload["raw_authoritative_sources"] == evidence["sources"]
        )


def test_zip_expansion_bound():
    import io

    raw = io.BytesIO()
    with ZipFile(raw, "w") as z:
        for i in range(513):
            z.writestr(str(i), "")
    with pytest.raises(ValueError, match="EXPANSION"):
        extract_docx(raw.getvalue())


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"success": True, "data": []},
        {"success": True, "data": {"count": False}},
        {"success": True, "data": {"count": -1}},
    ],
)
def test_schema_requires_real_nonnegative_integer_evidence(payload):
    schema = ToolResult(
        "schema",
        "run_command",
        ToolResultKind.SUCCESS,
        json.dumps(
            {
                "exit_code": 0,
                "timed_out": False,
                "truncated": False,
                "output": json.dumps(payload),
            }
        ),
    )
    provider = FakeProvider([])
    verifier = CompletionVerifier(provider, artifact="report.docx", sources=())
    result = asyncio.run(
        verifier.judge(
            goal="goal",
            candidate="done",
            evidence={"checker_issues": [], "artifact": {}, "sources": {}},
            schema=schema,
            recent_results=(),
        )
    )
    assert result.status == "INSUFFICIENT_EVIDENCE"
    assert not provider.requests


@pytest.mark.parametrize(
    "finish,usage,expected",
    [
        (LLMFinishReason.STOP, LLMUsage(10, 2, 0, 10), "PASS"),
        (LLMFinishReason.LENGTH, LLMUsage(10, 2, 0, 10), "VERIFIER_FAILED"),
        (LLMFinishReason.STOP, None, "VERIFIER_FAILED"),
    ],
)
def test_judge_accepts_only_completed_accounted_verdict(finish, usage, expected):
    schema = ToolResult(
        "s",
        "run_command",
        ToolResultKind.SUCCESS,
        json.dumps(
            {
                "exit_code": 0,
                "timed_out": False,
                "truncated": False,
                "output": json.dumps({"success": True, "data": {"count": 0}}),
            }
        ),
    )
    completion = LLMResponseCompleted(
        '{"status":"PASS","issues":[]}', (), finish, usage, "FAKE"
    )
    provider = FakeProvider([ScriptedProviderRound(events=(completion,))])
    v = CompletionVerifier(provider, artifact="report.docx", sources=())
    verdict = asyncio.run(
        v.judge(
            goal="goal",
            candidate="claimed done",
            evidence={
                "checker_issues": [],
                "artifact": {"full_text": "actual"},
                "sources": {},
            },
            schema=schema,
            recent_results=(),
        )
    )
    assert verdict.status == expected
