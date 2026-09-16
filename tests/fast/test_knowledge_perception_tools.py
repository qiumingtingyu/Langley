"""Public tool contracts, error ownership and direct-read evidence registration."""

import asyncio
import json

import pytest

from langley.answering import knowledge_tools
from langley.answering.contracts import ToolCall, ToolResultKind
from langley.answering.errors import WorkflowFailure
from langley.answering.knowledge_tools import InspectKnowledgeTool, ReadKnowledgeTool
from langley.answering.tools import ToolContext, ToolExecutor
from langley.knowledge.contracts import (
    KnowledgeLocator,
    KnowledgePerceptionError,
    KnowledgeReadResult,
    PdfPageRegion,
)


def result(locator=None):
    return KnowledgeReadResult(
        locator=locator
        or KnowledgeLocator(document_id=12, kind="pages", page_start=2, page_end=2),
        document_version_id=23,
        content="published text",
        source_display_name="notes.pdf",
        source_sha256="a" * 64,
        heading_path=(),
        source_regions=(PdfPageRegion(1, 3),),
    )


def tools():
    return (
        InspectKnowledgeTool(None, None),
        ReadKnowledgeTool(None, None, max_content_bytes=128),
    )


def execute(tool, arguments, context):
    return asyncio.run(
        ToolExecutor(tools=(tool,)).execute_batch(
            (ToolCall("call", tool.spec.name, json.dumps(arguments)),), context=context
        )
    )[0]


def test_inspect_locator_round_trip_and_single_read_handle(monkeypatch):
    inspect_tool, read_tool = tools()
    context = ToolContext(1, 2, 3)
    locator = {"document_id": 12, "kind": "pages", "page_start": 2, "page_end": 2}

    async def inspect(*args, **kwargs):
        assert kwargs == {"user_id": 2, "knowledge_base_id": 3, "document_id": 12}
        return {"detected_regions": [{"label": "display only", "locator": locator}]}

    async def read(*args, **kwargs):
        assert kwargs["locator"].model_dump(exclude_none=True) == locator
        assert kwargs["user_id"] == 2 and kwargs["knowledge_base_id"] == 3
        assert kwargs["max_content_bytes"] == 128
        return result(kwargs["locator"])

    monkeypatch.setattr(knowledge_tools, "inspect_knowledge", inspect)
    monkeypatch.setattr(knowledge_tools, "read_knowledge", read)
    navigation = execute(inspect_tool, {"document_id": 12}, context)
    assert context.knowledge_evidence.evidence == ()
    payload = json.loads(navigation.content)
    observation = execute(
        read_tool, {"locator": payload["detected_regions"][0]["locator"]}, context
    )
    assert observation.kind is ToolResultKind.SUCCESS
    (evidence,) = context.knowledge_evidence.evidence
    assert evidence.knowledge_chunk_id is None and evidence.chunk_ordinal is None
    assert not context.knowledge_evidence.has_chunk_evidence
    payload = json.loads(observation.content)
    assert payload["evidence_handle"] == "K1" and payload["heading_path"] == []
    assert payload["actual_page_regions"] == [
        {"kind": "pdf_page", "page_start": 1, "page_end": 3}
    ]
    assert all(
        key not in payload
        for key in ("document_version_id", "source_sha256", "knowledge_chunk_id")
    )


@pytest.mark.parametrize("kb", [None, 44])
def test_production_factory_wires_scoped_perception(tmp_path, monkeypatch, kb):
    from test_learning_assistant_workflow import (
        _DIRECT_READ_ARGUMENTS,
        _completion,
        _execute_workflow,
        _perception_round,
    )
    from test_skill_app_wiring import app_flow, app_settings

    from langley.answering.fake_provider import FakeProvider, ScriptedProviderRound
    from langley.infrastructure.local_file_storage import LocalFileStorage
    from langley.main import create_app

    called = []

    async def read(session_factory, storage, **kwargs):
        assert isinstance(storage, LocalFileStorage)
        assert kwargs["max_content_bytes"] == 128
        called.append(kwargs["knowledge_base_id"])
        return result(kwargs["locator"])

    monkeypatch.setattr(knowledge_tools, "read_knowledge", read)
    settings = app_settings(tmp_path).model_copy(
        update={"knowledge_read_max_content_bytes": 128}
    )
    rounds = (
        []
        if kb is None
        else [_perception_round("read_knowledge", _DIRECT_READ_ARGUMENTS)]
    )
    provider = FakeProvider(
        [
            *rounds,
            ScriptedProviderRound(
                events=(_completion(content="answer" if kb is None else "answer [K1]"),)
            ),
        ]
    )
    app = create_app(settings, provider=provider)
    try:
        asyncio.run(_execute_workflow(app_flow(app), knowledge_base_id=kb))
    finally:
        asyncio.run(app.state.database_engine.dispose())
    first = provider.requests[0]
    names = {tool.name for tool in first.allowed_tools}
    assert ("inspect_knowledge" in names) == (kb is not None)
    assert ("read_knowledge" in names) == (kb is not None)
    assert called == ([] if kb is None else [44])
    if kb is not None:
        schema = next(
            tool.arguments_schema
            for tool in first.allowed_tools
            if tool.name == "read_knowledge"
        )
        locator = schema["$defs"]["KnowledgeLocator"]
        assert set(locator["properties"]) == {
            "document_id",
            "kind",
            "heading_path",
            "page_start",
            "page_end",
        }
        assert locator["additionalProperties"] is False


@pytest.mark.parametrize(
    "locator",
    [
        {"document_id": True, "kind": "document"},
        {"document_id": "12", "kind": "document"},
        {"document_id": 12, "kind": "pages", "page_start": True, "page_end": 2},
        {"document_id": 12, "kind": "document", "document_version_id": 23},
        {"document_id": 12, "kind": "heading", "heading_path": [12]},
    ],
)
def test_strict_locator_schema_rejects_untrusted_identity(locator):
    _, tool = tools()
    output = execute(tool, {"locator": locator}, ToolContext(1, 2, 3))
    assert output.kind is ToolResultKind.INVALID_ARGUMENTS


def test_tools_without_scope_do_not_call_backend():
    for tool, args in zip(
        tools(), ({}, {"locator": {"document_id": 12, "kind": "document"}})
    ):
        output = execute(tool, args, ToolContext(1, 2, None))
        assert (
            json.loads(output.content)["error"]["code"] == "KNOWLEDGE_SCOPE_UNAVAILABLE"
        )


@pytest.mark.parametrize(
    "failure",
    [
        KnowledgePerceptionError("LOCATION_NOT_FOUND", "Inspect again."),
        KnowledgePerceptionError("KNOWLEDGE_READ_SCOPE_TOO_LARGE", "Narrow the scope."),
        asyncio.CancelledError(),
        RuntimeError("unexpected"),
    ],
)
def test_failed_read_never_registers_evidence_and_preserves_cancellation(
    monkeypatch, failure
):
    async def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(knowledge_tools, "read_knowledge", fail)
    _, tool = tools()
    context = ToolContext(1, 2, 3)
    args = {"locator": {"document_id": 12, "kind": "document"}}
    if isinstance(failure, KnowledgePerceptionError):
        output = execute(tool, args, context)
        assert json.loads(output.content)["error"]["code"] == failure.code
    else:
        expected = (
            asyncio.CancelledError
            if isinstance(failure, asyncio.CancelledError)
            else WorkflowFailure
        )
        with pytest.raises(expected):
            execute(tool, args, context)
    assert context.knowledge_evidence.evidence == ()
