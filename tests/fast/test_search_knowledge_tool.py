"""Focused contracts for the isolated Task 7.1 knowledge Tool adapter."""

import asyncio
import json
from dataclasses import dataclass, field

import pytest

from langley.answering.contracts import ToolCall, ToolResultKind
from langley.answering.errors import RunErrorCode, WorkflowFailure
from langley.answering.tools import (
    CurrentTimeTool,
    ExpandEvidenceTool,
    SearchKnowledgeTool,
    ToolContext,
    ToolExecutor,
)
from langley.answering.tracing import KnowledgeSearchOrigin
from langley.knowledge import retrieval_service
from langley.knowledge.reads import (
    AdjacentKnowledgeAnchorChangedError,
    AdjacentKnowledgeChunkRead,
)
from langley.knowledge.retrieval import (
    IndexNotReadyError,
    RetrievalHit,
    RetrievalResult,
)
from langley.knowledge.retrieval_service import (
    KnowledgeRetrievalService,
    KnowledgeSearchError,
)


def _search_call(raw_arguments: str = '{"query":" TCP ","top_k":2}') -> ToolCall:
    return ToolCall("search-1", "search_knowledge", raw_arguments)


def _hit() -> RetrievalHit:
    return RetrievalHit(
        knowledge_chunk_id=11,
        rank=1,
        retrieval_rank=1,
        score=0.9,
        rerank_score=None,
        chunk_ordinal=4,
        content="TIME-WAIT prevents delayed packets from being misread.",
        heading_path=("TCP", "TIME-WAIT"),
        source_regions=(),
        document_id=12,
        document_version_id=13,
        source_display_name="tcp.md",
        source_sha256="a" * 64,
    )


@dataclass
class _FakeRetrievalService:
    result: RetrievalResult | None = None
    error: Exception | None = None
    calls: list[tuple[int, int, str, int]] = field(default_factory=list)

    async def search(
        self, *, user_id: int, knowledge_base_id: int, query: str, top_k: int
    ) -> RetrievalResult:
        self.calls.append((user_id, knowledge_base_id, query, top_k))
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


def _service() -> _FakeRetrievalService:
    return _FakeRetrievalService(
        result=RetrievalResult(
            knowledge_base_id=23,
            hits=(_hit(),),
        )
    )


def _executor(service: _FakeRetrievalService) -> ToolExecutor:
    return ToolExecutor(
        tools=(CurrentTimeTool(), SearchKnowledgeTool(service))  # type: ignore[arg-type]
    )


def _context(knowledge_base_id: int | None = 43) -> ToolContext:
    return ToolContext(run_id=41, user_id=42, knowledge_base_id=knowledge_base_id)


@pytest.mark.anyio
async def test_explicit_registry_keeps_time_and_exposes_search() -> None:
    executor = _executor(_service())

    assert tuple(tool.name for tool in ToolExecutor().allowed_tools) == (
        "get_current_time",
    )
    assert tuple(tool.name for tool in executor.allowed_tools) == (
        "get_current_time",
        "search_knowledge",
    )
    with pytest.raises(ValueError, match="duplicate tool registration"):
        ToolExecutor(
            tools=(
                CurrentTimeTool(),
                SearchKnowledgeTool(_service()),  # type: ignore[arg-type]
                SearchKnowledgeTool(_service()),  # type: ignore[arg-type]
            )
        )


def test_search_schema_exposes_declared_constraints() -> None:
    schema = SearchKnowledgeTool.spec.arguments_schema

    assert schema["additionalProperties"] is False
    query = schema["properties"]["query"]
    top_k = schema["properties"]["top_k"]
    assert query["type"] == "string"
    assert query["minLength"] == 1
    assert top_k["type"] == "integer"
    assert top_k["default"] == 5
    assert top_k["minimum"] == 1
    assert top_k["maximum"] == 5


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("raw_arguments", "expected_kind"),
    [
        ('{"query":"useful"}', ToolResultKind.SUCCESS),
        ('{"query":"   "}', ToolResultKind.INVALID_ARGUMENTS),
        ('{"query":"useful","top_k":6}', ToolResultKind.INVALID_ARGUMENTS),
        ('{"query":"useful","top_k":"2"}', ToolResultKind.INVALID_ARGUMENTS),
        ('{"query":"useful","user_id":99}', ToolResultKind.INVALID_ARGUMENTS),
    ],
)
async def test_search_argument_boundary(
    raw_arguments: str, expected_kind: ToolResultKind
) -> None:
    service = _service()
    result = (
        await _executor(service).execute_batch(
            (_search_call(raw_arguments),), context=_context(3)
        )
    )[0]

    assert result.kind is expected_kind


@pytest.mark.anyio
async def test_search_without_kb_scope_fails_closed_without_service_call() -> None:
    service = _service()
    executor = _executor(service)
    result = (await executor.execute_batch((_search_call(),)))[0]

    assert result.kind is ToolResultKind.TOOL_ERROR
    assert json.loads(result.content) == {
        "error": {"code": "KNOWLEDGE_SCOPE_UNAVAILABLE", "retryable": False}
    }
    assert service.calls == []

    result = (await executor.execute_batch((_search_call(),), context=_context(None)))[
        0
    ]
    assert result.kind is ToolResultKind.TOOL_ERROR
    assert service.calls == []


@pytest.mark.anyio
async def test_search_uses_only_server_context_and_safe_evidence() -> None:
    service = _service()
    result = (
        await _executor(service).execute_batch(
            (_search_call('{"query":" TCP ","top_k":2,"knowledge_base_id":999}'),),
            context=_context(),
        )
    )[0]

    assert result.kind is ToolResultKind.INVALID_ARGUMENTS
    assert service.calls == []

    result = (
        await _executor(service).execute_batch((_search_call(),), context=_context())
    )[0]
    assert result.kind is ToolResultKind.SUCCESS
    assert service.calls == [(42, 43, "TCP", 2)]
    assert result.content == (
        '{"evidence":[{"evidence_handle":"K1","content":"TIME-WAIT prevents '
        'delayed packets from being misread.","source_display_name":"tcp.md",'
        '"heading_path":["TCP","TIME-WAIT"]}]}'
    )


def _neighbor(
    chunk_id: int,
    ordinal: int,
    position: str,
) -> AdjacentKnowledgeChunkRead:
    return AdjacentKnowledgeChunkRead(
        position=position,  # type: ignore[arg-type]
        knowledge_chunk_id=chunk_id,
        chunk_ordinal=ordinal,
        document_id=12,
        document_version_id=13,
        content=f"chunk {ordinal}",
        heading_path=("TCP", f"Part {ordinal}"),
        source_regions=(
            {
                "kind": "text_span",
                "start_byte": ordinal,
                "end_byte": ordinal + 1,
            },
        ),
        source_display_name="tcp.md",
        source_sha256="a" * 64,
    )


@pytest.mark.anyio
async def test_expand_evidence_is_handle_only_cumulative_and_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed = False

    async def read_neighbors(*args: object, **kwargs: object):
        del args
        if changed:
            raise AdjacentKnowledgeAnchorChangedError
        anchor_id = kwargs["anchor_knowledge_chunk_id"]
        return {
            11: (_neighbor(10, 3, "previous"), _neighbor(12, 5, "next")),
            10: (_neighbor(11, 4, "next"),),
            12: (_neighbor(11, 4, "previous"), _neighbor(13, 6, "next")),
            13: (),
        }[anchor_id]

    monkeypatch.setattr(
        "langley.answering.tools.read_adjacent_knowledge_chunks", read_neighbors
    )
    context = _context()
    context.knowledge_evidence.register_hit(_hit())
    executor = ToolExecutor(
        tools=(ExpandEvidenceTool(None),)  # type: ignore[arg-type]
    )

    schema = ExpandEvidenceTool.spec.arguments_schema
    assert schema["additionalProperties"] is False
    assert schema["properties"]["evidence_handle"]["pattern"] == "^K[1-9][0-9]*$"
    for raw_arguments in (
        '{"evidence_handle":2}',
        '{"evidence_handle":"K0"}',
        '{"evidence_handle":"K1","ordinal":4}',
    ):
        result = (
            await executor.execute_batch(
                (ToolCall("expand-invalid", "expand_evidence", raw_arguments),),
                context=context,
            )
        )[0]
        assert result.kind is ToolResultKind.INVALID_ARGUMENTS

    unknown = (
        await executor.execute_batch(
            (
                ToolCall(
                    "expand-unknown",
                    "expand_evidence",
                    '{"evidence_handle":"K9"}',
                ),
            ),
            context=context,
        )
    )[0]
    assert unknown.kind is ToolResultKind.TOOL_ERROR
    assert json.loads(unknown.content) == {
        "error": {"code": "KNOWLEDGE_EVIDENCE_UNAVAILABLE", "retryable": False}
    }

    first = (
        await executor.execute_batch(
            (ToolCall("expand-1", "expand_evidence", '{"evidence_handle":"K1"}'),),
            context=context,
        )
    )[0]
    assert json.loads(first.content) == {
        "anchor": "K1",
        "neighbors": [
            {
                "position": "previous",
                "evidence_handle": "K2",
                "status": "new",
                "content": "chunk 3",
                "source_display_name": "tcp.md",
                "heading_path": ["TCP", "Part 3"],
            },
            {
                "position": "next",
                "evidence_handle": "K3",
                "status": "new",
                "content": "chunk 5",
                "source_display_name": "tcp.md",
                "heading_path": ["TCP", "Part 5"],
            },
        ],
    }

    repeated = (
        await executor.execute_batch(
            (ToolCall("expand-2", "expand_evidence", '{"evidence_handle":"K1"}'),),
            context=context,
        )
    )[0]
    assert json.loads(repeated.content) == {
        "anchor": "K1",
        "neighbors": [
            {"position": "previous", "evidence_handle": "K2", "status": "existing"},
            {"position": "next", "evidence_handle": "K3", "status": "existing"},
        ],
    }

    chained = (
        await executor.execute_batch(
            (ToolCall("expand-3", "expand_evidence", '{"evidence_handle":"K3"}'),),
            context=context,
        )
    )[0]
    assert json.loads(chained.content)["neighbors"] == [
        {"position": "previous", "evidence_handle": "K1", "status": "existing"},
        {
            "position": "next",
            "evidence_handle": "K4",
            "status": "new",
            "content": "chunk 6",
            "source_display_name": "tcp.md",
            "heading_path": ["TCP", "Part 6"],
        },
    ]
    boundary = (
        await executor.execute_batch(
            (ToolCall("expand-4", "expand_evidence", '{"evidence_handle":"K4"}'),),
            context=context,
        )
    )[0]
    assert json.loads(boundary.content) == {"anchor": "K4", "neighbors": []}

    changed = True
    stale = (
        await executor.execute_batch(
            (ToolCall("expand-5", "expand_evidence", '{"evidence_handle":"K1"}'),),
            context=context,
        )
    )[0]
    assert json.loads(stale.content) == {
        "error": {"code": "KNOWLEDGE_EVIDENCE_CHANGED", "retryable": False}
    }


@pytest.mark.anyio
async def test_known_retrieval_error_maps_to_safe_tool_error() -> None:
    service = _service()
    service.error = KnowledgeSearchError("KNOWLEDGE_INDEX_NOT_READY", retryable=False)
    result = (
        await _executor(service).execute_batch((_search_call(),), context=_context(3))
    )[0]

    assert result.kind is ToolResultKind.TOOL_ERROR
    assert json.loads(result.content) == {
        "error": {"code": "KNOWLEDGE_INDEX_NOT_READY", "retryable": False}
    }


@pytest.mark.anyio
async def test_unexpected_and_cancelled_tool_failures_remain_fail_closed() -> None:
    service = _service()
    service.error = RuntimeError("internal detail")
    with pytest.raises(WorkflowFailure) as raised:
        await _executor(service).execute_batch((_search_call(),), context=_context(3))
    assert raised.value.error_code is RunErrorCode.TOOL_EXECUTION_FAILED

    service.error = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await _executor(service).execute_batch((_search_call(),), context=_context(3))


@pytest.mark.anyio
async def test_retrieval_service_maps_retrieval_error_to_stable_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def failing_retrieve(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise IndexNotReadyError()

    monkeypatch.setattr(retrieval_service, "retrieve_dense", failing_retrieve)
    service = KnowledgeRetrievalService(None, None)  # type: ignore[arg-type]
    with pytest.raises(KnowledgeSearchError) as raised:
        await service.search(
            user_id=42,
            knowledge_base_id=43,
            query="TCP",
            top_k=2,
        )

    assert raised.value.code == "KNOWLEDGE_INDEX_NOT_READY"
    assert raised.value.retryable is False


@pytest.mark.anyio
async def test_retrieval_service_uses_narrow_explicit_required_trace_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = RetrievalResult(43, (_hit(),))

    async def retrieve(*args: object, **kwargs: object) -> RetrievalResult:
        del args, kwargs
        return result

    observations: list[dict[str, object]] = []

    class SearchTrace:
        def finish(self, hit_count: int | None, error_code: str | None = None) -> None:
            observations.append({"hit_count": hit_count, "error_code": error_code})

    class Parent:
        def begin_knowledge_search(self, **kwargs: object) -> SearchTrace:
            observations.append(dict(kwargs))
            return SearchTrace()

    monkeypatch.setattr(retrieval_service, "retrieve_dense", retrieve)
    service = KnowledgeRetrievalService(None, None)  # type: ignore[arg-type]

    returned = await service.search(
        user_id=42,
        knowledge_base_id=43,
        query="TCP",
        top_k=5,
        trace_parent=Parent(),
        origin=KnowledgeSearchOrigin.HARNESS_REQUIRED,
    )

    assert returned is result
    assert observations == [
        {
            "origin": KnowledgeSearchOrigin.HARNESS_REQUIRED,
            "knowledge_base_id": 43,
            "top_k": 5,
            "query": "TCP",
        },
        {"hit_count": 1, "error_code": None},
    ]
