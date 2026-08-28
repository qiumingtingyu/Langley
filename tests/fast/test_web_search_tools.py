"""Focused Web provider, tool contract, and run-local capability tests."""

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from langley.answering.contracts import ToolCall, ToolResultKind
from langley.answering.tools import (
    ReadWebpageTool,
    SearchWebTool,
    ToolContext,
    ToolExecutor,
)
from langley.answering.web import (
    TavilyWebProvider,
    WebExtractResponse,
    WebProviderError,
    WebSearchResponse,
    WebSearchResult,
    WebToolSession,
)


@dataclass
class _FakeWebProvider:
    search_response: WebSearchResponse = WebSearchResponse(
        results=(
            WebSearchResult(
                title="Official One",
                url="https://docs.example.test/one",
                domain="docs.example.test",
                snippet="discovery only",
                provider_score=0.9,
            ),
            WebSearchResult(
                title="Official Two",
                url="https://docs.example.test/two",
                domain="docs.example.test",
                snippet="second discovery",
                provider_score=0.8,
            ),
        ),
        response_time_seconds=0.25,
        credits=1.0,
        request_id="search-request",
    )
    extract_response: WebExtractResponse = WebExtractResponse(
        contents=("trusted facts from the selected page",),
        response_time_seconds=0.4,
        credits=0.0,
        request_id="extract-request",
    )
    search_error: WebProviderError | None = None
    extract_error: WebProviderError | None = None
    search_queries: list[str] = field(default_factory=list)
    extract_calls: list[tuple[str, str]] = field(default_factory=list)

    async def search(self, query: str) -> WebSearchResponse:
        self.search_queries.append(query)
        if self.search_error is not None:
            raise self.search_error
        return self.search_response

    async def extract(self, url: str, focus: str) -> WebExtractResponse:
        self.extract_calls.append((url, focus))
        if self.extract_error is not None:
            raise self.extract_error
        return self.extract_response


class _RecordingToolTrace:
    def __init__(self) -> None:
        self.metadata: dict[str, object] | None = None

    def finish(
        self,
        result: object,
        error_code: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        del result, error_code
        self.metadata = metadata


class _RecordingExecutionTrace:
    def __init__(self) -> None:
        self.tool = _RecordingToolTrace()

    def begin_tool(self, call: ToolCall, tool_calls_used: int) -> _RecordingToolTrace:
        del call, tool_calls_used
        return self.tool


def _executor(provider: _FakeWebProvider) -> ToolExecutor:
    return ToolExecutor(tools=(SearchWebTool(provider), ReadWebpageTool(provider)))


def _context(session: WebToolSession | None = None) -> ToolContext:
    return ToolContext(
        run_id=101,
        user_id=1,
        knowledge_base_id=None,
        web_session=session or WebToolSession(),
    )


async def _search(
    executor: ToolExecutor,
    context: ToolContext,
    call_id: str = "search-1",
    trace: Any = None,
):
    return (
        await executor.execute_batch(
            (ToolCall(call_id, "search_web", '{"query":"latest facts"}'),),
            context=context,
            trace=trace,
        )
    )[0]


async def _read(
    executor: ToolExecutor,
    context: ToolContext,
    call_id: str = "read-1",
    result_id: str = "W1",
):
    return (
        await executor.execute_batch(
            (
                ToolCall(
                    call_id,
                    "read_webpage",
                    json.dumps({"result_id": result_id, "focus": "relevant facts"}),
                ),
            ),
            context=context,
        )
    )[0]


@pytest.mark.anyio
@pytest.mark.parametrize(
    "raw_arguments",
    ("{}", '{"query":"  "}', '{"query":"ok","extra":true}', "[]"),
)
async def test_search_web_rejects_invalid_arguments(raw_arguments: str) -> None:
    result = (
        await _executor(_FakeWebProvider()).execute_batch(
            (ToolCall("search", "search_web", raw_arguments),),
            context=_context(),
        )
    )[0]

    assert result.kind is ToolResultKind.INVALID_ARGUMENTS


@pytest.mark.anyio
async def test_search_web_assigns_deterministic_run_local_handles() -> None:
    provider = _FakeWebProvider()
    context = _context()
    trace = _RecordingExecutionTrace()

    result = await _search(_executor(provider), context, trace=trace)

    assert result.kind is ToolResultKind.SUCCESS
    assert json.loads(result.content) == {
        "results": [
            {
                "result_id": "W1",
                "title": "Official One",
                "url": "https://docs.example.test/one",
                "domain": "docs.example.test",
                "snippet": "discovery only",
                "score": 0.9,
            },
            {
                "result_id": "W2",
                "title": "Official Two",
                "url": "https://docs.example.test/two",
                "domain": "docs.example.test",
                "snippet": "second discovery",
                "score": 0.8,
            },
        ]
    }
    assert provider.search_queries == ["latest facts"]
    assert [source.result_id for source in context.web_session.sources] == ["W1", "W2"]
    assert trace.tool.metadata is not None
    assert trace.tool.metadata["success"] is True


@pytest.mark.anyio
async def test_search_web_maps_provider_failure_to_safe_tool_error() -> None:
    provider = _FakeWebProvider(
        search_error=WebProviderError("WEB_PROVIDER_UNAVAILABLE", retryable=True)
    )
    trace = _RecordingExecutionTrace()

    result = await _search(_executor(provider), _context(), trace=trace)

    assert result.kind is ToolResultKind.TOOL_ERROR
    assert json.loads(result.content) == {
        "error": {"code": "WEB_PROVIDER_UNAVAILABLE", "retryable": True}
    }
    assert "exception" not in result.content.lower()
    assert trace.tool.metadata == {
        "success": False,
        "tool_error_code": "WEB_PROVIDER_UNAVAILABLE",
        "retryable": True,
    }
    assert "latest facts" not in repr(trace.tool.metadata)


@pytest.mark.anyio
async def test_read_webpage_rejects_unknown_or_other_run_handle() -> None:
    provider = _FakeWebProvider()
    executor = _executor(provider)
    first_run = _context()
    second_run = _context()
    await _search(executor, first_run)

    unknown = await _read(executor, first_run, result_id="W99")
    other_run = await _read(executor, second_run, result_id="W1")

    assert unknown.kind is ToolResultKind.TOOL_ERROR
    assert other_run.kind is ToolResultKind.TOOL_ERROR
    assert json.loads(unknown.content)["error"]["code"] == "WEB_RESULT_NOT_AVAILABLE"
    assert json.loads(other_run.content)["error"]["code"] == "WEB_RESULT_NOT_AVAILABLE"
    assert provider.extract_calls == []


@pytest.mark.anyio
async def test_read_webpage_uses_stored_url_and_marks_evidence_untrusted() -> None:
    provider = _FakeWebProvider()
    executor = _executor(provider)
    context = _context()
    await _search(executor, context)

    result = await _read(executor, context)

    assert result.kind is ToolResultKind.SUCCESS
    assert provider.extract_calls == [
        ("https://docs.example.test/one", "relevant facts")
    ]
    assert json.loads(result.content) == {
        "source": {
            "result_id": "W1",
            "title": "Official One",
            "url": "https://docs.example.test/one",
            "domain": "docs.example.test",
        },
        "evidence": [
            {
                "evidence_handle": "W1:E1",
                "content": "trusted facts from the selected page",
            }
        ],
        "untrusted_external_content": True,
    }


@pytest.mark.anyio
async def test_web_session_enforces_one_successful_search_and_two_reads() -> None:
    provider = _FakeWebProvider()
    executor = _executor(provider)
    context = _context()
    assert (await _search(executor, context)).kind is ToolResultKind.SUCCESS

    second_search = await _search(executor, context, call_id="search-2")
    assert (
        await _read(executor, context, call_id="read-1")
    ).kind is ToolResultKind.SUCCESS
    assert (
        await _read(executor, context, call_id="read-2")
    ).kind is ToolResultKind.SUCCESS
    third_read = await _read(executor, context, call_id="read-3")

    assert json.loads(second_search.content)["error"]["code"] == (
        "WEB_SEARCH_BUDGET_EXHAUSTED"
    )
    assert json.loads(third_read.content)["error"]["code"] == (
        "WEB_READ_BUDGET_EXHAUSTED"
    )
    assert context.web_session.successful_searches == 1
    assert context.web_session.successful_reads == 2


class _RecordingTavilyClient:
    def __init__(self) -> None:
        self.search_kwargs: dict[str, Any] | None = None
        self.extract_kwargs: dict[str, Any] | None = None

    async def search(self, **kwargs: Any) -> dict[str, Any]:
        self.search_kwargs = kwargs
        return {
            "results": [
                {
                    "title": " Official docs ",
                    "url": "https://official.example.test/page",
                    "content": " snippet ",
                    "score": 0.7,
                }
            ],
            "response_time": 0.2,
            "usage": {"credits": 1},
            "request_id": "request-1",
        }

    async def extract(self, **kwargs: Any) -> dict[str, Any]:
        self.extract_kwargs = kwargs
        return {
            "results": [{"raw_content": " A [...] B [...] C "}],
            "response_time": 0.3,
            "usage": {"credits": 0},
            "request_id": "request-2",
        }


@pytest.mark.anyio
async def test_tavily_adapter_freezes_v1_parameters_and_normalizes_schema() -> None:
    client = _RecordingTavilyClient()
    provider = TavilyWebProvider("test-key", client=client)

    search = await provider.search("query")
    extract = await provider.extract("https://official.example.test/page", "focus")

    assert client.search_kwargs == {
        "query": "query",
        "search_depth": "basic",
        "max_results": 5,
        "topic": "general",
        "include_answer": False,
        "include_raw_content": False,
        "include_images": False,
        "auto_parameters": False,
        "include_usage": True,
    }
    assert client.extract_kwargs == {
        "urls": "https://official.example.test/page",
        "query": "focus",
        "chunks_per_source": 5,
        "extract_depth": "basic",
        "format": "markdown",
        "include_usage": True,
    }
    assert search.results[0].title == "Official docs"
    assert search.results[0].domain == "official.example.test"
    assert extract.contents == ("A", "B", "C")


@pytest.mark.anyio
async def test_tavily_literal_chunks_receive_granular_evidence_handles() -> None:
    client = _RecordingTavilyClient()
    provider = TavilyWebProvider("test-key", client=client)
    executor = ToolExecutor(tools=(SearchWebTool(provider), ReadWebpageTool(provider)))
    context = _context()

    assert (await _search(executor, context)).kind is ToolResultKind.SUCCESS
    result = await _read(executor, context)

    assert result.kind is ToolResultKind.SUCCESS
    assert [
        item["evidence_handle"] for item in json.loads(result.content)["evidence"]
    ] == ["W1:E1", "W1:E2", "W1:E3"]
