"""Run-local Web search capability and Tavily provider boundary."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, NoReturn, Protocol
from urllib.parse import urlparse

from tavily import AsyncTavilyClient  # type: ignore[import-untyped]

from langley.answering.errors import (
    InvalidResponseSubtype,
    RunErrorCode,
    WorkflowFailure,
)

_WEB_CITATION_CANDIDATE = re.compile(r"\[W[^\]\r\n]*\]")
_WEB_CITATION = re.compile(r"\[W([1-9][0-9]*):E([1-9][0-9]*)\]")
_WEB_URL = re.compile(r"https?://", re.IGNORECASE)
_TAVILY_CHUNK_SEPARATOR = "[...]"


class WebProviderError(Exception):
    """Safe normalized failure from the external Web provider."""

    def __init__(self, code: str, *, retryable: bool) -> None:
        super().__init__(code)
        self.code = code
        self.retryable = retryable


@dataclass(frozen=True)
class WebSearchResult:
    """One normalized source-discovery result."""

    title: str
    url: str
    domain: str
    snippet: str
    provider_score: float | None


@dataclass(frozen=True)
class WebSearchResponse:
    """Normalized Web search response plus safe provider metadata."""

    results: tuple[WebSearchResult, ...]
    response_time_seconds: float | None
    credits: float | None
    request_id: str | None


@dataclass(frozen=True)
class WebExtractResponse:
    """Normalized extracted evidence plus safe provider metadata."""

    contents: tuple[str, ...]
    response_time_seconds: float | None
    credits: float | None
    request_id: str | None


class WebProvider(Protocol):
    """The complete provider seam needed by Web Search V1."""

    async def search(self, query: str) -> WebSearchResponse: ...

    async def extract(self, url: str, focus: str) -> WebExtractResponse: ...


class TavilyWebProvider:
    """Official Tavily SDK adapter with frozen V1 request parameters."""

    def __init__(self, api_key: str, *, client: Any | None = None) -> None:
        self._client = client or AsyncTavilyClient(api_key=api_key)

    async def search(self, query: str) -> WebSearchResponse:
        try:
            response = await self._client.search(
                query=query,
                search_depth="basic",
                max_results=5,
                topic="general",
                include_answer=False,
                include_raw_content=False,
                include_images=False,
                auto_parameters=False,
                include_usage=True,
            )
            return _normalize_search_response(response)
        except WebProviderError:
            raise
        except Exception as error:
            raise WebProviderError(
                "WEB_PROVIDER_UNAVAILABLE", retryable=True
            ) from error

    async def extract(self, url: str, focus: str) -> WebExtractResponse:
        try:
            response = await self._client.extract(
                urls=url,
                query=focus,
                chunks_per_source=5,
                extract_depth="basic",
                format="markdown",
                include_usage=True,
            )
            return _normalize_extract_response(response)
        except WebProviderError:
            raise
        except Exception as error:
            raise WebProviderError(
                "WEB_PROVIDER_UNAVAILABLE", retryable=True
            ) from error


@dataclass(frozen=True)
class WebSource:
    """One result registered as a capability in the current Run."""

    result_id: str
    title: str
    url: str
    domain: str
    snippet: str
    provider_score: float | None


@dataclass(frozen=True)
class WebEvidence:
    """One current-Run evidence block created only by a successful read."""

    evidence_handle: str
    result_id: str
    content: str


class WebSessionError(Exception):
    """Expected run-local capability or budget rejection."""

    def __init__(self, code: str, *, retryable: bool) -> None:
        super().__init__(code)
        self.code = code
        self.retryable = retryable


class WebToolSession:
    """One-run registry for result handles, evidence, and Web budgets."""

    def __init__(self) -> None:
        self.search_attempted = False
        self.successful_searches = 0
        self.successful_reads = 0
        self._sources: dict[str, WebSource] = {}
        self._evidence: dict[str, WebEvidence] = {}

    @property
    def sources(self) -> tuple[WebSource, ...]:
        return tuple(self._sources.values())

    @property
    def evidence(self) -> tuple[WebEvidence, ...]:
        return tuple(self._evidence.values())

    def start_search(self) -> None:
        self.search_attempted = True
        if self.successful_searches >= 1:
            raise WebSessionError("WEB_SEARCH_BUDGET_EXHAUSTED", retryable=False)

    def register_search(
        self, results: tuple[WebSearchResult, ...]
    ) -> tuple[WebSource, ...]:
        if self.successful_searches >= 1:
            raise WebSessionError("WEB_SEARCH_BUDGET_EXHAUSTED", retryable=False)
        self.successful_searches += 1
        sources = tuple(
            WebSource(
                result_id=f"W{index}",
                title=result.title,
                url=result.url,
                domain=result.domain,
                snippet=result.snippet,
                provider_score=result.provider_score,
            )
            for index, result in enumerate(results, start=1)
        )
        self._sources = {source.result_id: source for source in sources}
        return sources

    def resolve_source(self, result_id: str) -> WebSource:
        source = self._sources.get(result_id)
        if source is None:
            raise WebSessionError("WEB_RESULT_NOT_AVAILABLE", retryable=False)
        if self.successful_reads >= 2:
            raise WebSessionError("WEB_READ_BUDGET_EXHAUSTED", retryable=False)
        return source

    def register_read(
        self, result_id: str, contents: tuple[str, ...]
    ) -> tuple[WebEvidence, ...]:
        if self.successful_reads >= 2:
            raise WebSessionError("WEB_READ_BUDGET_EXHAUSTED", retryable=False)
        source = self._sources.get(result_id)
        if source is None:
            raise WebSessionError("WEB_RESULT_NOT_AVAILABLE", retryable=False)
        normalized = tuple(content.strip() for content in contents if content.strip())
        if not normalized:
            raise WebSessionError("WEB_EMPTY_EVIDENCE", retryable=True)

        next_ordinal = 1 + sum(
            evidence.result_id == result_id for evidence in self._evidence.values()
        )
        evidence = tuple(
            WebEvidence(
                evidence_handle=f"{result_id}:E{next_ordinal + index}",
                result_id=result_id,
                content=content,
            )
            for index, content in enumerate(normalized)
        )
        for item in evidence:
            self._evidence[item.evidence_handle] = item
        self.successful_reads += 1
        return evidence

    def evidence_for(self, handle: str) -> WebEvidence | None:
        return self._evidence.get(handle)

    def source_for(self, result_id: str) -> WebSource | None:
        return self._sources.get(result_id)


def validated_web_answer(
    content: str,
    session: WebToolSession,
    *,
    requires_citation: bool,
) -> str:
    """Validate current-Run W evidence handles and append trusted source URLs."""

    validated_content = content
    for source in sorted(session.sources, key=lambda item: len(item.url), reverse=True):
        validated_content = validated_content.replace(
            source.url, f"source {source.result_id}"
        )
    if session.sources and _WEB_URL.search(validated_content):
        _invalid_web_citation(InvalidResponseSubtype.UNKNOWN_CITATION_HANDLE)

    cited_evidence: list[WebEvidence] = []
    for candidate in _WEB_CITATION_CANDIDATE.findall(validated_content):
        match = _WEB_CITATION.fullmatch(candidate)
        if match is None:
            _invalid_web_citation(InvalidResponseSubtype.UNKNOWN_CITATION_HANDLE)
        evidence = session.evidence_for(f"W{match.group(1)}:E{match.group(2)}")
        if evidence is None:
            _invalid_web_citation(InvalidResponseSubtype.UNKNOWN_CITATION_HANDLE)
        cited_evidence.append(evidence)

    if requires_citation and not cited_evidence:
        _invalid_web_citation(InvalidResponseSubtype.MISSING_REQUIRED_CITATION)
    if not cited_evidence:
        return validated_content

    sources: list[WebSource] = []
    seen_result_ids: set[str] = set()
    for evidence in cited_evidence:
        if evidence.result_id in seen_result_ids:
            continue
        resolved_source = session.source_for(evidence.result_id)
        if resolved_source is None:
            _invalid_web_citation(InvalidResponseSubtype.UNKNOWN_CITATION_HANDLE)
        sources.append(resolved_source)
        seen_result_ids.add(evidence.result_id)

    footer = "\n".join(
        f"- [{source.result_id}] {source.title} — {source.url}" for source in sources
    )
    return f"{validated_content.rstrip()}\n\n来源：\n{footer}"


def _invalid_web_citation(subtype: InvalidResponseSubtype) -> NoReturn:
    raise WorkflowFailure(
        RunErrorCode.LLM_RESPONSE_INVALID,
        invalid_response_subtype=subtype,
    )


def _normalize_search_response(response: Any) -> WebSearchResponse:
    if not isinstance(response, dict) or not isinstance(response.get("results"), list):
        raise WebProviderError("WEB_PROVIDER_RESPONSE_INVALID", retryable=False)
    results: list[WebSearchResult] = []
    for item in response["results"]:
        if not isinstance(item, dict):
            raise WebProviderError("WEB_PROVIDER_RESPONSE_INVALID", retryable=False)
        title = item.get("title")
        url = item.get("url")
        snippet = item.get("content", "")
        score = item.get("score")
        if (
            not isinstance(title, str)
            or not title.strip()
            or not isinstance(url, str)
            or not _safe_web_url(url)
            or not isinstance(snippet, str)
            or not (score is None or isinstance(score, int | float))
        ):
            raise WebProviderError("WEB_PROVIDER_RESPONSE_INVALID", retryable=False)
        results.append(
            WebSearchResult(
                title=" ".join(title.split()),
                url=url,
                domain=urlparse(url).netloc.lower(),
                snippet=snippet.strip(),
                provider_score=None if score is None else float(score),
            )
        )
    return WebSearchResponse(
        results=tuple(results),
        response_time_seconds=_optional_number(response.get("response_time")),
        credits=_credits(response),
        request_id=_optional_string(response.get("request_id")),
    )


def _normalize_extract_response(response: Any) -> WebExtractResponse:
    if not isinstance(response, dict) or not isinstance(response.get("results"), list):
        raise WebProviderError("WEB_PROVIDER_RESPONSE_INVALID", retryable=False)
    contents: list[str] = []
    for item in response["results"]:
        if not isinstance(item, dict):
            raise WebProviderError("WEB_PROVIDER_RESPONSE_INVALID", retryable=False)
        raw_content = item.get("raw_content")
        if isinstance(raw_content, str):
            contents.extend(
                chunk.strip()
                for chunk in raw_content.split(_TAVILY_CHUNK_SEPARATOR)
                if chunk.strip()
            )
        elif isinstance(raw_content, list) and all(
            isinstance(content, str) for content in raw_content
        ):
            contents.extend(
                content.strip() for content in raw_content if content.strip()
            )
        elif raw_content is not None:
            raise WebProviderError("WEB_PROVIDER_RESPONSE_INVALID", retryable=False)
    return WebExtractResponse(
        contents=tuple(contents),
        response_time_seconds=_optional_number(response.get("response_time")),
        credits=_credits(response),
        request_id=_optional_string(response.get("request_id")),
    )


def _safe_web_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _credits(response: dict[str, Any]) -> float | None:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return None
    return _optional_number(usage.get("credits"))


def _optional_number(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) else None


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) else None
