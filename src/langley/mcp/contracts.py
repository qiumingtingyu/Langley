"""Strict external addresses and fixed, bounded error results."""

from typing import Annotated, Literal, Sequence

from mcp_types import CallToolResult, TextContent
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from langley.knowledge.contracts import (
    KnowledgeLocator,
    PdfPageRegion,
    TextSpanRegion,
    decode_source_region,
    encode_source_region,
)

Identifier = Annotated[int, Field(ge=1, le=9_007_199_254_740_991)]
Revision = Annotated[int, Field(ge=0, le=9_007_199_254_740_991)]
Heading = Annotated[str, Field(min_length=1, max_length=256)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class DocumentLocation(StrictModel):
    kind: Literal["document"]


class HeadingLocation(StrictModel):
    kind: Literal["heading"]
    heading_path: list[Heading] = Field(min_length=1, max_length=16)

    @field_validator("heading_path")
    @classmethod
    def nonblank(cls, value: list[str]) -> list[str]:
        if any(not part.strip() for part in value):
            raise ValueError("INVALID_ARGUMENTS")
        return value


class PagesLocation(StrictModel):
    kind: Literal["pages"]
    page_start: int = Field(ge=1, le=1_000_000)
    page_end: int = Field(ge=1, le=1_000_000)

    @model_validator(mode="after")
    def span(self) -> "PagesLocation":
        if not 0 <= self.page_end - self.page_start < 10:
            raise ValueError("INVALID_ARGUMENTS")
        return self


class Locator(StrictModel):
    v: int = Field(ge=1, le=1)
    kb_id: Identifier
    document_id: Identifier
    document_version_id: Identifier
    chunk_revision: Revision
    location: Annotated[
        DocumentLocation | HeadingLocation | PagesLocation, Field(discriminator="kind")
    ]

    def native(self) -> KnowledgeLocator:
        return KnowledgeLocator.model_validate(
            {"document_id": self.document_id, **self.location.model_dump()}
        )


class SearchArguments(StrictModel):
    query: str = Field(min_length=1, max_length=2048)
    kb_scope: Identifier
    top_k: int = Field(default=5, ge=1, le=5)

    @field_validator("query")
    @classmethod
    def bounded_query(cls, value: str) -> str:
        value = value.strip()
        if not value or len(value.encode("utf-8")) > 8192:
            raise ValueError("INVALID_ARGUMENTS")
        return value


class ReadArguments(StrictModel):
    locator: Locator
    max_bytes: int = Field(default=16384, ge=256, le=16384)


class InspectArguments(StrictModel):
    locator: Locator
    offset: int = Field(default=0, ge=0, le=999)
    limit: int = Field(default=20, ge=1, le=50)

    @field_validator("locator")
    @classmethod
    def document_only(cls, value: Locator) -> Locator:
        if value.location.kind != "document":
            raise ValueError("INVALID_ARGUMENTS")
        return value


ARGUMENTS: dict[str, type[StrictModel]] = {
    "search_knowledge": SearchArguments,
    "read_knowledge": ReadArguments,
    "inspect_knowledge": InspectArguments,
}


def encoded_regions(values: Sequence[object]) -> list[dict[str, object]]:
    if not 1 <= len(values) <= 128:
        raise GateError("RESPONSE_TOO_LARGE")
    result = []
    kind = None
    for value in values:
        region = (
            value
            if isinstance(value, (TextSpanRegion, PdfPageRegion))
            else decode_source_region(value)
        )
        if kind is not None and type(region) is not kind:
            raise GateError("READ_UNAVAILABLE")
        kind = type(region)
        if isinstance(region, TextSpanRegion):
            if not 0 <= region.start_byte < region.end_byte <= 5242880:
                raise GateError("READ_UNAVAILABLE")
        elif not 1 <= region.page_start <= region.page_end <= 1000000:
            raise GateError("READ_UNAVAILABLE")
        result.append(encode_source_region(region))
    return result


class GateError(Exception):
    def __init__(self, code: str, retry_after_ms: int | None = None):
        super().__init__(code)
        self.code = code
        self.retry_after_ms = retry_after_ms


def failure(request_id: str, error: GateError) -> CallToolResult:
    return CallToolResult(
        is_error=True,
        content=[TextContent(type="text", text="Knowledge request failed.")],
        structured_content={
            "ok": False,
            "request_id": request_id,
            "error": {
                "code": error.code,
                "message": "Knowledge request could not be completed.",
                "retryable": error.code in {"RATE_LIMITED", "SERVER_BUSY"},
                "retry_after_ms": error.retry_after_ms,
            },
        },
    )


def success(request_id: str, data: dict) -> CallToolResult:
    result = CallToolResult(
        content=[TextContent(type="text", text="Untrusted Knowledge data returned.")],
        structured_content={"ok": True, "request_id": request_id, "data": data},
    )
    if len(result.model_dump_json(by_alias=True).encode("utf-8")) > 32768:
        raise GateError("RESPONSE_TOO_LARGE")
    return result
