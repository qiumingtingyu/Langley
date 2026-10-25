"""Three read-only adapters over the existing detached Knowledge services."""

import os
from hashlib import sha256

from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from langley.infrastructure.database import (
    create_database_engine,
    create_session_factory,
)
from langley.infrastructure.local_file_storage import LocalFileStorage
from langley.knowledge import perception
from langley.knowledge.contracts import (
    FileStorage,
    KnowledgePerceptionError,
)
from langley.knowledge.embedding_runtime import KnowledgeEmbeddingRuntime
from langley.knowledge.index_build import KnowledgeIndexBuildRuntime
from langley.knowledge.retrieval_service import (
    KnowledgeRetrievalService,
    KnowledgeSearchError,
)
from langley.mcp.access import Access
from langley.mcp.config import Principal
from langley.mcp.contracts import (
    GateError,
    InspectArguments,
    Locator,
    ReadArguments,
    SearchArguments,
    encoded_regions,
    success,
)
from langley.settings import Settings


def bounded_title(value: str) -> str:
    if not 1 <= len(value) <= 255:
        raise GateError("RESPONSE_TOO_LARGE")
    return value


class KnowledgeAdapter:
    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        storage: FileStorage,
        retrieval: KnowledgeRetrievalService,
        engine: AsyncEngine | None = None,
    ):
        self.sessions, self.storage, self.retrieval, self.engine = (
            sessions,
            storage,
            retrieval,
            engine,
        )
        self.access = Access(sessions)

    async def user_exists(self, user_id: int) -> bool:
        return await self.access.user_exists(user_id)

    async def close(self) -> None:
        if self.engine is not None:
            await self.engine.dispose()

    async def execute(self, principal: Principal, name: str, arguments: dict) -> dict:
        try:
            if name == "search_knowledge":
                return await self.search(
                    principal, SearchArguments.model_validate(arguments)
                )
            if name == "read_knowledge":
                return await self.read(
                    principal, ReadArguments.model_validate(arguments)
                )
            if name == "inspect_knowledge":
                return await self.inspect(
                    principal, InspectArguments.model_validate(arguments)
                )
            raise GateError("NOT_ALLOWED")
        except KnowledgePerceptionError as error:
            code = {
                "DOCUMENT_NOT_AVAILABLE": "RESOURCE_UNAVAILABLE",
                "KNOWLEDGE_READ_SCOPE_TOO_LARGE": "READ_SCOPE_TOO_LARGE",
                "LOCATION_NOT_FOUND": "LOCATION_NOT_FOUND",
                "INVALID_LOCATOR": "INVALID_ARGUMENTS",
            }.get(error.code, "READ_UNAVAILABLE")
            raise GateError(code) from None
        except KnowledgeSearchError as error:
            code = {
                "KNOWLEDGE_SCOPE_UNAVAILABLE": "RESOURCE_UNAVAILABLE",
                "KNOWLEDGE_INDEX_NOT_READY": "INDEX_NOT_READY",
                "KNOWLEDGE_SEARCH_CHANGED": "RESOURCE_CHANGED",
            }.get(error.code, "SEARCH_UNAVAILABLE")
            raise GateError(code) from None

    async def search(self, principal: Principal, args: SearchArguments) -> dict:
        await self.access.kb(principal, args.kb_scope)
        result = await self.retrieval.search(
            user_id=principal.user_id,
            knowledge_base_id=args.kb_scope,
            query=args.query,
            top_k=args.top_k,
        )
        if len(result.hits) > args.top_k:
            raise GateError("SEARCH_UNAVAILABLE")
        locators = await self.access.hits(principal, args.kb_scope, result.hits)
        items = [
            {
                "title": bounded_title(hit.source_display_name),
                "snippet": hit.content[:512],
                "snippet_truncated": len(hit.content) > 512,
                "locator": loc.model_dump(),
                "source_regions": encoded_regions(hit.source_regions),
                "coverage": "search_hit_regions",
            }
            for hit, loc in zip(result.hits, locators)
        ]
        return {
            "items": items,
            "returned_count": len(items),
            "truncated": any(i["snippet_truncated"] for i in items),
            "has_more": None,
        }

    async def read(self, principal: Principal, args: ReadArguments) -> dict:
        loc = args.locator
        before = await self.access.document(principal, loc)
        result = await perception.read_knowledge(
            self.sessions,
            self.storage,
            user_id=principal.user_id,
            knowledge_base_id=loc.kb_id,
            locator=loc.native(),
            max_content_bytes=args.max_bytes,
        )
        if (
            result.document_version_id != loc.document_version_id
            or result.source_sha256 != before.source_sha256
            or result.locator != loc.native()
        ):
            raise GateError("RESOURCE_CHANGED")
        after = await self.access.document(principal, loc)
        if after != before:
            raise GateError("RESOURCE_CHANGED")
        content_bytes = result.content.encode("utf-8")
        if not content_bytes or len(content_bytes) > args.max_bytes:
            raise GateError("READ_SCOPE_TOO_LARGE")
        actual = encoded_regions(result.source_regions)
        markdown = before.media == "text/markdown"
        citation = {
            "title": bounded_title(result.source_display_name),
            "locator": loc.model_dump(),
            "source_regions": actual,
            "evidence_sha256": sha256(content_bytes).hexdigest(),
            "coverage": "exact_source_span" if markdown else "published_regions",
        }
        return {
            "content": result.content,
            "content_kind": "markdown_source" if markdown else "pdf_extracted_text",
            "locator": loc.model_dump(),
            "citation": citation,
            "actual_source_regions": actual,
            "byte_count": len(content_bytes),
            "truncated": False,
        }

    async def inspect(self, principal: Principal, args: InspectArguments) -> dict:
        loc = args.locator
        before = await self.access.document(principal, loc)
        result = await perception.inspect_knowledge(
            self.sessions,
            self.storage,
            user_id=principal.user_id,
            knowledge_base_id=loc.kb_id,
            document_id=loc.document_id,
        )
        if await self.access.document(principal, loc) != before:
            raise GateError("RESOURCE_CHANGED")
        raw = result.get("headings", result.get("detected_regions", []))
        if not isinstance(raw, list) or len(raw) > 1000:
            raise GateError("RESOURCE_TOO_LARGE")
        entries = []
        for item in raw[args.offset : args.offset + args.limit]:
            label = item.get("label", "Published PDF region")
            if not isinstance(label, str) or len(label) > 1024:
                raise GateError("RESPONSE_TOO_LARGE")
            native = item.get("locator")
            address = None
            reason: str | None = "EMPTY_OR_AMBIGUOUS_HEADING"
            if native is not None:
                try:
                    if native.get("document_id") != loc.document_id:
                        raise GateError("RESOURCE_CHANGED")
                    address = Locator.model_validate(
                        {
                            **loc.model_dump(),
                            "location": {
                                k: v for k, v in native.items() if k != "document_id"
                            },
                        }
                    ).model_dump()
                    reason = None
                except ValidationError:
                    reason = "LOCATION_TOO_COMPLEX"
            entries.append(
                {
                    "label": label,
                    "readable": address is not None,
                    "locator": address,
                    "reason": reason,
                }
            )
        data = {
            "title": bounded_title(str(result["label"])),
            "locator": loc.model_dump(),
            "format": result["format"],
            "entries": entries,
            "offset": args.offset,
            "next_offset": None,
            "truncated": False,
        }
        while True:
            next_offset = args.offset + len(entries)
            data["next_offset"] = next_offset if next_offset < len(raw) else None
            data["truncated"] = data["next_offset"] is not None
            try:
                # Reserve room for SDK metadata before the final actual wire check.
                result_model = success("0" * 32, data)
                if len(result_model.model_dump_json(by_alias=True).encode()) > 31000:
                    raise GateError("RESPONSE_TOO_LARGE")
                return data
            except GateError:
                if len(entries) <= 1:
                    raise
                entries.pop()


def create_backend() -> KnowledgeAdapter:
    settings = Settings()
    if not settings.database_url:
        raise ValueError("MCP_DATABASE_REQUIRED")
    # This dedicated local process must never fetch a model on first search.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    engine = create_database_engine(settings.database_url)
    sessions = create_session_factory(engine)
    embedding = KnowledgeEmbeddingRuntime(device=settings.knowledge_embedding_device)
    runtime = KnowledgeIndexBuildRuntime(
        sessions, settings, embedding_runtime=embedding
    )
    return KnowledgeAdapter(
        sessions,
        LocalFileStorage(settings.knowledge_storage_root),
        KnowledgeRetrievalService(sessions, runtime),
        engine,
    )
