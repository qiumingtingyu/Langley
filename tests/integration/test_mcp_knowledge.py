"""Real MySQL adapter checks with additive fixtures; NEVER reset or migrate."""

import asyncio
import uuid
from contextlib import asynccontextmanager
from hashlib import sha256
from pathlib import Path

import pytest
from sqlalchemy import delete, update

from langley.business_time import utc_now
from langley.infrastructure.database import (
    create_database_engine,
    create_session_factory,
)
from langley.infrastructure.local_file_storage import LocalFileStorage
from langley.infrastructure.models import (
    Document,
    DocumentVersion,
    KnowledgeBase,
    KnowledgeChunk,
    User,
)
from langley.knowledge.retrieval import RetrievalHit, RetrievalResult
from langley.mcp.config import Principal
from langley.mcp.contracts import GateError
from langley.mcp.tools import KnowledgeAdapter


class FixtureRetrieval:
    def __init__(self, hit, kb):
        self.hit, self.kb = hit, kb

    async def search(self, **kwargs):
        return RetrievalResult(self.kb, (self.hit,))


def synthetic_pdf() -> bytes:
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [4 0 R 6 0 R 8 0 R 10 0 R] /Count 4 >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for page in range(4):
        objects.append(
            (
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                "/Resources << /Font << /F1 3 0 R >> >> "
                f"/Contents {5 + 2 * page} 0 R >>"
            ).encode()
        )
        body = (
            (
                b"BT /F1 12 Tf 72 720 Td "
                b"(Published PDF extraction for pages 2 to 4.) Tj ET"
            )
            if page == 1
            else b""
        )
        objects.append(
            f"<< /Length {len(body)} >>\nstream\n".encode() + body + b"\nendstream"
        )
    data = b"%PDF-1.4\n"
    offsets = []
    for number, obj in enumerate(objects, 1):
        offsets.append(len(data))
        data += f"{number} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(data)
    data += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    data += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    return (
        data
        + (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n"
        ).encode()
    )


@asynccontextmanager
async def fixture(database_url: str, path: Path):
    from sqlalchemy.engine import make_url

    assert make_url(database_url).database == "langley_test"
    engine = create_database_engine(database_url)
    sessions = create_session_factory(engine)
    storage = LocalFileStorage(path)
    users, kbs, docs, versions, chunks = [], [], [], [], []
    marker = "mcp-fixture-" + uuid.uuid4().hex
    source = "# Exact\n真实来源 UTF-8 marker.\n"
    big = "# Small\nsmall section\n# Large\n" + ("large body\n" * 3000)
    try:
        async with sessions() as s, s.begin():
            for _ in range(2):
                u = User(created_at=utc_now(), auto_memory_enabled=False)
                s.add(u)
                await s.flush()
                users.append(u.id)
            for owner in [users[0], users[1], users[0]]:
                kb = KnowledgeBase(user_id=owner, name=marker, created_at=utc_now())
                s.add(kb)
                await s.flush()
                kbs.append(kb.id)
        for index, (kb, content, media) in enumerate(
            [
                (kbs[0], source, "text/markdown"),
                (kbs[1], "# Private B\nB data\n", "text/markdown"),
                (kbs[0], big, "text/markdown"),
                (
                    kbs[0],
                    "%PDF-1.4\n% controlled published fixture\n",
                    "application/pdf",
                ),
            ]
        ):
            stored = await storage.store_source(
                users[1] if index == 1 else users[0],
                synthetic_pdf() if index == 3 else content.encode(),
            )
            async with sessions() as s, s.begin():
                doc = Document(
                    knowledge_base_id=kb, name=marker + str(index), created_at=utc_now()
                )
                s.add(doc)
                await s.flush()
                docs.append(doc.id)
                version = DocumentVersion(
                    document_id=doc.id,
                    source_filename="synthetic.md" if index != 3 else "synthetic.pdf",
                    source_media_type=media,
                    source_sha256=stored.sha256,
                    source_size_bytes=stored.size_bytes,
                    storage_key=stored.storage_key,
                    chunk_revision=1,
                    chunk_set_sha256=sha256(content.encode()).hexdigest(),
                    indexed_chunk_revision=1,
                    created_at=utc_now(),
                )
                s.add(version)
                await s.flush()
                versions.append(version.id)
                chunk = KnowledgeChunk(
                    document_version_id=version.id,
                    ordinal=1,
                    content=content
                    if index != 3
                    else "Published PDF extraction for pages 2 to 4.",
                    heading_path=["Exact"] if index == 0 else [],
                    source_regions=[
                        {"kind": "pdf_page", "page_start": 2, "page_end": 4}
                    ]
                    if index == 3
                    else [
                        {
                            "kind": "text_span",
                            "start_byte": 0,
                            "end_byte": len(content.encode()),
                        }
                    ],
                    created_at=utc_now(),
                )
                s.add(chunk)
                await s.flush()
                chunks.append(chunk.id)
        hit = RetrievalHit(
            chunks[0],
            1,
            1,
            0.9,
            None,
            1,
            source,
            ("Exact",),
            ({"kind": "text_span", "start_byte": 0, "end_byte": len(source.encode())},),
            docs[0],
            versions[0],
            marker + "0",
            sha256(source.encode()).hexdigest(),
        )
        adapter = KnowledgeAdapter(sessions, storage, FixtureRetrieval(hit, kbs[0]))

        def locator(index=0):
            return {
                "v": 1,
                "kb_id": kbs[1] if index == 1 else kbs[0],
                "document_id": docs[index],
                "document_version_id": versions[index],
                "chunk_revision": 1,
                "location": {"kind": "document"},
            }

        yield {
            "adapter": adapter,
            "sessions": sessions,
            "users": users,
            "kbs": kbs,
            "docs": docs,
            "versions": versions,
            "chunks": chunks,
            "source": source,
            "locator": locator,
            "principal": Principal("A", users[0], frozenset({kbs[0]})),
            "other": Principal("B", users[1], frozenset({kbs[1]})),
        }
    finally:
        # Exact IDs created by this fixture only; unrelated rows are never reset.
        async with sessions() as s, s.begin():
            for model, ids in [
                (KnowledgeChunk, chunks),
                (DocumentVersion, versions),
                (Document, docs),
                (KnowledgeBase, kbs),
                (User, users),
            ]:
                if ids:
                    await s.execute(delete(model).where(model.id.in_(ids)))
        await engine.dispose()


def test_real_access_read_inspect_membership_and_revision(
    test_database_url, tmp_path, monkeypatch
):
    async def scenario():
        async with fixture(test_database_url, tmp_path) as f:
            adapter, principal = f["adapter"], f["principal"]
            found = await adapter.execute(
                principal,
                "search_knowledge",
                {"query": "marker", "kb_scope": f["kbs"][0]},
            )
            loc = found["items"][0]["locator"]
            assert loc == f["locator"]()
            read = await adapter.execute(principal, "read_knowledge", {"locator": loc})
            assert read["content"] == f["source"]
            assert (
                read["citation"]["evidence_sha256"]
                == sha256(f["source"].encode()).hexdigest()
            )
            assert read["actual_source_regions"] == [
                {
                    "kind": "text_span",
                    "start_byte": 0,
                    "end_byte": len(f["source"].encode()),
                }
            ]
            for denied in [
                f["other"],
                Principal("A2", f["users"][0], frozenset({f["kbs"][2]})),
            ]:
                for tool, args in [
                    ("search_knowledge", {"query": "x", "kb_scope": f["kbs"][0]}),
                    ("read_knowledge", {"locator": loc}),
                    ("inspect_knowledge", {"locator": loc}),
                ]:
                    with pytest.raises(GateError, match="RESOURCE_UNAVAILABLE"):
                        await adapter.execute(denied, tool, args)
            # Grant alone never overrides DB ownership.
            forged = Principal("forged-grant", f["users"][1], frozenset({f["kbs"][0]}))
            with pytest.raises(GateError, match="RESOURCE_UNAVAILABLE"):
                await adapter.execute(forged, "read_knowledge", {"locator": loc})
            big = f["locator"](2)
            with pytest.raises(GateError, match="READ_SCOPE_TOO_LARGE"):
                await adapter.execute(principal, "read_knowledge", {"locator": big})
            structure = await adapter.execute(
                principal, "inspect_knowledge", {"locator": big, "limit": 1}
            )
            assert structure["next_offset"] == 1
            narrow = await adapter.execute(
                principal,
                "read_knowledge",
                {"locator": structure["entries"][0]["locator"]},
            )
            assert narrow["content"] == "# Small\nsmall section\n"
            pdf = f["locator"](3)
            pdf["location"] = {"kind": "pages", "page_start": 3, "page_end": 3}
            pdf_read = await adapter.execute(
                principal, "read_knowledge", {"locator": pdf}
            )
            assert pdf_read["content_kind"] == "pdf_extracted_text"
            assert pdf_read["actual_source_regions"] == [
                {"kind": "pdf_page", "page_start": 2, "page_end": 4}
            ]
            assert pdf_read["citation"]["coverage"] == "published_regions"
            # Publish revision changes during the service, before adapter postcheck.
            from langley.knowledge import perception

            native = perception.read_knowledge

            async def race(*args, **kwargs):
                result = await native(*args, **kwargs)
                async with f["sessions"]() as s, s.begin():
                    await s.execute(
                        update(DocumentVersion)
                        .where(DocumentVersion.id == f["versions"][0])
                        .values(chunk_revision=2)
                    )
                return result

            monkeypatch.setattr(perception, "read_knowledge", race)
            with pytest.raises(GateError, match="RESOURCE_CHANGED"):
                await adapter.execute(principal, "read_knowledge", {"locator": loc})
            with pytest.raises(GateError, match="RESOURCE_CHANGED"):
                await adapter.execute(principal, "inspect_knowledge", {"locator": loc})
            with pytest.raises(GateError, match="RESOURCE_CHANGED"):
                await adapter.execute(
                    principal,
                    "search_knowledge",
                    {"query": "marker", "kb_scope": f["kbs"][0]},
                )

    asyncio.run(scenario())
