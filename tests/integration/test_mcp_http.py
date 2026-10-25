"""Opt-in real loopback SDK + MySQL + cached BGE + disposable Qdrant acceptance."""

import asyncio
import hashlib
import json
import os
import secrets
import socket
import subprocess
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx2
import pytest
import uvicorn
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from sqlalchemy import select, update
from test_mcp_knowledge import fixture

from langley.infrastructure.models import DocumentVersion, KnowledgeBase, KnowledgeChunk
from langley.knowledge.document_index_contract import build_source_context_v1
from langley.knowledge.embedding_runtime import KnowledgeEmbeddingRuntime
from langley.knowledge.index_build import COLLECTION_NAME, KnowledgeIndexBuildRuntime
from langley.knowledge.retrieval_service import KnowledgeRetrievalService
from langley.mcp.audit import silence_payload_logs
from langley.mcp.config import MCPConfig
from langley.mcp.server import build_app
from langley.settings import Settings


@asynccontextmanager
async def running_app(adapter, credentials):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    config = MCPConfig.parse({"port": port, "credentials": credentials})
    app = build_app(config, adapter)
    silence_payload_logs()
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_config=None,
            access_log=False,
            proxy_headers=False,
        )
    )
    task = asyncio.create_task(server.serve(sockets=[sock]))
    for _ in range(250):
        if server.started or task.done():
            break
        await asyncio.sleep(0.02)
    assert server.started
    try:
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        server.should_exit = True
        await task
        sock.close()


@asynccontextmanager
async def client(url, token, mode, wire):
    async def record(request):
        if request.method == "POST":
            wire.append({"request": json.loads(request.content)})

    async with httpx2.AsyncClient(
        headers={"Authorization": "Bearer " + token},
        event_hooks={"request": [record]},
        trust_env=False,
    ) as http:
        async with Client(
            streamable_http_client(url, http_client=http), mode=mode
        ) as c:
            yield c


def checked(result, wire):
    payload = result.model_dump(mode="json", by_alias=True, exclude_none=True)
    wire.append({"response": payload})
    assert len(json.dumps(payload, ensure_ascii=False).encode()) <= 32768
    return result.structured_content


@pytest.mark.skipif(
    os.environ.get("LANGLEY_MCP_REAL_ACCEPTANCE") != "1",
    reason="explicit real-local gate",
)
def test_real_standard_client_chain(test_database_url, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "1")
    evidence = Path(os.environ["LANGLEY_MCP_EVIDENCE_DIR"]).resolve()
    evidence.mkdir(parents=True, exist_ok=True)
    container = "langley-mcp-p0-" + uuid.uuid4().hex[:12]
    container_id = None
    wire = []
    tokens = [secrets.token_urlsafe(32) for _ in range(4)]
    try:
        # No existing container/volume is reused or deleted. Never pull an image.
        container_id = subprocess.check_output(
            [
                "docker",
                "run",
                "-d",
                "--pull=never",
                "--name",
                container,
                "--label",
                "langley.mcp-p0=isolated-acceptance",
                "--cpus",
                "1",
                "--memory",
                "1g",
                "-p",
                "127.0.0.1::6333",
                "--tmpfs",
                "/qdrant/storage",
                "qdrant/qdrant:v1.19.0",
            ],
            text=True,
        ).strip()
        info = json.loads(subprocess.check_output(["docker", "inspect", container_id]))[
            0
        ]
        port = info["NetworkSettings"]["Ports"]["6333/tcp"][0]["HostPort"]

        async def scenario():
            from qdrant_client import AsyncQdrantClient
            from qdrant_client.http import models as qm

            qdrant = AsyncQdrantClient(url="http://127.0.0.1:" + port)
            try:
                for attempt in range(30):
                    try:
                        await qdrant.get_collections()
                        break
                    except Exception:
                        if attempt == 29:
                            raise
                        await asyncio.sleep(0.3)
                async with fixture(test_database_url, tmp_path / "sources") as f:
                    settings = Settings(
                        database_url=test_database_url,
                        qdrant_url="http://127.0.0.1:" + port,
                        tracing_enabled=False,
                    )
                    embedding = KnowledgeEmbeddingRuntime(
                        device=settings.knowledge_embedding_device
                    )
                    async with f["sessions"]() as s:
                        rows = (
                            await s.scalars(
                                select(KnowledgeChunk).where(
                                    KnowledgeChunk.id.in_(
                                        [f["chunks"][0], f["chunks"][1]]
                                    )
                                )
                            )
                        ).all()
                    vectors = await asyncio.to_thread(
                        embedding.encode_documents,
                        [
                            build_source_context_v1(c.content, c.heading_path)
                            for c in rows
                        ],
                        model=settings.knowledge_embedding_model,
                        revision=settings.knowledge_embedding_revision,
                        dimension=settings.knowledge_embedding_dimension,
                    )
                    await qdrant.create_collection(
                        COLLECTION_NAME,
                        vectors_config=qm.VectorParams(
                            size=1024, distance=qm.Distance.COSINE
                        ),
                    )
                    points = []
                    for chunk, vector in zip(rows, vectors):
                        index = 0 if chunk.id == f["chunks"][0] else 1
                        points.append(
                            qm.PointStruct(
                                id=chunk.id,
                                vector=vector,
                                payload={
                                    "knowledge_chunk_id": chunk.id,
                                    "user_id": f["users"][index],
                                    "knowledge_base_id": f["kbs"][index],
                                },
                            )
                        )
                    await qdrant.upsert(COLLECTION_NAME, points=points, wait=True)
                    async with f["sessions"]() as s, s.begin():
                        await s.execute(
                            update(KnowledgeBase)
                            .where(KnowledgeBase.id.in_(f["kbs"][:2]))
                            .values(
                                index_status="READY",
                                active_embedding_model=settings.knowledge_embedding_model,
                                active_embedding_revision=settings.knowledge_embedding_revision,
                                active_embedding_dimension=1024,
                                active_embedding_representation="source_context_v1",
                            )
                        )
                    runtime = KnowledgeIndexBuildRuntime(
                        f["sessions"], settings, embedding_runtime=embedding
                    )
                    adapter = f["adapter"]
                    adapter.retrieval = KnowledgeRetrievalService(
                        f["sessions"], runtime
                    )
                    # Existing Web router and real retrieval, isolated from Web jobs.
                    import httpx
                    from fastapi import FastAPI

                    from langley.api import dependencies
                    from langley.api.knowledge import router

                    web = FastAPI()
                    web.include_router(router)
                    web.dependency_overrides[dependencies.get_session_factory] = (
                        lambda: f["sessions"]
                    )
                    web.dependency_overrides[dependencies.get_current_user_id] = (
                        lambda: f["users"][0]
                    )
                    web.dependency_overrides[
                        dependencies.get_knowledge_index_runtime
                    ] = lambda: runtime
                    async with httpx.AsyncClient(
                        transport=httpx.ASGITransport(app=web), base_url="http://test"
                    ) as web_client:
                        for kb, expected in [(f["kbs"][0], 200), (f["kbs"][1], 404)]:
                            response = await web_client.post(
                                f"/api/knowledge-bases/{kb}/retrieval",
                                json={"query": "marker", "top_k": 1},
                            )
                            assert response.status_code == expected
                            if expected == 200:
                                assert (
                                    response.json()["hits"][0]["content"] == f["source"]
                                )
                            wire.append({"web_regression_status": expected})
                    credentials = [
                        {
                            "principal_id": label,
                            "user_id": user,
                            "allowed_kb_ids": [kb],
                            "token_digest": hashlib.sha256(token.encode()).hexdigest(),
                        }
                        for token, label, user, kb in zip(
                            tokens,
                            ["A", "B", "A2", "C"],
                            [
                                f["users"][0],
                                f["users"][1],
                                f["users"][0],
                                f["users"][0],
                            ],
                            [f["kbs"][0], f["kbs"][1], f["kbs"][2], f["kbs"][0]],
                        )
                    ]
                    for mode in ["auto", "legacy"]:
                        async with running_app(adapter, credentials) as url:
                            async with (
                                client(url, tokens[0], mode, wire) as a,
                                client(url, tokens[1], mode, wire) as b,
                                client(url, tokens[2], mode, wire) as a2,
                            ):
                                listed = await a.list_tools()
                                assert {t.name for t in listed.tools} == {
                                    "search_knowledge",
                                    "read_knowledge",
                                    "inspect_knowledge",
                                }
                                found = checked(
                                    await a.call_tool(
                                        "search_knowledge",
                                        {
                                            "query": "真实来源 marker",
                                            "kb_scope": f["kbs"][0],
                                        },
                                    ),
                                    wire,
                                )
                                assert found["ok"] and len(found["data"]["items"]) == 1
                                loc = found["data"]["items"][0]["locator"]
                                ra, rb = await asyncio.gather(
                                    a.call_tool("read_knowledge", {"locator": loc}),
                                    b.call_tool(
                                        "read_knowledge", {"locator": f["locator"](1)}
                                    ),
                                )
                                actual = checked(ra, wire)
                                assert actual["data"]["content"] == f["source"]
                                assert actual["data"]["citation"]["locator"] == loc
                                assert (
                                    actual["data"]["citation"]["evidence_sha256"]
                                    == hashlib.sha256(f["source"].encode()).hexdigest()
                                )
                                assert (
                                    checked(rb, wire)["data"]["content"]
                                    == "# Private B\nB data\n"
                                )
                                for c in [b, a2]:
                                    for tool, args in [
                                        (
                                            "search_knowledge",
                                            {"query": "x", "kb_scope": f["kbs"][0]},
                                        ),
                                        ("read_knowledge", {"locator": loc}),
                                    ]:
                                        denied = checked(
                                            await c.call_tool(tool, args), wire
                                        )
                                        assert (
                                            denied["error"]["code"]
                                            == "RESOURCE_UNAVAILABLE"
                                        )
                                oversized = checked(
                                    await a.call_tool(
                                        "read_knowledge", {"locator": f["locator"](2)}
                                    ),
                                    wire,
                                )
                                assert (
                                    oversized["error"]["code"] == "READ_SCOPE_TOO_LARGE"
                                )
                                structure = checked(
                                    await a.call_tool(
                                        "inspect_knowledge",
                                        {"locator": f["locator"](2), "limit": 1},
                                    ),
                                    wire,
                                )
                                narrow = checked(
                                    await a.call_tool(
                                        "read_knowledge",
                                        {
                                            "locator": structure["data"]["entries"][0][
                                                "locator"
                                            ]
                                        },
                                    ),
                                    wire,
                                )
                                assert (
                                    narrow["data"]["content"]
                                    == "# Small\nsmall section\n"
                                )
                                limited = checked(
                                    await a.call_tool(
                                        "read_knowledge", {"locator": loc}
                                    ),
                                    wire,
                                )
                                assert limited["error"]["code"] == "RATE_LIMITED"
                            async with client(url, tokens[3], mode, wire) as c:
                                unknown = checked(
                                    await c.call_tool("missing_tool", {}), wire
                                )
                                assert unknown["error"]["code"] == "NOT_ALLOWED"
                                pdf_loc = f["locator"](3)
                                pdf_loc["location"] = {
                                    "kind": "pages",
                                    "page_start": 3,
                                    "page_end": 3,
                                }
                                pdf = checked(
                                    await c.call_tool(
                                        "read_knowledge", {"locator": pdf_loc}
                                    ),
                                    wire,
                                )
                                assert pdf["data"]["actual_source_regions"] == [
                                    {"kind": "pdf_page", "page_start": 2, "page_end": 4}
                                ]
                                assert (
                                    pdf["data"]["citation"]["coverage"]
                                    == "published_regions"
                                )
                                async with f["sessions"]() as s, s.begin():
                                    await s.execute(
                                        update(DocumentVersion)
                                        .where(DocumentVersion.id == f["versions"][3])
                                        .values(chunk_revision=2)
                                    )
                                stale = checked(
                                    await c.call_tool(
                                        "read_knowledge", {"locator": pdf_loc}
                                    ),
                                    wire,
                                )
                                assert stale["error"]["code"] == "RESOURCE_CHANGED"
                                async with f["sessions"]() as s, s.begin():
                                    await s.execute(
                                        update(DocumentVersion)
                                        .where(DocumentVersion.id == f["versions"][3])
                                        .values(chunk_revision=1)
                                    )
                            async with httpx2.AsyncClient(trust_env=False) as http:
                                for headers, expected in [
                                    ({}, 401),
                                    (
                                        {
                                            "Authorization": "Bearer " + tokens[3],
                                            "Host": "evil.test",
                                        },
                                        421,
                                    ),
                                    (
                                        {
                                            "Authorization": "Bearer " + tokens[3],
                                            "Origin": "null",
                                        },
                                        403,
                                    ),
                                ]:
                                    response = await http.post(
                                        url, headers=headers, json={}
                                    )
                                    http_status = response.status_code
                                    wire.append(
                                        {
                                            "negative_http_status": http_status,
                                            "body": response.json(),
                                        }
                                    )
                                    assert response.status_code == expected
                                oversized = await http.post(
                                    url,
                                    headers={"Authorization": "Bearer " + tokens[3]},
                                    content=b"x" * 17000,
                                )
                                assert oversized.status_code == 413
                                wire.append({"oversized_http_status": 413})
                    summary = {
                        "outcome": "PASS",
                        "modern": "2026-07-28",
                        "legacy": "2025-11-25",
                        "model": settings.knowledge_embedding_model,
                        "revision": settings.knowledge_embedding_revision,
                        "device": settings.knowledge_embedding_device,
                        "embedding": "REAL_LOCAL",
                        "qdrant_container_id": container_id,
                        "mysql": "langley_test",
                        "source_sha256": hashlib.sha256(
                            f["source"].encode()
                        ).hexdigest(),
                    }
                    (evidence / "wire-summary.json").write_text(
                        json.dumps(summary, indent=2)
                    )
            finally:
                await qdrant.close()

        asyncio.run(scenario())
    finally:
        (evidence / "wire-requests-responses.json").write_text(
            json.dumps(wire, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if container_id:
            # Verify ID and label before deleting our disposable container.
            info = json.loads(
                subprocess.check_output(["docker", "inspect", container_id])
            )[0]
            assert info["Id"] == container_id and info["Name"] == "/" + container
            assert (
                info["Config"]["Labels"].get("langley.mcp-p0") == "isolated-acceptance"
            )
            subprocess.run(
                ["docker", "rm", "-f", container_id], check=True, capture_output=True
            )
        captured = capsys.readouterr()
        assert all(token not in captured.out + captured.err for token in tokens)
        (evidence / "audit.log").write_text(captured.out, encoding="utf-8")
