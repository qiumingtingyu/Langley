"""Deterministic admission, strict contract and actual-work capacity tests."""

import asyncio
import hashlib
import secrets
import threading

import pytest
from mcp_types import CLIENT_CAPABILITIES_META_KEY, PROTOCOL_VERSION_META_KEY
from pydantic import ValidationError
from starlette.testclient import TestClient

from langley.mcp.config import MCPConfig
from langley.mcp.contracts import GateError, ReadArguments, SearchArguments
from langley.mcp.security import ToolCapacity
from langley.mcp.server import build_app


class FakeBackend:
    def __init__(self):
        self.calls = []

    async def user_exists(self, user_id):
        return user_id in {1, 2}

    async def execute(self, principal, name, arguments):
        self.calls.append((principal, name, arguments))
        if arguments.get("kb_scope") not in principal.allowed_kb_ids:
            raise GateError("RESOURCE_UNAVAILABLE")
        return {"items": [], "returned_count": 0, "truncated": False, "has_more": None}

    async def close(self):
        pass


def setup_gate():
    tokens = [secrets.token_urlsafe(32) for _ in range(3)]
    config = MCPConfig.parse(
        {
            "port": 8766,
            "credentials": [
                {
                    "principal_id": label,
                    "user_id": user,
                    "allowed_kb_ids": [kb],
                    "token_digest": hashlib.sha256(token.encode()).hexdigest(),
                }
                for token, label, user, kb in zip(
                    tokens, ["A", "B", "A2"], [1, 2, 1], [11, 22, 33]
                )
            ],
        }
    )
    backend = FakeBackend()
    return build_app(config, backend), backend, tokens


def call(client, token, arguments=None, name="search_knowledge", headers=None):
    return client.post(
        "/mcp",
        headers={
            "Authorization": "Bearer " + token,
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": "2026-07-28",
            "Mcp-Method": "tools/call",
            "Mcp-Name": name,
            **(headers or {}),
        },
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "_meta": {
                    PROTOCOL_VERSION_META_KEY: "2026-07-28",
                    CLIENT_CAPABILITIES_META_KEY: {},
                },
                "name": name,
                "arguments": arguments or {"query": "marker", "kb_scope": 11},
            },
        },
    )


def test_gate_auth_schema_grants_and_audit_canary(capsys):
    app, backend, tokens = setup_gate()
    with TestClient(app, base_url="http://127.0.0.1:8766") as c:
        assert c.post("/mcp", json={}).status_code == 401
        assert call(c, tokens[0], headers={"Host": "evil.test"}).status_code == 421
        assert call(c, tokens[0], headers={"Origin": "null"}).status_code == 403
        canary = "CANARY_PRIVATE_ARGUMENT_82384"
        bad = call(c, tokens[0], {"query": canary, "kb_scope": True})
        assert canary not in bad.text
        assert (
            bad.json()["result"]["structuredContent"]["error"]["code"]
            == "INVALID_ARGUMENTS"
        )
        unknown = call(c, tokens[0], name="not_allowed_tool")
        assert (
            unknown.json()["result"]["structuredContent"]["error"]["code"]
            == "NOT_ALLOWED"
        )
        assert not backend.calls
        good = call(c, tokens[0])
        assert good.json()["result"]["structuredContent"]["ok"] is True
        denied = call(c, tokens[2])
        assert (
            denied.json()["result"]["structuredContent"]["error"]["code"]
            == "RESOURCE_UNAVAILABLE"
        )
        assert backend.calls[-1][0].allowed_kb_ids == frozenset({33})
    logs = capsys.readouterr().out
    assert canary not in logs and all(token not in logs for token in tokens)
    assert "INVALID_ARGUMENTS" in logs and "RESOURCE_UNAVAILABLE" in logs


@pytest.mark.parametrize("value", [True, 1.0, "1", 0, 9007199254740992])
def test_strict_identifiers(value):
    with pytest.raises(ValidationError):
        SearchArguments.model_validate({"query": "x", "kb_scope": value})


def test_locator_no_arbitrary_paths_or_oversized_pages():
    locator = {
        "v": 1,
        "kb_id": 1,
        "document_id": 1,
        "document_version_id": 1,
        "chunk_revision": 0,
        "location": {"kind": "pages", "page_start": 1, "page_end": 11},
    }
    with pytest.raises(ValidationError):
        ReadArguments.model_validate({"locator": locator})
    locator["location"] = {"kind": "document", "path": "../../private"}
    with pytest.raises(ValidationError):
        ReadArguments.model_validate({"locator": locator})


def test_timeout_and_cancellation_keep_actual_thread_capacity():
    async def scenario():
        capacity = ToolCapacity()
        capacity.search_timeout = 0.03
        release = threading.Event()
        entered = threading.Event()

        def slow():
            entered.set()
            release.wait(3)

        async def work():
            await asyncio.to_thread(slow)

        try:
            with pytest.raises(GateError, match="TOOL_TIMEOUT"):
                await capacity.run("A", True, work)
            assert entered.is_set() and len(capacity.tasks) == 1
            with pytest.raises(GateError, match="SERVER_BUSY"):
                await capacity.run("B", True, work)
        finally:
            release.set()
            await capacity.drain()
        assert not capacity.tasks and capacity.searches == 0
        release.clear()
        waiting = asyncio.create_task(capacity.run("A", False, work))
        await asyncio.sleep(0.02)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        assert len(capacity.tasks) == 1
        release.set()
        await capacity.drain()
        assert not capacity.tasks

    asyncio.run(scenario())


def test_http_actual_bytes_duplicate_credentials_and_rate():
    app, backend, tokens = setup_gate()
    with TestClient(app, base_url="http://127.0.0.1:8766") as c:
        duplicate = c.post(
            "/mcp",
            headers=[
                ("Authorization", "Bearer " + tokens[0]),
                ("Authorization", "Bearer " + tokens[1]),
            ],
            content=b"{}",
        )
        assert duplicate.status_code == 400
        too_large = c.post(
            "/mcp",
            headers={
                "Authorization": "Bearer " + tokens[0],
                "Content-Type": "application/json",
            },
            content=iter([b"x" * 9000, b"x" * 9000]),
        )
        assert too_large.status_code == 413
        assert not backend.calls
        for _ in range(5):
            assert call(c, tokens[0]).json()["result"]["structuredContent"]["ok"]
        limited = call(c, tokens[0]).json()["result"]["structuredContent"]["error"]
        assert limited["code"] == "RATE_LIMITED" and limited["retry_after_ms"] > 0
        statuses = [
            c.get("/mcp", headers={"Authorization": "Bearer " + tokens[0]}).status_code
            for _ in range(20)
        ]
        assert 429 in statuses
        assert len(backend.calls) == 5


def test_output_cap_and_backend_error_do_not_leak(capsys, caplog):
    app, backend, tokens = setup_gate()
    canary = "CANARY_EXCEPTION_PRIVATE_583713"

    async def exploding(*args):
        raise RuntimeError(canary)

    backend.execute = exploding
    with TestClient(app, base_url="http://127.0.0.1:8766") as c:
        failed = call(c, tokens[0])
        assert (
            failed.json()["result"]["structuredContent"]["error"]["code"]
            == "INTERNAL_ERROR"
        )
        assert canary not in failed.text

        # JSON escaping creates more wire bytes than Unicode character count.
        async def huge(*args):
            return {"content": "\x00" * 10000}

        backend.execute = huge
        oversized = call(c, tokens[0])
        assert (
            oversized.json()["result"]["structuredContent"]["error"]["code"]
            == "RESPONSE_TOO_LARGE"
        )
        assert len(oversized.content) < 32768
    assert canary not in capsys.readouterr().out + caplog.text


def test_missing_server_owned_context_fails_closed():
    from types import SimpleNamespace

    from langley.mcp.security import request_state

    for request in [None, SimpleNamespace(scope={"user_id": 1, "principal": "A"})]:
        with pytest.raises(GateError, match="NOT_ALLOWED"):
            request_state(SimpleNamespace(request=request))


def test_multiple_tokens_for_principal_share_bucket():
    from langley.mcp.security import Bucket

    # Same principal keys, not credential digests, own the rate state.
    first, second = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    cfg = MCPConfig.parse(
        {
            "credentials": [
                {
                    "principal_id": "shared",
                    "user_id": 1,
                    "allowed_kb_ids": [11],
                    "token_digest": hashlib.sha256(t.encode()).hexdigest(),
                }
                for t in [first, second]
            ]
        }
    )
    app = build_app(cfg, FakeBackend())
    app.buckets["shared"] = Bucket(1, 1)
    with TestClient(app, base_url="http://127.0.0.1:8766") as c:
        assert (
            c.get("/mcp", headers={"Authorization": "Bearer " + first}).status_code
            == 405
        )
        assert (
            c.get("/mcp", headers={"Authorization": "Bearer " + second}).status_code
            == 429
        )
