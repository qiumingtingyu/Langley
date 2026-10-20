"""Real loopback proxy/core traces plus fail-closed synthetic boundary cases."""

import asyncio
import json
import socket
from types import SimpleNamespace

import httpx
import pytest
import structlog
from pydantic import SecretStr
from test_provider_connect_retry import collect, provider, response

from langley.answering.errors import RunErrorCode, WorkflowFailure
from langley.infrastructure import openai_compatible_provider as adapter
from langley.infrastructure.deepseek_provider import DeepSeekProvider


async def no_wait(_):
    pass


@pytest.mark.parametrize(
    "kind",
    [
        "absent",
        "partial",
        "reordered",
        "unrelated_failure",
        "wrong_host",
        "wrong_connect_target",
        "malformed",
    ],
)
def test_missing_or_invalid_evidence_is_unknown_and_not_retried(kind):
    async def handler(request):
        trace = request.extensions["trace"]
        error = httpx.ConnectError("unknown phase", request=request)
        if kind == "reordered":
            await trace("connection.connect_tcp.failed", {"exception": error})
        if kind != "absent":
            await trace(
                "connection.connect_tcp.started", {"host": "provider.test", "port": 443}
            )
        if kind in ("unrelated_failure", "wrong_host"):
            await trace("connection.connect_tcp.complete", {"return_value": None})
            await trace(
                "connection.start_tls.started",
                {
                    "server_hostname": "wrong.test"
                    if kind == "wrong_host"
                    else "provider.test"
                },
            )
            await trace(
                "connection.start_tls.failed",
                {
                    "exception": ValueError("unrelated")
                    if kind == "unrelated_failure"
                    else error
                },
            )
        if kind == "malformed":
            await trace("connection.connect_tcp.failed", None)
        if kind == "wrong_connect_target":
            await trace("connection.connect_tcp.complete", {"return_value": None})
            await trace(
                "http11.send_request_headers.started",
                {
                    "request": SimpleNamespace(
                        method=b"CONNECT", url=SimpleNamespace(target=b"wrong.test:443")
                    ),
                },
            )
            await trace("http11.send_request_headers.failed", {"exception": error})
        raise error

    p, clients = provider(handler, evidence=False)
    with (
        structlog.testing.capture_logs() as logs,
        pytest.raises(WorkflowFailure) as exc,
    ):
        asyncio.run(collect(p))
    assert exc.value.error_code is RunErrorCode.LLM_PROVIDER_FAILED
    assert len(clients) == 1
    assert logs[-1]["retry_safety"] == "UNKNOWN"
    assert logs[-1]["retry_reason"] == "INSUFFICIENT_TRANSPORT_EVIDENCE"


@pytest.mark.parametrize("protocol", ["http11", "http2"])
@pytest.mark.parametrize(
    "stage", ["send_request_headers", "send_request_body", "receive_response_headers"]
)
def test_target_post_started_even_with_connect_error_never_retries(protocol, stage):
    async def handler(request):
        await request.extensions["trace"](
            protocol + "." + stage + ".started",
            {"request": SimpleNamespace(method=b"POST")},
        )
        raise httpx.ConnectError("sent", request=request)

    p, clients = provider(handler, evidence=False)
    with structlog.testing.capture_logs() as logs, pytest.raises(WorkflowFailure):
        asyncio.run(collect(p))
    assert len(clients) == 1 and logs[-1]["retry_safety"] == "SENT_OR_RESPONSE_STARTED"


@pytest.mark.parametrize(
    "stage", ["send_request_headers", "send_request_body", "receive_response_headers"]
)
def test_connect_handshake_failure_has_positive_pre_post_evidence(monkeypatch, stage):
    calls = []

    async def handler(request):
        calls.append(request.content)
        if len(calls) > 1:
            return response()
        trace = request.extensions["trace"]
        await trace(
            "connection.connect_tcp.started", {"host": "proxy.test", "port": 8080}
        )
        await trace("connection.connect_tcp.complete", {"return_value": None})
        connect = SimpleNamespace(
            method=b"CONNECT", url=SimpleNamespace(target=b"provider.test:443")
        )
        error = httpx.ConnectError("handshake failure", request=request)
        for name in (
            "send_request_headers",
            "send_request_body",
            "receive_response_headers",
        ):
            await trace("http11." + name + ".started", {"request": connect})
            if name == stage:
                await trace("http11." + name + ".failed", {"exception": error})
                raise error
            await trace("http11." + name + ".complete", {"return_value": None})
        raise AssertionError("unreachable")

    monkeypatch.setattr(adapter.asyncio, "sleep", no_wait)
    p, clients = provider(handler, evidence=False)
    assert asyncio.run(collect(p))[-1].assistant_content == "done"
    assert len(clients) == 2 and calls[0] == calls[1]


@pytest.mark.parametrize("reject", [False, True])
def test_real_http_proxy_connect_then_tls_failure_or_rejection(
    monkeypatch, tmp_path, reject
):
    """No external traffic: proxy closes after receiving the TLS ClientHello."""
    monkeypatch.setattr(adapter.asyncio, "sleep", no_wait)

    async def run():
        methods = []
        hellos = []
        clients = []
        jobs = set()
        observed_traces = []

        async def proxy(reader, writer):
            task = asyncio.current_task()
            jobs.add(task)
            try:
                headers = await reader.readuntil(b"\r\n\r\n")
                methods.append(headers.split(b" ", 1)[0].decode("ascii"))
                if reject:
                    writer.write(
                        b"HTTP/1.1 407 Proxy Authentication Required\r\n"
                        b"Content-Length: 0\r\n\r\n"
                    )
                    await writer.drain()
                else:
                    writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                    await writer.drain()
                    first = await reader.readexactly(5)
                    hellos.append(first[0] == 22)
            finally:
                writer.close()
                await writer.wait_closed()
                jobs.discard(task)

        server = await asyncio.start_server(proxy, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]

        async def record_request(request):
            original = request.extensions["trace"]
            events = []
            observed_traces.append(events)

            async def record(name, info):
                method = getattr(info.get("request"), "method", None)
                events.append(
                    {"name": name, "method": method.decode() if method else None}
                )
                await original(name, info)

            request.extensions["trace"] = record

        def factory():
            client = httpx.AsyncClient(
                proxy=f"http://127.0.0.1:{port}",
                trust_env=False,
                timeout=1.0,
                event_hooks={"request": [record_request]},
            )
            clients.append(client)
            return client

        p = DeepSeekProvider(
            api_key=SecretStr("offline"),
            base_url="https://model.test",
            model="fake",
            client_factory=factory,
            connect_max_retries=2,
        )
        try:
            with (
                structlog.testing.capture_logs() as logs,
                pytest.raises(WorkflowFailure) as exc,
            ):
                async with asyncio.timeout(5):
                    await collect(p)
            assert exc.value.error_code is RunErrorCode.LLM_PROVIDER_FAILED
            assert len(clients) == (1 if reject else 3)
            assert methods == ["CONNECT"] * len(clients)
            assert hellos == ([] if reject else [True] * 3)
            if not reject:
                assert all(x["retry_safety"] == "PROVEN_PRE_SEND" for x in logs)
                assert [x["will_retry"] for x in logs] == [True, True, False]
            evidence = {
                "real_loopback_proxy": True,
                "CONNECT_count": len(methods),
                "TLS_ClientHello_count": len(hellos),
                "target_POST_count": 0,
                "rejected_407": reject,
                "attempt_logs": logs,
                "actual_trace_events": observed_traces,
            }
            (tmp_path / "proxy-evidence.json").write_text(
                json.dumps(evidence, indent=2)
            )
        finally:
            server.close()
            await server.wait_closed()
            if jobs:
                await asyncio.gather(*jobs)

    asyncio.run(run())


def test_real_tcp_refusal_has_complete_connection_failure_trace(monkeypatch):
    monkeypatch.setattr(adapter.asyncio, "sleep", no_wait)
    # Bound but not listening: reserve a private local port without a server race.
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        port = reserved.getsockname()[1]
        clients = []

        def factory():
            c = httpx.AsyncClient(trust_env=False, timeout=0.5)
            clients.append(c)
            return c

        p = DeepSeekProvider(
            api_key=SecretStr("offline"),
            base_url=f"https://127.0.0.1:{port}",
            model="fake",
            client_factory=factory,
            connect_max_retries=2,
        )
        with structlog.testing.capture_logs() as logs, pytest.raises(WorkflowFailure):
            asyncio.run(collect(p))
        assert len(clients) == 3
        assert all(x["retry_safety"] == "PROVEN_PRE_SEND" for x in logs)
