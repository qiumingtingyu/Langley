"""Whole-endpoint authentication and bounded, lifetime-owned tool execution."""

import asyncio
import json
import math
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from mcp.server.context import CallNext, HandlerResult, ServerRequestContext
from pydantic import ValidationError
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from langley.mcp import audit
from langley.mcp.config import MCPConfig, Principal
from langley.mcp.contracts import ARGUMENTS, GateError, failure

SCOPE_KEY = "langley.mcp.request"


@dataclass
class RequestState:
    request_id: str
    principal: Principal
    input_bytes: int
    tool: str | None = None
    decision: str = "ALLOW"
    reason: str = "OK"
    status: str = "SUCCEEDED"


def request_state(ctx: ServerRequestContext) -> RequestState:
    request = ctx.request
    state = request.scope.get(SCOPE_KEY) if request is not None else None
    if not isinstance(state, RequestState):
        raise GateError("NOT_ALLOWED")
    return state


class Bucket:
    def __init__(self, per_minute: int, burst: int):
        self.rate = per_minute / 60
        self.burst = burst
        self.tokens = float(burst)
        self.updated = time.monotonic()

    def take(self) -> None:
        now = time.monotonic()
        self.tokens = min(self.burst, self.tokens + (now - self.updated) * self.rate)
        self.updated = now
        if self.tokens < 1:
            raise GateError(
                "RATE_LIMITED",
                min(60000, math.ceil((1 - self.tokens) / self.rate * 1000)),
            )
        self.tokens -= 1


class ToolCapacity:
    """Never cancel underlying work on client deadline/disconnect.

    Detached work owns its admission until its coroutine, including awaited local
    threads, really completes. No queue, replay or replacement task is created.
    """

    def __init__(self) -> None:
        self.tasks: set[asyncio.Task] = set()
        self.principals: set[str] = set()
        self.searches = 0
        self.search_timeout = 30.0
        self.read_timeout = 10.0

    async def run(
        self, principal: str, search: bool, work: Callable[[], Awaitable[Any]]
    ) -> Any:
        if (
            len(self.tasks) >= 2
            or principal in self.principals
            or (search and self.searches)
        ):
            raise GateError("SERVER_BUSY", 1000)
        self.principals.add(principal)
        self.searches += int(search)

        async def supervised() -> Any:
            return await work()

        task: asyncio.Task[Any] = asyncio.create_task(supervised())
        self.tasks.add(task)

        def finished(done: asyncio.Task) -> None:
            self.tasks.remove(done)
            self.principals.remove(principal)
            self.searches -= int(search)
            if not done.cancelled():
                done.exception()  # Consume late errors without logging payloads.

        task.add_done_callback(finished)
        try:
            return await asyncio.wait_for(
                asyncio.shield(task),
                timeout=self.search_timeout if search else self.read_timeout,
            )
        except TimeoutError:
            raise GateError("TOOL_TIMEOUT") from None

    async def drain(self) -> None:
        if self.tasks:
            await asyncio.gather(*tuple(self.tasks), return_exceptions=True)


class ToolPolicy:
    def __init__(self, config: MCPConfig):
        self.buckets = {p.principal_id: Bucket(30, 5) for _, p in config.credentials}
        self.capacity = ToolCapacity()

    async def __call__(
        self, ctx: ServerRequestContext, call_next: CallNext
    ) -> HandlerResult:
        state = request_state(ctx)
        if ctx.method != "tools/call":
            return await call_next(ctx)
        params = ctx.params or {}
        name = params.get("name")
        state.tool = (
            name if isinstance(name, str) and name in ARGUMENTS else "UNRECOGNIZED"
        )
        try:
            if not isinstance(name, str) or name not in ARGUMENTS:
                raise GateError("NOT_ALLOWED")
            ARGUMENTS[name].model_validate(params.get("arguments", {}))
            self.buckets[state.principal.principal_id].take()
            result = await self.capacity.run(
                state.principal.principal_id,
                name == "search_knowledge",
                lambda: call_next(ctx),
            )
            return result
        except ValidationError:
            error = GateError("INVALID_ARGUMENTS")
        except GateError as caught:
            error = caught
        except asyncio.CancelledError:
            state.reason, state.status = "CANCELLED", "FAILED"
            raise
        except Exception:
            error = GateError("INTERNAL_ERROR")
        state.reason = error.code
        state.decision = "DENY"
        state.status = "REJECTED"
        if error.code in {"INTERNAL_ERROR", "TOOL_TIMEOUT"}:
            state.decision, state.status = "ALLOW", "FAILED"
        return failure(state.request_id, error)


class HTTPGate:
    def __init__(
        self,
        app: ASGIApp,
        config: MCPConfig,
        user_exists: Callable[[int], Awaitable[bool]],
    ):
        self.app, self.config, self.user_exists = app, config, user_exists
        self.buckets = {p.principal_id: Bucket(120, 20) for _, p in config.credentials}
        self.unauthenticated = Bucket(60, 10)
        self.inflight = 0

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = time.monotonic()
        request_id = uuid.uuid4().hex
        state: RequestState | None = None
        body_size = output_size = 0
        decision, reason, status = "DENY", "INTERNAL_ERROR", "REJECTED"
        entered = False

        async def reject(code: str, http_status: int, retry: int | None = None) -> None:
            nonlocal reason, output_size
            reason = code
            data = json.dumps({"error": code, "request_id": request_id}).encode()
            headers = [
                (b"content-type", b"application/json"),
                (b"x-request-id", request_id.encode()),
            ]
            if retry is not None:
                headers.append(
                    (b"retry-after", str(max(1, math.ceil(retry / 1000))).encode())
                )
            await send(
                {
                    "type": "http.response.start",
                    "status": http_status,
                    "headers": headers,
                }
            )
            await send({"type": "http.response.body", "body": data})
            output_size = len(data)

        try:
            if self.inflight >= 8:
                await reject("SERVER_BUSY", 503)
                return
            self.inflight += 1
            entered = True
            headers: dict[bytes, bytes] = {}
            critical = {
                b"authorization",
                b"host",
                b"origin",
                b"content-length",
                b"content-type",
                b"transfer-encoding",
                b"content-encoding",
                b"mcp-protocol-version",
                b"mcp-method",
                b"mcp-name",
                b"mcp-session-id",
            }
            for key, value in scope["headers"]:
                key = key.lower()
                if key in critical and key in headers:
                    await reject("INVALID_HEADERS", 400)
                    return
                headers[key] = value
            if headers.get(b"host", b"").decode("latin1") not in self.config.hosts:
                await reject("HOST_DENIED", 421)
                return
            origin = headers.get(b"origin")
            if origin is not None and origin.decode("latin1") not in [
                "http://" + h for h in self.config.hosts
            ]:
                await reject("ORIGIN_DENIED", 403)
                return
            principal = self.config.authenticate(
                headers.get(b"authorization", b"").decode("latin1")
            )
            if principal is None:
                self.unauthenticated.take()
                await reject("UNAUTHENTICATED", 401)
                return
            state = RequestState(request_id, principal, 0)
            self.buckets[principal.principal_id].take()
            async with asyncio.timeout(5):
                exists = await self.user_exists(principal.user_id)
            if not exists:
                await reject("UNAUTHENTICATED", 401)
                return
            if scope["path"] != "/mcp" or scope.get("query_string"):
                await reject("NOT_FOUND", 404)
                return
            if scope["method"] != "POST":
                await reject("METHOD_NOT_ALLOWED", 405)
                return
            if b"content-encoding" in headers:
                await reject("INVALID_HEADERS", 400)
                return
            length = headers.get(b"content-length")
            if length is not None and (not length.isdigit() or int(length) > 16384):
                await reject("BODY_TOO_LARGE", 413)
                return
            chunks = []
            async with asyncio.timeout(5):
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        reason = "DISCONNECTED"
                        return
                    chunk = message.get("body", b"")
                    body_size += len(chunk)
                    if body_size > 16384:
                        await reject("BODY_TOO_LARGE", 413)
                        return
                    chunks.append(chunk)
                    if not message.get("more_body", False):
                        break
            body = b"".join(chunks)
            state.input_bytes = body_size
            if body.lstrip().startswith(b"["):
                await reject("INVALID_REQUEST", 400)
                return
            delivered = False

            async def replay() -> Message:
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {"type": "http.request", "body": body, "more_body": False}
                return await receive()

            response_start: Message = {}
            response_parts: list[bytes] = []

            async def capture(message: Message) -> None:
                nonlocal output_size, response_start
                if message["type"] == "http.response.start":
                    response_start = message
                elif message["type"] == "http.response.body":
                    output_size += len(message.get("body", b""))
                    if output_size > 65536:
                        raise GateError("RESPONSE_TOO_LARGE")
                    response_parts.append(message.get("body", b""))

            await self.app({**scope, SCOPE_KEY: state}, replay, capture)
            if response_start.get("status", 500) >= 400:
                await reject("PROTOCOL_REJECTED", response_start.get("status", 500))
                return
            data = b"".join(response_parts)
            if data:
                parsed = json.loads(data)
                if "error" in parsed:
                    parsed["error"] = {
                        "code": parsed["error"].get("code", -32603),
                        "message": "MCP request rejected.",
                    }
                    state.decision, state.reason, state.status = (
                        "DENY",
                        "PROTOCOL_REJECTED",
                        "REJECTED",
                    )
                result = parsed.get("result", {})
                if (
                    state.tool is not None
                    and len(json.dumps(result, ensure_ascii=False).encode()) > 32768
                ):
                    parsed["result"] = failure(
                        request_id, GateError("RESPONSE_TOO_LARGE")
                    ).model_dump(mode="json", by_alias=True)
                    state.reason, state.status = "RESPONSE_TOO_LARGE", "FAILED"
                data = json.dumps(
                    parsed, ensure_ascii=False, separators=(",", ":")
                ).encode()
            response_start["headers"] = [
                (k, v)
                for k, v in response_start.get("headers", [])
                if k.lower() not in {b"content-length", b"access-control-allow-origin"}
            ]
            response_start["headers"].extend(
                [
                    (b"content-length", str(len(data)).encode()),
                    (b"x-request-id", request_id.encode()),
                ]
            )
            await send(response_start)
            await send({"type": "http.response.body", "body": data})
            output_size = len(data)
            decision, reason, status = state.decision, state.reason, state.status
        except GateError as error:
            await reject(
                error.code,
                429 if error.code == "RATE_LIMITED" else 503,
                error.retry_after_ms,
            )
        except TimeoutError:
            await reject("REQUEST_TIMEOUT", 408)
        except asyncio.CancelledError:
            reason = "CANCELLED"
            raise
        except Exception:
            await reject("INTERNAL_ERROR", 500)
        finally:
            if entered:
                self.inflight -= 1
            audit.event(
                request_id=request_id,
                principal=state.principal.principal_id if state else None,
                tool=state.tool if state else None,
                decision=decision,
                reason=reason,
                status=status,
                duration_ms=(time.monotonic() - started) * 1000,
                input_bytes=body_size,
                output_bytes=output_size,
            )
