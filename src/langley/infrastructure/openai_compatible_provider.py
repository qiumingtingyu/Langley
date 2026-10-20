"""Shared OpenAI-compatible streaming adapter.

The adapter normalizes only one provider round.  It neither runs an Agent loop
nor validates or executes Tool calls.
"""

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
import structlog
from pydantic import SecretStr

from langley.answering.contracts import (
    AssistantContentDelta,
    AssistantRuntimeMessage,
    LLMFinishReason,
    LLMRequest,
    LLMResponseCompleted,
    LLMStreamEvent,
    LLMUsage,
    ToolCall,
    ToolResult,
    UserRuntimeMessage,
)
from langley.answering.errors import RunErrorCode, WorkflowFailure

HTTPClientFactory = Callable[[], httpx.AsyncClient]
logger = structlog.get_logger(__name__)


@dataclass
class _ConnectEvidence:
    """Bounded evidence for httpcore 1.0.x pre-POST failures, not absent sends.

    CONNECT reuses the target request's trace callback. Only exact known
    prefixes ending in a paired, causally matching failure prove pre-send.
    New/missing/reordered transport events therefore fail closed.
    """

    target: httpx.URL
    events: list[str] = field(default_factory=list)
    target_started: bool = False
    invalid: bool = False
    failure: BaseException | None = None

    def observe(self, name: str, info: dict[str, Any]) -> None:
        if not isinstance(name, str) or not isinstance(info, dict):
            self.invalid = True
            return
        if len(self.events) >= 16:
            self.invalid = True
        else:
            self.events.append(name)
        if name.endswith(
            (
                "send_request_headers.started",
                "send_request_body.started",
                "receive_response_headers.started",
            )
        ):
            request = info.get("request")
            method = getattr(request, "method", None)
            if method == b"POST":
                self.target_started = True
            elif method == b"CONNECT" and name.startswith("http11."):
                authority = (
                    self.target.raw_host + b":" + str(self.target.port or 443).encode()
                )
                self.invalid |= (
                    getattr(getattr(request, "url", None), "target", None) != authority
                )
            else:
                self.invalid = True
        if name.endswith("connect_tcp.started"):
            self.invalid |= (
                not isinstance(info.get("host"), str)
                or type(info.get("port")) is not int
            )
        if name.endswith("start_tls.started"):
            self.invalid |= info.get("server_hostname") != self.target.host
        if name == "http11.receive_response_headers.complete":
            value = info.get("return_value")
            self.invalid |= not (
                isinstance(value, tuple)
                and len(value) == 4
                and type(value[1]) is int
                and 200 <= value[1] < 300
            )
        if name.endswith(".failed"):
            error = info.get("exception")
            self.failure = error if isinstance(error, BaseException) else None

    def classify(self, error: BaseException) -> str:
        if self.target_started:
            return "SENT_OR_RESPONSE_STARTED"
        if self.invalid or self.target.scheme != "https":
            return "UNKNOWN"
        tcp = ("connection.connect_tcp.started", "connection.connect_tcp.complete")
        tunnel = (
            "http11.send_request_headers.started",
            "http11.send_request_headers.complete",
            "http11.send_request_body.started",
            "http11.send_request_body.complete",
            "http11.receive_response_headers.started",
            "http11.receive_response_headers.complete",
        )
        prefixes = [
            (tcp[0], "connection.connect_tcp.failed"),
            tcp + ("connection.start_tls.started", "connection.start_tls.failed"),
            tcp + tunnel + ("proxy.start_tls.started", "proxy.start_tls.failed"),
        ]
        # CONNECT itself may fail, but only a complete prefix before target POST.
        for offset in (0, 2, 4):
            stage = tunnel[offset].removesuffix(".started")
            prefixes.append(
                tcp + tunnel[:offset] + (stage + ".started", stage + ".failed")
            )
        cause: BaseException | None = error
        seen: set[int] = set()
        while cause is not None and id(cause) not in seen:
            if cause is self.failure and tuple(self.events) in prefixes:
                return "PROVEN_PRE_SEND"
            seen.add(id(cause))
            cause = cause.__cause__
        return "UNKNOWN"


@dataclass
class _ToolCallParts:
    """Losslessly collect one provider-emitted Tool argument stream."""

    call_id: str | None = None
    name: str | None = None
    argument_fragments: list[str] = field(default_factory=list)


class OpenAICompatibleProvider:
    """Translate Chat Completions HTTP/SSE into Langley's normalized contract."""

    def __init__(
        self,
        *,
        api_key: SecretStr,
        base_url: str,
        model: str,
        client_factory: HTTPClientFactory | None = None,
        connect_max_retries: int = 0,
    ) -> None:
        if type(connect_max_retries) is not int or not 0 <= connect_max_retries <= 2:
            raise ValueError("connect_max_retries must be an integer from 0 to 2")
        self._connect_max_retries = connect_max_retries
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._client_factory = (
            client_factory
            if client_factory is not None
            else lambda: httpx.AsyncClient(timeout=httpx.Timeout(60.0))
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        """Send one streaming Chat Completions request and normalize its SSE data."""

        payload = self._request_payload(request)
        headers = {
            "Authorization": f"Bearer {self._api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        for attempt in range(1, self._connect_max_retries + 2):
            content_parts: list[str] = []
            tool_parts: dict[int, _ToolCallParts] = {}
            finish_reason = LLMFinishReason.UNKNOWN
            usage: LLMUsage | None = None
            provider_model: str | None = None
            saw_done = False

            response_received = False
            evidence = _ConnectEvidence(httpx.URL(f"{self._base_url}/chat/completions"))
            request_hooks_finished = False
            opening_stream = False

            async def response_received_hook(response: httpx.Response) -> None:
                nonlocal response_received
                response_received = True

            async def request_ready_hook(request: httpx.Request) -> None:
                nonlocal request_hooks_finished
                request_hooks_finished = True

            async def transport_trace(name: str, info: dict[str, Any]) -> None:
                evidence.observe(name, info)

            try:
                async with self._client_factory() as client:
                    client.event_hooks["response"].insert(0, response_received_hook)
                    client.event_hooks["request"].append(request_ready_hook)
                    opening_stream = True
                    async with client.stream(
                        "POST",
                        f"{self._base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                        extensions={"trace": transport_trace},
                    ) as response:
                        response_received = True
                        opening_stream = False
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            if not line.startswith("data:"):
                                continue
                            data = line.removeprefix("data:").strip()
                            if data == "[DONE]":
                                saw_done = True
                                break
                            event = self._parse_event(data)
                            event_model = event.get("model")
                            if event_model is not None:
                                if (
                                    not isinstance(event_model, str)
                                    or not event_model.strip()
                                ):
                                    raise WorkflowFailure(
                                        RunErrorCode.LLM_RESPONSE_INVALID
                                    )
                                if (
                                    provider_model is not None
                                    and provider_model != event_model
                                ):
                                    raise WorkflowFailure(
                                        RunErrorCode.LLM_RESPONSE_INVALID
                                    )
                                provider_model = event_model
                            event_usage = event.get("usage")
                            if event_usage is not None:
                                usage = self._normalize_usage(event_usage)
                            choices = event.get("choices")
                            if choices is None:
                                continue
                            if not isinstance(choices, list):
                                raise WorkflowFailure(RunErrorCode.LLM_RESPONSE_INVALID)
                            if not choices:
                                continue
                            if len(choices) != 1:
                                raise WorkflowFailure(RunErrorCode.LLM_RESPONSE_INVALID)
                            choice = choices[0]
                            if not isinstance(choice, dict):
                                raise WorkflowFailure(RunErrorCode.LLM_RESPONSE_INVALID)
                            delta = choice.get("delta")
                            if delta is not None:
                                if not isinstance(delta, dict):
                                    raise WorkflowFailure(
                                        RunErrorCode.LLM_RESPONSE_INVALID
                                    )
                                self._validate_delta(delta)
                                content = delta.get("content")
                                if content is not None:
                                    if not isinstance(content, str):
                                        raise WorkflowFailure(
                                            RunErrorCode.LLM_RESPONSE_INVALID
                                        )
                                    content_parts.append(content)
                                    yield AssistantContentDelta(content=content)
                                self._append_tool_call_fragments(
                                    tool_parts, delta.get("tool_calls")
                                )
                            raw_finish_reason = choice.get("finish_reason")
                            if raw_finish_reason is not None:
                                finish_reason = self._normalize_finish_reason(
                                    raw_finish_reason
                                )
                if not saw_done:
                    raise WorkflowFailure(RunErrorCode.LLM_RESPONSE_INVALID)

                tool_calls = tuple(
                    self._complete_tool_call(parts)
                    for _, parts in sorted(tool_parts.items(), key=lambda item: item[0])
                )
                if len({tool_call.call_id for tool_call in tool_calls}) != len(
                    tool_calls
                ):
                    raise WorkflowFailure(RunErrorCode.LLM_RESPONSE_INVALID)
                if attempt > 1:
                    self._record_attempt(attempt, "RECOVERED", usage=usage)
                yield LLMResponseCompleted(
                    assistant_content="".join(content_parts),
                    tool_calls=tool_calls,
                    finish_reason=finish_reason,
                    usage=usage if attempt == 1 else None,
                    provider_model=provider_model,
                )
                return
            except (httpx.ConnectError, httpx.ConnectTimeout) as error:
                safety = (
                    "SENT_OR_RESPONSE_STARTED"
                    if response_received
                    else evidence.classify(error)
                    if opening_stream and request_hooks_finished
                    else "UNKNOWN"
                )
                retry = (
                    safety == "PROVEN_PRE_SEND" and attempt <= self._connect_max_retries
                )
                self._record_attempt(
                    attempt,
                    "RETRYING" if retry else "FAILED",
                    error_type=type(error).__name__,
                    will_retry=retry,
                    retry_safety=safety,
                    retry_reason=(
                        "SAFE_CONNECT_RETRY"
                        if retry
                        else "RETRY_BUDGET_EXHAUSTED"
                        if safety == "PROVEN_PRE_SEND"
                        else "TARGET_STARTED"
                        if safety == "SENT_OR_RESPONSE_STARTED"
                        else "INSUFFICIENT_TRANSPORT_EVIDENCE"
                    ),
                )
                if not retry:
                    raise WorkflowFailure(RunErrorCode.LLM_PROVIDER_FAILED) from error
                try:
                    await asyncio.sleep(2 ** (attempt - 1))
                except asyncio.CancelledError:
                    self._record_attempt(attempt, "CANCELLED")
                    raise
            except asyncio.CancelledError:
                self._record_attempt(
                    attempt, "CANCELLED", retry_reason="CANCELLED_OR_DEADLINE"
                )
                raise
            except WorkflowFailure as error:
                self._record_attempt(
                    attempt,
                    "FAILED",
                    error_type=error.error_code.value,
                    retry_reason="RESPONSE_VALIDATION_FAILED",
                )
                raise
            except httpx.HTTPStatusError as error:
                self._record_attempt(
                    attempt,
                    "FAILED",
                    error_type=type(error).__name__,
                    retry_safety="SENT_OR_RESPONSE_STARTED",
                    retry_reason="HTTP_STATUS_ERROR",
                )
                error_code = (
                    RunErrorCode.ANSWER_CONTEXT_TOO_LARGE
                    if self._is_context_capacity_rejection(error.response)
                    else RunErrorCode.LLM_PROVIDER_FAILED
                )
                raise WorkflowFailure(error_code) from error
            except httpx.HTTPError as error:
                self._record_attempt(
                    attempt,
                    "FAILED",
                    error_type=type(error).__name__,
                    retry_reason="NON_RETRYABLE_HTTP_ERROR",
                )
                raise WorkflowFailure(RunErrorCode.LLM_PROVIDER_FAILED) from error

    def _record_attempt(
        self,
        attempt: int,
        outcome: str,
        *,
        error_type: str | None = None,
        will_retry: bool = False,
        usage: LLMUsage | None = None,
        retry_safety: str | None = None,
        retry_reason: str | None = None,
    ) -> None:
        # Never log exception strings, headers, payloads or response contents.
        # Failed connection attempts have UNKNOWN Usage, never zero. After a
        # retry the logical completion therefore has usage=None; known final
        # attempt tokens remain here, separate from a complete request total.
        logger.info(
            "llm.provider_attempt",
            provider=type(self).__name__,
            model=self._model,
            attempt=attempt,
            outcome=outcome,
            error_type=error_type,
            will_retry=will_retry,
            retry_safety=retry_safety,
            retry_reason=retry_reason,
            usage_unknown=(
                usage is None
                or usage.input_tokens is None
                or usage.output_tokens is None
            ),
            logical_usage_complete=False,
            successful_attempt_input_tokens=usage.input_tokens if usage else None,
            successful_attempt_output_tokens=usage.output_tokens if usage else None,
            successful_attempt_cache_hit_tokens=usage.cache_hit_tokens
            if usage
            else None,
            successful_attempt_cache_miss_tokens=usage.cache_miss_tokens
            if usage
            else None,
        )

    def _request_payload(self, request: LLMRequest) -> dict[str, Any]:
        """Map Langley's normalized request fields to Chat Completions."""

        system_input = request.system_input
        if request.active_skill is not None:
            system_input += (
                f"\n\n[Active Skill Instructions: {request.active_skill.name}]\n"
                + request.active_skill.instructions
                + "\n[End Active Skill Instructions]"
            )
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_input}]
        for index, item in enumerate(request.transcript):
            if isinstance(item, UserRuntimeMessage):
                content: object = item.content
                if index == request.current_user_message_index:
                    content = json.dumps(
                        {
                            "personal_context_status": (
                                "unavailable"
                                if request.personal_context is None
                                else "available"
                            ),
                            "personal_context": (
                                None
                                if request.personal_context is None
                                else list(request.personal_context)
                            ),
                            **(
                                {
                                    "conversation_compact_context": (
                                        request.conversation_compact_context
                                    )
                                }
                                if request.conversation_compact_context is not None
                                else {}
                            ),
                            "current_user_request": item.content,
                            **(
                                {
                                    "active_skill_resources": [
                                        {
                                            "path": resource.path,
                                            "byte_size": resource.byte_size,
                                        }
                                        for resource in request.active_skill_resources
                                    ]
                                }
                                if request.active_skill is not None
                                and request.active_skill_resources
                                else {}
                            ),
                            **(
                                {
                                    "available_skills": [
                                        {
                                            "name": skill.name,
                                            "description": skill.description,
                                        }
                                        for skill in request.available_skills
                                    ]
                                }
                                if request.available_skills
                                and request.active_skill is None
                                else {}
                            ),
                            **(
                                {"evidence_context": request.evidence_context}
                                if request.evidence_context is not None
                                else {}
                            ),
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                messages.append({"role": "user", "content": content})
            elif isinstance(item, AssistantRuntimeMessage):
                message: dict[str, Any] = {"role": "assistant", "content": item.content}
                if item.tool_calls:
                    message["tool_calls"] = [
                        {
                            "id": tool_call.call_id,
                            "type": "function",
                            "function": {
                                "name": tool_call.name,
                                "arguments": tool_call.raw_arguments,
                            },
                        }
                        for tool_call in item.tool_calls
                    ]
                messages.append(message)
            elif isinstance(item, ToolResult):
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": item.call_id,
                        "content": item.content,
                    }
                )
            else:
                raise TypeError("unsupported normalized runtime transcript item")

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.arguments_schema,
                    },
                }
                for tool in request.allowed_tools
            ],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        self._configure_payload(payload)
        return payload

    def _configure_payload(self, payload: dict[str, Any]) -> None:
        raise NotImplementedError

    def _validate_delta(self, delta: dict[str, Any]) -> None:
        """Allow adapters to reject unsupported protocol modes."""

    @staticmethod
    def _parse_event(data: str) -> dict[str, Any]:
        try:
            event = json.loads(data)
        except json.JSONDecodeError as error:
            raise WorkflowFailure(RunErrorCode.LLM_RESPONSE_INVALID) from error
        if not isinstance(event, dict):
            raise WorkflowFailure(RunErrorCode.LLM_RESPONSE_INVALID)
        if "error" in event:
            raise WorkflowFailure(RunErrorCode.LLM_PROVIDER_FAILED)
        return event

    @staticmethod
    def _normalize_usage(raw_usage: object) -> LLMUsage | None:
        if not isinstance(raw_usage, dict):
            return None
        input_tokens = raw_usage.get("prompt_tokens")
        output_tokens = raw_usage.get("completion_tokens")
        if (input_tokens is not None and not isinstance(input_tokens, int)) or (
            output_tokens is not None and not isinstance(output_tokens, int)
        ):
            return None

        def cache_count(key: str) -> int | None:
            value = raw_usage.get(key)
            return value if type(value) is int and value >= 0 else None

        return LLMUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_hit_tokens=cache_count("prompt_cache_hit_tokens"),
            cache_miss_tokens=cache_count("prompt_cache_miss_tokens"),
        )

    @staticmethod
    def _normalize_finish_reason(raw_finish_reason: object) -> LLMFinishReason:
        if raw_finish_reason == "stop":
            return LLMFinishReason.STOP
        if raw_finish_reason in {"tool_calls", "function_call"}:
            return LLMFinishReason.TOOL_CALLS
        if raw_finish_reason == "length":
            return LLMFinishReason.LENGTH
        if raw_finish_reason in {"content_filter", "filtered"}:
            return LLMFinishReason.FILTERED
        return LLMFinishReason.UNKNOWN

    @staticmethod
    def _is_context_capacity_rejection(response: httpx.Response) -> bool:
        """Recognize only explicit provider context-capacity rejections.

        The raw provider payload remains inside this Adapter and is never put
        into a stable Run error, user-visible response, or trace payload.
        """

        if response.status_code not in {400, 413, 422}:
            return False
        try:
            payload = response.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            return False
        if not isinstance(payload, dict):
            return False
        error = payload.get("error", payload)
        if not isinstance(error, dict):
            return False
        text = " ".join(
            value.lower()
            for value in (error.get("code"), error.get("message"), error.get("type"))
            if isinstance(value, str)
        )
        return any(
            marker in text
            for marker in (
                "context_length",
                "context length",
                "maximum context",
                "max context",
                "input length",
                "input token",
                "token limit",
            )
        )

    @staticmethod
    def _append_tool_call_fragments(
        tool_parts: dict[int, _ToolCallParts], raw_tool_calls: object
    ) -> None:
        if raw_tool_calls is None:
            return
        if not isinstance(raw_tool_calls, list):
            raise WorkflowFailure(RunErrorCode.LLM_RESPONSE_INVALID)
        for raw_tool_call in raw_tool_calls:
            if not isinstance(raw_tool_call, dict):
                raise WorkflowFailure(RunErrorCode.LLM_RESPONSE_INVALID)
            index = raw_tool_call.get("index")
            if not isinstance(index, int):
                raise WorkflowFailure(RunErrorCode.LLM_RESPONSE_INVALID)
            parts = tool_parts.setdefault(index, _ToolCallParts())
            call_id = raw_tool_call.get("id")
            if call_id is not None:
                if not isinstance(call_id, str):
                    raise WorkflowFailure(RunErrorCode.LLM_RESPONSE_INVALID)
                # Qwen emits an empty ``id`` for later fragments of a Tool
                # call whose initial fragment already supplied its identity.
                # It is an omitted fragment field, not a second identity.
                if call_id:
                    if parts.call_id is not None and parts.call_id != call_id:
                        raise WorkflowFailure(RunErrorCode.LLM_RESPONSE_INVALID)
                    parts.call_id = call_id
            function = raw_tool_call.get("function")
            if function is None:
                continue
            if not isinstance(function, dict):
                raise WorkflowFailure(RunErrorCode.LLM_RESPONSE_INVALID)
            name = function.get("name")
            if name is not None:
                if not isinstance(name, str) or not name:
                    raise WorkflowFailure(RunErrorCode.LLM_RESPONSE_INVALID)
                if parts.name is not None and parts.name != name:
                    raise WorkflowFailure(RunErrorCode.LLM_RESPONSE_INVALID)
                parts.name = name
            arguments = function.get("arguments")
            if arguments is not None:
                if not isinstance(arguments, str):
                    raise WorkflowFailure(RunErrorCode.LLM_RESPONSE_INVALID)
                parts.argument_fragments.append(arguments)

    @staticmethod
    def _complete_tool_call(parts: _ToolCallParts) -> ToolCall:
        if parts.call_id is None or parts.name is None:
            raise WorkflowFailure(RunErrorCode.LLM_RESPONSE_INVALID)
        return ToolCall(
            call_id=parts.call_id,
            name=parts.name,
            raw_arguments="".join(parts.argument_fragments),
        )
