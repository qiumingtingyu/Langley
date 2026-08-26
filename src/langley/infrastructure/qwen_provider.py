"""Qwen OpenAI-compatible streaming adapter.

The adapter normalizes only one provider round.  It neither runs an Agent loop
nor validates or executes Tool calls.
"""

import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
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


@dataclass
class _ToolCallParts:
    """Losslessly collect one provider-emitted Tool argument stream."""

    call_id: str | None = None
    name: str | None = None
    argument_fragments: list[str] = field(default_factory=list)


class QwenProvider:
    """Translate Qwen's HTTP/SSE protocol into Langley's normalized contract."""

    def __init__(
        self,
        *,
        api_key: SecretStr,
        base_url: str,
        model: str,
        client_factory: HTTPClientFactory | None = None,
    ) -> None:
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
        content_parts: list[str] = []
        tool_parts: dict[int, _ToolCallParts] = {}
        finish_reason = LLMFinishReason.UNKNOWN
        usage: LLMUsage | None = None
        provider_model: str | None = None
        saw_done = False

        try:
            async with self._client_factory() as client:
                async with client.stream(
                    "POST",
                    f"{self._base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                ) as response:
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
                                raise WorkflowFailure(RunErrorCode.LLM_RESPONSE_INVALID)
                            if (
                                provider_model is not None
                                and provider_model != event_model
                            ):
                                raise WorkflowFailure(RunErrorCode.LLM_RESPONSE_INVALID)
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
                                raise WorkflowFailure(RunErrorCode.LLM_RESPONSE_INVALID)
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
        except WorkflowFailure:
            raise
        except httpx.HTTPStatusError as error:
            error_code = (
                RunErrorCode.ANSWER_CONTEXT_TOO_LARGE
                if self._is_context_capacity_rejection(error.response)
                else RunErrorCode.LLM_PROVIDER_FAILED
            )
            raise WorkflowFailure(error_code) from error
        except httpx.HTTPError as error:
            raise WorkflowFailure(RunErrorCode.LLM_PROVIDER_FAILED) from error

        if not saw_done:
            raise WorkflowFailure(RunErrorCode.LLM_RESPONSE_INVALID)

        tool_calls = tuple(
            self._complete_tool_call(parts)
            for _, parts in sorted(tool_parts.items(), key=lambda item: item[0])
        )
        if len({tool_call.call_id for tool_call in tool_calls}) != len(tool_calls):
            raise WorkflowFailure(RunErrorCode.LLM_RESPONSE_INVALID)
        yield LLMResponseCompleted(
            assistant_content="".join(content_parts),
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            provider_model=provider_model,
        )

    def _request_payload(self, request: LLMRequest) -> dict[str, Any]:
        """Map only Langley's normalized request fields to Qwen's wire shape."""

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": request.system_input}
        ]
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
                            "current_user_request": item.content,
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

        return {
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
            "enable_thinking": False,
            "parallel_tool_calls": False,
        }

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
        return LLMUsage(input_tokens=input_tokens, output_tokens=output_tokens)

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
