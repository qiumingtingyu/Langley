"""DeepSeek Chat Completions adapter supporting non-thinking mode only."""

from typing import Any

from langley.answering.contracts import LLMFinishReason
from langley.answering.errors import RunErrorCode, WorkflowFailure
from langley.infrastructure.openai_compatible_provider import OpenAICompatibleProvider


class DeepSeekProvider(OpenAICompatibleProvider):
    """Reuse the streaming protocol without retaining reasoning content."""

    def _configure_payload(self, payload: dict[str, Any]) -> None:
        payload["thinking"] = {"type": "disabled"}
        if not payload["tools"]:
            del payload["tools"]

    def _validate_delta(self, delta: dict[str, Any]) -> None:
        if delta.get("reasoning_content") not in (None, ""):
            raise WorkflowFailure(RunErrorCode.LLM_RESPONSE_INVALID)

    @staticmethod
    def _normalize_finish_reason(raw_finish_reason: object) -> LLMFinishReason:
        if raw_finish_reason in ("insufficient_system_resource", "aborted"):
            raise WorkflowFailure(RunErrorCode.LLM_PROVIDER_FAILED)
        return OpenAICompatibleProvider._normalize_finish_reason(raw_finish_reason)
