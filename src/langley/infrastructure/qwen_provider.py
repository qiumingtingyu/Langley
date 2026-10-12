"""Qwen adapter retaining its existing non-thinking wire configuration."""

from typing import Any

from langley.infrastructure.openai_compatible_provider import OpenAICompatibleProvider


class QwenProvider(OpenAICompatibleProvider):
    """Use the shared protocol with Qwen-specific request controls."""

    def _configure_payload(self, payload: dict[str, Any]) -> None:
        payload["enable_thinking"] = False
        payload["parallel_tool_calls"] = False
