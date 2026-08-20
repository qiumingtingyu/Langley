"""Fast checks for optional LangSmith tracing."""

import asyncio
import threading

import pytest
from langsmith import Client

from langley.answering.contracts import (
    LLMFinishReason,
    LLMRequest,
    LLMResponseCompleted,
    UserRuntimeMessage,
)
from langley.answering.tracing import LangSmithTracer


def test_disabled_tracing_does_not_create_a_client() -> None:
    def forbidden() -> Client:
        raise AssertionError("client should not be created")

    trace = LangSmithTracer(
        enabled=False, project=None, client_factory=forbidden
    ).start(1, "qwen", "test", False)
    trace.success("answer")


@pytest.mark.anyio
async def test_content_disabled_trace_does_not_export_personal_context() -> None:
    class RecordingClient:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []
            self.ready = threading.Event()

        def create_run(self, **kwargs: object) -> None:
            self.calls.append(kwargs)
            if len(self.calls) == 2:
                self.ready.set()

        def update_run(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

    client = RecordingClient()
    trace = LangSmithTracer(
        enabled=True,
        project=None,
        client_factory=lambda: client,  # type: ignore[arg-type]
    ).start(1, "qwen", "test", False)
    trace.llm(
        LLMRequest(
            system_input="system",
            transcript=(UserRuntimeMessage(content="request"),),
            allowed_tools=(),
            personal_context=("private preference",),
            current_user_message_index=0,
        ),
        LLMResponseCompleted(
            assistant_content="answer",
            tool_calls=(),
            finish_reason=LLMFinishReason.STOP,
            usage=None,
        ),
        1,
    )
    assert await asyncio.to_thread(client.ready.wait, 1)

    assert len(client.calls) == 2
    assert client.calls[1]["inputs"] == {}
