"""Fast checks for optional LangSmith tracing."""

from langsmith import Client

from langley.answering.tracing import LangSmithTracer


def test_disabled_tracing_does_not_create_a_client() -> None:
    def forbidden() -> Client:
        raise AssertionError("client should not be created")

    trace = LangSmithTracer(
        enabled=False, project=None, client_factory=forbidden
    ).start(1, "qwen", "test", False)
    trace.success("answer")
