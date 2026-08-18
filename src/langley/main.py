"""FastAPI application factory."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from time import perf_counter
from uuid import uuid4

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from langley.answer_execution import AnswerExecutionManager
from langley.answer_lifecycle import interrupt_active_runs
from langley.answering.context_builder import AnswerContextBuilder
from langley.answering.contracts import LLMProvider, LLMRequest, LLMStreamEvent
from langley.answering.errors import RunErrorCode, WorkflowFailure
from langley.answering.tools import ToolExecutor
from langley.answering.tracing import LangSmithTracer, Tracer
from langley.answering.workflow import LearningAssistantWorkflow
from langley.api.conversations import router as conversations_router
from langley.api.health import router as health_router
from langley.api.runs import router as runs_router
from langley.infrastructure.database import (
    create_database_engine,
    create_session_factory,
    dispose_database_engine,
)
from langley.infrastructure.qwen_provider import QwenProvider
from langley.observability import configure_logging
from langley.settings import Settings

logger = structlog.get_logger(__name__)


class _UnavailableProvider:
    """Fail safely when the deployment has not configured Qwen credentials."""

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        del request
        return self._stream()

    @staticmethod
    async def _stream() -> AsyncIterator[LLMStreamEvent]:
        raise WorkflowFailure(RunErrorCode.LLM_PROVIDER_FAILED)
        yield


def _provider_for(settings: Settings, override: LLMProvider | None) -> LLMProvider:
    """Create the configured real Provider once at the application composition root."""

    if override is not None:
        return override
    if settings.qwen_api_key is None or settings.qwen_base_url is None:
        return _UnavailableProvider()
    return QwenProvider(
        api_key=settings.qwen_api_key,
        base_url=settings.qwen_base_url,
        model=settings.llm_model,
    )


def _workflow_factory_for(
    settings: Settings,
    provider: LLMProvider,
    tracer: Tracer | None,
):
    """Assemble one production Workflow factory without giving routes AI authority."""

    context_builder = AnswerContextBuilder(
        history_estimated_token_budget=settings.history_estimated_token_budget
    )
    tool_executor = ToolExecutor()
    resolved_tracer = tracer or LangSmithTracer(
        enabled=settings.tracing_enabled,
        project=settings.langsmith_project,
    )

    def factory() -> LearningAssistantWorkflow:
        return LearningAssistantWorkflow(
            context_builder=context_builder,
            provider=provider,
            tool_executor=tool_executor,
            max_llm_rounds=settings.max_llm_rounds,
            max_tool_calls=settings.max_tool_calls,
            overall_deadline_seconds=settings.overall_workflow_deadline_seconds,
            provider_name=settings.llm_provider,
            model=settings.llm_model,
            trace_content_enabled=settings.trace_content_enabled,
            tracer=resolved_tracer,
        )

    return factory


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Record application lifecycle events without external side effects."""

    logger.info("application.started")
    try:
        session_factory = getattr(app.state, "session_factory", None)
        execution_manager = getattr(app.state, "execution_manager", None)
        if session_factory is not None:
            await interrupt_active_runs(session_factory)
        yield
    finally:
        if session_factory is not None:
            repaired_run_ids = await interrupt_active_runs(session_factory)
            if execution_manager is not None:
                await execution_manager.stop_interrupted_runs(repaired_run_ids)
        database_engine = getattr(app.state, "database_engine", None)
        if database_engine is not None:
            await dispose_database_engine(database_engine)
        logger.info("application.stopped")


def create_app(
    settings: Settings | None = None,
    *,
    provider: LLMProvider | None = None,
    tracer: Tracer | None = None,
) -> FastAPI:
    """Create the Langley FastAPI application."""

    resolved_settings = settings if settings is not None else Settings()
    configure_logging(resolved_settings)
    app = FastAPI(lifespan=lifespan)
    app.state.settings = resolved_settings
    if resolved_settings.database_url is not None:
        database_engine = create_database_engine(resolved_settings.database_url)
        app.state.database_engine = database_engine
        app.state.session_factory = create_session_factory(database_engine)
        configured_provider = _provider_for(resolved_settings, provider)
        app.state.execution_manager = AnswerExecutionManager(
            app.state.session_factory,
            _workflow_factory_for(resolved_settings, configured_provider, tracer),
        )

    @app.middleware("http")
    async def observe_http_request(request: Request, call_next):
        structlog.contextvars.clear_contextvars()
        request_id = uuid4().hex
        structlog.contextvars.bind_contextvars(request_id=request_id)
        started_at = perf_counter()

        try:
            try:
                response = await call_next(request)
            except Exception:
                duration_ms = round((perf_counter() - started_at) * 1000, 3)
                logger.error(
                    "http.request.failed",
                    **{
                        "http.method": request.method,
                        "http.path": request.url.path,
                        "http.status_code": 500,
                        "duration_ms": duration_ms,
                    },
                )
                response = JSONResponse(
                    status_code=500,
                    content={"detail": "Internal Server Error"},
                )
            else:
                duration_ms = round((perf_counter() - started_at) * 1000, 3)
                logger.info(
                    "http.request.completed",
                    **{
                        "http.method": request.method,
                        "http.path": request.url.path,
                        "http.status_code": response.status_code,
                        "duration_ms": duration_ms,
                    },
                )

            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            structlog.contextvars.clear_contextvars()

    app.include_router(health_router)
    app.include_router(conversations_router)
    app.include_router(runs_router)
    return app
