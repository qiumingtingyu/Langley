"""FastAPI application factory."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from time import perf_counter
from uuid import uuid4

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from langley.answer_execution import AnswerExecutionManager
from langley.answer_lifecycle import interrupt_active_runs
from langley.answering.contracts import LLMProvider, LLMRequest, LLMStreamEvent
from langley.answering.conversation_context import LLMConversationCompactor
from langley.answering.conversation_context_builder import ConversationContextBuilder
from langley.answering.errors import RunErrorCode, WorkflowFailure
from langley.answering.tools import (
    AgentTool,
    CurrentTimeTool,
    ReadWebpageTool,
    SearchKnowledgeTool,
    SearchWebTool,
    ToolExecutor,
)
from langley.answering.tracing import LangSmithTracer, Tracer
from langley.answering.web import TavilyWebProvider
from langley.answering.workflow import LearningAssistantWorkflow
from langley.api.conversations import router as conversations_router
from langley.api.health import router as health_router
from langley.api.knowledge import router as knowledge_router
from langley.api.memories import router as memories_router
from langley.api.runs import router as runs_router
from langley.infrastructure.database import (
    create_database_engine,
    create_session_factory,
    dispose_database_engine,
)
from langley.infrastructure.local_file_storage import LocalFileStorage
from langley.infrastructure.qwen_provider import QwenProvider
from langley.knowledge.document_processing import (
    reconcile_interrupted_document_processing_jobs,
)
from langley.knowledge.index_build import (
    KnowledgeIndexBuildRuntime,
    reconcile_interrupted_index_builds,
    reconcile_stale_ready_index_configurations,
)
from langley.knowledge.reranking import LocalBGEReranker, Reranker
from langley.knowledge.retrieval_service import KnowledgeRetrievalService
from langley.memory.events import MemoryEventSubscribers
from langley.memory.policy import MemoryPolicy, MemoryPolicyStatus
from langley.memory.processing import (
    BACKGROUND_BATCH_LIMIT,
    PRE_ANSWER_CATCHUP_LIMIT,
    PRE_ANSWER_CATCHUP_TIMEOUT_SECONDS,
    capture_memory_high_water,
    process_memory_through,
)
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


def _memory_provider_for(
    settings: Settings, override: LLMProvider | None
) -> LLMProvider | None:
    """Construct only the explicitly configured Memory Policy provider."""

    if settings.memory_policy_model is None:
        return None
    if override is not None:
        return override
    if settings.qwen_api_key is None or settings.qwen_base_url is None:
        return _UnavailableProvider()
    return QwenProvider(
        api_key=settings.qwen_api_key,
        base_url=settings.qwen_base_url,
        model=settings.memory_policy_model,
    )


def _memory_policy_status_for(
    settings: Settings, provider: LLMProvider | None
) -> MemoryPolicyStatus:
    """Describe static readiness from the provider built by composition."""

    if settings.memory_policy_model is None:
        return MemoryPolicyStatus.NOT_CONFIGURED
    if provider is None or isinstance(provider, _UnavailableProvider):
        return MemoryPolicyStatus.PROVIDER_CONFIGURATION_UNAVAILABLE
    return MemoryPolicyStatus.READY


def _conversation_compactor_provider_for(
    settings: Settings, override: LLMProvider | None
) -> LLMProvider:
    """Construct the configured internal conversation compactor provider."""

    if override is not None:
        return override
    if settings.qwen_api_key is None or settings.qwen_base_url is None:
        return _UnavailableProvider()
    return QwenProvider(
        api_key=settings.qwen_api_key,
        base_url=settings.qwen_base_url,
        model=settings.conversation_compactor_model,
    )


def _reranker_for(settings: Settings) -> Reranker | None:
    """Construct one unloaded local reranker only when explicitly enabled."""

    if not settings.knowledge_reranking_enabled:
        return None
    if settings.knowledge_reranker_model_path is None:
        raise ValueError(
            "knowledge_reranker_model_path is required when reranking is enabled"
        )
    return LocalBGEReranker(
        model_path=settings.knowledge_reranker_model_path,
        device=settings.knowledge_reranker_device,
        max_length=2048,
    )


def _memory_lifecycle_callbacks(
    settings: Settings,
    session_factory,
    memory_policy: MemoryPolicy,
    lane: asyncio.Lock | None = None,
    outcome_callback=None,
):
    """Build concrete optional-Memory callbacks around T4's one processor."""

    resolved_lane = lane or asyncio.Lock()

    async def catch_up(command) -> None:
        boundary = command.memory_catchup_through_message_id
        if boundary is None:
            return
        try:
            async with asyncio.timeout(PRE_ANSWER_CATCHUP_TIMEOUT_SECONDS):
                result = await process_memory_through(
                    session_factory,
                    user_id=command.user_id,
                    through_message_id=boundary,
                    policy=memory_policy,
                    local_timezone=settings.local_timezone,
                    lane=resolved_lane,
                    limit=PRE_ANSWER_CATCHUP_LIMIT,
                    outcome_callback=outcome_callback,
                )
        except TimeoutError:
            logger.warning(
                "memory.catch_up.timed_out",
                run_id=command.run.id,
                conversation_id=command.run.conversation_id,
            )
            return
        if not result.complete:
            logger.info(
                "memory.catch_up.incomplete",
                run_id=command.run.id,
                conversation_id=command.run.conversation_id,
            )

    async def capture_boundary(user_id: int) -> int | None:
        return await capture_memory_high_water(session_factory, user_id=user_id)

    async def background_drain(user_id: int, boundary: int) -> None:
        result = await process_memory_through(
            session_factory,
            user_id=user_id,
            through_message_id=boundary,
            policy=memory_policy,
            local_timezone=settings.local_timezone,
            lane=resolved_lane,
            limit=BACKGROUND_BATCH_LIMIT,
            outcome_callback=outcome_callback,
        )
        if not result.complete:
            logger.info("memory.background.incomplete", user_id=user_id)

    return catch_up, capture_boundary, background_drain


def _workflow_factory_for(
    settings: Settings,
    provider: LLMProvider,
    tracer: Tracer | None,
    session_factory,
    knowledge_index_runtime: KnowledgeIndexBuildRuntime,
    conversation_compactor_provider: LLMProvider | None = None,
):
    """Assemble one production Workflow factory without giving routes AI authority."""

    resolved_compactor_provider = (
        conversation_compactor_provider
        if conversation_compactor_provider is not None
        else _conversation_compactor_provider_for(settings, None)
    )
    compactor = LLMConversationCompactor(
        provider=resolved_compactor_provider,
        model=settings.conversation_compactor_model,
        compact_state_target_estimate=settings.compact_state_target_estimate,
    )
    context_builder = ConversationContextBuilder(
        working_context_budget_estimate=settings.working_context_budget_estimate,
        conversation_compaction_trigger_estimate=(
            settings.conversation_compaction_trigger_estimate
        ),
        recent_raw_target_estimate=settings.recent_raw_target_estimate,
        compact_state_target_estimate=settings.compact_state_target_estimate,
        memory_estimated_token_budget=settings.memory_estimated_token_budget,
        compactor=compactor,
    )
    retrieval_service = KnowledgeRetrievalService(
        session_factory,
        knowledge_index_runtime,
        reranker=_reranker_for(settings),
        reranker_candidate_k=settings.knowledge_reranker_candidate_k,
    )
    tools: list[AgentTool] = [
        CurrentTimeTool(),
        SearchKnowledgeTool(retrieval_service),
    ]
    if settings.web_search_enabled:
        assert settings.tavily_api_key is not None
        web_provider = TavilyWebProvider(settings.tavily_api_key.get_secret_value())
        tools.extend((SearchWebTool(web_provider), ReadWebpageTool(web_provider)))
    tool_executor = ToolExecutor(tools=tools)
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
            retrieval_service=retrieval_service,
        )

    return factory


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Record application lifecycle events without external side effects."""

    session_factory = None
    execution_manager = None
    logger.info("application.started")
    try:
        local_file_storage = getattr(app.state, "local_file_storage", None)
        if local_file_storage is not None:
            await local_file_storage.cleanup_partial_sources()
        session_factory = getattr(app.state, "session_factory", None)
        execution_manager = getattr(app.state, "execution_manager", None)
        if session_factory is not None:
            await interrupt_active_runs(session_factory)
            await reconcile_interrupted_document_processing_jobs(session_factory)
            await reconcile_interrupted_index_builds(session_factory)
            await reconcile_stale_ready_index_configurations(
                session_factory, settings=app.state.settings
            )
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
    memory_provider: LLMProvider | None = None,
    conversation_compactor_provider: LLMProvider | None = None,
    tracer: Tracer | None = None,
    knowledge_index_runtime: KnowledgeIndexBuildRuntime | None = None,
) -> FastAPI:
    """Create the Langley FastAPI application."""

    resolved_settings = settings if settings is not None else Settings()
    configure_logging(resolved_settings)
    app = FastAPI(lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def request_validation_error(
        request: Request, error: RequestValidationError
    ) -> JSONResponse:
        del request, error
        return JSONResponse(
            status_code=422, content={"detail": {"code": "VALIDATION_ERROR"}}
        )

    app.state.settings = resolved_settings
    app.state.local_file_storage = LocalFileStorage(
        resolved_settings.knowledge_storage_root
    )
    if resolved_settings.database_url is not None:
        database_engine = create_database_engine(resolved_settings.database_url)
        app.state.database_engine = database_engine
        app.state.session_factory = create_session_factory(database_engine)
        app.state.memory_lane = asyncio.Lock()
        app.state.memory_subscribers = MemoryEventSubscribers()
        configured_provider = _provider_for(resolved_settings, provider)
        configured_memory_provider = _memory_provider_for(
            resolved_settings, memory_provider
        )
        app.state.memory_policy_status = _memory_policy_status_for(
            resolved_settings, configured_memory_provider
        )
        configured_compactor_provider = _conversation_compactor_provider_for(
            resolved_settings,
            (
                conversation_compactor_provider
                if conversation_compactor_provider is not None
                else provider
            ),
        )
        memory_callbacks = None
        if configured_memory_provider is not None:
            memory_policy = MemoryPolicy(
                provider=configured_memory_provider,
                memory_policy_estimated_token_budget=(
                    resolved_settings.memory_policy_estimated_token_budget
                ),
            )
            app.state.memory_policy = memory_policy
            memory_callbacks = _memory_lifecycle_callbacks(
                resolved_settings,
                app.state.session_factory,
                memory_policy,
                app.state.memory_lane,
                app.state.memory_subscribers.publish,
            )
        app.state.knowledge_index_runtime = (
            knowledge_index_runtime
            if knowledge_index_runtime is not None
            else KnowledgeIndexBuildRuntime(
                app.state.session_factory, resolved_settings
            )
        )
        app.state.execution_manager = AnswerExecutionManager(
            app.state.session_factory,
            _workflow_factory_for(
                resolved_settings,
                configured_provider,
                tracer,
                app.state.session_factory,
                app.state.knowledge_index_runtime,
                configured_compactor_provider,
            ),
            memory_catch_up=memory_callbacks[0]
            if memory_callbacks is not None
            else None,
            memory_boundary_capture=(
                memory_callbacks[1] if memory_callbacks is not None else None
            ),
            memory_background_drain=(
                memory_callbacks[2] if memory_callbacks is not None else None
            ),
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
    app.include_router(memories_router)
    app.include_router(knowledge_router)
    return app
