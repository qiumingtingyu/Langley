"""Tests for application settings."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from langley.settings import Settings


def test_settings_use_safe_defaults_without_dotenv(monkeypatch) -> None:
    for variable in (
        "LANGLEY_ENVIRONMENT",
        "LANGLEY_LOG_LEVEL",
        "LANGLEY_LOG_FORMAT",
        "LANGLEY_DATABASE_URL",
        "LANGLEY_TEST_DATABASE_URL",
        "LANGLEY_KNOWLEDGE_STORAGE_ROOT",
        "LANGLEY_KNOWLEDGE_RERANKING_ENABLED",
        "LANGLEY_KNOWLEDGE_RERANKER_MODEL_PATH",
        "LANGLEY_KNOWLEDGE_RERANKER_DEVICE",
        "LANGLEY_KNOWLEDGE_RERANKER_CANDIDATE_K",
        "LANGLEY_LOCAL_USER_ID",
        "LANGLEY_LLM_PROVIDER",
        "LANGLEY_QWEN_API_KEY",
        "LANGLEY_QWEN_BASE_URL",
        "LANGLEY_LLM_MODEL",
        "LANGLEY_WORKING_CONTEXT_BUDGET_ESTIMATE",
        "LANGLEY_CONVERSATION_COMPACTION_TRIGGER_ESTIMATE",
        "LANGLEY_RECENT_RAW_TARGET_ESTIMATE",
        "LANGLEY_COMPACT_STATE_TARGET_ESTIMATE",
        "LANGLEY_CONVERSATION_COMPACTOR_MODEL",
        "LANGLEY_MEMORY_ESTIMATED_TOKEN_BUDGET",
        "LANGLEY_MEMORY_POLICY_ESTIMATED_TOKEN_BUDGET",
        "LANGLEY_MEMORY_POLICY_MODEL",
        "LANGLEY_LOCAL_TIMEZONE",
        "LANGLEY_MAX_LLM_ROUNDS",
        "LANGLEY_MAX_TOOL_CALLS",
        "LANGLEY_OVERALL_WORKFLOW_DEADLINE_SECONDS",
        "LANGLEY_TRACING_ENABLED",
        "LANGLEY_TRACE_CONTENT_ENABLED",
        "LANGLEY_LANGSMITH_PROJECT",
        "LANGLEY_WEB_SEARCH_ENABLED",
        "LANGLEY_TAVILY_API_KEY",
        "TAVILY_API_KEY",
    ):
        monkeypatch.delenv(variable, raising=False)

    with TemporaryDirectory(dir=Path.cwd()) as temporary_directory:
        temporary_path = Path(temporary_directory)
        (temporary_path / ".env").write_text(
            "LANGLEY_ENVIRONMENT=from-dotenv\nLANGLEY_DATABASE_URL=mysql://unexpected\n",
            encoding="utf-8",
        )
        with monkeypatch.context() as scoped_monkeypatch:
            scoped_monkeypatch.chdir(temporary_path)
            settings = Settings()

    assert settings.environment == "development"
    assert settings.log_level == "INFO"
    assert settings.log_format == "console"
    assert settings.database_url is None
    assert settings.test_database_url is None
    assert settings.knowledge_storage_root == Path("data/knowledge")
    assert settings.knowledge_reranking_enabled is False
    assert settings.knowledge_reranker_model_path is None
    assert settings.knowledge_reranker_device == "cuda:0"
    assert settings.knowledge_reranker_candidate_k == 20
    assert settings.local_user_id is None
    assert settings.llm_provider == "qwen"
    assert settings.qwen_api_key is None
    assert settings.qwen_base_url is None
    assert settings.llm_model == "qwen3.7-plus-2026-05-26"
    assert settings.working_context_budget_estimate == 16_000
    assert settings.conversation_compaction_trigger_estimate == 12_000
    assert settings.recent_raw_target_estimate == 6_000
    assert settings.compact_state_target_estimate == 2_000
    assert settings.conversation_compactor_model == "qwen3.7-plus-2026-05-26"
    assert settings.memory_estimated_token_budget == 8_192
    assert settings.memory_policy_estimated_token_budget is None
    assert settings.memory_policy_model is None
    assert settings.local_timezone == "UTC"
    assert settings.max_llm_rounds == 4
    assert settings.max_tool_calls == 3
    assert settings.overall_workflow_deadline_seconds == 180.0
    assert settings.tracing_enabled is False
    assert settings.trace_content_enabled is False
    assert settings.langsmith_project is None
    assert settings.web_search_enabled is False
    assert settings.tavily_api_key is None


def test_settings_read_langley_prefixed_environment_variables(monkeypatch) -> None:
    monkeypatch.setenv("LANGLEY_ENVIRONMENT", "test")
    monkeypatch.setenv("LANGLEY_LOG_LEVEL", "WARNING")
    monkeypatch.setenv("LANGLEY_LOG_FORMAT", "json")
    monkeypatch.setenv("LANGLEY_DATABASE_URL", "mysql://application")
    monkeypatch.setenv("LANGLEY_TEST_DATABASE_URL", "mysql://test")
    monkeypatch.setenv("LANGLEY_KNOWLEDGE_STORAGE_ROOT", "temporary/knowledge")
    monkeypatch.setenv("LANGLEY_KNOWLEDGE_RERANKING_ENABLED", "true")
    monkeypatch.setenv("LANGLEY_KNOWLEDGE_RERANKER_MODEL_PATH", "models/test-reranker")
    monkeypatch.setenv("LANGLEY_KNOWLEDGE_RERANKER_DEVICE", "cpu")
    monkeypatch.setenv("LANGLEY_KNOWLEDGE_RERANKER_CANDIDATE_K", "24")
    monkeypatch.setenv("LANGLEY_LOCAL_USER_ID", "17")
    monkeypatch.setenv("LANGLEY_LLM_PROVIDER", "qwen")
    monkeypatch.setenv("LANGLEY_QWEN_API_KEY", "test-qwen-key")
    monkeypatch.setenv("LANGLEY_QWEN_BASE_URL", "https://qwen.example.test/v1")
    monkeypatch.setenv("LANGLEY_LLM_MODEL", "qwen-test-model")
    monkeypatch.setenv("LANGLEY_WORKING_CONTEXT_BUDGET_ESTIMATE", "20000")
    monkeypatch.setenv("LANGLEY_CONVERSATION_COMPACTION_TRIGGER_ESTIMATE", "14000")
    monkeypatch.setenv("LANGLEY_RECENT_RAW_TARGET_ESTIMATE", "7000")
    monkeypatch.setenv("LANGLEY_COMPACT_STATE_TARGET_ESTIMATE", "2500")
    monkeypatch.setenv("LANGLEY_CONVERSATION_COMPACTOR_MODEL", "compact-test-model")
    monkeypatch.setenv("LANGLEY_MEMORY_ESTIMATED_TOKEN_BUDGET", "8192")
    monkeypatch.setenv("LANGLEY_MEMORY_POLICY_ESTIMATED_TOKEN_BUDGET", "24576")
    monkeypatch.setenv("LANGLEY_MEMORY_POLICY_MODEL", "memory-policy-test-model")
    monkeypatch.setenv("LANGLEY_LOCAL_TIMEZONE", "Asia/Shanghai")
    monkeypatch.setenv("LANGLEY_MAX_LLM_ROUNDS", "5")
    monkeypatch.setenv("LANGLEY_MAX_TOOL_CALLS", "4")
    monkeypatch.setenv("LANGLEY_OVERALL_WORKFLOW_DEADLINE_SECONDS", "12.5")
    monkeypatch.setenv("LANGLEY_TRACING_ENABLED", "true")
    monkeypatch.setenv("LANGLEY_TRACE_CONTENT_ENABLED", "true")
    monkeypatch.setenv("LANGLEY_LANGSMITH_PROJECT", "langley-test")
    monkeypatch.setenv("LANGLEY_WEB_SEARCH_ENABLED", "true")
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-key")

    settings = Settings()

    assert settings.environment == "test"
    assert settings.log_level == "WARNING"
    assert settings.log_format == "json"
    assert settings.database_url == "mysql://application"
    assert settings.test_database_url == "mysql://test"
    assert settings.knowledge_storage_root == Path("temporary/knowledge")
    assert settings.knowledge_reranking_enabled is True
    assert settings.knowledge_reranker_model_path == Path("models/test-reranker")
    assert settings.knowledge_reranker_device == "cpu"
    assert settings.knowledge_reranker_candidate_k == 24
    assert settings.local_user_id == 17
    assert settings.llm_provider == "qwen"
    assert settings.qwen_api_key is not None
    assert settings.qwen_api_key.get_secret_value() == "test-qwen-key"
    assert settings.qwen_base_url == "https://qwen.example.test/v1"
    assert settings.llm_model == "qwen-test-model"
    assert settings.working_context_budget_estimate == 20_000
    assert settings.conversation_compaction_trigger_estimate == 14_000
    assert settings.recent_raw_target_estimate == 7_000
    assert settings.compact_state_target_estimate == 2_500
    assert settings.conversation_compactor_model == "compact-test-model"
    assert settings.memory_estimated_token_budget == 8_192
    assert settings.memory_policy_estimated_token_budget == 24_576
    assert settings.memory_policy_model == "memory-policy-test-model"
    assert settings.local_timezone == "Asia/Shanghai"
    assert settings.max_llm_rounds == 5
    assert settings.max_tool_calls == 4
    assert settings.overall_workflow_deadline_seconds == 12.5
    assert settings.tracing_enabled is True
    assert settings.trace_content_enabled is True
    assert settings.langsmith_project == "langley-test"
    assert settings.web_search_enabled is True
    assert settings.tavily_api_key is not None
    assert settings.tavily_api_key.get_secret_value() == "test-tavily-key"
    assert "test-qwen-key" not in repr(settings)
    assert "test-qwen-key" not in str(settings)
    assert "test-tavily-key" not in repr(settings)
    assert "test-tavily-key" not in str(settings)


def test_settings_reject_an_invalid_local_timezone() -> None:
    with pytest.raises(ValueError, match="IANA timezone"):
        Settings(local_timezone="not-a-timezone")


def test_settings_allow_memory_policy_budget_without_model() -> None:
    settings = Settings(memory_policy_estimated_token_budget=24_576)

    assert settings.memory_policy_model is None
    assert settings.memory_policy_estimated_token_budget == 24_576


def test_settings_reject_memory_policy_model_without_budget() -> None:
    with pytest.raises(
        ValueError, match="LANGLEY_MEMORY_POLICY_ESTIMATED_TOKEN_BUDGET"
    ):
        Settings(memory_policy_model="qwen3.7-plus-2026-05-26")


def test_settings_reject_enabled_web_search_without_tavily_key(monkeypatch) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("LANGLEY_TAVILY_API_KEY", raising=False)

    with pytest.raises(ValueError, match="TAVILY_API_KEY"):
        Settings(web_search_enabled=True)
