"""Tests for application settings."""

from pathlib import Path
from tempfile import TemporaryDirectory

from langley.settings import Settings


def test_settings_use_safe_defaults_without_dotenv(monkeypatch) -> None:
    for variable in (
        "LANGLEY_ENVIRONMENT",
        "LANGLEY_LOG_LEVEL",
        "LANGLEY_LOG_FORMAT",
        "LANGLEY_DATABASE_URL",
        "LANGLEY_TEST_DATABASE_URL",
        "LANGLEY_LOCAL_USER_ID",
        "LANGLEY_LLM_PROVIDER",
        "LANGLEY_QWEN_API_KEY",
        "LANGLEY_QWEN_BASE_URL",
        "LANGLEY_LLM_MODEL",
        "LANGLEY_HISTORY_ESTIMATED_TOKEN_BUDGET",
        "LANGLEY_MAX_LLM_ROUNDS",
        "LANGLEY_MAX_TOOL_CALLS",
        "LANGLEY_OVERALL_WORKFLOW_DEADLINE_SECONDS",
        "LANGLEY_TRACING_ENABLED",
        "LANGLEY_TRACE_CONTENT_ENABLED",
        "LANGLEY_LANGSMITH_PROJECT",
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
    assert settings.local_user_id is None
    assert settings.llm_provider == "qwen"
    assert settings.qwen_api_key is None
    assert settings.qwen_base_url is None
    assert settings.llm_model == "qwen3.7-plus-2026-05-26"
    assert settings.history_estimated_token_budget == 16_000
    assert settings.max_llm_rounds == 4
    assert settings.max_tool_calls == 3
    assert settings.overall_workflow_deadline_seconds == 180.0
    assert settings.tracing_enabled is False
    assert settings.trace_content_enabled is False
    assert settings.langsmith_project is None


def test_settings_read_langley_prefixed_environment_variables(monkeypatch) -> None:
    monkeypatch.setenv("LANGLEY_ENVIRONMENT", "test")
    monkeypatch.setenv("LANGLEY_LOG_LEVEL", "WARNING")
    monkeypatch.setenv("LANGLEY_LOG_FORMAT", "json")
    monkeypatch.setenv("LANGLEY_DATABASE_URL", "mysql://application")
    monkeypatch.setenv("LANGLEY_TEST_DATABASE_URL", "mysql://test")
    monkeypatch.setenv("LANGLEY_LOCAL_USER_ID", "17")
    monkeypatch.setenv("LANGLEY_LLM_PROVIDER", "qwen")
    monkeypatch.setenv("LANGLEY_QWEN_API_KEY", "test-qwen-key")
    monkeypatch.setenv("LANGLEY_QWEN_BASE_URL", "https://qwen.example.test/v1")
    monkeypatch.setenv("LANGLEY_LLM_MODEL", "qwen-test-model")
    monkeypatch.setenv("LANGLEY_HISTORY_ESTIMATED_TOKEN_BUDGET", "20000")
    monkeypatch.setenv("LANGLEY_MAX_LLM_ROUNDS", "5")
    monkeypatch.setenv("LANGLEY_MAX_TOOL_CALLS", "4")
    monkeypatch.setenv("LANGLEY_OVERALL_WORKFLOW_DEADLINE_SECONDS", "12.5")
    monkeypatch.setenv("LANGLEY_TRACING_ENABLED", "true")
    monkeypatch.setenv("LANGLEY_TRACE_CONTENT_ENABLED", "true")
    monkeypatch.setenv("LANGLEY_LANGSMITH_PROJECT", "langley-test")

    settings = Settings()

    assert settings.environment == "test"
    assert settings.log_level == "WARNING"
    assert settings.log_format == "json"
    assert settings.database_url == "mysql://application"
    assert settings.test_database_url == "mysql://test"
    assert settings.local_user_id == 17
    assert settings.llm_provider == "qwen"
    assert settings.qwen_api_key is not None
    assert settings.qwen_api_key.get_secret_value() == "test-qwen-key"
    assert settings.qwen_base_url == "https://qwen.example.test/v1"
    assert settings.llm_model == "qwen-test-model"
    assert settings.history_estimated_token_budget == 20_000
    assert settings.max_llm_rounds == 5
    assert settings.max_tool_calls == 4
    assert settings.overall_workflow_deadline_seconds == 12.5
    assert settings.tracing_enabled is True
    assert settings.trace_content_enabled is True
    assert settings.langsmith_project == "langley-test"
    assert "test-qwen-key" not in repr(settings)
    assert "test-qwen-key" not in str(settings)
