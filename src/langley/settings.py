"""Application configuration."""

from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

LogFormat = Literal["console", "json"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
LLMProviderName = Literal["qwen"]

_DEFAULT_QWEN_MODEL = "qwen3.7-plus-2026-05-26"


class Settings(BaseSettings):
    """Settings loaded from LANGLEY_-prefixed environment variables."""

    model_config = SettingsConfigDict(env_prefix="LANGLEY_", env_file=None)

    environment: str = "development"
    log_level: LogLevel = "INFO"
    log_format: LogFormat = "console"
    database_url: str | None = None
    test_database_url: str | None = None
    local_user_id: int | None = None
    llm_provider: LLMProviderName = "qwen"
    qwen_api_key: SecretStr | None = None
    qwen_base_url: str | None = None
    llm_model: str = _DEFAULT_QWEN_MODEL
    history_estimated_token_budget: int = Field(default=16_000, ge=1)
    max_llm_rounds: int = Field(default=4, ge=1)
    max_tool_calls: int = Field(default=3, ge=1)
    overall_workflow_deadline_seconds: float = Field(default=180.0, gt=0)
    tracing_enabled: bool = False
    trace_content_enabled: bool = False
    langsmith_project: str | None = None
