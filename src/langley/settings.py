"""Application configuration."""

from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LogFormat = Literal["console", "json"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
LLMProviderName = Literal["qwen"]
KnowledgeEmbeddingRepresentation = Literal["content_only"]

_DEFAULT_QWEN_MODEL = "qwen3.7-plus-2026-05-26"


class Settings(BaseSettings):
    """Settings loaded from LANGLEY_-prefixed environment variables."""

    model_config = SettingsConfigDict(env_prefix="LANGLEY_", env_file=None)

    environment: str = "development"
    log_level: LogLevel = "INFO"
    log_format: LogFormat = "console"
    database_url: str | None = None
    test_database_url: str | None = None
    knowledge_storage_root: Path = Path("data/knowledge")
    qdrant_url: str = "http://127.0.0.1:6333"
    knowledge_embedding_model: str = "BAAI/bge-m3"
    knowledge_embedding_revision: str = "5617a9f61b028005a4858fdac845db406aefb181"
    knowledge_embedding_device: str = "cuda:0"
    knowledge_embedding_dimension: int = Field(default=1024, ge=1)
    knowledge_embedding_representation: KnowledgeEmbeddingRepresentation = (
        "content_only"
    )
    knowledge_index_build_concurrency: int = Field(default=1, ge=1, le=4)
    knowledge_reranking_enabled: bool = False
    knowledge_reranker_model_path: Path | None = None
    knowledge_reranker_device: str = "cuda:0"
    knowledge_reranker_candidate_k: int = Field(default=20, ge=1)
    local_user_id: int | None = None
    llm_provider: LLMProviderName = "qwen"
    qwen_api_key: SecretStr | None = None
    qwen_base_url: str | None = None
    llm_model: str = _DEFAULT_QWEN_MODEL
    working_context_budget_estimate: int = Field(default=16_000, ge=1)
    conversation_compaction_trigger_estimate: int = Field(default=12_000, ge=1)
    recent_raw_target_estimate: int = Field(default=6_000, ge=1)
    compact_state_target_estimate: int = Field(default=2_000, ge=1)
    conversation_compactor_model: str = _DEFAULT_QWEN_MODEL
    memory_estimated_token_budget: int = Field(default=8_192, ge=1)
    memory_policy_estimated_token_budget: int | None = Field(default=None, ge=1)
    memory_policy_model: str | None = None
    local_timezone: str = "UTC"
    max_llm_rounds: int = Field(default=4, ge=1)
    max_tool_calls: int = Field(default=3, ge=1)
    overall_workflow_deadline_seconds: float = Field(default=180.0, gt=0)
    tracing_enabled: bool = False
    trace_content_enabled: bool = False
    langsmith_project: str | None = None
    web_search_enabled: bool = False
    tavily_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "TAVILY_API_KEY", "LANGLEY_TAVILY_API_KEY", "tavily_api_key"
        ),
    )

    @field_validator("local_timezone")
    @classmethod
    def local_timezone_must_be_an_iana_timezone(cls, value: str) -> str:
        """Reject invalid timezone names without consulting the host timezone."""
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as error:
            raise ValueError("local_timezone must be a valid IANA timezone") from error
        return value

    @model_validator(mode="after")
    def enabled_web_search_requires_tavily_key(self) -> "Settings":
        """Fail at startup instead of exposing a misconfigured capability."""
        if self.web_search_enabled and self.tavily_api_key is None:
            raise ValueError("TAVILY_API_KEY is required when Web search is enabled")
        return self
