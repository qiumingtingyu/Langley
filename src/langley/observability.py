"""Structured logging configuration for HTTP observability."""

import logging
from typing import Any

import structlog
from structlog.typing import EventDict, Processor

from langley.settings import Settings

_SENSITIVE_FIELDS = frozenset(
    {
        "password",
        "token",
        "secret",
        "api_key",
        "authorization",
        "cookie",
        "database_url",
    }
)


def redact_sensitive_fields(_: Any, __: str, event_dict: EventDict) -> EventDict:
    """Replace known top-level sensitive fields with a safe marker."""

    for field_name in event_dict:
        if field_name.lower() in _SENSITIVE_FIELDS:
            event_dict[field_name] = "[REDACTED]"
    return event_dict


def configure_logging(settings: Settings) -> None:
    """Configure the Langley structlog and standard-library logging pipeline."""

    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        redact_sensitive_fields,
    ]
    renderer = (
        structlog.processors.JSONRenderer()
        if settings.log_format == "json"
        else structlog.dev.ConsoleRenderer()
    )
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.ExtraAdder(),
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            redact_sensitive_fields,
        ],
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    langley_logger = logging.getLogger("langley")
    langley_logger.handlers.clear()
    langley_logger.addHandler(handler)
    langley_logger.setLevel(settings.log_level)
    langley_logger.propagate = False

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,
    )
