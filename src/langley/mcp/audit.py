"""Allowlisted best-effort audit events, with no raw request or error values."""

import logging

import structlog


def event(
    *,
    request_id: str,
    principal: str | None,
    tool: str | None,
    decision: str,
    reason: str,
    status: str,
    duration_ms: float,
    input_bytes: int,
    output_bytes: int,
) -> None:
    try:
        structlog.get_logger("langley.mcp").info(
            "mcp.operation",
            request_id=request_id,
            principal=principal,
            tool=tool,
            decision=decision,
            reason=reason,
            status=status,
            duration_ms=round(duration_ms, 2),
            input_bytes=input_bytes,
            output_bytes=output_bytes,
        )
    except Exception:
        # Total Spec: observation failure must not become business authority.
        pass


def silence_payload_logs() -> None:
    """Only called by the dedicated process entry, never on module import."""
    prefixes = (
        "mcp",
        "httpx",
        "httpx2",
        "httpcore",
        "httpcore2",
        "sqlalchemy",
        "uvicorn",
    )
    names = set(prefixes) | {
        name
        for name in logging.root.manager.loggerDict
        if name.startswith(tuple(p + "." for p in prefixes))
    }
    for name in names:
        logger = logging.getLogger(name)
        logger.handlers = [logging.NullHandler()]
        logger.propagate = False
