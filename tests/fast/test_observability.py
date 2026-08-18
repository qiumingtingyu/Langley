"""Tests for HTTP observability."""

import asyncio
import json
import logging

import httpx
import structlog
from fastapi.testclient import TestClient

from langley.main import create_app
from langley.settings import Settings


def _json_events(capsys) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in capsys.readouterr().err.splitlines()
        if line.strip()
    ]


def test_success_and_framework_404_responses_have_distinct_request_ids() -> None:
    with TestClient(create_app()) as client:
        health_response = client.get("/health")
        missing_response = client.get("/missing")

    health_request_id = health_response.headers["X-Request-ID"]
    missing_request_id = missing_response.headers["X-Request-ID"]
    assert health_response.status_code == 200
    assert missing_response.status_code == 404
    assert health_request_id != missing_request_id


def test_unhandled_exception_returns_safe_500_with_request_id(capsys) -> None:
    app = create_app(Settings(log_format="json"))

    @app.get("/_test-unhandled")
    def raise_unhandled_exception() -> None:
        raise RuntimeError("internal test failure")

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/_test-unhandled")

    events = _json_events(capsys)
    failed_events = [
        event for event in events if event["event"] == "http.request.failed"
    ]
    completed_events = [
        event for event in events if event["event"] == "http.request.completed"
    ]

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal Server Error"}
    assert "internal test failure" not in response.text
    assert len(failed_events) == 1
    assert not completed_events
    assert failed_events[0]["request_id"] == response.headers["X-Request-ID"]


def test_http_completed_event_contains_required_fields_and_lifecycle_events(
    capsys,
) -> None:
    with TestClient(create_app(Settings(log_format="json"))) as client:
        response = client.get(
            "/health?token=query-token",
            headers={
                "Authorization": "Bearer authorization-value",
                "Cookie": "session=cookie-value",
            },
        )

    events = _json_events(capsys)
    completed_event = next(
        event for event in events if event["event"] == "http.request.completed"
    )

    assert {"application.started", "application.stopped"} <= {
        event["event"] for event in events
    }
    assert completed_event["request_id"] == response.headers["X-Request-ID"]
    assert completed_event["http.method"] == "GET"
    assert completed_event["http.path"] == "/health"
    assert completed_event["http.status_code"] == 200
    assert isinstance(completed_event["duration_ms"], float)
    assert completed_event["duration_ms"] >= 0
    assert {"timestamp", "level", "event", "logger"} <= completed_event.keys()
    assert all(
        value not in str(events)
        for value in ("query-token", "authorization-value", "cookie-value")
    )


def test_console_renderer_and_standard_library_logging_are_rendered(capsys) -> None:
    create_app(Settings(log_format="console"))
    logging.getLogger("langley.tests").info("python.logging.event")

    output = capsys.readouterr().err

    assert "python.logging.event" in output
    assert "langley.tests" in output
    assert not output.lstrip().startswith("{")


def test_json_renderer_redacts_known_sensitive_fields(capsys) -> None:
    create_app(Settings(log_format="json"))
    structlog.get_logger("langley.tests").info(
        "security.check",
        password="password-value",
        token="token-value",
        secret="secret-value",
        api_key="api-key-value",
        authorization="authorization-value",
        cookie="cookie-value",
        database_url="mysql://sensitive",
    )

    events = _json_events(capsys)
    security_event = next(
        event for event in events if event["event"] == "security.check"
    )

    assert all(
        value not in str(events)
        for value in ("password-value", "token-value", "mysql://sensitive")
    )
    assert all(
        security_event[field] == "[REDACTED]"
        for field in (
            "password",
            "token",
            "secret",
            "api_key",
            "authorization",
            "cookie",
            "database_url",
        )
    )


def test_standard_library_extra_fields_are_redacted(capsys) -> None:
    create_app(Settings(log_format="json"))
    logging.getLogger("langley.tests").info(
        "python.logging.security.check",
        extra={"password": "password-value", "token": "token-value"},
    )

    events = _json_events(capsys)
    security_event = next(
        event for event in events if event["event"] == "python.logging.security.check"
    )

    assert "password-value" not in str(events)
    assert "token-value" not in str(events)
    assert security_event["password"] == "[REDACTED]"
    assert security_event["token"] == "[REDACTED]"


def test_concurrent_requests_do_not_share_request_ids(capsys) -> None:
    app = create_app(Settings(log_format="json"))

    async def send_requests() -> list[httpx.Response]:
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            return await asyncio.gather(*[client.get("/health") for _ in range(10)])

    responses = asyncio.run(send_requests())
    request_ids = [response.headers["X-Request-ID"] for response in responses]
    events = _json_events(capsys)
    completed_request_ids = [
        event["request_id"]
        for event in events
        if event["event"] == "http.request.completed"
    ]

    assert all(response.status_code == 200 for response in responses)
    assert len(request_ids) == len(set(request_ids))
    assert len(completed_request_ids) == len(request_ids)
    assert set(completed_request_ids) == set(request_ids)
