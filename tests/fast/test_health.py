"""Tests for the health endpoint."""

from fastapi.testclient import TestClient

from langley.main import create_app
from langley.settings import Settings


def test_create_app_uses_explicit_settings() -> None:
    settings = Settings(environment="test")

    app = create_app(settings=settings)

    assert app.state.settings is settings


def test_health_returns_contract_without_database_configuration() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/health", headers={"X-Request-ID": "client-value"})

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "langley"}
    assert response.headers["X-Request-ID"] != "client-value"
