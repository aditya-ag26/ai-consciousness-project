import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.security import (
    SECURITY_HEADERS,
    RateLimitMiddleware,
    SecurityHeadersMiddleware,
    allowed_origins,
    docs_enabled,
    rate_limit_settings,
)


def build_app(**rate_limit) -> FastAPI:
    app = FastAPI()
    app.add_middleware(SecurityHeadersMiddleware)
    if rate_limit:
        app.add_middleware(RateLimitMiddleware, **rate_limit)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/thing")
    def thing():
        return {"ok": True}

    return app


def test_security_headers_are_present_on_responses():
    client = TestClient(build_app())

    response = client.get("/thing")

    for header, value in SECURITY_HEADERS.items():
        assert response.headers[header] == value


def test_requests_under_the_limit_pass():
    client = TestClient(build_app(max_requests=3, window_seconds=60))

    for _ in range(3):
        assert client.get("/thing").status_code == 200


def test_requests_over_the_limit_are_rejected():
    client = TestClient(build_app(max_requests=2, window_seconds=60))

    client.get("/thing")
    client.get("/thing")
    response = client.get("/thing")

    assert response.status_code == 429
    assert response.headers["Retry-After"] == "60"


def test_health_is_never_rate_limited():
    client = TestClient(build_app(max_requests=1, window_seconds=60))

    for _ in range(5):
        assert client.get("/health").status_code == 200


def test_expired_hits_free_up_the_window():
    limiter = RateLimitMiddleware(build_app(), max_requests=1, window_seconds=60)

    assert limiter._is_over_limit("1.2.3.4") is False
    assert limiter._is_over_limit("1.2.3.4") is True

    # Backdate the recorded hit past the window rather than waiting it out.
    limiter._hits["1.2.3.4"][0] -= 61

    assert limiter._is_over_limit("1.2.3.4") is False


def test_clients_are_limited_independently():
    limiter = RateLimitMiddleware(build_app(), max_requests=1, window_seconds=60)

    assert limiter._is_over_limit("1.1.1.1") is False
    assert limiter._is_over_limit("1.1.1.1") is True
    assert limiter._is_over_limit("2.2.2.2") is False


def test_allowed_origins_defaults_to_the_dev_server(monkeypatch):
    monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)

    assert allowed_origins() == ["http://localhost:5173"]


def test_allowed_origins_splits_and_trims(monkeypatch):
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://a.example , https://b.example")

    assert allowed_origins() == ["https://a.example", "https://b.example"]


@pytest.mark.parametrize(
    "value,expected",
    [("true", True), ("false", False), ("1", True), ("no", False)],
)
def test_docs_toggle(monkeypatch, value, expected):
    monkeypatch.setenv("ENABLE_DOCS", value)

    assert docs_enabled() is expected


def test_rate_limit_settings_read_from_env(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "5")
    monkeypatch.setenv("RATE_LIMIT_WINDOW_SECONDS", "10")

    assert rate_limit_settings() == (5, 10)
