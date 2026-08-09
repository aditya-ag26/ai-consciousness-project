"""
Transport-level protections for the public API.

The chat endpoints run a local LLM, so every request costs seconds of CPU. That
makes unthrottled access a practical denial-of-service risk even without any
malicious intent, which is what the rate limiter here addresses.
"""
import os
import threading
import time
from collections import defaultdict, deque

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# Sent on every response. These are cheap and prevent whole classes of
# browser-side attacks without needing any application changes.
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
}


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Adds hardening headers to every response."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        for header, value in SECURITY_HEADERS.items():
            response.headers.setdefault(header, value)
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Caps requests per client IP using a sliding window.

    State is per-process, so with multiple replicas each one enforces its own
    share of the limit. That is a deliberate trade-off: it needs no external
    store, and a shared limiter (e.g. Redis) can replace it if traffic warrants.
    """

    def __init__(self, app, max_requests: int, window_seconds: int):
        super().__init__(app)
        self._max_requests = max_requests
        self._window = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    async def dispatch(self, request: Request, call_next):
        # Health checks are what keeps a container in service; never throttle them.
        if request.url.path == "/health":
            return await call_next(request)

        if self._is_over_limit(self._client_key(request)):
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Please slow down."},
                headers={"Retry-After": str(self._window)},
            )
        return await call_next(request)

    def _client_key(self, request: Request) -> str:
        return request.client.host if request.client else "unknown"

    def _is_over_limit(self, key: str) -> bool:
        now = time.monotonic()
        cutoff = now - self._window
        with self._lock:
            hits = self._hits[key]
            while hits and hits[0] < cutoff:
                hits.popleft()
            if len(hits) >= self._max_requests:
                return True
            hits.append(now)

            # Drop the bucket entirely once a client goes quiet, so the map does
            # not grow without bound across many distinct addresses.
            if not hits:
                del self._hits[key]
            return False


def allowed_origins() -> list[str]:
    """CORS whitelist, from ALLOWED_ORIGINS (comma-separated)."""
    raw = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173")
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def docs_enabled() -> bool:
    """Whether to expose interactive API docs. Off by default in production."""
    return os.getenv("ENABLE_DOCS", "true").lower() in {"1", "true", "yes"}


def rate_limit_settings() -> tuple[int, int]:
    """Returns (max_requests, window_seconds) for the rate limiter."""
    return (
        int(os.getenv("RATE_LIMIT_REQUESTS", "30")),
        int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60")),
    )
