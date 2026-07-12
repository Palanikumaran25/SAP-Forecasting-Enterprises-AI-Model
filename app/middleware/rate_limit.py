"""
Rate Limiting Middleware  –  Phase 12
---------------------------------------
Simple in-process sliding-window rate limiter (per IP, per minute).
For production, replace with a Redis-backed implementation.

Default: 120 requests / 60 seconds per IP.
Auth endpoints: 10 requests / 60 seconds per IP.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Deque, Dict

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

# Paths with tighter limits (brute-force protection)
_STRICT_PATHS = {"/api/v1/auth/login", "/api/v1/auth/refresh"}
_GLOBAL_LIMIT = 600       # requests per window (raised for dev — SPA makes many parallel calls)
_STRICT_LIMIT = 30        # requests per window for auth paths
_WINDOW_SECONDS = 60


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding-window rate limiter stored in memory (per-process).
    Not suitable for multi-worker deployments without a shared store (Redis).
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)
        # { ip: deque of timestamps }
        self._store: Dict[str, Deque[float]] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next) -> Response:
        client_ip = _get_client_ip(request)
        path = request.url.path

        limit = _STRICT_LIMIT if path in _STRICT_PATHS else _GLOBAL_LIMIT
        now = time.monotonic()
        window_start = now - _WINDOW_SECONDS

        key = f"{client_ip}:{path if path in _STRICT_PATHS else 'global'}"
        queue = self._store[key]

        # Evict timestamps outside the current window
        while queue and queue[0] < window_start:
            queue.popleft()

        if len(queue) >= limit:
            retry_after = int(_WINDOW_SECONDS - (now - queue[0]))
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Too many requests. Please slow down.",
                    "retry_after_seconds": retry_after,
                },
                headers={"Retry-After": str(retry_after)},
            )

        queue.append(now)
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(limit - len(queue))
        return response


def _get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"
