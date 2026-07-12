"""
Audit Middleware  –  Phase 12
-------------------------------
FastAPI middleware that automatically records every request/response
to the AuditLog table.

Records:
  - HTTP method + path
  - Response status code
  - Authenticated user (if JWT present)
  - Client IP address
  - Processing duration (ms)

XSS / Security headers are also injected via SecurityHeadersMiddleware.
"""
from __future__ import annotations

import logging
import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.core.database import AsyncSessionLocal
from app.repositories.audit_repository import AuditLogRepository

logger = logging.getLogger(__name__)

# Actions to skip auditing (health-check noise)
_SKIP_PATHS = {"/health", "/docs", "/redoc", "/openapi.json"}


class AuditMiddleware(BaseHTTPMiddleware):
    """
    Logs every HTTP request to the audit_logs table.
    Silently swallows DB errors to never block the main request pipeline.
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

        path = request.url.path
        if path in _SKIP_PATHS:
            return response

        # Resolve user_id from request state (set by auth dependency if present)
        user_id = getattr(request.state, "user_id", None)
        client_ip = _get_client_ip(request)

        try:
            async with AsyncSessionLocal() as db:
                repo = AuditLogRepository(db)
                await repo.create_log(
                    action=f"{request.method} {path}",
                    details={
                        "status_code": response.status_code,
                        "duration_ms": elapsed_ms,
                        "query_params": str(request.query_params),
                    },
                    user_id=user_id,
                    ip_address=client_ip,
                )
                await db.commit()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Audit log write failed: %s", exc)

        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Injects security-hardening HTTP headers on every response:
      - X-Content-Type-Options      (prevent MIME sniffing)
      - X-Frame-Options             (clickjacking protection)
      - X-XSS-Protection            (legacy XSS filter)
      - Strict-Transport-Security   (HSTS)
      - Referrer-Policy
      - Content-Security-Policy     (strict CSP)
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains; preload"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "connect-src 'self';"
        )
        return response


def _get_client_ip(request: Request) -> str:
    """Extract real IP from X-Forwarded-For or fall back to direct connection."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"
