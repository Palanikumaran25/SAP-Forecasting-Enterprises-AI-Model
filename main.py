"""
SAP Enterprise FI Forecasting Platform
=======================================
FastAPI application entrypoint.
"""
from __future__ import annotations
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.core.config import settings
from app.core.database import engine, Base
from app.middleware.audit_middleware import AuditMiddleware, SecurityHeadersMiddleware
from app.middleware.rate_limit import RateLimitMiddleware

# --- API Routers ---
from app.api.v1.auth import router as auth_router
from app.api.v1.users import router as users_router
from app.api.v1.ingestion import router as ingestion_router
from app.api.v1.forecasts import router as forecasts_router
from app.api.v1.analysis import router as analysis_router
from app.api.v1.dashboard import router as dashboard_router
from app.api.v1.compatibility import router as compatibility_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan: create DB tables on startup & seed default users
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("Starting SAP FI Forecasting Platform …")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables verified / created.")

    # Seed default users
    from app.core.database import SessionLocal
    from app.domain.models import User
    from app.domain.enums import UserRole
    from app.core.security import get_password_hash
    from sqlalchemy import select

    async with SessionLocal() as db:
        result = await db.execute(select(User))
        if not result.scalars().first():
            logger.info("Seeding default users...")
            default_users = [
                User(username="sap_admin", email="admin@company.com", hashed_password=get_password_hash("password"), role=UserRole.ADMIN, is_active=True),
                User(username="finance_controller", email="controller@company.com", hashed_password=get_password_hash("password"), role=UserRole.FINANCE_MANAGER, is_active=True),
                User(username="finance_analyst", email="finance_analyst@company.com", hashed_password=get_password_hash("password"), role=UserRole.ANALYST, is_active=True),
                User(username="board_member", email="board@company.com", hashed_password=get_password_hash("password"), role=UserRole.VIEWER, is_active=True)
            ]
            db.add_all(default_users)
            await db.commit()
            logger.info("Seeding complete.")

    yield
    logger.info("Application shutdown.")
    await engine.dispose()


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SAP Enterprise FI Forecasting Platform",
    description=(
        "Enterprise-grade financial forecasting platform integrating SAP FI/COPA data "
        "with ARIMA, ETS, Random Forest, XGBoost models and Gemini/OpenAI LLM analysis. "
        "Includes approval workflows, role-based access control, and real-time dashboards."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Middleware (order matters – outermost runs first)
# ---------------------------------------------------------------------------

# 1. CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Rate limiting  (before audit to avoid logging blocked requests)
app.add_middleware(RateLimitMiddleware)

# 3. Security headers
app.add_middleware(SecurityHeadersMiddleware)

# 4. Audit logging  (innermost – captures real response code)
app.add_middleware(AuditMiddleware)

# ---------------------------------------------------------------------------
# Global exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled exception on %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred. Please contact support."},
    )


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

API_PREFIX = "/api/v1"

app.include_router(auth_router,      prefix=API_PREFIX)
app.include_router(ingestion_router, prefix=API_PREFIX)
app.include_router(forecasts_router, prefix=API_PREFIX)
app.include_router(analysis_router,  prefix=API_PREFIX)
app.include_router(dashboard_router, prefix=API_PREFIX)
app.include_router(users_router,     prefix=API_PREFIX)
app.include_router(compatibility_router)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Health"], summary="Health check")
async def health_check() -> dict:
    return {"status": "ok", "service": "SAP FI Forecasting Platform", "version": "1.0.0"}


# ---------------------------------------------------------------------------
# Entrypoint  (python main.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_excludes=["*.db", "*.db-wal", "*.db-shm"],
        log_level="info",
    )
