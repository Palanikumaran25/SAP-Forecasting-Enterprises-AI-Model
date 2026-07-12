from typing import AsyncGenerator
import os
from pathlib import Path
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase
from app.core.config import settings

# Automatically create parent directories for SQLite database if needed
if settings.DATABASE_URL.startswith("sqlite+aiosqlite:///"):
    db_path = Path(settings.DATABASE_URL.replace("sqlite+aiosqlite:///", ""))
    if db_path.parent:
        os.makedirs(db_path.parent, exist_ok=True)

# Create async engine for PostgreSQL
# pool_pre_ping=True enables testing of connections before using them
# future=True ensures SQLAlchemy 2.0 style APIs are used
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    future=True
)

# Create session maker for async sessions
SessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)

AsyncSessionLocal = SessionLocal


# Modern SQLAlchemy 2.0 Declarative Base class
class Base(DeclarativeBase):
    pass


# Dependency to yield database sessions per request, ensuring clean teardown
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with SessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
