"""
Audit Log Repository
--------------------
Persists structured audit trail entries for every significant system action.
"""
from __future__ import annotations

from typing import Optional, Sequence
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models import AuditLog
from app.repositories.base import BaseRepository


class AuditLogRepository(BaseRepository[AuditLog]):
    def __init__(self, db: AsyncSession) -> None:
        super().__init__(AuditLog, db)

    async def get_by_user(
        self, user_id: int, skip: int = 0, limit: int = 100
    ) -> Sequence[AuditLog]:
        query = (
            select(AuditLog)
            .where(AuditLog.user_id == user_id)
            .order_by(AuditLog.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await self.db.execute(query)
        return result.scalars().all()

    async def get_by_action(
        self, action: str, skip: int = 0, limit: int = 100
    ) -> Sequence[AuditLog]:
        query = (
            select(AuditLog)
            .where(AuditLog.action == action)
            .order_by(AuditLog.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await self.db.execute(query)
        return result.scalars().all()

    async def create_log(
        self,
        action: str,
        details: dict,
        user_id: Optional[int] = None,
        ip_address: Optional[str] = None,
    ) -> AuditLog:
        log = AuditLog(
            action=action,
            details=details,
            user_id=user_id,
            ip_address=ip_address,
        )
        self.db.add(log)
        await self.db.flush()
        return log
