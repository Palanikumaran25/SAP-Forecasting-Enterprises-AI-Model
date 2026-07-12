from typing import Optional, List, Sequence
from datetime import datetime, timezone
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.repositories.base import BaseRepository
from app.domain.models import ForecastRun, LLMAnalysis
from app.domain.enums import ForecastStatus


class ForecastRunRepository(BaseRepository[ForecastRun]):
    def __init__(self, db: AsyncSession):
        super().__init__(ForecastRun, db)

    async def get_all_runs(self, skip: int = 0, limit: int = 100) -> Sequence[ForecastRun]:
        """Retrieve all forecast runs, ordered by creation date descending."""
        query = (
            select(ForecastRun)
            .order_by(ForecastRun.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await self.db.execute(query)
        return result.scalars().all()

    async def get_runs_by_status(self, status: ForecastStatus, skip: int = 0, limit: int = 100) -> Sequence[ForecastRun]:
        """Filter forecast runs by approval workflow status."""
        query = (
            select(ForecastRun)
            .where(ForecastRun.status == status)
            .order_by(ForecastRun.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await self.db.execute(query)
        return result.scalars().all()

    async def update_approval_status(
        self, run_id: int, status: ForecastStatus, approved_by: Optional[int] = None, comments: Optional[str] = None
    ) -> Optional[ForecastRun]:
        """Transition a forecast run through approval workflow states."""
        run = await self.get(run_id)
        if run:
            run.status = status
            if approved_by is not None:
                run.approved_by = approved_by
                run.approved_at = datetime.now(timezone.utc)
            if comments is not None:
                run.comments = comments
            await self.db.flush()
        return run


class LLMAnalysisRepository(BaseRepository[LLMAnalysis]):
    def __init__(self, db: AsyncSession):
        super().__init__(LLMAnalysis, db)

    async def get_by_run_id(self, run_id: int) -> Sequence[LLMAnalysis]:
        """Fetch LLM analysis reports linked to a specific forecast run."""
        query = select(LLMAnalysis).where(LLMAnalysis.forecast_run_id == run_id)
        result = await self.db.execute(query)
        return result.scalars().all()
