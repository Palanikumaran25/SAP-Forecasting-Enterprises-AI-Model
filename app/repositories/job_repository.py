from typing import Optional, Sequence
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.repositories.base import BaseRepository
from app.domain.models import BackgroundJob
from app.domain.enums import JobStatus, JobType


class BackgroundJobRepository(BaseRepository[BackgroundJob]):
    def __init__(self, db: AsyncSession):
        super().__init__(BackgroundJob, db)

    async def update_job_status(
        self, job_id: int, status: JobStatus, result: Optional[dict] = None, error_message: Optional[str] = None
    ) -> Optional[BackgroundJob]:
        """Update status, output result, or error message of a background job."""
        job = await self.get(job_id)
        if job:
            job.status = status
            if result is not None:
                job.result = result
            if error_message is not None:
                job.error_message = error_message
            await self.db.flush()
        return job

    async def get_by_creator(self, user_id: int, skip: int = 0, limit: int = 100) -> Sequence[BackgroundJob]:
        """Fetch background jobs created by a specific user."""
        query = (
            select(BackgroundJob)
            .where(BackgroundJob.created_by == user_id)
            .order_by(BackgroundJob.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await self.db.execute(query)
        return result.scalars().all()


JobRepository = BackgroundJobRepository
