from typing import Optional, List, Sequence
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from app.repositories.base import BaseRepository
from app.domain.models import IngestionBatch, FinancialData


class IngestionBatchRepository(BaseRepository[IngestionBatch]):
    def __init__(self, db: AsyncSession):
        super().__init__(IngestionBatch, db)

    async def get_all_batches(self, skip: int = 0, limit: int = 100) -> Sequence[IngestionBatch]:
        """Fetch all ingestion batches sorted by upload time descending."""
        query = (
            select(IngestionBatch)
            .order_by(IngestionBatch.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await self.db.execute(query)
        return result.scalars().all()


class FinancialDataRepository(BaseRepository[FinancialData]):
    def __init__(self, db: AsyncSession):
        super().__init__(FinancialData, db)

    async def bulk_create(self, records: List[FinancialData]) -> List[FinancialData]:
        """Add a list of financial data records in bulk."""
        self.db.add_all(records)
        await self.db.flush()
        return records

    async def get_by_batch_id(self, batch_id: int) -> Sequence[FinancialData]:
        """Retrieve all financial records uploaded in a specific batch."""
        query = select(FinancialData).where(FinancialData.ingestion_batch_id == batch_id)
        result = await self.db.execute(query)
        return result.scalars().all()

    async def delete_by_batch_id(self, batch_id: int) -> int:
        """Delete all financial records associated with a specific batch, returning the count deleted."""
        stmt = delete(FinancialData).where(FinancialData.ingestion_batch_id == batch_id)
        result = await self.db.execute(stmt)
        return result.rowcount
