import asyncio
from sqlalchemy.future import select
from app.core.database import SessionLocal
from app.domain.models import ForecastRun

async def main():
    async with SessionLocal() as db:
        result = await db.execute(select(ForecastRun))
        runs = result.scalars().all()
        print(f"Total runs found: {len(runs)}")
        for r in runs:
            print(f"Run ID: {r.id}, Model: {r.model_name}, Best: {r.is_best_model}, Metrics: {r.metrics}")

if __name__ == "__main__":
    asyncio.run(main())
