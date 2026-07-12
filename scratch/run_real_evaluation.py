import asyncio
from sqlalchemy.future import select
from app.core.database import SessionLocal
from app.api.v1.compatibility import _run_and_save_model, _load_monthly_series
from app.domain.models import User

async def main():
    async with SessionLocal() as db:
        # Get a user ID to use for the run
        result = await db.execute(select(User))
        user = result.scalars().first()
        if not user:
            print("No users found in database!")
            return
        
        series = await _load_monthly_series(db, None, None)
        print(f"Loaded series length: {len(series)}")
        
        for model in ["ARIMA", "ETS", "RandomForest", "XGBoost"]:
            print(f"\nRunning model: {model}")
            res = await _run_and_save_model(db, model, series, 12, user.id)
            print(f"Run ID: {res['run_id']}")
            print(f"Metrics: {res['metrics']}")

if __name__ == "__main__":
    asyncio.run(main())
