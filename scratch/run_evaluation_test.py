import asyncio
import pandas as pd
import numpy as np
from sqlalchemy.future import select
from app.core.database import SessionLocal
from app.repositories.data_repository import FinancialDataRepository
from app.services.forecast_orchestrator import _build_monthly_series
from app.services.forecasting_service import MODEL_REGISTRY
from app.services.comparison_service import compute_metrics, _r2

async def main():
    async with SessionLocal() as db:
        data_repo = FinancialDataRepository(db)
        records = await data_repo.get_multi(skip=0, limit=100_000)
        if not records:
            print("No records found in DB!")
            return
        
        series = _build_monthly_series(records, None, None)
        print(f"Total series length: {len(series)}")
        
        for test_size in [1, 2, 3]:
            print(f"\n=================== test_size: {test_size} ===================")
            n = len(series)
            train_series = series.iloc[:n - test_size]
            test_series = series.iloc[n - test_size:]
            test_vals = test_series.values.astype(float)
            print(f"test_vals: {test_vals}")
            
            for model_name, forecaster in MODEL_REGISTRY.items():
                try:
                    fit_kwargs = {}
                    n_train = len(train_series)
                    if model_name in ("XGBoost", "RandomForest"):
                        fit_kwargs["n_lags"] = max(1, n_train // 2 - 1)
                    elif model_name == "ARIMA" and n_train < 12:
                        fit_kwargs["order"] = (1, 0, 0)
                    
                    test_preds_raw, _ = forecaster.fit_predict(train_series, horizon=test_size, **fit_kwargs)
                    pred_vals = np.array([p["value"] for p in test_preds_raw], dtype=float)
                    print(f"  Model: {model_name} | pred_vals: {pred_vals} | R2: {_r2(test_vals, pred_vals)}")
                except Exception as e:
                    print(f"  Model: {model_name} | Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
