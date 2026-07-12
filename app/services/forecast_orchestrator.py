"""
Forecast Orchestrator
---------------------
Runs all registered forecasting models as a background job.
Responsible for:
  1. Loading financial data from PostgreSQL
  2. Preprocessing / aggregating into a monthly time-series
  3. Running each model wrapper (ARIMA, ETS, RF, XGBoost)
  4. Persisting ForecastRun records per model
  5. Updating BackgroundJob status throughout execution
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import AsyncSessionLocal
from app.domain.enums import ForecastSchedule, ForecastStatus, JobStatus, JobType
from app.domain.models import BackgroundJob, ForecastRun
from app.repositories.data_repository import FinancialDataRepository
from app.repositories.forecast_repository import ForecastRunRepository
from app.repositories.job_repository import JobRepository
from app.services.forecasting_service import MODEL_REGISTRY, ForecastResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry-point called by FastAPI BackgroundTasks
# ---------------------------------------------------------------------------

async def run_forecast_job(
    job_id: int,
    created_by: int,
    schedule_type: str = ForecastSchedule.AD_HOC.value,
    horizon: int = 12,
    gl_account_filter: Optional[str] = None,
    cost_center_filter: Optional[str] = None,
) -> None:
    """
    Background entry-point.  Executes all models and persists results.
    Called from the forecasting API router via FastAPI BackgroundTasks.
    """
    async with AsyncSessionLocal() as db:
        job_repo = JobRepository(db)
        forecast_repo = ForecastRunRepository(db)
        data_repo = FinancialDataRepository(db)

        # --- Mark job as processing ---
        await _update_job(job_repo, db, job_id, JobStatus.PROCESSING)

        try:
            # 1. Load financial data
            records = await data_repo.get_multi(skip=0, limit=100_000)
            if not records:
                raise ValueError("No financial data available to build a forecast series.")

            # 2. Build monthly time-series
            series = _build_monthly_series(
                records,
                gl_account_filter=gl_account_filter,
                cost_center_filter=cost_center_filter,
            )
            if len(series) < 6:
                raise ValueError(
                    f"Insufficient data points ({len(series)}) for forecasting. "
                    "Minimum 6 monthly observations required."
                )

            # 3. Resolve schedule enum
            schedule_enum = _resolve_schedule(schedule_type)

            # 4. Run each model
            run_ids: List[int] = []
            for model_name, forecaster in MODEL_REGISTRY.items():
                try:
                    logger.info("Running model: %s", model_name)

                    # Adaptive kwargs per model type
                    fit_kwargs: Dict[str, Any] = {}
                    n = len(series)
                    if model_name in ("XGBoost", "RandomForest"):
                        safe_lags = max(1, n // 2 - 1)
                        fit_kwargs["n_lags"] = safe_lags
                    elif model_name == "ARIMA" and n < 12:
                        fit_kwargs["order"] = (1, 0, 0)

                    forecast_values, parameters = forecaster.fit_predict(
                        series, horizon=horizon, **fit_kwargs
                    )
                    run = ForecastRun(
                        model_name=model_name,
                        schedule_type=schedule_enum,
                        parameters=parameters,
                        metrics={},          # filled by comparison service below
                        forecast_values=forecast_values,
                        status=ForecastStatus.DRAFT,
                        created_by=created_by,
                    )
                    db.add(run)
                    await db.flush()        # get run.id
                    run_ids.append(run.id)
                    logger.info("Persisted ForecastRun id=%d model=%s", run.id, model_name)
                except Exception as model_exc:  # noqa: BLE001
                    logger.error("Model %s failed: %s", model_name, model_exc, exc_info=True)

            await db.commit()

            # 4b. Compute hold-out metrics for all runs via comparison service
            if run_ids:
                try:
                    from app.services.comparison_service import compare_forecast_runs
                    hold_out = max(2, int(len(series) * 0.20))
                    comparison_result = await compare_forecast_runs(
                        db, run_ids, series, hold_out_periods=hold_out
                    )
                    logger.info(
                        "Comparison complete — best model run_id=%s",
                        comparison_result.get("best_model_run_id"),
                    )
                except Exception as cmp_exc:
                    logger.error("Comparison service failed: %s", cmp_exc, exc_info=True)

            # 5. Mark job complete
            result_payload: Dict[str, Any] = {
                "forecast_run_ids": run_ids,
                "models_executed": list(MODEL_REGISTRY.keys()),
                "horizon": horizon,
                "series_length": len(series),
            }
            await _update_job(
                job_repo, db, job_id, JobStatus.COMPLETED, result=result_payload
            )

        except Exception as exc:  # noqa: BLE001
            logger.error("Forecast job %d failed: %s", job_id, exc, exc_info=True)
            await _update_job(
                job_repo, db, job_id, JobStatus.FAILED, error_message=str(exc)
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_monthly_series(
    records: Any,
    gl_account_filter: Optional[str],
    cost_center_filter: Optional[str],
) -> pd.Series:
    """Aggregate FinancialData records into a monthly amount series."""
    rows = []
    for rec in records:
        if gl_account_filter and rec.gl_account != gl_account_filter:
            continue
        if cost_center_filter and rec.cost_center != cost_center_filter:
            continue
        rows.append({"date": rec.posting_date, "amount": float(rec.amount)})

    if not rows:
        raise ValueError("No records match the provided filters.")

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    monthly = (
        df.set_index("date")
        .resample("MS")["amount"]
        .sum()
        .sort_index()
    )
    return monthly


def _resolve_schedule(schedule_type: str) -> ForecastSchedule:
    """Safely convert a raw string to ForecastSchedule enum."""
    for member in ForecastSchedule:
        if member.value == schedule_type or member.name == schedule_type:
            return member
    return ForecastSchedule.AD_HOC


async def _update_job(
    job_repo: JobRepository,
    db: AsyncSession,
    job_id: int,
    status: JobStatus,
    result: Optional[Dict[str, Any]] = None,
    error_message: Optional[str] = None,
) -> None:
    job = await job_repo.get(job_id)
    if job:
        job.status = status
        if result is not None:
            job.result = result
        if error_message is not None:
            job.error_message = error_message
        await db.commit()
