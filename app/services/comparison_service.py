"""
Forecast Comparison Service
----------------------------
Computes accuracy metrics (MAE, RMSE, MAPE) for all completed ForecastRuns
that belong to the same job, then:
  1. Updates each ForecastRun.metrics in PostgreSQL
  2. Marks the best-performing run (lowest MAPE) with is_best_model=True
  3. Returns a structured comparison report

Metric definitions
------------------
MAE   = mean absolute error
RMSE  = root mean squared error
MAPE  = mean absolute percentage error  (%)
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models import ForecastRun
from app.repositories.forecast_repository import ForecastRunRepository

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric calculations
# ---------------------------------------------------------------------------

def _mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(math.sqrt(np.mean((actual - predicted) ** 2)))


def _mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """MAPE – skips zero actual values to avoid division-by-zero."""
    mask = actual != 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def _r2(actual: np.ndarray, predicted: np.ndarray) -> Optional[float]:
    """Return R² score, or None when it is mathematically undefined.

    R² is undefined when all actual values are identical (ss_tot == 0)
    or when fewer than 2 observations are supplied.  Returning None (rather
    than a synthetic 0.0) lets callers distinguish a truly-computed value
    from an uncomputable one.
    """
    if len(actual) < 2:
        return None
    ss_res = float(np.sum((actual - predicted) ** 2))
    ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
    if ss_tot == 0.0:
        return None  # undefined: all actuals are equal, denominator is zero
    return float(1.0 - ss_res / ss_tot)


def compute_metrics(
    actual: Sequence[float],
    predicted: Sequence[float],
) -> Dict[str, Any]:
    """Return MAE, RMSE, MAPE, R2 for a single model's in-sample or hold-out predictions.

    R2 may be None when it is mathematically undefined (fewer than 2 points
    or all actual values are identical).  All other metrics are always floats.
    """
    a = np.array(actual, dtype=float)
    p = np.array(predicted, dtype=float)

    # helper for checking nan/inf before rounding — leaves None unchanged
    def safe_round(val: Optional[float], decimals: int) -> Optional[float]:
        if val is None:
            return None
        if math.isnan(val) or math.isinf(val):
            return None
        return round(val, decimals)

    return {
        "MAE":  safe_round(_mae(a, p),  4),
        "RMSE": safe_round(_rmse(a, p), 4),
        "MAPE": safe_round(_mape(a, p), 4),
        "R2":   safe_round(_r2(a, p),   4),
    }


# ---------------------------------------------------------------------------
# Comparison service
# ---------------------------------------------------------------------------

class ForecastComparisonService:
    """
    Loads a group of ForecastRun rows, computes hold-out metrics,
    ranks models, and marks the winner.
    """

    def __init__(self, db: AsyncSession) -> None:
        self._repo = ForecastRunRepository(db)
        self._db = db

    async def compare_and_rank(
        self,
        run_ids: List[int],
        actual_series: pd.Series,
        hold_out_periods: int = 3,
    ) -> Dict[str, Any]:
        """
        Compare multiple ForecastRun rows against *actual_series*.

        Parameters
        ----------
        run_ids          : IDs of ForecastRun rows to evaluate.
        actual_series    : Full monthly actual series (DatetimeIndex, float values).
        hold_out_periods : Last N periods used as the evaluation window.

        Returns a ranked comparison report.
        """
        if hold_out_periods >= len(actual_series):
            raise ValueError("hold_out_periods must be less than the series length.")

        actual_holdout = actual_series.values[-hold_out_periods:]
        train_series = actual_series.iloc[:-hold_out_periods]

        results: List[Dict[str, Any]] = []
        best_run_id: Optional[int] = None
        best_mape: float = float("inf")

        for run_id in run_ids:
            run = await self._repo.get(run_id)
            if run is None:
                logger.warning("ForecastRun %d not found – skipping.", run_id)
                continue

            # Compute hold-out predictions by re-fitting the model on train_series
            predicted = None
            try:
                from app.services.forecasting_service import get_forecaster
                forecaster = get_forecaster(run.model_name)
                
                fit_kwargs = {}
                n = len(train_series)
                if run.model_name in ("XGBoost", "RandomForest"):
                    safe_lags = max(1, n // 2 - 1)
                    fit_kwargs["n_lags"] = safe_lags
                elif run.model_name == "ARIMA":
                    if n < 12:
                        fit_kwargs["order"] = (1, 0, 0)

                test_preds_raw, _ = forecaster.fit_predict(train_series, horizon=hold_out_periods, **fit_kwargs)
                predicted = np.array([p["value"] for p in test_preds_raw], dtype=float)
            except Exception as exc:
                logger.warning("Could not compute hold-out predictions for run %d model %s: %s", run.id, run.model_name, exc)

            if predicted is None or len(predicted) != len(actual_holdout):
                logger.warning("Could not extract predictions for run %d – skipping.", run_id)
                continue

            metrics = compute_metrics(actual_holdout, predicted)
            run.metrics = metrics
            await self._db.flush()

            results.append({
                "run_id": run.id,
                "model_name": run.model_name,
                **metrics,
            })

            if metrics["MAPE"] < best_mape:
                best_mape = metrics["MAPE"]
                best_run_id = run.id

        # Mark best model
        for run_id in run_ids:
            run = await self._repo.get(run_id)
            if run:
                run.is_best_model = run.id == best_run_id
                await self._db.flush()

        await self._db.commit()

        # Sort by MAPE ascending
        results.sort(key=lambda x: x.get("MAPE", float("inf")))

        return {
            "best_model_run_id": best_run_id,
            "hold_out_periods": hold_out_periods,
            "ranking": results,
        }


def _extract_holdout_predictions(
    run: ForecastRun,
    hold_out_periods: int,
) -> Optional[np.ndarray]:
    """
    Fallback extraction for legacy support.
    """
    fv = run.forecast_values or []
    if len(fv) < hold_out_periods:
        return None
    values = [entry["value"] for entry in fv[:hold_out_periods]]
    return np.array(values, dtype=float)


# ---------------------------------------------------------------------------
# Convenience function for use by comparison API endpoint
# ---------------------------------------------------------------------------

async def compare_forecast_runs(
    db: AsyncSession,
    run_ids: List[int],
    actual_series: pd.Series,
    hold_out_periods: int = 3,
) -> Dict[str, Any]:
    """Top-level helper used by the API router."""
    svc = ForecastComparisonService(db)
    return await svc.compare_and_rank(run_ids, actual_series, hold_out_periods)
