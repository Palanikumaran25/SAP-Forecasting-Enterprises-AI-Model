"""
Forecasting Engine Service
--------------------------
Provides a unified interface for all supported forecasting models:
  - ARIMA  (statsmodels)
  - ETS / Exponential Smoothing  (statsmodels)
  - Random Forest  (scikit-learn)
  - XGBoost  (xgboost)

Each model wrapper implements the `BaseForecaster` protocol so the
comparison service can call them interchangeably.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared result type
# ---------------------------------------------------------------------------

class ForecastResult:
    """Container returned by every model wrapper."""

    def __init__(
        self,
        model_name: str,
        forecast_values: List[Dict[str, Any]],   # [{"period": "2024-01", "value": 123.45}, ...]
        parameters: Dict[str, Any],
        metrics: Dict[str, float],               # MAE, RMSE, MAPE filled by comparison layer
        is_best_model: bool = False,
    ) -> None:
        self.model_name = model_name
        self.forecast_values = forecast_values
        self.parameters = parameters
        self.metrics = metrics
        self.is_best_model = is_best_model

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "forecast_values": self.forecast_values,
            "parameters": self.parameters,
            "metrics": self.metrics,
            "is_best_model": self.is_best_model,
        }


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseForecaster(ABC):
    """Protocol every model wrapper must satisfy."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def fit_predict(
        self,
        series: pd.Series,
        horizon: int,
        **kwargs: Any,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Fit on *series* and forecast *horizon* steps ahead.

        Returns
        -------
        forecast_values : list of {period, value} dicts
        parameters      : model hyper-params / fitted values (serialisable)
        """
        ...


# ---------------------------------------------------------------------------
# ARIMA wrapper
# ---------------------------------------------------------------------------

class ARIMAForecaster(BaseForecaster):
    """ARIMA via statsmodels.  Auto-selects order with pmdarima if available,
    otherwise falls back to sensible defaults (1,1,1)."""

    name = "ARIMA"

    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1)) -> None:
        self._order = order

    def fit_predict(
        self,
        series: pd.Series,
        horizon: int,
        **kwargs: Any,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        try:
            from statsmodels.tsa.arima.model import ARIMA  # type: ignore
        except ImportError as exc:
            raise RuntimeError("statsmodels is required for ARIMA forecasting") from exc

        order = kwargs.get("order", self._order)
        if len(series) < 12 and order == self._order:
            order = (1, 0, 0)

        logger.info("Fitting ARIMA%s on %d observations", order, len(series))

        fitted = None
        last_exc = None
        # Try fitting with requested order, fallback progressively on failure
        for attempt_order in [order, (1, 0, 0), (0, 1, 0), (0, 0, 1)]:
            try:
                model = ARIMA(series, order=attempt_order)
                fitted = model.fit()
                order = attempt_order
                break
            except Exception as e:
                last_exc = e
                logger.warning("ARIMA%s fit failed: %s", attempt_order, e)

        if fitted is None:
            raise RuntimeError(f"ARIMA fitting failed on all fallbacks. Last error: {last_exc}")

        forecast = fitted.forecast(steps=horizon)
        forecast_values = _build_forecast_list(series, forecast)
        parameters = {
            "order": list(order),
            "aic": round(float(fitted.aic), 4) if hasattr(fitted, "aic") else 0.0,
            "bic": round(float(fitted.bic), 4) if hasattr(fitted, "bic") else 0.0,
        }
        return forecast_values, parameters


# ---------------------------------------------------------------------------
# ETS wrapper
# ---------------------------------------------------------------------------

class ETSForecaster(BaseForecaster):
    """Exponential Smoothing (Holt-Winters) via statsmodels."""

    name = "ETS"

    def __init__(
        self,
        trend: Optional[str] = "add",
        seasonal: Optional[str] = "add",
        seasonal_periods: int = 12,
    ) -> None:
        self._trend = trend
        self._seasonal = seasonal
        self._seasonal_periods = seasonal_periods

    def fit_predict(
        self,
        series: pd.Series,
        horizon: int,
        **kwargs: Any,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing  # type: ignore
        except ImportError as exc:
            raise RuntimeError("statsmodels is required for ETS forecasting") from exc

        trend = kwargs.get("trend", self._trend)
        seasonal = kwargs.get("seasonal", self._seasonal)
        sp = kwargs.get("seasonal_periods", self._seasonal_periods)

        # ETS requires at least 2 full seasonal cycles
        if len(series) < sp * 2:
            seasonal = None
        # ETS with trend requires at least 4 observations
        if len(series) < 4:
            trend = None

        logger.info(
            "Fitting ETS (trend=%s, seasonal=%s, sp=%d) on %d observations",
            trend, seasonal, sp, len(series),
        )

        fitted = None
        try:
            model = ExponentialSmoothing(
                series,
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=sp if seasonal else None,
                initialization_method="estimated",
            )
            fitted = model.fit(optimized=True)
        except Exception as e:
            logger.warning("ETS fit with trend=%s failed: %s. Falling back to Simple Exponential Smoothing.", trend, e)
            model = ExponentialSmoothing(
                series,
                trend=None,
                seasonal=None,
                initialization_method="estimated",
            )
            fitted = model.fit(optimized=True)
            trend = None
            seasonal = None

        forecast = fitted.forecast(horizon)
        forecast_values = _build_forecast_list(series, forecast)
        parameters = {
            "trend": trend,
            "seasonal": seasonal,
            "seasonal_periods": sp,
            "alpha": round(float(fitted.params.get("smoothing_level", 0)), 4),
            "beta": round(float(fitted.params.get("smoothing_trend", 0)), 4),
            "gamma": round(float(fitted.params.get("smoothing_seasonal", 0)), 4),
        }
        return forecast_values, parameters


# ---------------------------------------------------------------------------
# Random Forest wrapper
# ---------------------------------------------------------------------------

class RandomForestForecaster(BaseForecaster):
    """Random Forest regressor with lag-feature engineering."""

    name = "RandomForest"

    def __init__(self, n_estimators: int = 200, max_depth: int = 8, n_lags: int = 12) -> None:
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._n_lags = n_lags

    def fit_predict(
        self,
        series: pd.Series,
        horizon: int,
        **kwargs: Any,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        try:
            from sklearn.ensemble import RandomForestRegressor  # type: ignore
        except ImportError as exc:
            raise RuntimeError("scikit-learn is required for Random Forest forecasting") from exc

        n_lags = kwargs.get("n_lags", self._n_lags)
        n_estimators = kwargs.get("n_estimators", self._n_estimators)
        max_depth = kwargs.get("max_depth", self._max_depth)

        # Adaptively adjust lags if series is too short
        if len(series) <= n_lags:
            n_lags = max(1, len(series) // 2 - 1)

        logger.info(
            "Fitting RandomForest (n_estimators=%d, n_lags=%d) on %d observations",
            n_estimators, n_lags, len(series),
        )

        X, y = _create_lag_features(series, n_lags)
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X, y)

        forecast_values = _recursive_forecast(series, model, horizon, n_lags)
        parameters = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "n_lags": n_lags,
            "feature_importances": model.feature_importances_.tolist(),
        }
        return forecast_values, parameters


# ---------------------------------------------------------------------------
# XGBoost wrapper
# ---------------------------------------------------------------------------

class XGBoostForecaster(BaseForecaster):
    """XGBoost regressor with lag-feature engineering."""

    name = "XGBoost"

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        n_lags: int = 12,
    ) -> None:
        self._n_estimators = n_estimators
        self._learning_rate = learning_rate
        self._max_depth = max_depth
        self._n_lags = n_lags

    def fit_predict(
        self,
        series: pd.Series,
        horizon: int,
        **kwargs: Any,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        try:
            import xgboost as xgb  # type: ignore
        except ImportError as exc:
            raise RuntimeError("xgboost is required for XGBoost forecasting") from exc

        n_lags = kwargs.get("n_lags", self._n_lags)
        n_estimators = kwargs.get("n_estimators", self._n_estimators)
        learning_rate = kwargs.get("learning_rate", self._learning_rate)
        max_depth = kwargs.get("max_depth", self._max_depth)

        # Adaptively adjust lags if series is too short
        if len(series) <= n_lags:
            n_lags = max(1, len(series) // 2 - 1)

        logger.info(
            "Fitting XGBoost (n_estimators=%d, lr=%s, n_lags=%d) on %d observations",
            n_estimators, learning_rate, n_lags, len(series),
        )

        X, y = _create_lag_features(series, n_lags)
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42,
            verbosity=0,
        )
        model.fit(X, y)

        forecast_values = _recursive_forecast(series, model, horizon, n_lags)
        parameters = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "n_lags": n_lags,
        }
        return forecast_values, parameters


# ---------------------------------------------------------------------------
# Utility helpers (private)
# ---------------------------------------------------------------------------

def _build_forecast_list(
    series: pd.Series,
    forecast: pd.Series,
) -> List[Dict[str, Any]]:
    """Convert a statsmodels forecast Series to the canonical list format."""
    result: List[Dict[str, Any]] = []
    for ts, val in forecast.items():
        label = ts.strftime("%Y-%m") if hasattr(ts, "strftime") else str(ts)
        result.append({"period": label, "value": round(float(val), 2)})
    return result


def _create_lag_features(
    series: pd.Series, n_lags: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a supervised lag-feature matrix for ML regressors."""
    df = pd.DataFrame({"y": series.values})
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df["y"].shift(lag)
    df.dropna(inplace=True)
    X = df.drop(columns=["y"]).values
    y = df["y"].values
    return X, y


def _recursive_forecast(
    series: pd.Series,
    model: Any,
    horizon: int,
    n_lags: int,
) -> List[Dict[str, Any]]:
    """Iteratively extend the series one step at a time using the fitted model."""
    history = list(series.values)
    forecast_index = pd.date_range(
        start=series.index[-1] + pd.offsets.MonthBegin(1),
        periods=horizon,
        freq="MS",
    )
    result: List[Dict[str, Any]] = []

    for ts in forecast_index:
        lags = history[-n_lags:]
        lags_arr = np.array(lags[::-1]).reshape(1, -1)
        pred = float(model.predict(lags_arr)[0])
        history.append(pred)
        result.append({"period": ts.strftime("%Y-%m"), "value": round(pred, 2)})

    return result


# ---------------------------------------------------------------------------
# Registry – maps model names to their default instances
# ---------------------------------------------------------------------------

MODEL_REGISTRY: Dict[str, BaseForecaster] = {
    "ARIMA": ARIMAForecaster(),
    "ETS": ETSForecaster(),
    "RandomForest": RandomForestForecaster(),
    "XGBoost": XGBoostForecaster(),
}


def get_forecaster(model_name: str) -> BaseForecaster:
    """Resolve a forecaster by name (case-insensitive)."""
    key = model_name.strip()
    forecaster = MODEL_REGISTRY.get(key)
    if forecaster is None:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{key}'. Available: {available}")
    return forecaster


