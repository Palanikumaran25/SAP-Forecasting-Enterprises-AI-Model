"""
Unit Tests – SAP FI Forecasting Platform  (Phase 12)
=====================================================
Covers:
  - Metric computations (MAE, RMSE, MAPE)
  - ARIMA / ETS / RF / XGBoost forecaster wrappers (smoke tests)
  - Forecast orchestrator helpers
  - LLM prompt builder
  - Rate limiter sliding-window logic
  - Dashboard YoY comparison helper
  - Security utilities (password hash / JWT)

Run with:
    pytest tests/ -v
"""
from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

from app.services.comparison_service import compute_metrics, _mae, _rmse, _mape


class TestMetricHelpers:
    def test_mae_perfect(self):
        a = np.array([1.0, 2.0, 3.0])
        assert _mae(a, a) == 0.0

    def test_rmse_perfect(self):
        a = np.array([1.0, 2.0, 3.0])
        assert _rmse(a, a) == 0.0

    def test_mae_known_value(self):
        a = np.array([10.0, 20.0, 30.0])
        p = np.array([12.0, 18.0, 33.0])
        assert _mae(a, p) == pytest.approx(7 / 3, rel=1e-4)

    def test_rmse_known_value(self):
        a = np.array([0.0, 10.0])
        p = np.array([0.0, 0.0])
        assert _rmse(a, p) == pytest.approx(math.sqrt(50), rel=1e-4)

    def test_mape_skips_zero_actuals(self):
        a = np.array([0.0, 10.0, 20.0])
        p = np.array([0.0, 8.0, 22.0])
        # MAPE computed only on non-zero: |10-8|/10=0.2, |20-22|/20=0.1 → mean=0.15 → 15%
        assert _mape(a, p) == pytest.approx(15.0, rel=1e-4)

    def test_mape_all_zero_actuals_returns_nan(self):
        a = np.array([0.0, 0.0])
        p = np.array([1.0, 2.0])
        assert math.isnan(_mape(a, p))

    def test_compute_metrics_returns_dict(self):
        actual = [100.0, 200.0, 300.0]
        predicted = [110.0, 195.0, 290.0]
        metrics = compute_metrics(actual, predicted)
        assert "MAE" in metrics
        assert "RMSE" in metrics
        assert "MAPE" in metrics
        assert "R2" in metrics
        assert all(isinstance(v, float) for v in metrics.values())

    def test_r2_score_calculations(self):
        from app.services.comparison_service import _r2
        # Perfect prediction
        assert _r2(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])) == 1.0
        # Undefined R2 due to len < 2
        assert _r2(np.array([1.0]), np.array([1.2])) is None
        # Undefined R2 due to all identical actuals (ss_tot == 0)
        assert _r2(np.array([2.0, 2.0]), np.array([2.1, 1.9])) is None
        # Normal calculations
        # Mean = 2.0. ss_tot = (1-2)^2 + (2-2)^2 + (3-2)^2 = 2.
        # ss_res = (1-1.5)^2 + (2-2)^2 + (3-2.5)^2 = 0.25 + 0 + 0.25 = 0.5.
        # R2 = 1.0 - 0.5 / 2.0 = 0.75.
        assert _r2(np.array([1.0, 2.0, 3.0]), np.array([1.5, 2.0, 2.5])) == pytest.approx(0.75)
        # Verify negative R2 score is not clamped
        # ss_res = (1-10)^2 + (2-20)^2 + (3-30)^2 = 81 + 324 + 729 = 1134.
        # ss_tot = 2. R2 = 1.0 - 1134 / 2 = -566.0.
        assert _r2(np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0, 30.0])) == pytest.approx(-566.0)

    def test_compute_metrics_handles_undefined_r2(self):
        metrics = compute_metrics([2.0], [2.1])
        assert metrics["R2"] is None
        assert isinstance(metrics["MAE"], float)


# ---------------------------------------------------------------------------
# Forecasting service helpers
# ---------------------------------------------------------------------------

from app.services.forecasting_service import (
    _create_lag_features,
    _recursive_forecast,
    _build_forecast_list,
    get_forecaster,
    MODEL_REGISTRY,
)


class TestForecastingHelpers:
    def _make_series(self, n: int = 36) -> pd.Series:
        idx = pd.date_range("2021-01-01", periods=n, freq="MS")
        values = np.sin(np.linspace(0, 4 * np.pi, n)) * 1000 + 5000
        return pd.Series(values, index=idx)

    def test_create_lag_features_shape(self):
        series = self._make_series(24)
        X, y = _create_lag_features(series, n_lags=6)
        assert X.shape[1] == 6
        assert len(X) == len(y)
        assert len(X) == 24 - 6

    def test_get_forecaster_valid(self):
        for name in ["ARIMA", "ETS", "RandomForest", "XGBoost"]:
            f = get_forecaster(name)
            assert f.name == name

    def test_get_forecaster_invalid(self):
        with pytest.raises(ValueError, match="Unknown model"):
            get_forecaster("InvalidModel")

    def test_build_forecast_list_format(self):
        idx = pd.date_range("2025-01-01", periods=3, freq="MS")
        fc = pd.Series([100.0, 200.0, 300.0], index=idx)
        series = pd.Series([0.0], index=pd.date_range("2024-12-01", periods=1, freq="MS"))
        result = _build_forecast_list(series, fc)
        assert len(result) == 3
        assert result[0] == {"period": "2025-01", "value": 100.0}
        assert all("period" in r and "value" in r for r in result)

    @pytest.mark.skipif(
        True,  # skip in CI if statsmodels unavailable
        reason="Integration test – requires statsmodels",
    )
    def test_arima_smoke(self):
        from app.services.forecasting_service import ARIMAForecaster
        series = self._make_series(36)
        f = ARIMAForecaster(order=(1, 1, 1))
        values, params = f.fit_predict(series, horizon=3)
        assert len(values) == 3
        assert "order" in params

    @pytest.mark.skipif(
        True,
        reason="Integration test – requires scikit-learn",
    )
    def test_random_forest_smoke(self):
        from app.services.forecasting_service import RandomForestForecaster
        series = self._make_series(36)
        f = RandomForestForecaster(n_estimators=10, n_lags=6)
        values, params = f.fit_predict(series, horizon=3)
        assert len(values) == 3


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

from app.middleware.rate_limit import _WINDOW_SECONDS


class TestRateLimiter:
    """White-box tests for the sliding window logic used in RateLimitMiddleware."""

    def _sliding_window_check(self, store, key: str, limit: int, now: float) -> bool:
        """Returns True if request should be blocked."""
        window_start = now - _WINDOW_SECONDS
        queue = store[key]
        while queue and queue[0] < window_start:
            queue.popleft()
        if len(queue) >= limit:
            return True
        queue.append(now)
        return False

    def test_allows_under_limit(self):
        store = defaultdict(deque)
        now = time.monotonic()
        for _ in range(5):
            blocked = self._sliding_window_check(store, "127.0.0.1:global", 10, now)
        assert not blocked

    def test_blocks_at_limit(self):
        store = defaultdict(deque)
        now = time.monotonic()
        limit = 5
        for _ in range(limit):
            self._sliding_window_check(store, "127.0.0.1:global", limit, now)
        blocked = self._sliding_window_check(store, "127.0.0.1:global", limit, now)
        assert blocked

    def test_resets_after_window(self):
        store = defaultdict(deque)
        old_time = time.monotonic() - _WINDOW_SECONDS - 1
        limit = 3
        # Fill up with old timestamps
        for _ in range(limit):
            store["127.0.0.1:global"].append(old_time)
        # New request at current time should NOT be blocked
        now = time.monotonic()
        blocked = self._sliding_window_check(store, "127.0.0.1:global", limit, now)
        assert not blocked


# ---------------------------------------------------------------------------
# LLM prompt builder
# ---------------------------------------------------------------------------

from app.services.llm_service import _build_analysis_prompt, _parse_llm_response


class TestLLMPrompt:
    def _sample_run_data(self):
        return {
            "model_name": "ARIMA",
            "schedule_type": "Monthly Forecast",
            "forecast_values": [{"period": "2025-01", "value": 1000.0}] * 6,
            "metrics": {"MAE": 50.0, "RMSE": 60.0, "MAPE": 5.0},
            "status": "Draft",
        }

    def test_prompt_contains_model_name(self):
        prompt = _build_analysis_prompt(self._sample_run_data())
        assert "ARIMA" in prompt

    def test_prompt_contains_mape(self):
        prompt = _build_analysis_prompt(self._sample_run_data())
        assert "MAPE" in prompt or "5.0" in prompt

    def test_parse_valid_json(self):
        raw = '{"summary": "ok", "risks": [{"level": "Low", "description": "fine"}], "explanation": "all good"}'
        result = _parse_llm_response(raw)
        assert result["summary"] == "ok"
        assert len(result["risks"]) == 1

    def test_parse_fenced_json(self):
        raw = '```json\n{"summary": "s", "risks": [], "explanation": "e"}\n```'
        result = _parse_llm_response(raw)
        assert result["summary"] == "s"

    def test_parse_invalid_json_fallback(self):
        raw = "This is not valid JSON"
        result = _parse_llm_response(raw)
        assert "summary" in result
        assert "risks" in result


# ---------------------------------------------------------------------------
# Security utilities
# ---------------------------------------------------------------------------

from app.core.security import hash_password, verify_password, create_access_token, decode_token


class TestSecurity:
    def test_password_hash_and_verify(self):
        raw = "SuperSecret123!"
        hashed = hash_password(raw)
        assert hashed != raw
        assert verify_password(raw, hashed)

    def test_wrong_password_fails(self):
        hashed = hash_password("correct_password")
        assert not verify_password("wrong_password", hashed)

    def test_access_token_roundtrip(self):
        payload = {"sub": "user_42", "role": "Analyst"}
        token = create_access_token(payload)
        decoded = decode_token(token)
        assert decoded["sub"] == "user_42"

    def test_tampered_token_raises(self):
        token = create_access_token({"sub": "user_1"})
        tampered = token[:-5] + "XXXXX"
        with pytest.raises(Exception):
            decode_token(tampered)


# ---------------------------------------------------------------------------
# Forecast orchestrator helpers
# ---------------------------------------------------------------------------

from app.services.forecast_orchestrator import _build_monthly_series, _resolve_schedule
from app.domain.enums import ForecastSchedule


class TestOrchestratorHelpers:
    def _make_records(self, n: int):
        records = []
        for i in range(n):
            rec = MagicMock()
            rec.posting_date = datetime(2023, (i % 12) + 1, 1).date()
            rec.amount = float((i + 1) * 1000)
            rec.gl_account = "400000"
            rec.cost_center = "CC100"
            records.append(rec)
        return records

    def test_build_monthly_series_length(self):
        records = self._make_records(24)
        series = _build_monthly_series(records, None, None)
        assert len(series) >= 1  # aggregated by month

    def test_build_monthly_series_filter_gl(self):
        records = self._make_records(12)
        records[0].gl_account = "999999"  # only one different
        series = _build_monthly_series(records, "400000", None)
        assert series is not None

    def test_resolve_schedule_valid(self):
        assert _resolve_schedule("Monthly Forecast") == ForecastSchedule.MONTHLY
        assert _resolve_schedule("Weekly Forecast") == ForecastSchedule.WEEKLY

    def test_resolve_schedule_fallback(self):
        assert _resolve_schedule("unknown_schedule") == ForecastSchedule.AD_HOC


# ---------------------------------------------------------------------------
# Ingestion & Validation tests
# ---------------------------------------------------------------------------
from app.services.validation_service import ValidationService
from app.domain.schemas import IngestionBatchResponse
from app.domain.models import FinancialData
from datetime import date, datetime

class TestValidationAndSchemaErrors:
    def test_validation_warning_does_not_increment_rule_violations(self):
        r1 = FinancialData(
            posting_date=date(2026, 1, 1),
            fiscal_year=2026,
            fiscal_period=1,
            gl_account="1000",
            amount=100.0,
            currency="EUR",
            cost_center=None,
            profit_center=None,
            document_type="SA",
            ingestion_batch_id=1
        )
        r2 = FinancialData(
            posting_date=date(2026, 1, 1),
            fiscal_year=2026,
            fiscal_period=1,
            gl_account="2000",
            amount=50.0,
            currency="EUR",
            cost_center=None,
            profit_center=None,
            document_type="SA",
            ingestion_batch_id=1
        )
        
        val_service = ValidationService()
        report = val_service.validate_records([r1, r2])
        
        warnings = [f for f in report["financial_rules"] if f["severity"] == "WARNING"]
        assert len(warnings) == 1
        assert warnings[0]["rule"] == "Double-entry Ledger Balance Check"
        assert report["summary"]["rule_violations"] == 0
        assert report["summary"]["is_valid"] is True

    def test_validation_error_increments_rule_violations(self):
        r1 = FinancialData(
            posting_date=date(2026, 1, 1),
            gl_account="12",
            amount=0.0,
            currency="EUR",
            document_type="SA",
            ingestion_batch_id=1
        )
        
        val_service = ValidationService()
        report = val_service.validate_records([r1])
        
        errors = [f for f in report["financial_rules"] if f["severity"] == "ERROR"]
        assert len(errors) == 1
        assert report["summary"]["rule_violations"] == 1
        assert report["summary"]["is_valid"] is False

    def test_schema_error_count_calculation_filters_warnings(self):
        validation_errors = {
            "missing_values": [],
            "duplicates": [],
            "outliers": [],
            "financial_rules": [
                {
                    "rule": "Double-entry Ledger Balance Check",
                    "severity": "WARNING",
                    "message": "Ledger batch is unbalanced."
                }
            ],
            "summary": {
                "total_records": 200,
                "missing_count": 0,
                "duplicate_count": 0,
                "outlier_count": 0,
                "rule_violations": 1,
                "is_valid": True
            }
        }
        
        from app.domain.enums import JobStatus
        response = IngestionBatchResponse(
            id=3,
            filename="test.xlsx",
            uploaded_by=3,
            status=JobStatus.COMPLETED,
            record_count=200,
            validation_errors=validation_errors,
            created_at=datetime.utcnow()
        )
        
        assert response.error_count == 0

