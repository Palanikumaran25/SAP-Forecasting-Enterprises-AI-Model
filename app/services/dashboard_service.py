"""
Dashboard Analytics Service  –  Phase 11
-----------------------------------------
Computes KPIs, trend analysis, and period-over-period comparisons
from the financial data stored in PostgreSQL.

All methods return plain Python dicts – serialisable directly to JSON.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models import FinancialData, ForecastRun
from app.domain.enums import ForecastStatus

logger = logging.getLogger(__name__)


class DashboardService:
    """Provides analytics aggregations for the dashboard API."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    # ------------------------------------------------------------------
    # 1. Summary KPIs
    # ------------------------------------------------------------------

    async def get_summary_kpis(self) -> Dict[str, Any]:
        """
        Returns top-level KPIs:
          - total_revenue, total_expenses, net_income
          - record_count
          - forecast_run counts by status
        """
        # Aggregate financial data
        result = await self._db.execute(
            select(
                func.sum(FinancialData.amount).label("total_amount"),
                func.count(FinancialData.id).label("record_count"),
            )
        )
        row = result.one()
        total_amount = float(row.total_amount or 0)
        record_count = int(row.record_count or 0)

        # Forecast run counts by status
        fc_result = await self._db.execute(
            select(ForecastRun.status, func.count(ForecastRun.id).label("count"))
            .group_by(ForecastRun.status)
        )
        status_counts = {r.status.value: r.count for r in fc_result}

        return {
            "total_amount": round(total_amount, 2),
            "record_count": record_count,
            "forecast_run_counts": status_counts,
        }

    # ------------------------------------------------------------------
    # 2. Monthly trend
    # ------------------------------------------------------------------

    async def get_monthly_trend(
        self,
        fiscal_year: Optional[int] = None,
        gl_account: Optional[str] = None,
        cost_center: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Monthly aggregated amounts – useful for trend charts.
        Filtered optionally by fiscal year, GL account, or cost center.
        """
        query = select(
            FinancialData.fiscal_year,
            FinancialData.fiscal_period,
            func.sum(FinancialData.amount).label("total"),
            func.count(FinancialData.id).label("count"),
        )
        if fiscal_year:
            query = query.where(FinancialData.fiscal_year == fiscal_year)
        if gl_account:
            query = query.where(FinancialData.gl_account == gl_account)
        if cost_center:
            query = query.where(FinancialData.cost_center == cost_center)

        query = query.group_by(
            FinancialData.fiscal_year, FinancialData.fiscal_period
        ).order_by(FinancialData.fiscal_year, FinancialData.fiscal_period)

        result = await self._db.execute(query)
        rows = result.all()

        return [
            {
                "fiscal_year": r.fiscal_year,
                "fiscal_period": r.fiscal_period,
                "period_label": f"{r.fiscal_year}-{str(r.fiscal_period).zfill(2)}",
                "total_amount": round(float(r.total), 2),
                "record_count": r.count,
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # 3. GL Account breakdown
    # ------------------------------------------------------------------

    async def get_gl_account_breakdown(
        self, top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """Top-N GL accounts by absolute total amount."""
        query = (
            select(
                FinancialData.gl_account,
                func.sum(FinancialData.amount).label("total"),
                func.count(FinancialData.id).label("count"),
            )
            .group_by(FinancialData.gl_account)
            .order_by(func.abs(func.sum(FinancialData.amount)).desc())
            .limit(top_n)
        )
        result = await self._db.execute(query)
        return [
            {
                "gl_account": r.gl_account,
                "total_amount": round(float(r.total), 2),
                "record_count": r.count,
            }
            for r in result.all()
        ]

    # ------------------------------------------------------------------
    # 4. Cost centre breakdown
    # ------------------------------------------------------------------

    async def get_cost_center_breakdown(
        self, top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """Top-N cost centres by total amount."""
        query = (
            select(
                FinancialData.cost_center,
                func.sum(FinancialData.amount).label("total"),
                func.count(FinancialData.id).label("count"),
            )
            .where(FinancialData.cost_center.isnot(None))
            .group_by(FinancialData.cost_center)
            .order_by(func.abs(func.sum(FinancialData.amount)).desc())
            .limit(top_n)
        )
        result = await self._db.execute(query)
        return [
            {
                "cost_center": r.cost_center,
                "total_amount": round(float(r.total), 2),
                "record_count": r.count,
            }
            for r in result.all()
        ]

    # ------------------------------------------------------------------
    # 5. Best forecast run overview
    # ------------------------------------------------------------------

    async def get_best_forecast_overview(self) -> Optional[Dict[str, Any]]:
        """Return the most recently approved best-model forecast run."""
        query = (
            select(ForecastRun)
            .where(
                ForecastRun.is_best_model == True,  # noqa: E712
                ForecastRun.status == ForecastStatus.APPROVED,
            )
            .order_by(ForecastRun.created_at.desc())
            .limit(1)
        )
        result = await self._db.execute(query)
        run = result.scalar_one_or_none()
        if not run:
            return None
        return {
            "run_id": run.id,
            "model_name": run.model_name,
            "schedule_type": run.schedule_type.value,
            "metrics": run.metrics,
            "forecast_values": run.forecast_values,
            "approved_at": run.approved_at.isoformat() if run.approved_at else None,
        }

    # ------------------------------------------------------------------
    # 6. Period-over-period YoY comparison
    # ------------------------------------------------------------------

    async def get_yoy_comparison(
        self,
        current_year: int,
        previous_year: int,
    ) -> Dict[str, Any]:
        """Year-over-year comparison of monthly totals."""
        trend_current = await self.get_monthly_trend(fiscal_year=current_year)
        trend_previous = await self.get_monthly_trend(fiscal_year=previous_year)

        prev_by_period = {r["fiscal_period"]: r["total_amount"] for r in trend_previous}

        comparison = []
        for row in trend_current:
            period = row["fiscal_period"]
            prev = prev_by_period.get(period, 0.0)
            curr = row["total_amount"]
            change_pct = round(((curr - prev) / prev * 100) if prev != 0 else 0.0, 2)
            comparison.append({
                "fiscal_period": period,
                "period_label": row["period_label"],
                f"amount_{current_year}": curr,
                f"amount_{previous_year}": prev,
                "change_pct": change_pct,
            })

        return {
            "current_year": current_year,
            "previous_year": previous_year,
            "comparison": comparison,
        }
