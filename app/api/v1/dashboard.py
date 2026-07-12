"""
Dashboard API Router  –  /api/v1/dashboard
-------------------------------------------
Endpoints:
  GET /kpis                   – Summary KPIs
  GET /trend/monthly          – Monthly trend (with optional filters)
  GET /breakdown/gl-account   – Top-N GL account breakdown
  GET /breakdown/cost-center  – Top-N cost centre breakdown
  GET /forecast/best          – Best approved forecast overview
  GET /yoy                    – Year-over-year comparison
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_active_user
from app.core.database import get_db
from app.domain.models import User
from app.services.dashboard_service import DashboardService

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])
logger = logging.getLogger(__name__)


def _svc(db: AsyncSession) -> DashboardService:
    return DashboardService(db)


@router.get("/kpis", summary="Summary KPIs")
async def summary_kpis(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """Returns total amount, record count and forecast run status counts."""
    return await _svc(db).get_summary_kpis()


@router.get("/trend/monthly", summary="Monthly aggregated amounts (trend)")
async def monthly_trend(
    fiscal_year: Optional[int] = Query(None, description="Filter by fiscal year"),
    gl_account: Optional[str] = Query(None, description="Filter by GL account"),
    cost_center: Optional[str] = Query(None, description="Filter by cost center"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    data = await _svc(db).get_monthly_trend(
        fiscal_year=fiscal_year,
        gl_account=gl_account,
        cost_center=cost_center,
    )
    return {"trend": data, "count": len(data)}


@router.get("/breakdown/gl-account", summary="Top-N GL account breakdown")
async def gl_account_breakdown(
    top_n: int = Query(10, ge=1, le=50, description="Number of top GL accounts"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    data = await _svc(db).get_gl_account_breakdown(top_n=top_n)
    return {"breakdown": data, "count": len(data)}


@router.get("/breakdown/cost-center", summary="Top-N cost centre breakdown")
async def cost_center_breakdown(
    top_n: int = Query(10, ge=1, le=50, description="Number of top cost centres"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    data = await _svc(db).get_cost_center_breakdown(top_n=top_n)
    return {"breakdown": data, "count": len(data)}


@router.get("/forecast/best", summary="Best approved forecast run overview")
async def best_forecast(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    data = await _svc(db).get_best_forecast_overview()
    if data is None:
        return {"message": "No approved best-model forecast run found.", "run": None}
    return {"run": data}


@router.get("/yoy", summary="Year-over-year comparison")
async def yoy_comparison(
    current_year: int = Query(..., description="Current fiscal year"),
    previous_year: int = Query(..., description="Previous fiscal year to compare against"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    return await _svc(db).get_yoy_comparison(
        current_year=current_year,
        previous_year=previous_year,
    )
