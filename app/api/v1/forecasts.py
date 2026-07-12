"""
Forecasting API Router  –  /api/v1/forecasts
---------------------------------------------
Endpoints:
  POST   /run          – Trigger forecast job (all models, background)
  GET    /             – List all forecast runs (paginated)
  GET    /{run_id}     – Get single run detail
  GET    /status/{job_id} – Poll background job status
  PATCH  /{run_id}/approve    – Approve forecast (RBAC: Finance Manager, Admin)
  PATCH  /{run_id}/reject     – Reject forecast  (RBAC: Finance Manager, Admin)
  PATCH  /{run_id}/submit     – Submit draft for approval (Analyst, Admin)
"""
from __future__ import annotations

import logging
from typing import Annotated, Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_active_user, require_roles
from app.core.database import get_db
from app.domain.enums import ForecastSchedule, ForecastStatus, JobStatus, JobType, UserRole
from app.domain.models import BackgroundJob, User
from app.repositories.forecast_repository import ForecastRunRepository
from app.repositories.job_repository import JobRepository
from app.services.forecast_orchestrator import run_forecast_job

router = APIRouter(prefix="/forecasts", tags=["Forecasting"])
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / Response Schemas (local – no circular imports)
# ---------------------------------------------------------------------------

class ForecastRunRequest(BaseModel):
    schedule_type: ForecastSchedule = Field(ForecastSchedule.AD_HOC, description="Forecast schedule")
    horizon: int = Field(12, ge=1, le=36, description="Number of periods ahead to forecast")
    gl_account: Optional[str] = Field(None, description="Optional GL account filter")
    cost_center: Optional[str] = Field(None, description="Optional cost center filter")


class ApprovalRequest(BaseModel):
    comments: Optional[str] = Field(None, max_length=1000, description="Reviewer comments")


class ForecastRunResponse(BaseModel):
    id: int
    model_name: str
    schedule_type: str
    status: str
    horizon: int = 0
    parameters: Dict[str, Any]
    metrics: Dict[str, Any]
    forecast_values: list
    is_best_model: bool
    created_by: int
    approved_by: Optional[int]
    comments: Optional[str]

    model_config = {"from_attributes": True}


class JobStatusResponse(BaseModel):
    job_id: int
    status: str
    result: Optional[Dict[str, Any]]
    error_message: Optional[str]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/run",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger a forecasting job (all models)",
)
async def trigger_forecast(
    request: ForecastRunRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles(UserRole.ANALYST, UserRole.FINANCE_MANAGER, UserRole.ADMIN)),
) -> Dict[str, Any]:
    """
    Enqueues a background job that runs ARIMA, ETS, Random Forest and XGBoost
    against the ingested financial data, then persists a ForecastRun row for
    each model.
    """
    # Create background job record
    job = BackgroundJob(
        job_type=JobType.FORECASTING,
        status=JobStatus.PENDING,
        payload={
            "schedule_type": request.schedule_type.value,
            "horizon": request.horizon,
            "gl_account": request.gl_account,
            "cost_center": request.cost_center,
        },
        created_by=current_user.id,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    # Dispatch background task
    background_tasks.add_task(
        run_forecast_job,
        job_id=job.id,
        created_by=current_user.id,
        schedule_type=request.schedule_type.value,
        horizon=request.horizon,
        gl_account_filter=request.gl_account,
        cost_center_filter=request.cost_center,
    )

    logger.info(
        "Forecast job %d queued by user %d (schedule=%s, horizon=%d)",
        job.id, current_user.id, request.schedule_type.value, request.horizon,
    )
    return {
        "message": "Forecast job queued successfully.",
        "job_id": job.id,
        "schedule_type": request.schedule_type.value,
        "horizon": request.horizon,
    }


@router.get("/status/{job_id}", response_model=JobStatusResponse, summary="Poll background job status")
async def get_job_status(
    job_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> JobStatusResponse:
    job_repo = JobRepository(db)
    job = await job_repo.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return JobStatusResponse(
        job_id=job.id,
        status=job.status.value,
        result=job.result,
        error_message=job.error_message,
    )


@router.get("/", summary="List all forecast runs (paginated)")
async def list_forecast_runs(
    status_filter: Optional[ForecastStatus] = Query(None, alias="status"),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    repo = ForecastRunRepository(db)
    if status_filter:
        runs = await repo.get_runs_by_status(status_filter, skip=skip, limit=limit)
    else:
        runs = await repo.get_all_runs(skip=skip, limit=limit)
    return {
        "total": len(runs),
        "skip": skip,
        "limit": limit,
        "runs": [
            {
                "id": r.id,
                "model_name": r.model_name,
                "schedule_type": r.schedule_type.value,
                "status": r.status.value,
                "is_best_model": r.is_best_model,
                "created_by": r.created_by,
                "approved_by": r.approved_by,
                "created_at": r.created_at.isoformat(),
            }
            for r in runs
        ],
    }


@router.get("/{run_id}", summary="Get a single forecast run with full detail")
async def get_forecast_run(
    run_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    repo = ForecastRunRepository(db)
    run = await repo.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Forecast run not found.")
    return {
        "id": run.id,
        "model_name": run.model_name,
        "schedule_type": run.schedule_type.value,
        "status": run.status.value,
        "parameters": run.parameters,
        "metrics": run.metrics,
        "forecast_values": run.forecast_values,
        "is_best_model": run.is_best_model,
        "created_by": run.created_by,
        "approved_by": run.approved_by,
        "approved_at": run.approved_at.isoformat() if run.approved_at else None,
        "comments": run.comments,
        "created_at": run.created_at.isoformat(),
    }


@router.patch("/{run_id}/submit", summary="Submit draft forecast for approval")
async def submit_for_approval(
    run_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles(UserRole.ANALYST, UserRole.FINANCE_MANAGER, UserRole.ADMIN)),
) -> Dict[str, Any]:
    repo = ForecastRunRepository(db)
    run = await repo.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Forecast run not found.")
    if run.status != ForecastStatus.DRAFT:
        raise HTTPException(
            status_code=400,
            detail=f"Only DRAFT runs can be submitted. Current status: {run.status.value}",
        )
    run = await repo.update_approval_status(run_id, ForecastStatus.PENDING_APPROVAL)
    await db.commit()
    return {"message": "Forecast submitted for approval.", "run_id": run_id, "status": ForecastStatus.PENDING_APPROVAL.value}


@router.patch("/{run_id}/approve", summary="Approve a pending forecast run")
async def approve_forecast(
    run_id: int,
    body: ApprovalRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles(UserRole.FINANCE_MANAGER, UserRole.ADMIN)),
) -> Dict[str, Any]:
    repo = ForecastRunRepository(db)
    run = await repo.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Forecast run not found.")
    if run.status != ForecastStatus.PENDING_APPROVAL:
        raise HTTPException(
            status_code=400,
            detail=f"Only PENDING_APPROVAL runs can be approved. Current status: {run.status.value}",
        )
    await repo.update_approval_status(
        run_id, ForecastStatus.APPROVED, approved_by=current_user.id, comments=body.comments
    )
    await db.commit()
    logger.info("ForecastRun %d approved by user %d", run_id, current_user.id)
    return {"message": "Forecast approved.", "run_id": run_id, "status": ForecastStatus.APPROVED.value}


@router.patch("/{run_id}/reject", summary="Reject a pending forecast run")
async def reject_forecast(
    run_id: int,
    body: ApprovalRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles(UserRole.FINANCE_MANAGER, UserRole.ADMIN)),
) -> Dict[str, Any]:
    repo = ForecastRunRepository(db)
    run = await repo.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Forecast run not found.")
    if run.status != ForecastStatus.PENDING_APPROVAL:
        raise HTTPException(
            status_code=400,
            detail=f"Only PENDING_APPROVAL runs can be rejected. Current status: {run.status.value}",
        )
    await repo.update_approval_status(
        run_id, ForecastStatus.REJECTED, approved_by=current_user.id, comments=body.comments
    )
    await db.commit()
    logger.info("ForecastRun %d rejected by user %d", run_id, current_user.id)
    return {"message": "Forecast rejected.", "run_id": run_id, "status": ForecastStatus.REJECTED.value}
