"""
LLM Analysis API Router  –  /api/v1/analysis
---------------------------------------------
Endpoints:
  POST  /run/{run_id}          – Trigger LLM analysis (background)
  GET   /{run_id}/analyses     – List all LLM analyses for a run
  GET   /status/{job_id}       – Poll LLM job status
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Path, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_active_user, require_roles
from app.core.database import get_db
from app.domain.enums import JobStatus, JobType, UserRole
from app.domain.models import BackgroundJob, User
from app.repositories.forecast_repository import LLMAnalysisRepository
from app.repositories.job_repository import JobRepository
from app.services.llm_service import run_llm_analysis_job

router = APIRouter(prefix="/analysis", tags=["LLM Analysis"])
logger = logging.getLogger(__name__)


class LLMRunRequest(BaseModel):
    provider: str = Field("gemini", pattern="^(gemini|openai)$", description="LLM provider to use")


class LLMAnalysisResponse(BaseModel):
    id: int
    forecast_run_id: int
    provider: str
    summary: str
    risks_detected: List[Dict[str, Any]]
    explanation: str
    created_at: str

    model_config = {"from_attributes": True}


@router.post(
    "/run/{run_id}",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger LLM analysis for a forecast run",
)
async def trigger_llm_analysis(
    run_id: int = Path(..., description="ForecastRun ID to analyse"),
    body: LLMRunRequest = ...,
    background_tasks: BackgroundTasks = ...,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles(UserRole.ANALYST, UserRole.FINANCE_MANAGER, UserRole.ADMIN)),
) -> Dict[str, Any]:
    # Create job record
    job = BackgroundJob(
        job_type=JobType.LLM_ANALYSIS,
        status=JobStatus.PENDING,
        payload={"run_id": run_id, "provider": body.provider},
        created_by=current_user.id,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    background_tasks.add_task(
        run_llm_analysis_job,
        job_id=job.id,
        run_id=run_id,
        provider=body.provider,
    )

    logger.info(
        "LLM analysis job %d queued for ForecastRun %d (provider=%s)",
        job.id, run_id, body.provider,
    )
    return {
        "message": "LLM analysis queued.",
        "job_id": job.id,
        "run_id": run_id,
        "provider": body.provider,
    }


@router.get("/{run_id}/analyses", summary="List LLM analyses for a forecast run")
async def list_analyses(
    run_id: int = Path(..., description="ForecastRun ID"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    repo = LLMAnalysisRepository(db)
    analyses = await repo.get_by_run_id(run_id)
    return {
        "run_id": run_id,
        "total": len(analyses),
        "analyses": [
            {
                "id": a.id,
                "provider": a.provider,
                "summary": a.summary,
                "risks_detected": a.risks_detected,
                "explanation": a.explanation,
                "created_at": a.created_at.isoformat(),
            }
            for a in analyses
        ],
    }


@router.get("/status/{job_id}", summary="Poll LLM analysis background job status")
async def get_llm_job_status(
    job_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    repo = JobRepository(db)
    job = await repo.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {
        "job_id": job.id,
        "status": job.status.value,
        "result": job.result,
        "error_message": job.error_message,
    }
