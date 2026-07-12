"""
LLM Analysis Service  –  Phase 10
-----------------------------------
Supports both Google Gemini and OpenAI (GPT-4o) as interchangeable backends.

Responsibilities
----------------
1. Build a structured prompt from ForecastRun data
2. Call the selected LLM provider
3. Parse the response into:
   - Markdown summary
   - Risk list  [{level, description}, ...]
   - Narrative explanation
4. Persist an LLMAnalysis row linked to the ForecastRun
"""
from __future__ import annotations

import json
import logging
import textwrap
from typing import Any, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.domain.enums import JobStatus
from app.domain.models import LLMAnalysis
from app.repositories.forecast_repository import ForecastRunRepository, LLMAnalysisRepository
from app.repositories.job_repository import JobRepository

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_analysis_prompt(run_data: Dict[str, Any]) -> str:
    forecast_preview = json.dumps(run_data["forecast_values"][:6], indent=2)
    metrics_str = json.dumps(run_data["metrics"], indent=2)

    return textwrap.dedent(f"""
    You are a senior SAP FI financial controller and forecasting expert.

    A machine-learning forecast has been generated with the following details:

    ## Forecast Run Summary
    - Model        : {run_data["model_name"]}
    - Schedule     : {run_data["schedule_type"]}
    - Horizon      : {len(run_data["forecast_values"])} periods
    - Status       : {run_data["status"]}

    ## Performance Metrics
    {metrics_str}

    ## Forecast Values (first 6 periods)
    {forecast_preview}

    ## Your Task
    Respond in **valid JSON only** (no markdown fences) with exactly this structure:
    {{
      "summary": "<2-3 sentence executive Markdown summary>",
      "risks": [
        {{"level": "High|Medium|Low", "description": "<risk description>"}},
        ...
      ],
      "explanation": "<detailed paragraph explaining the forecast trend, seasonality, and key drivers>"
    }}

    Be concise, use financial terminology, and flag any MAPE > 15% as a High risk.
    """).strip()


# ---------------------------------------------------------------------------
# Provider clients
# ---------------------------------------------------------------------------

async def _call_gemini(prompt: str) -> str:
    """Call Google Gemini via the google-generativeai SDK."""
    try:
        import google.generativeai as genai  # type: ignore
    except ImportError as exc:
        raise RuntimeError("google-generativeai package is required for Gemini.") from exc

    genai.configure(api_key=settings.GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text


async def _call_openai(prompt: str) -> str:
    """Call OpenAI GPT-4o via the openai SDK."""
    try:
        from openai import AsyncOpenAI  # type: ignore
    except ImportError as exc:
        raise RuntimeError("openai package is required for GPT-4o.") from exc

    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert financial forecasting analyst."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=1024,
    )
    return response.choices[0].message.content or ""


def _parse_llm_response(raw: str) -> Dict[str, Any]:
    """Parse JSON response from LLM. Gracefully handles wrapped text."""
    raw = raw.strip()
    # Strip markdown code fences if model disobeys instructions
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("LLM did not return valid JSON; using fallback extraction.")
        return {
            "summary": raw[:500],
            "risks": [{"level": "Low", "description": "LLM response could not be parsed."}],
            "explanation": raw,
        }


# ---------------------------------------------------------------------------
# Public service
# ---------------------------------------------------------------------------

class LLMAnalysisService:
    """Orchestrates LLM calls and persists results."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db
        self._forecast_repo = ForecastRunRepository(db)
        self._llm_repo = LLMAnalysisRepository(db)

    async def analyse_forecast(
        self,
        run_id: int,
        provider: str = "gemini",  # "gemini" | "openai"
    ) -> LLMAnalysis:
        """
        Run LLM analysis for a single ForecastRun and persist the result.
        """
        run = await self._forecast_repo.get(run_id)
        if run is None:
            raise ValueError(f"ForecastRun {run_id} not found.")

        run_data = {
            "model_name": run.model_name,
            "schedule_type": run.schedule_type.value,
            "forecast_values": run.forecast_values,
            "metrics": run.metrics,
            "status": run.status.value,
        }

        prompt = _build_analysis_prompt(run_data)
        provider_lower = provider.lower()

        logger.info("Calling LLM provider '%s' for ForecastRun %d", provider, run_id)
        if provider_lower == "gemini":
            raw = await _call_gemini(prompt)
        elif provider_lower == "openai":
            raw = await _call_openai(prompt)
        else:
            raise ValueError(f"Unknown provider '{provider}'. Use 'gemini' or 'openai'.")

        parsed = _parse_llm_response(raw)

        analysis = LLMAnalysis(
            forecast_run_id=run_id,
            provider=provider_lower,
            summary=parsed.get("summary", ""),
            risks_detected=parsed.get("risks", []),
            explanation=parsed.get("explanation", ""),
        )
        self._db.add(analysis)
        await self._db.commit()
        await self._db.refresh(analysis)
        logger.info("LLMAnalysis %d created for ForecastRun %d", analysis.id, run_id)
        return analysis


# ---------------------------------------------------------------------------
# Background entry-point (called from API router)
# ---------------------------------------------------------------------------

async def run_llm_analysis_job(
    job_id: int,
    run_id: int,
    provider: str,
) -> None:
    """Background task wrapper for LLM analysis with job status tracking."""
    async with AsyncSessionLocal() as db:
        job_repo = JobRepository(db)
        job = await job_repo.get(job_id)
        if job:
            job.status = JobStatus.PROCESSING
            await db.commit()

        try:
            svc = LLMAnalysisService(db)
            analysis = await svc.analyse_forecast(run_id, provider=provider)

            if job:
                job.status = JobStatus.COMPLETED
                job.result = {"analysis_id": analysis.id, "run_id": run_id, "provider": provider}
                await db.commit()

        except Exception as exc:  # noqa: BLE001
            logger.error("LLM analysis job %d failed: %s", job_id, exc, exc_info=True)
            if job:
                job.status = JobStatus.FAILED
                job.error_message = str(exc)
                await db.commit()
