import logging
from typing import List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.api.dependencies import get_current_user, RoleChecker
from app.domain.models import User, IngestionBatch, BackgroundJob
from app.domain.enums import UserRole, JobType, JobStatus
from app.domain.schemas import IngestionBatchResponse, BackgroundJobResponse
from app.repositories.data_repository import IngestionBatchRepository
from app.repositories.job_repository import BackgroundJobRepository
from app.services.ingestion_service import IngestionService
from app.services.background_worker import process_ingestion_job

logger = logging.getLogger(__name__)

router = APIRouter()

# Enforce RBAC: Analysts, Finance Managers, and Admins can upload
upload_roles = [UserRole.ANALYST, UserRole.FINANCE_MANAGER, UserRole.ADMIN]
upload_permission = Depends(RoleChecker(upload_roles))

# View permission is available to all authenticated roles (including Viewer)
authenticated_user = Depends(get_current_user)


@router.post("/upload", response_model=BackgroundJobResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_financial_data(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: User = upload_permission,
    db: AsyncSession = Depends(get_db)
):
    """Upload SAP FI / SAP COPA csv/excel files. Triggers parsing and validation in the background."""
    filename = file.filename or "uploaded_file.csv"
    if not filename.lower().endswith((".csv", ".xlsx", ".xls")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file format. Only CSV (.csv) and Excel (.xlsx, .xls) files are supported."
        )

    # Read raw bytes before handing over to the background task
    file_bytes = await file.read()

    # Pre-validate CSV/XLSX columns before creating database jobs
    try:
        ingestion_service = IngestionService(db)
        df = ingestion_service.parse_file_to_dataframe(file_bytes, filename)
        ingestion_service.normalize_dataframe(df)
    except ValueError as val_err:
        logger.warning(f"File column validation failed for '{filename}': {val_err}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err)
        )
    except Exception as exc:
        logger.exception(f"Error reading file structure for '{filename}': {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read file structure: {str(exc)}"
        )

    batch_repo = IngestionBatchRepository(db)
    job_repo = BackgroundJobRepository(db)

    # 1. Create IngestionBatch metadata record
    batch = IngestionBatch(
        filename=filename,
        uploaded_by=current_user.id,
        status=JobStatus.PENDING,
        record_count=0
    )
    batch = await batch_repo.create(batch)

    # 2. Create BackgroundJob record
    job = BackgroundJob(
        job_type=JobType.VALIDATION,
        status=JobStatus.PENDING,
        payload={"batch_id": batch.id, "filename": filename},
        created_by=current_user.id
    )
    job = await job_repo.create(job)
    
    # Flush changes to obtain generated IDs
    await db.commit()

    # 3. Schedule execution as a background task
    background_tasks.add_task(
        process_ingestion_job,
        batch_id=batch.id,
        job_id=job.id,
        file_bytes=file_bytes,
        filename=filename
    )

    return job


@router.get("/batches", response_model=List[IngestionBatchResponse])
async def list_batches(
    skip: int = 0,
    limit: int = 100,
    current_user: User = authenticated_user,
    db: AsyncSession = Depends(get_db)
):
    """Retrieve all uploaded data batches (any authenticated user)."""
    batch_repo = IngestionBatchRepository(db)
    return await batch_repo.get_all_batches(skip=skip, limit=limit)


@router.get("/batches/{batch_id}", response_model=IngestionBatchResponse)
async def get_batch(
    batch_id: int,
    current_user: User = authenticated_user,
    db: AsyncSession = Depends(get_db)
):
    """Retrieve details of a specific data batch."""
    batch_repo = IngestionBatchRepository(db)
    batch = await batch_repo.get(batch_id)
    if not batch:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Ingestion batch not found"
        )
    return batch


@router.get("/jobs/{job_id}", response_model=BackgroundJobResponse)
async def get_job_status(
    job_id: int,
    current_user: User = authenticated_user,
    db: AsyncSession = Depends(get_db)
):
    """Query the execution status and output result of a background task."""
    job_repo = BackgroundJobRepository(db)
    job = await job_repo.get(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Background job not found"
        )
    return job
