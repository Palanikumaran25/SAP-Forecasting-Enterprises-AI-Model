import logging
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import SessionLocal
from app.services.ingestion_service import IngestionService
from app.repositories.job_repository import BackgroundJobRepository
from app.repositories.data_repository import IngestionBatchRepository
from app.domain.enums import JobStatus, JobType

logger = logging.getLogger(__name__)


async def process_ingestion_job(batch_id: int, job_id: int, file_bytes: bytes, filename: str):
    """Background task to parse raw file bytes, save financial records, and run validations."""
    async with SessionLocal() as db:
        try:
            job_repo = BackgroundJobRepository(db)
            batch_repo = IngestionBatchRepository(db)
            ingestion_service = IngestionService(db)

            # 1. Update job to PROCESSING
            await job_repo.update_job_status(job_id, JobStatus.PROCESSING)
            await db.commit()

            # 2. Run file parsing and database bulk writing
            success, count, error_msg, records = await ingestion_service.process_ingestion(
                batch_id, file_bytes, filename
            )

            if not success:
                # Update batch and job to FAILED
                batch = await batch_repo.get(batch_id)
                if batch:
                    batch.status = JobStatus.FAILED
                    batch.validation_errors = {"parsing_error": error_msg}
                await job_repo.update_job_status(
                    job_id, JobStatus.FAILED, error_message=error_msg
                )
                await db.commit()
                logger.error(f"Ingestion job {job_id} failed: {error_msg}")
                return

            # 3. Trigger Validation Engine
            from app.services.validation_service import ValidationService
            validation_service = ValidationService()
            validation_report = validation_service.validate_records(records or [])

            # Detailed validation logging for troubleshooting
            for rule in validation_report.get("financial_rules", []):
                severity = rule.get("severity", "INFO")
                msg = f"Validation {severity}: {rule.get('rule')} - {rule.get('message')}"
                if severity == "ERROR":
                    logger.error(msg)
                elif severity == "WARNING":
                    logger.warning(msg)
                else:
                    logger.info(msg)

            for missing in validation_report.get("missing_values", []):
                logger.error(f"Validation ERROR: Missing fields at record index {missing.get('record_index')}: {missing.get('missing_fields')}")
            for dup in validation_report.get("duplicates", []):
                logger.warning(f"Validation WARNING: Duplicate at record index {dup.get('record_index')}: {dup.get('details')}")
            for outlier in validation_report.get("outliers", []):
                logger.warning(f"Validation WARNING: Outlier at record index {outlier.get('record_index')}: {outlier.get('details')}")

            # Update batch metadata
            batch = await batch_repo.get(batch_id)
            if batch:
                batch.record_count = count
                batch.validation_errors = validation_report
                
                # If validation determines data contains fatal formatting/structural errors
                if not validation_report["summary"]["is_valid"]:
                    batch.status = JobStatus.FAILED
                    err_msg = "Validation Engine: Critical formatting or financial rule violations detected."
                    batch.validation_errors["error_message"] = err_msg
                    await job_repo.update_job_status(
                        job_id, JobStatus.FAILED, 
                        result={"record_count": count, "validation": validation_report},
                        error_message=err_msg
                    )
                else:
                    batch.status = JobStatus.COMPLETED
                    await job_repo.update_job_status(
                        job_id, JobStatus.COMPLETED, 
                        result={"record_count": count, "validation": validation_report}
                    )

            await db.commit()
            logger.info(f"Ingestion job {job_id} processed cleanly with status {batch.status if batch else 'UNKNOWN'}. Saved {count} records.")

        except Exception as e:
            logger.exception(f"Unexpected error in background ingestion job {job_id}: {e}")
            await db.rollback()
            # Attempt to record failure cleanly in database
            try:
                async with SessionLocal() as fail_db:
                    f_job_repo = BackgroundJobRepository(fail_db)
                    f_batch_repo = IngestionBatchRepository(fail_db)
                    await f_job_repo.update_job_status(
                        job_id, JobStatus.FAILED, error_message=f"System error: {str(e)}"
                    )
                    batch = await f_batch_repo.get(batch_id)
                    if batch:
                        batch.status = JobStatus.FAILED
                        batch.validation_errors = {"system_error": str(e), "error_message": f"System error: {str(e)}"}
                    await fail_db.commit()
            except Exception:
                logger.exception("Failed to write ingestion job error status to database")
