import uuid
from datetime import datetime, date
from typing import Optional, List, Dict, Any
from sqlalchemy import String, Date, DateTime, Numeric, Text, JSON, Enum, ForeignKey, func, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.core.database import Base
from app.domain.enums import UserRole, ForecastStatus, ForecastSchedule, JobType, JobStatus


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, index=True, nullable=False)
    email: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[UserRole] = mapped_column(Enum(UserRole), default=UserRole.VIEWER, nullable=False)
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # Relationships
    ingestion_batches: Mapped[List["IngestionBatch"]] = relationship("IngestionBatch", back_populates="uploader")


class TokenBlocklist(Base):
    __tablename__ = "token_blocklist"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    jti: Mapped[str] = mapped_column(String(36), index=True, unique=True, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class IngestionBatch(Base):
    __tablename__ = "ingestion_batches"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    uploaded_by: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    status: Mapped[JobStatus] = mapped_column(Enum(JobStatus), default=JobStatus.PENDING, nullable=False)
    record_count: Mapped[int] = mapped_column(default=0, nullable=False)
    validation_errors: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    uploader: Mapped["User"] = relationship("User", back_populates="ingestion_batches")
    financial_records: Mapped[List["FinancialData"]] = relationship(
        "FinancialData", back_populates="batch", cascade="all, delete-orphan"
    )


class FinancialData(Base):
    __tablename__ = "financial_data"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    posting_date: Mapped[date] = mapped_column(Date, index=True, nullable=False)
    fiscal_year: Mapped[int] = mapped_column(index=True, nullable=False)
    fiscal_period: Mapped[int] = mapped_column(index=True, nullable=False)
    gl_account: Mapped[str] = mapped_column(String(50), index=True, nullable=False)
    amount: Mapped[float] = mapped_column(Numeric(18, 2), nullable=False)  # Precise currency representation
    currency: Mapped[str] = mapped_column(String(10), default="EUR", nullable=False)
    cost_center: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, index=True)
    profit_center: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, index=True)
    document_type: Mapped[str] = mapped_column(String(10), nullable=False)
    ingestion_batch_id: Mapped[int] = mapped_column(ForeignKey("ingestion_batches.id"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    batch: Mapped["IngestionBatch"] = relationship("IngestionBatch", back_populates="financial_records")


class ForecastRun(Base):
    __tablename__ = "forecast_runs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    version: Mapped[int] = mapped_column(default=1, nullable=False)
    model_name: Mapped[str] = mapped_column(String(50), nullable=False)
    schedule_type: Mapped[ForecastSchedule] = mapped_column(
        Enum(ForecastSchedule), default=ForecastSchedule.AD_HOC, nullable=False
    )
    parameters: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    metrics: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    forecast_values: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, nullable=False)  # Forecast timeline values
    is_best_model: Mapped[bool] = mapped_column(default=False, nullable=False)
    status: Mapped[ForecastStatus] = mapped_column(Enum(ForecastStatus), default=ForecastStatus.DRAFT, nullable=False)
    
    created_by: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    approved_by: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"), nullable=True)
    approved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    comments: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    creator: Mapped["User"] = relationship("User", foreign_keys=[created_by])
    approver: Mapped[Optional["User"]] = relationship("User", foreign_keys=[approved_by])
    llm_analyses: Mapped[List["LLMAnalysis"]] = relationship(
        "LLMAnalysis", back_populates="forecast_run", cascade="all, delete-orphan"
    )


class LLMAnalysis(Base):
    __tablename__ = "llm_analyses"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    forecast_run_id: Mapped[int] = mapped_column(ForeignKey("forecast_runs.id"), nullable=False)
    provider: Mapped[str] = mapped_column(String(20), nullable=False)  # "gemini", "openai"
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    risks_detected: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, nullable=False)
    explanation: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    forecast_run: Mapped["ForecastRun"] = relationship("ForecastRun", back_populates="llm_analyses")


class BackgroundJob(Base):
    __tablename__ = "background_jobs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    job_type: Mapped[JobType] = mapped_column(Enum(JobType), nullable=False)
    status: Mapped[JobStatus] = mapped_column(Enum(JobStatus), default=JobStatus.PENDING, nullable=False)
    payload: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    result: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_by: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # Relationships
    creator: Mapped["User"] = relationship("User")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"), nullable=True)
    action: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    ip_address: Mapped[Optional[str]] = mapped_column(String(45), nullable=True)
    details: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    user: Mapped[Optional["User"]] = relationship("User")


# Compound indices for faster filtering / dashboard aggregation
Index("ix_financial_data_year_period", FinancialData.fiscal_year, FinancialData.fiscal_period)
Index("ix_financial_data_gl_date", FinancialData.gl_account, FinancialData.posting_date)
