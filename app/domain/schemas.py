from datetime import datetime, date
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, EmailStr, Field, ConfigDict, computed_field
from app.domain.enums import UserRole, ForecastStatus, ForecastSchedule, JobType, JobStatus


# =====================================================================
# USER SCHEMAS
# =====================================================================
class UserBase(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr


class UserCreate(UserBase):
    password: str = Field(..., min_length=6, max_length=100)
    role: Optional[UserRole] = UserRole.VIEWER


class UserUpdate(BaseModel):
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    email: Optional[EmailStr] = None
    password: Optional[str] = Field(None, min_length=6, max_length=100)
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None


class UserResponse(UserBase):
    id: int
    role: UserRole
    is_active: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class PasswordResetRequest(BaseModel):
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    token: str
    new_password: str = Field(..., min_length=6, max_length=100)


# =====================================================================
# AUTHENTICATION SCHEMAS
# =====================================================================
class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class TokenPayload(BaseModel):
    sub: Optional[str] = None
    exp: Optional[int] = None
    role: Optional[str] = None


class RefreshTokenRequest(BaseModel):
    refresh_token: str


# =====================================================================
# SAP DATA INGESTION SCHEMAS
# =====================================================================
class FinancialDataResponse(BaseModel):
    id: int
    posting_date: date
    fiscal_year: int
    fiscal_period: int
    gl_account: str
    amount: float
    currency: str
    cost_center: Optional[str] = None
    profit_center: Optional[str] = None
    document_type: str
    ingestion_batch_id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class IngestionBatchResponse(BaseModel):
    id: int
    filename: str
    uploaded_by: int
    status: JobStatus
    record_count: int
    validation_errors: Optional[Dict[str, Any]] = None
    created_at: datetime

    @computed_field
    @property
    def error_count(self) -> int:
        if not self.validation_errors or not isinstance(self.validation_errors, dict):
            return 0
        summary = self.validation_errors.get("summary", {})
        if isinstance(summary, dict) and summary:
            financial_rules = self.validation_errors.get("financial_rules", [])
            rule_errors = sum(
                1 for rule in financial_rules 
                if isinstance(rule, dict) and rule.get("severity") == "ERROR"
            )
            return summary.get("missing_count", 0) + rule_errors
        if any(k in self.validation_errors for k in ["error_message", "parsing_error", "system_error"]):
            return 1
        return 0

    @computed_field
    @property
    def error_message(self) -> Optional[str]:
        if not self.validation_errors or not isinstance(self.validation_errors, dict):
            return None
        return (self.validation_errors.get("error_message") or 
                self.validation_errors.get("parsing_error") or 
                self.validation_errors.get("system_error"))

    model_config = ConfigDict(from_attributes=True)


# =====================================================================
# FORECAST SCHEMAS
# =====================================================================
class ForecastRunCreate(BaseModel):
    model_name: str = Field(..., description="ARIMA, ETS, Random Forest, XGBoost, or Best Model Selector")
    schedule_type: ForecastSchedule = ForecastSchedule.AD_HOC
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Hyperparameters for the models")


class ForecastRunResponse(BaseModel):
    id: int
    version: int
    model_name: str
    schedule_type: ForecastSchedule
    parameters: Dict[str, Any]
    metrics: Dict[str, Any]
    forecast_values: List[Dict[str, Any]]
    is_best_model: bool
    status: ForecastStatus
    created_by: int
    approved_by: Optional[int] = None
    approved_at: Optional[datetime] = None
    comments: Optional[str] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ForecastApprovalRequest(BaseModel):
    status: ForecastStatus = Field(..., description="APPROVED or REJECTED status to transition")
    comments: Optional[str] = Field(None, max_length=500, description="Approver feedback comment")


class ForecastCompareRequest(BaseModel):
    historical_start_date: date
    historical_end_date: date
    test_periods: int = Field(12, ge=2, description="Periods to reserve for backtesting metrics")
    forecast_periods: int = Field(12, ge=1, description="Periods to forecast out-of-sample")


# =====================================================================
# LLM LAYER SCHEMAS
# =====================================================================
class LLMAnalysisResponse(BaseModel):
    id: int
    forecast_run_id: int
    provider: str
    summary: str
    risks_detected: List[Dict[str, Any]]
    explanation: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# =====================================================================
# BACKGROUND JOB SCHEMAS
# =====================================================================
class BackgroundJobResponse(BaseModel):
    id: int
    job_type: JobType
    status: JobStatus
    payload: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created_by: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


# =====================================================================
# AUDIT AND MONITORING SCHEMAS
# =====================================================================
class AuditLogResponse(BaseModel):
    id: int
    user_id: Optional[int] = None
    action: str
    ip_address: Optional[str] = None
    details: Dict[str, Any]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# =====================================================================
# DASHBOARD SCHEMAS
# =====================================================================
class KPISummary(BaseModel):
    total_actual_revenue: float
    total_actual_expense: float
    net_actual_income: float
    total_forecasted_revenue: float
    total_forecasted_expense: float
    net_forecasted_income: float
    variance: float


class TrendDataPoint(BaseModel):
    date: date
    actual_amount: Optional[float] = None
    forecasted_amount: Optional[float] = None


class TrendResponse(BaseModel):
    data: List[TrendDataPoint]
