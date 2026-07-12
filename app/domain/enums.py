from enum import Enum

class UserRole(str, Enum):
    ADMIN = "Admin"
    FINANCE_MANAGER = "Finance Manager"
    ANALYST = "Analyst"
    VIEWER = "Viewer"


class ForecastStatus(str, Enum):
    DRAFT = "Draft"
    PENDING_APPROVAL = "Pending Approval"
    APPROVED = "Approved"
    REJECTED = "Rejected"


class ForecastSchedule(str, Enum):
    WEEKLY = "Weekly Forecast"
    MONTHLY = "Monthly Forecast"
    QUARTERLY = "Quarterly Forecast"
    AD_HOC = "AdHoc"


class JobType(str, Enum):
    VALIDATION = "Data Validation"
    FORECASTING = "Forecast Generation"
    LLM_ANALYSIS = "LLM Analysis"


class JobStatus(str, Enum):
    PENDING = "Pending"
    PROCESSING = "Processing"
    COMPLETED = "Completed"
    FAILED = "Failed"
