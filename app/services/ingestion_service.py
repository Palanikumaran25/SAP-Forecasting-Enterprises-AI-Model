import io
import logging
import pandas as pd
from datetime import datetime, date
from typing import Tuple, List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from app.repositories.data_repository import IngestionBatchRepository, FinancialDataRepository
from app.domain.models import IngestionBatch, FinancialData
from app.domain.enums import JobStatus

logger = logging.getLogger(__name__)


class IngestionService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.batch_repo = IngestionBatchRepository(db)
        self.data_repo = FinancialDataRepository(db)

    def parse_file_to_dataframe(self, file_bytes: bytes, filename: str) -> pd.DataFrame:
        """Parse raw file bytes to pandas DataFrame based on file extension."""
        filename_lower = filename.lower()
        if filename_lower.endswith(".csv"):
            return pd.read_csv(io.BytesIO(file_bytes))
        elif filename_lower.endswith((".xlsx", ".xls")):
            return pd.read_excel(io.BytesIO(file_bytes))
        else:
            raise ValueError("Unsupported file format. Only CSV (.csv) and Excel (.xlsx, .xls) files are supported.")

    def normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize dataframe column names and apply semantic mapping to database columns."""
        # Convert column headers to lowercase, stripped, and underscore-spaced
        df.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]

        # Define aliases commonly found in SAP exports (FBL3N, CO-PA, standard GL)
        column_mapping = {
            "posting_date": ["posting_date", "postingdate", "date", "pstng_date", "posting_dt", "bldat", "budat", "document_date"],
            "gl_account": ["gl_account", "glaccount", "account", "g_l_acct", "h_g_l_acct", "gl_acc", "glacc", "saknr", "hkont", "gl_number"],
            "amount": ["amount", "value", "amt", "amount_in_lc", "amount_in_local_currency", "dmbtr", "wrbtr", "val_in_lc"],
            "document_type": ["document_type", "doctype", "doc_type", "doc_t", "blart", "type"],
            "cost_center": ["cost_center", "costcenter", "cost_ctr", "cost_centre", "kostl"],
            "profit_center": ["profit_center", "profitcenter", "profit_ctr", "profit_centre", "prctr"],
            "currency": ["currency", "curr", "waers", "curr_key", "loc_curr"]
        }

        # Apply mapping
        new_cols = {}
        for std_name, aliases in column_mapping.items():
            for alias in aliases:
                if alias in df.columns:
                    new_cols[alias] = std_name
                    break

        df = df.rename(columns=new_cols)

        # Check for required columns
        required_cols = ["posting_date", "gl_account", "amount", "document_type"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Required columns missing from file. Expected columns: posting_date (or Bldat/Budat/Date), "
                f"gl_account (or Saknr/Account), amount (or Dmbtr/Value), document_type (or Blart/DocType). "
                f"Missing: {', '.join(missing_cols)}"
            )

        return df

    async def process_ingestion(
        self, batch_id: int, file_bytes: bytes, filename: str
    ) -> Tuple[bool, int, Optional[str], Optional[List[FinancialData]]]:
        """Parse, normalize, and bulk insert financial records associated with an ingestion batch."""
        try:
            df = self.parse_file_to_dataframe(file_bytes, filename)
            df = self.normalize_dataframe(df)

            # Assign default values for optional columns
            if "currency" not in df.columns:
                df["currency"] = "EUR"
            if "cost_center" not in df.columns:
                df["cost_center"] = None
            if "profit_center" not in df.columns:
                df["profit_center"] = None

            # Remove rows where required columns contain NaN
            df = df.dropna(subset=["posting_date", "gl_account", "amount", "document_type"])

            records = []
            for idx, row in df.iterrows():
                # Parse date string or timestamp to Python date
                p_date_raw = row["posting_date"]
                if isinstance(p_date_raw, (datetime, pd.Timestamp)):
                    p_date = p_date_raw.date()
                elif isinstance(p_date_raw, date):
                    p_date = p_date_raw
                else:
                    try:
                        p_date = pd.to_datetime(p_date_raw).date()
                    except Exception:
                        continue  # Skip unparseable date row

                # Parse float amount
                try:
                    amount_val = float(row["amount"])
                except ValueError:
                    continue  # Skip rows with invalid numbers

                # Extract and clean values
                gl_acct_val = str(row["gl_account"]).strip()
                doc_type_val = str(row["document_type"]).strip()
                cc_val = str(row["cost_center"]).strip() if pd.notna(row["cost_center"]) and str(row["cost_center"]).strip() != "" else None
                pc_val = str(row["profit_center"]).strip() if pd.notna(row["profit_center"]) and str(row["profit_center"]).strip() != "" else None
                curr_val = str(row["currency"]).strip() if pd.notna(row["currency"]) and str(row["currency"]).strip() != "" else "EUR"

                # Standard fiscal year and period (month 1-12)
                fiscal_yr = p_date.year
                fiscal_pd = p_date.month

                record = FinancialData(
                    posting_date=p_date,
                    fiscal_year=fiscal_yr,
                    fiscal_period=fiscal_pd,
                    gl_account=gl_acct_val,
                    amount=amount_val,
                    currency=curr_val,
                    cost_center=cc_val,
                    profit_center=pc_val,
                    document_type=doc_type_val,
                    ingestion_batch_id=batch_id
                )
                records.append(record)

            if not records:
                return False, 0, "No valid data rows found in the uploaded file.", None

            created_records = await self.data_repo.bulk_create(records)
            return True, len(created_records), None, created_records

        except Exception as e:
            logger.exception(f"Error during ingestion processing for file '{filename}': {e}")
            return False, 0, str(e), None
