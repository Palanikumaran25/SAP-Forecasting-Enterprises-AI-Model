import pandas as pd
import numpy as np
from typing import List
from app.domain.models import FinancialData


class FeatureService:
    def prepare_monthly_series(self, records: List[FinancialData]) -> pd.DataFrame:
        """Aggregates raw transactional data into a chronologically sorted monthly series.
        
        Resamples the series to 'MS' (Month Start) to ensure there are no missing monthly periods
        in the timeline, filling empty months with 0.0.
        """
        if not records:
            # Return empty dataframe with correct index structure
            return pd.DataFrame(columns=["amount"], index=pd.DatetimeIndex([]))

        # Convert records to dictionary payloads
        data = [
            {
                "year": r.fiscal_year,
                "period": r.fiscal_period,
                "amount": float(r.amount)
            }
            for r in records
        ]
        df_raw = pd.DataFrame(data)

        # Sum amounts grouping by year and period (month)
        df_monthly = df_raw.groupby(["year", "period"]).agg({"amount": "sum"}).reset_index()

        # Construct date stamps representing the start of each month
        df_monthly["date_str"] = df_monthly.apply(
            lambda row: f"{int(row['year'])}-{int(row['period']):02d}-01", axis=1
        )
        df_monthly["date"] = pd.to_datetime(df_monthly["date_str"])

        # Sort and index by the DatetimeIndex
        df_monthly = df_monthly.sort_values("date").set_index("date")
        df_monthly = df_monthly.drop(columns=["date_str"])

        # Resample to Month Start to pad any skipped months with 0.0
        df_monthly = df_monthly.resample("MS").agg({"amount": "sum"}).fillna(0.0)
        return df_monthly

    def engineer_features(self, df_monthly: pd.DataFrame) -> pd.DataFrame:
        """Calculates lag features, rolling window metrics, and seasonal variables.
        
        Note: Shift(1) is applied prior to rolling window calculations to prevent
        data leakage of the current target value into the predictor features.
        """
        df = df_monthly.copy()

        # 1. Lag Features (shift by N months)
        df["lag_1"] = df["amount"].shift(1)
        df["lag_2"] = df["amount"].shift(2)
        df["lag_3"] = df["amount"].shift(3)
        df["lag_12"] = df["amount"].shift(12)  # Seasonality lag (e.g. same month previous year)

        # 2. Rolling Window Features (3-month and 6-month windows)
        df["rolling_mean_3"] = df["amount"].shift(1).rolling(window=3).mean()
        df["rolling_std_3"] = df["amount"].shift(1).rolling(window=3).std().fillna(0.0)

        df["rolling_mean_6"] = df["amount"].shift(1).rolling(window=6).mean()
        df["rolling_std_6"] = df["amount"].shift(1).rolling(window=6).std().fillna(0.0)

        # 3. Calendar Seasonality Features
        df["month"] = df.index.month
        df["quarter"] = df.index.quarter

        return df
