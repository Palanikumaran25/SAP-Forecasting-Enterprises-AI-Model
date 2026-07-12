import numpy as np
from typing import List, Dict, Any
from app.domain.models import FinancialData


class ValidationService:
    def validate_records(self, records: List[FinancialData]) -> Dict[str, Any]:
        """Verify financial records for missing values, duplicates, outliers, and double-entry consistency."""
        report = {
            "missing_values": [],
            "duplicates": [],
            "outliers": [],
            "financial_rules": [],
            "summary": {
                "total_records": len(records),
                "missing_count": 0,
                "duplicate_count": 0,
                "outlier_count": 0,
                "rule_violations": 0,
                "is_valid": True
            }
        }

        if not records:
            report["summary"]["is_valid"] = True
            return report

        # Collect data details for validation calculations
        amounts = [float(r.amount) for r in records]
        currency = records[0].currency if records else "EUR"

        # 1. Missing Values Verification
        missing_count = 0
        for idx, r in enumerate(records):
            missing_fields = []
            if not r.gl_account or r.gl_account.strip() == "":
                missing_fields.append("gl_account")
            if r.amount is None:
                missing_fields.append("amount")
            if not r.posting_date:
                missing_fields.append("posting_date")
            if not r.document_type or r.document_type.strip() == "":
                missing_fields.append("document_type")

            if missing_fields:
                missing_count += 1
                report["missing_values"].append({
                    "record_index": idx,
                    "missing_fields": missing_fields
                })

        report["summary"]["missing_count"] = missing_count

        # 2. Duplicate Detection
        # Finds identical transaction lines in the same uploaded batch
        seen = {}
        duplicate_count = 0
        for idx, r in enumerate(records):
            # Key based on posting date, G/L account, amount, and doc type
            key = (r.posting_date, r.gl_account, float(r.amount), r.document_type)
            if key in seen:
                duplicate_count += 1
                report["duplicates"].append({
                    "record_index": idx,
                    "duplicate_of_index": seen[key],
                    "details": f"Duplicate found: GL Account '{r.gl_account}' with amount {r.amount} on posting date {r.posting_date}"
                })
            else:
                seen[key] = idx

        report["summary"]["duplicate_count"] = duplicate_count

        # 3. Outlier Detection (Interquartile Range - IQR)
        # Identifies abnormal transaction posting amounts compared to the batch distribution
        if len(amounts) >= 5:
            q1 = np.percentile(amounts, 25)
            q3 = np.percentile(amounts, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outlier_count = 0
            for idx, r in enumerate(records):
                amt = float(r.amount)
                if amt < lower_bound or amt > upper_bound:
                    outlier_count += 1
                    report["outliers"].append({
                        "record_index": idx,
                        "amount": amt,
                        "bounds": [float(lower_bound), float(upper_bound)],
                        "details": f"Posting amount {amt:.2f} is outside normal bounds [{lower_bound:.2f}, {upper_bound:.2f}]"
                    })
            report["summary"]["outlier_count"] = outlier_count

        # 4. Financial Rule Validation
        rule_violations = 0

        # Rule A: Double-entry ledger balance check
        # For a full ledger batch upload, sum of all debit (positive) and credit (negative) entries should equal zero.
        net_balance = sum(amounts)
        if abs(net_balance) > 0.01:
            report["financial_rules"].append({
                "rule": "Double-entry Ledger Balance Check",
                "severity": "WARNING",
                "message": f"Ledger batch is unbalanced. Net postings total {net_balance:+.2f} {currency}. Postings should net to zero."
            })

        # Rule B: G/L Account Format Check
        # Standard G/L Accounts are digits (SAP FI charts of accounts usually use 4-10 digits).
        for idx, r in enumerate(records):
            gl = r.gl_account
            if not gl.isdigit() or len(gl) < 4:
                rule_violations += 1
                report["financial_rules"].append({
                    "rule": "G/L Account Formatting Check",
                    "severity": "ERROR",
                    "record_index": idx,
                    "message": f"G/L Account code '{gl}' is invalid. Standard codes should consist only of digits and be at least 4 digits long."
                })

        report["summary"]["rule_violations"] = rule_violations

        # The batch is rejected if there are formatting errors or missing fields
        has_errors = any(rule.get("severity") == "ERROR" for rule in report["financial_rules"])
        if has_errors or missing_count > 0:
            report["summary"]["is_valid"] = False

        return report
