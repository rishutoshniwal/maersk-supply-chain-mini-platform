"""
Standalone Data Quality (DQ) profiling and check runner.
Runs against raw CSV files prior to ingestion.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd


@dataclass
class DQCheckResult:
    check_name: str
    dataset: str
    severity: str  # 'critical' | 'warning' | 'info'
    status: str  # 'FAIL' | 'PASS'
    rows_affected: int
    total_rows: int
    percentage_affected: float
    recommended_action: str  # 'drop' | 'fix' | 'flag' | 'quarantine'
    reasoning: str
    sample_affected_ids: List[str] = field(default_factory=list)


@dataclass
class DQReport:
    generated_at: str
    input_directory: str
    total_datasets: int
    total_checks: int
    checks_passed: int
    checks_failed: int
    critical_issues: int
    warning_issues: int
    info_issues: int
    total_rows_inspected: int
    checks: List[DQCheckResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "input_directory": self.input_directory,
            "summary": {
                "total_datasets": self.total_datasets,
                "total_checks": self.total_checks,
                "checks_passed": self.checks_passed,
                "checks_failed": self.checks_failed,
                "critical_issues": self.critical_issues,
                "warning_issues": self.warning_issues,
                "info_issues": self.info_issues,
                "total_rows_inspected": self.total_rows_inspected,
            },
            "checks": [asdict(c) for c in self.checks],
        }


def find_data_file(input_path: Path, patterns: List[str]) -> Optional[Path]:
    """Resolve file from direct path or by searching patterns inside a directory."""
    if input_path.is_file():
        return input_path

    if input_path.is_dir():
        for pattern in patterns:
            # Check exact match
            exact = input_path / pattern
            if exact.is_file():
                return exact
            # Check glob match
            matches = list(input_path.glob(f"*{pattern}*"))
            if matches:
                # Prioritize shortest / most standard name
                matches.sort(key=lambda p: len(p.name))
                return matches[0]
    return None


def resolve_raw_files(input_dir: str | Path) -> Tuple[Path, Path, Path]:
    """Find ports, shipments, and port_events CSV files."""
    path = Path(input_dir)
    ports_path = find_data_file(path, ["ports.csv", "ports 2.csv", "ports"])
    shipments_path = find_data_file(path, ["shipments.csv", "shipments 2.csv", "shipments"])
    events_path = find_data_file(path, ["port_events.csv", "port_events 2.csv", "port_events"])

    missing = []
    if not ports_path:
        missing.append("ports.csv")
    if not shipments_path:
        missing.append("shipments.csv")
    if not events_path:
        missing.append("port_events.csv")

    if missing:
        raise FileNotFoundError(
            f"Could not locate required CSV files in '{path}': missing {missing}"
        )

    return ports_path, shipments_path, events_path


class DataQualityRunner:
    """Performs standalone verification of raw data files."""

    def __init__(self, ports_df: pd.DataFrame, shipments_df: pd.DataFrame, events_df: pd.DataFrame):
        self.ports = ports_df.copy()
        self.shipments = shipments_df.copy()
        self.events = events_df.copy()
        self.valid_port_codes: Set[str] = set(self.ports["port_code"].dropna().unique())
        self.checks: List[DQCheckResult] = []

    def _calc_pct(self, count: int, total: int) -> float:
        return round((count / total) * 100.0, 2) if total > 0 else 0.0

    # ------------------ PORTS CHECKS ------------------
    def check_ports_missing_country(self) -> DQCheckResult:
        total = len(self.ports)
        mask = self.ports["country"].isna()
        count = int(mask.sum())
        sample_ids = self.ports.loc[mask, "port_code"].head(5).tolist()
        return DQCheckResult(
            check_name="ports_missing_country",
            dataset="ports",
            severity="warning",
            status="FAIL" if count > 0 else "PASS",
            rows_affected=count,
            total_rows=total,
            percentage_affected=self._calc_pct(count, total),
            recommended_action="fix",
            reasoning="Reference port missing country code ('BEANR' Antwerp). Affects geopolitical grouping and reporting; fix via UN/LOCODE reference imputation.",
            sample_affected_ids=sample_ids,
        )

    # ------------------ SHIPMENTS CHECKS ------------------
    def check_shipments_duplicate_id(self) -> DQCheckResult:
        total = len(self.shipments)
        dup_mask = self.shipments["shipment_id"].duplicated(keep=False)
        count = int(dup_mask.sum())
        sample_ids = self.shipments.loc[dup_mask, "shipment_id"].unique()[:5].tolist()
        return DQCheckResult(
            check_name="shipments_duplicate_primary_key",
            dataset="shipments",
            severity="critical",
            status="FAIL" if count > 0 else "PASS",
            rows_affected=count,
            total_rows=total,
            percentage_affected=self._calc_pct(count, total),
            recommended_action="quarantine",
            reasoning="Multiple rows share identical shipment_id with conflicting metadata/weights. Deduplicate by keeping latest valid booking date and quarantine conflicting versions.",
            sample_affected_ids=sample_ids,
        )

    def check_shipments_invalid_origin_port(self) -> DQCheckResult:
        total = len(self.shipments)
        mask = ~self.shipments["origin_port"].isin(self.valid_port_codes)
        count = int(mask.sum())
        sample_ids = self.shipments.loc[mask, "shipment_id"].head(5).tolist()
        return DQCheckResult(
            check_name="shipments_invalid_origin_port",
            dataset="shipments",
            severity="critical",
            status="FAIL" if count > 0 else "PASS",
            rows_affected=count,
            total_rows=total,
            percentage_affected=self._calc_pct(count, total),
            recommended_action="quarantine",
            reasoning="Origin port codes not present in reference ports table (e.g. XXTST, ZZZZZ, UNKNW). Shipments cannot be assigned to physical terminal berths.",
            sample_affected_ids=sample_ids,
        )

    def check_shipments_invalid_destination_port(self) -> DQCheckResult:
        total = len(self.shipments)
        mask = ~self.shipments["destination_port"].isin(self.valid_port_codes)
        count = int(mask.sum())
        sample_ids = self.shipments.loc[mask, "shipment_id"].head(5).tolist()
        return DQCheckResult(
            check_name="shipments_invalid_destination_port",
            dataset="shipments",
            severity="critical",
            status="FAIL" if count > 0 else "PASS",
            rows_affected=count,
            total_rows=total,
            percentage_affected=self._calc_pct(count, total),
            recommended_action="quarantine",
            reasoning="Destination port codes not present in reference ports table (e.g. XXTST, UNKNW, ZZZZZ). Shipments cannot clear customs or schedule delivery.",
            sample_affected_ids=sample_ids,
        )

    def check_shipments_invalid_container_count(self) -> DQCheckResult:
        total = len(self.shipments)
        mask = self.shipments["container_count"] <= 0
        count = int(mask.sum())
        sample_ids = self.shipments.loc[mask, "shipment_id"].head(5).tolist()
        return DQCheckResult(
            check_name="shipments_non_positive_container_count",
            dataset="shipments",
            severity="critical",
            status="FAIL" if count > 0 else "PASS",
            rows_affected=count,
            total_rows=total,
            percentage_affected=self._calc_pct(count, total),
            recommended_action="quarantine",
            reasoning="Container count is <= 0. Booking with zero containers violates physical transport requirements and vessel load limits.",
            sample_affected_ids=sample_ids,
        )

    def check_shipments_negative_zero_weight(self) -> DQCheckResult:
        total = len(self.shipments)
        mask = self.shipments["weight_tons"].notna() & (self.shipments["weight_tons"] <= 0)
        count = int(mask.sum())
        sample_ids = self.shipments.loc[mask, "shipment_id"].head(5).tolist()
        return DQCheckResult(
            check_name="shipments_negative_or_zero_weight",
            dataset="shipments",
            severity="critical",
            status="FAIL" if count > 0 else "PASS",
            rows_affected=count,
            total_rows=total,
            percentage_affected=self._calc_pct(count, total),
            recommended_action="quarantine",
            reasoning="Cargo weight <= 0 is physically impossible and violates maritime safety (SOLAS VGM convention). Indicates sensor or feed failure.",
            sample_affected_ids=sample_ids,
        )

    def check_shipments_null_weight(self) -> DQCheckResult:
        total = len(self.shipments)
        mask = self.shipments["weight_tons"].isna()
        count = int(mask.sum())
        sample_ids = self.shipments.loc[mask, "shipment_id"].head(5).tolist()
        return DQCheckResult(
            check_name="shipments_null_weight",
            dataset="shipments",
            severity="warning",
            status="FAIL" if count > 0 else "PASS",
            rows_affected=count,
            total_rows=total,
            percentage_affected=self._calc_pct(count, total),
            recommended_action="flag",
            reasoning="Missing cargo weight. Impute with median cargo weight per container grouped by cargo_type and flag row as estimated.",
            sample_affected_ids=sample_ids,
        )

    def check_shipments_inconsistent_cargo_type(self) -> DQCheckResult:
        canonical_cargos = {
            "Furniture", "Automotive Parts", "Machinery", "Pharmaceuticals",
            "Electronics", "Chemicals", "Consumer Goods", "Apparel",
            "Food & Beverage", "Raw Materials"
        }
        total = len(self.shipments)
        mask = self.shipments["cargo_type"].notna() & ~self.shipments["cargo_type"].isin(canonical_cargos)
        count = int(mask.sum())
        sample_ids = self.shipments.loc[mask, "shipment_id"].head(5).tolist()
        return DQCheckResult(
            check_name="shipments_inconsistent_cargo_type_casing",
            dataset="shipments",
            severity="warning",
            status="FAIL" if count > 0 else "PASS",
            rows_affected=count,
            total_rows=total,
            percentage_affected=self._calc_pct(count, total),
            recommended_action="fix",
            reasoning="Cargo type values contain leading whitespace (' Chemicals') or non-standard casing ('FURNITURE', 'electronics'). Normalize via strip and title-case.",
            sample_affected_ids=sample_ids,
        )

    def check_shipments_missing_cargo_type(self) -> DQCheckResult:
        total = len(self.shipments)
        mask = self.shipments["cargo_type"].isna()
        count = int(mask.sum())
        sample_ids = self.shipments.loc[mask, "shipment_id"].head(5).tolist()
        return DQCheckResult(
            check_name="shipments_missing_cargo_type",
            dataset="shipments",
            severity="warning",
            status="FAIL" if count > 0 else "PASS",
            rows_affected=count,
            total_rows=total,
            percentage_affected=self._calc_pct(count, total),
            recommended_action="flag",
            reasoning="Cargo type is missing. Assign to 'Unknown' category to retain shipment for route-level metric aggregation.",
            sample_affected_ids=sample_ids,
        )

    def check_shipments_inconsistent_status(self) -> DQCheckResult:
        canonical_statuses = {"DELIVERED", "DELAYED", "CANCELLED"}
        total = len(self.shipments)
        mask = self.shipments["status"].notna() & ~self.shipments["status"].isin(canonical_statuses)
        count = int(mask.sum())
        sample_ids = self.shipments.loc[mask, "shipment_id"].head(5).tolist()
        return DQCheckResult(
            check_name="shipments_inconsistent_status_values",
            dataset="shipments",
            severity="warning",
            status="FAIL" if count > 0 else "PASS",
            rows_affected=count,
            total_rows=total,
            percentage_affected=self._calc_pct(count, total),
            recommended_action="fix",
            reasoning="Non-canonical status representations ('COMPLETED', 'Complete', 'delivered'). Standardize to canonical uppercase vocabulary.",
            sample_affected_ids=sample_ids,
        )

    def check_shipments_missing_status(self) -> DQCheckResult:
        total = len(self.shipments)
        mask = self.shipments["status"].isna()
        count = int(mask.sum())
        sample_ids = self.shipments.loc[mask, "shipment_id"].head(5).tolist()
        return DQCheckResult(
            check_name="shipments_missing_status",
            dataset="shipments",
            severity="warning",
            status="FAIL" if count > 0 else "PASS",
            rows_affected=count,
            total_rows=total,
            percentage_affected=self._calc_pct(count, total),
            recommended_action="fix",
            reasoning="Missing shipment status. Infer from actual delivery timestamps or classify as 'UNKNOWN' pending reconciliation.",
            sample_affected_ids=sample_ids,
        )

    def check_shipments_departure_before_booking(self) -> DQCheckResult:
        total = len(self.shipments)
        booking = pd.to_datetime(self.shipments["booking_date"], errors="coerce")
        actual_dep = pd.to_datetime(self.shipments["actual_departure"], errors="coerce")
        mask = actual_dep.notna() & booking.notna() & (actual_dep < booking)
        count = int(mask.sum())
        sample_ids = self.shipments.loc[mask, "shipment_id"].head(5).tolist()
        return DQCheckResult(
            check_name="shipments_departure_before_booking_anomaly",
            dataset="shipments",
            severity="critical",
            status="FAIL" if count > 0 else "PASS",
            rows_affected=count,
            total_rows=total,
            percentage_affected=self._calc_pct(count, total),
            recommended_action="quarantine",
            reasoning="Temporal paradox: vessel actual_departure is recorded before booking_date. Indicates back-dated booking or event clock desynchronization.",
            sample_affected_ids=sample_ids,
        )

    def check_shipments_status_actuals_contradiction(self) -> DQCheckResult:
        total = len(self.shipments)
        status_clean = self.shipments["status"].fillna("").astype(str).str.upper().str.strip()
        mask = status_clean.isin(["DELIVERED", "COMPLETED", "COMPLETE"]) & self.shipments["actual_arrival"].isna()
        count = int(mask.sum())
        sample_ids = self.shipments.loc[mask, "shipment_id"].head(5).tolist()
        return DQCheckResult(
            check_name="shipments_completed_status_missing_arrival_timestamp",
            dataset="shipments",
            severity="warning",
            status="FAIL" if count > 0 else "PASS",
            rows_affected=count,
            total_rows=total,
            percentage_affected=self._calc_pct(count, total),
            recommended_action="flag",
            reasoning="Shipment is marked delivered or completed, yet actual_arrival is NULL. Flag for logistics milestone reconciliation.",
            sample_affected_ids=sample_ids,
        )

    # ------------------ PORT EVENTS CHECKS ------------------
    def check_events_duplicate_event_id(self) -> DQCheckResult:
        total = len(self.events)
        dup_mask = self.events["event_id"].duplicated(keep=False)
        count = int(dup_mask.sum())
        sample_ids = self.events.loc[dup_mask, "event_id"].unique()[:5].tolist()
        return DQCheckResult(
            check_name="port_events_duplicate_event_id",
            dataset="port_events",
            severity="critical",
            status="FAIL" if count > 0 else "PASS",
            rows_affected=count,
            total_rows=total,
            percentage_affected=self._calc_pct(count, total),
            recommended_action="fix",
            reasoning="Event IDs are duplicated across distinct event payloads (ID generation collision). Generate synthetic composite UUIDs upon ingestion.",
            sample_affected_ids=sample_ids,
        )

    def check_events_missing_event_type(self) -> DQCheckResult:
        total = len(self.events)
        mask = self.events["event_type"].isna()
        count = int(mask.sum())
        sample_ids = self.events.loc[mask, "event_id"].head(5).tolist()
        return DQCheckResult(
            check_name="port_events_missing_event_type",
            dataset="port_events",
            severity="warning",
            status="FAIL" if count > 0 else "PASS",
            rows_affected=count,
            total_rows=total,
            percentage_affected=self._calc_pct(count, total),
            recommended_action="fix",
            reasoning="Event record missing event_type. Can be inferred from operational notes (e.g. weather/congestion notes -> DELAYED) or defaulted to UNKNOWN.",
            sample_affected_ids=sample_ids,
        )

    def check_events_missing_notes(self) -> DQCheckResult:
        total = len(self.events)
        mask = self.events["notes"].isna()
        count = int(mask.sum())
        sample_ids = self.events.loc[mask, "event_id"].head(5).tolist()
        return DQCheckResult(
            check_name="port_events_missing_notes",
            dataset="port_events",
            severity="info",
            status="FAIL" if count > 0 else "PASS",
            rows_affected=count,
            total_rows=total,
            percentage_affected=self._calc_pct(count, total),
            recommended_action="fix",
            reasoning="Notes column is NULL in event feed. Contextual text is optional; populate with empty string in curated table to ensure null-safety.",
            sample_affected_ids=sample_ids,
        )

    def run_all(self, input_dir_desc: str = "./data/raw") -> DQReport:
        """Execute all data quality checks and return comprehensive report."""
        self.checks = [
            self.check_ports_missing_country(),
            self.check_shipments_duplicate_id(),
            self.check_shipments_invalid_origin_port(),
            self.check_shipments_invalid_destination_port(),
            self.check_shipments_invalid_container_count(),
            self.check_shipments_negative_zero_weight(),
            self.check_shipments_null_weight(),
            self.check_shipments_inconsistent_cargo_type(),
            self.check_shipments_missing_cargo_type(),
            self.check_shipments_inconsistent_status(),
            self.check_shipments_missing_status(),
            self.check_shipments_departure_before_booking(),
            self.check_shipments_status_actuals_contradiction(),
            self.check_events_duplicate_event_id(),
            self.check_events_missing_event_type(),
            self.check_events_missing_notes(),
        ]

        total_checks = len(self.checks)
        passed = sum(1 for c in self.checks if c.status == "PASS")
        failed = sum(1 for c in self.checks if c.status == "FAIL")
        critical = sum(1 for c in self.checks if c.status == "FAIL" and c.severity == "critical")
        warning = sum(1 for c in self.checks if c.status == "FAIL" and c.severity == "warning")
        info = sum(1 for c in self.checks if c.status == "FAIL" and c.severity == "info")
        total_rows = len(self.ports) + len(self.shipments) + len(self.events)

        return DQReport(
            generated_at=datetime.now(timezone.utc).isoformat(),
            input_directory=str(input_dir_desc),
            total_datasets=3,
            total_checks=total_checks,
            checks_passed=passed,
            checks_failed=failed,
            critical_issues=critical,
            warning_issues=warning,
            info_issues=info,
            total_rows_inspected=total_rows,
            checks=self.checks,
        )


def run_dq_checks(input_dir: str | Path, output_file: Optional[str | Path] = None) -> DQReport:
    """Load raw CSV files and run standalone data quality checks."""
    ports_path, shipments_path, events_path = resolve_raw_files(input_dir)

    ports_df = pd.read_csv(ports_path)
    shipments_df = pd.read_csv(shipments_path)
    events_df = pd.read_csv(events_path)

    runner = DataQualityRunner(ports_df, shipments_df, events_df)
    report = runner.run_all(input_dir_desc=str(input_dir))

    if output_file:
        out_p = Path(output_file)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)

    return report


def print_report_summary(report: DQReport) -> None:
    """Print clean summary table to stdout."""
    print("=" * 105)
    print("  SUPPLY CHAIN INTELLIGENCE PLATFORM — DATA QUALITY PROFILING REPORT")
    print("=" * 105)
    print(f"Timestamp: {report.generated_at} | Input Directory: {report.input_directory}")
    print(f"Rows Inspected: {report.total_rows_inspected:,} | Checks: {report.total_checks} | "
          f"Failed: {report.checks_failed} (Critical: {report.critical_issues}, Warning: {report.warning_issues}, Info: {report.info_issues})")
    print("-" * 105)
    header = f"{'Check Name':<38} | {'Dataset':<11} | {'Severity':<8} | {'Status':<5} | {'Affected':<8} | {'% Total':<7} | {'Action':<10}"
    print(header)
    print("-" * 105)
    for c in report.checks:
        pct_str = f"{c.percentage_affected:.2f}%"
        print(f"{c.check_name:<38} | {c.dataset:<11} | {c.severity.upper():<8} | {c.status:<5} | {c.rows_affected:<8} | {pct_str:<7} | {c.recommended_action.upper():<10}")
    print("=" * 105)


def main():
    parser = argparse.ArgumentParser(description="Run standalone data quality checks on raw supply chain CSVs.")
    parser.add_argument(
        "--input",
        "-i",
        default="./data/raw",
        help="Path to directory containing raw CSV files or file path (default: ./data/raw)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="./reports/dq_report.json",
        help="Path where output JSON report will be saved (default: ./reports/dq_report.json)",
    )
    args = parser.parse_args()

    report = run_dq_checks(args.input, args.output)
    print_report_summary(report)
    print(f"\n[+] Detailed machine-readable JSON quality report generated at: {args.output}\n")


if __name__ == "__main__":
    main()
