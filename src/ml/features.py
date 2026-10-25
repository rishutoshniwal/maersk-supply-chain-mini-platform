"""
Feature extraction and leakage prevention for delay prediction model.
Strict rule: Booking-time features ONLY.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# Explicit blacklist of target leakage fields
LEAKAGE_COLUMNS = {
    "actual_departure",
    "actual_arrival",
    "actual_delay_hours",
    "on_time_flag",
    "transit_days_actual",
    "status",
    "notes",
    "delay_minutes",
}

NUMERICAL_FEATURES = [
    "container_count",
    "weight_tons",
    "weight_per_container",
    "planned_transit_days",
    "booking_lead_days",
    "origin_congestion_score",
    "dest_congestion_score",
    "booking_month",
    "booking_day_of_week",
    "planned_dep_month",
    "planned_dep_day_of_week",
]

CATEGORICAL_FEATURES = [
    "origin_port",
    "destination_port",
    "cargo_type",
    "origin_region",
    "dest_region",
]

ALL_FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


def validate_no_leakage(df: pd.DataFrame) -> None:
    """Raise ValueError if any forbidden post-booking columns are present."""
    leaked = set(df.columns).intersection(LEAKAGE_COLUMNS)
    if leaked:
        raise ValueError(
            f"DATA LEAKAGE DETECTED! Features contain forbidden post-booking fields: {sorted(leaked)}"
        )


def extract_booking_features(
    df: pd.DataFrame,
    port_meta: Optional[Dict[str, Dict[str, Any]]] = None,
) -> pd.DataFrame:
    """
    Extract clean booking-time feature matrix from raw/curated dataframe or dictionary records.
    Guarantee zero data leakage.
    """
    data = df.copy()

    # Convert timestamps
    booking_dt = pd.to_datetime(data["booking_date"], errors="coerce")
    planned_dep_dt = pd.to_datetime(data["planned_departure"], errors="coerce")
    planned_arr_dt = pd.to_datetime(data["planned_arrival"], errors="coerce")

    # Engineered time features
    data["planned_transit_days"] = (planned_arr_dt - planned_dep_dt).dt.total_seconds() / 86400.0
    data["booking_lead_days"] = (planned_dep_dt - booking_dt).dt.total_seconds() / 86400.0

    # Handle pandas dt warning by using day_of_week
    data["booking_month"] = booking_dt.dt.month.fillna(1).astype(int)
    data["booking_day_of_week"] = booking_dt.dt.day_of_week.fillna(0).astype(int)
    data["planned_dep_month"] = planned_dep_dt.dt.month.fillna(1).astype(int)
    data["planned_dep_day_of_week"] = planned_dep_dt.dt.day_of_week.fillna(0).astype(int)

    # Cargo and container ratios
    container_count = pd.to_numeric(data["container_count"], errors="coerce").fillna(1)
    container_count = np.where(container_count <= 0, 1, container_count)
    weight_tons = pd.to_numeric(data["weight_tons"], errors="coerce").fillna(500.0)
    weight_tons = np.where(weight_tons <= 0, 500.0, weight_tons)

    data["container_count"] = container_count
    data["weight_tons"] = weight_tons
    data["weight_per_container"] = weight_tons / container_count

    # Port metadata enrichment (congestion, region)
    if port_meta:
        data["origin_congestion_score"] = data["origin_port"].map(
            lambda p: port_meta.get(p, {}).get("avg_congestion_score", 0.70)
        )
        data["dest_congestion_score"] = data["destination_port"].map(
            lambda p: port_meta.get(p, {}).get("avg_congestion_score", 0.70)
        )
        data["origin_region"] = data["origin_port"].map(
            lambda p: port_meta.get(p, {}).get("region", "Unknown")
        )
        data["dest_region"] = data["destination_port"].map(
            lambda p: port_meta.get(p, {}).get("region", "Unknown")
        )
    else:
        if "origin_congestion_score" not in data.columns:
            data["origin_congestion_score"] = 0.70
        if "dest_congestion_score" not in data.columns:
            data["dest_congestion_score"] = 0.70
        if "origin_region" not in data.columns:
            data["origin_region"] = "Unknown"
        if "dest_region" not in data.columns:
            data["dest_region"] = "Unknown"

    # Fill defaults for categorical features
    data["cargo_type"] = data["cargo_type"].fillna("Unknown Cargo").astype(str).str.strip().str.title()
    data["origin_port"] = data["origin_port"].fillna("UNKNOWN").astype(str).str.strip()
    data["destination_port"] = data["destination_port"].fillna("UNKNOWN").astype(str).str.strip()
    data["origin_region"] = data["origin_region"].fillna("Unknown").astype(str)
    data["dest_region"] = data["dest_region"].fillna("Unknown").astype(str)

    # Impute negative/zero lead or transit days with median
    data["planned_transit_days"] = np.where(
        data["planned_transit_days"] <= 0, 14.0, data["planned_transit_days"].fillna(14.0)
    )
    data["booking_lead_days"] = np.where(
        data["booking_lead_days"] < 0, 7.0, data["booking_lead_days"].fillna(7.0)
    )

    feature_df = data[ALL_FEATURE_COLUMNS].copy()

    # Verify no leakage
    validate_no_leakage(feature_df)

    return feature_df
