"""
Unit tests for ML feature engineering and data leakage prevention.
"""

import pandas as pd
import pytest

from src.ml.features import (
    ALL_FEATURE_COLUMNS,
    LEAKAGE_COLUMNS,
    extract_booking_features,
    validate_no_leakage,
)


def test_leakage_detection_raises_error():
    # Construct dataframe with forbidden actuals
    df_leak = pd.DataFrame([{
        "shipment_id": "SHP-12345",
        "actual_arrival": "2024-06-25 10:00:00",
        "container_count": 20,
    }])
    with pytest.raises(ValueError, match="DATA LEAKAGE DETECTED"):
        validate_no_leakage(df_leak)


def test_no_leakage_in_extracted_features():
    raw_df = pd.DataFrame([{
        "shipment_id": "SHP-00100",
        "origin_port": "CNSHA",
        "destination_port": "NLRTM",
        "cargo_type": "Electronics",
        "container_count": 25,
        "weight_tons": 600.0,
        "booking_date": "2025-06-01 10:00:00",
        "planned_departure": "2025-06-07 10:00:00",
        "planned_arrival": "2025-07-02 10:00:00",
        "actual_departure": "2025-06-07 14:00:00",  # Raw data has actuals
        "actual_arrival": "2025-07-03 18:00:00",
    }])

    port_meta = {
        "CNSHA": {"avg_congestion_score": 0.85, "region": "APAC"},
        "NLRTM": {"avg_congestion_score": 0.65, "region": "EUR"},
    }

    features = extract_booking_features(raw_df, port_meta)

    # 1. Assert no forbidden columns exist in the output
    for col in LEAKAGE_COLUMNS:
        assert col not in features.columns, f"Leakage column {col} found in feature matrix!"

    # 2. Assert exact feature schema
    assert sorted(features.columns.tolist()) == sorted(ALL_FEATURE_COLUMNS)

    # 3. Check calculated ratio values
    assert features["weight_per_container"].iloc[0] == 600.0 / 25
    assert features["planned_transit_days"].iloc[0] == 25.0
    assert features["booking_lead_days"].iloc[0] == 6.0
    assert features["origin_congestion_score"].iloc[0] == 0.85
    assert features["dest_congestion_score"].iloc[0] == 0.65


def test_features_handle_null_and_zero_values():
    raw_df = pd.DataFrame([{
        "shipment_id": "SHP-00200",
        "origin_port": "UNKNOWN",
        "destination_port": "UNKNOWN",
        "cargo_type": None,
        "container_count": 0,
        "weight_tons": -100.0,
        "booking_date": "2025-01-01 00:00:00",
        "planned_departure": "2025-01-05 00:00:00",
        "planned_arrival": "2025-01-15 00:00:00",
    }])

    features = extract_booking_features(raw_df)
    assert features["container_count"].iloc[0] == 1  # Imputed positive
    assert features["weight_tons"].iloc[0] == 500.0  # Imputed default
    assert features["cargo_type"].iloc[0] == "Unknown Cargo"
    assert not features.isna().any().any()
