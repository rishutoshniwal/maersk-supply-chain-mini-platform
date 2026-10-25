"""
Unit tests for Data Quality (DQ) checks.
Runs in milliseconds without external database infrastructure.
"""

import pandas as pd
import pytest

from src.data.dq_check import DataQualityRunner


@pytest.fixture
def sample_ports_df():
    return pd.DataFrame([
        {"port_code": "SGSIN", "port_name": "Singapore", "country": "Singapore", "region": "APAC", "timezone": "UTC+8", "avg_congestion_score": 0.72},
        {"port_code": "BEANR", "port_name": "Antwerp", "country": None, "region": "EUR", "timezone": "UTC+1", "avg_congestion_score": 0.70},
        {"port_code": "CNSHA", "port_name": "Shanghai", "country": "China", "region": "APAC", "timezone": "UTC+8", "avg_congestion_score": 0.85},
    ])


@pytest.fixture
def sample_shipments_df():
    return pd.DataFrame([
        # Row 0: Valid shipment
        {
            "shipment_id": "SHP-00001", "customer_id": "CUST-1001",
            "origin_port": "SGSIN", "destination_port": "BEANR", "vessel_id": "VSL-0001",
            "booking_date": "2024-06-01 10:00:00", "planned_departure": "2024-06-05 10:00:00",
            "actual_departure": "2024-06-05 12:00:00", "planned_arrival": "2024-06-25 10:00:00",
            "actual_arrival": "2024-06-25 14:00:00", "container_count": 20, "cargo_type": "Electronics",
            "weight_tons": 450.0, "status": "DELIVERED"
        },
        # Row 1: Duplicate shipment ID with Row 0
        {
            "shipment_id": "SHP-00001", "customer_id": "CUST-1001",
            "origin_port": "SGSIN", "destination_port": "BEANR", "vessel_id": "VSL-0001",
            "booking_date": "2024-06-01 11:00:00", "planned_departure": "2024-06-05 10:00:00",
            "actual_departure": "2024-06-05 12:00:00", "planned_arrival": "2024-06-25 10:00:00",
            "actual_arrival": "2024-06-25 14:00:00", "container_count": 20, "cargo_type": "Electronics",
            "weight_tons": 460.0, "status": "DELIVERED"
        },
        # Row 2: Invalid origin port (XXTST)
        {
            "shipment_id": "SHP-00002", "customer_id": "CUST-1002",
            "origin_port": "XXTST", "destination_port": "BEANR", "vessel_id": "VSL-0002",
            "booking_date": "2024-06-02 10:00:00", "planned_departure": "2024-06-06 10:00:00",
            "actual_departure": "2024-06-06 12:00:00", "planned_arrival": "2024-06-26 10:00:00",
            "actual_arrival": "2024-06-26 14:00:00", "container_count": 10, "cargo_type": "Chemicals",
            "weight_tons": 250.0, "status": "DELIVERED"
        },
        # Row 3: Negative weight & zero container count
        {
            "shipment_id": "SHP-00003", "customer_id": "CUST-1003",
            "origin_port": "SGSIN", "destination_port": "CNSHA", "vessel_id": "VSL-0003",
            "booking_date": "2024-06-03 10:00:00", "planned_departure": "2024-06-07 10:00:00",
            "actual_departure": "2024-06-07 12:00:00", "planned_arrival": "2024-06-27 10:00:00",
            "actual_arrival": "2024-06-27 14:00:00", "container_count": 0, "cargo_type": " Chemicals",
            "weight_tons": -50.0, "status": "delivered"
        },
        # Row 4: Departure before booking anomaly
        {
            "shipment_id": "SHP-00004", "customer_id": "CUST-1004",
            "origin_port": "SGSIN", "destination_port": "CNSHA", "vessel_id": "VSL-0004",
            "booking_date": "2024-06-10 10:00:00", "planned_departure": "2024-06-15 10:00:00",
            "actual_departure": "2024-06-08 10:00:00", "planned_arrival": "2024-06-25 10:00:00",
            "actual_arrival": "2024-06-25 14:00:00", "container_count": 15, "cargo_type": "FURNITURE",
            "weight_tons": 300.0, "status": "COMPLETED"
        },
    ])


@pytest.fixture
def sample_events_df():
    return pd.DataFrame([
        {"event_id": "EVT-001", "event_timestamp": "2024-06-01 10:00:00", "port_code": "SGSIN", "vessel_id": "VSL-0001", "event_type": "ARRIVAL", "delay_minutes": 0, "notes": "On schedule"},
        {"event_id": "EVT-001", "event_timestamp": "2024-06-01 12:00:00", "port_code": "SGSIN", "vessel_id": "VSL-0001", "event_type": None, "delay_minutes": 30, "notes": "Weather delay – fog"},
        {"event_id": "EVT-002", "event_timestamp": "2024-06-02 10:00:00", "port_code": "CNSHA", "vessel_id": "VSL-0002", "event_type": "BERTHED", "delay_minutes": 0, "notes": None},
    ])


def test_ports_missing_country(sample_ports_df, sample_shipments_df, sample_events_df):
    runner = DataQualityRunner(sample_ports_df, sample_shipments_df, sample_events_df)
    res = runner.check_ports_missing_country()
    assert res.status == "FAIL"
    assert res.rows_affected == 1
    assert "BEANR" in res.sample_affected_ids
    assert res.severity == "warning"


def test_shipments_duplicate_primary_key(sample_ports_df, sample_shipments_df, sample_events_df):
    runner = DataQualityRunner(sample_ports_df, sample_shipments_df, sample_events_df)
    res = runner.check_shipments_duplicate_id()
    assert res.status == "FAIL"
    assert res.rows_affected == 2
    assert "SHP-00001" in res.sample_affected_ids
    assert res.severity == "critical"


def test_shipments_invalid_origin_port(sample_ports_df, sample_shipments_df, sample_events_df):
    runner = DataQualityRunner(sample_ports_df, sample_shipments_df, sample_events_df)
    res = runner.check_shipments_invalid_origin_port()
    assert res.status == "FAIL"
    assert res.rows_affected == 1
    assert "SHP-00002" in res.sample_affected_ids
    assert res.severity == "critical"


def test_shipments_invalid_container_and_weight(sample_ports_df, sample_shipments_df, sample_events_df):
    runner = DataQualityRunner(sample_ports_df, sample_shipments_df, sample_events_df)
    res_container = runner.check_shipments_invalid_container_count()
    res_weight = runner.check_shipments_negative_zero_weight()
    assert res_container.status == "FAIL"
    assert res_container.rows_affected == 1
    assert res_weight.status == "FAIL"
    assert res_weight.rows_affected == 1


def test_shipments_departure_before_booking(sample_ports_df, sample_shipments_df, sample_events_df):
    runner = DataQualityRunner(sample_ports_df, sample_shipments_df, sample_events_df)
    res = runner.check_shipments_departure_before_booking()
    assert res.status == "FAIL"
    assert res.rows_affected == 1
    assert "SHP-00004" in res.sample_affected_ids


def test_events_duplicate_and_missing_types(sample_ports_df, sample_shipments_df, sample_events_df):
    runner = DataQualityRunner(sample_ports_df, sample_shipments_df, sample_events_df)
    res_dup = runner.check_events_duplicate_event_id()
    res_type = runner.check_events_missing_event_type()
    res_notes = runner.check_events_missing_notes()
    assert res_dup.status == "FAIL"
    assert res_dup.rows_affected == 2
    assert res_type.status == "FAIL"
    assert res_type.rows_affected == 1
    assert res_notes.status == "FAIL"
    assert res_notes.rows_affected == 1
