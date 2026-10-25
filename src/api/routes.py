"""
REST API route definitions.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, status

from src.api.schemas import (
    HealthResponse,
    PaginatedShipmentsResponse,
    PredictDelayRequest,
    PredictDelayResponse,
    RouteStatsResponse,
    ShipmentDetail,
)
from src.data.db import get_read_connection
from src.data.dq_check import run_dq_checks
from src.ml.predict import get_predictor

router = APIRouter()
REPORT_PATH = Path("./reports/dq_report.json")


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness and readiness health check",
    tags=["System"],
)
def get_health() -> HealthResponse:
    """Check system health, database connection, and ML model status."""
    db_status = "connected"
    try:
        with get_read_connection() as conn:
            conn.execute("SELECT 1;").fetchone()
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"

    model_loaded = False
    try:
        predictor = get_predictor()
        model_loaded = predictor.model is not None
    except Exception:
        model_loaded = False

    overall_status = "healthy" if db_status == "connected" and model_loaded else "degraded"

    return HealthResponse(
        status=overall_status,
        version="1.0.0",
        service="supply-chain-intelligence-api",
        database=db_status,
        model_loaded=model_loaded,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.get(
    "/shipments/{shipment_id}",
    response_model=ShipmentDetail,
    summary="Get full details for a single shipment",
    tags=["Shipments"],
)
def get_shipment(shipment_id: str) -> ShipmentDetail:
    """Retrieve complete metadata and derived metrics for a shipment ID."""
    clean_id = shipment_id.strip()
    with get_read_connection() as conn:
        res = conn.execute(
            """
            SELECT
                shipment_id, customer_id, origin_port, destination_port, route_key,
                vessel_id, CAST(booking_date AS VARCHAR) AS booking_date,
                CAST(planned_departure AS VARCHAR) AS planned_departure,
                CAST(actual_departure AS VARCHAR) AS actual_departure,
                CAST(planned_arrival AS VARCHAR) AS planned_arrival,
                CAST(actual_arrival AS VARCHAR) AS actual_arrival,
                container_count, cargo_type, weight_tons, status,
                actual_delay_hours, on_time_flag, transit_days_planned,
                transit_days_actual, is_quarantined, quarantine_reasons
            FROM curated_shipments
            WHERE shipment_id = ?;
            """,
            [clean_id],
        ).fetchone()

    if not res:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Shipment '{clean_id}' not found",
        )

    columns = [
        "shipment_id", "customer_id", "origin_port", "destination_port", "route_key",
        "vessel_id", "booking_date", "planned_departure", "actual_departure",
        "planned_arrival", "actual_arrival", "container_count", "cargo_type",
        "weight_tons", "status", "actual_delay_hours", "on_time_flag",
        "transit_days_planned", "transit_days_actual", "is_quarantined", "quarantine_reasons"
    ]
    data = dict(zip(columns, res))
    return ShipmentDetail(**data)


@router.get(
    "/shipments",
    response_model=PaginatedShipmentsResponse,
    summary="List shipments with filters and pagination",
    tags=["Shipments"],
)
def list_shipments(
    origin: Optional[str] = Query(None, description="Origin port code filter (e.g. SGSIN, CNSHA)"),
    destination: Optional[str] = Query(None, description="Destination port code filter (e.g. NLRTM, DEHAM)"),
    status_filter: Optional[str] = Query(None, alias="status", description="Status filter (DELIVERED, DELAYED, CANCELLED)"),
    date_from: Optional[str] = Query(None, description="Planned departure start date (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="Planned departure end date (YYYY-MM-DD)"),
    page: int = Query(1, ge=1, description="Page number (1-based)"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    limit: Optional[int] = Query(None, ge=1, le=100, description="Optional override for limit"),
    offset: Optional[int] = Query(None, ge=0, description="Optional override for offset"),
) -> PaginatedShipmentsResponse:
    """List shipments with multi-field filtering and pagination."""
    # Determine pagination
    actual_limit = limit if limit is not None else page_size
    actual_offset = offset if offset is not None else (page - 1) * page_size
    current_page = (actual_offset // actual_limit) + 1

    where_clauses = ["1=1"]
    params: List[Any] = []

    if origin:
        where_clauses.append("origin_port = ?")
        params.append(origin.strip().upper())

    if destination:
        where_clauses.append("destination_port = ?")
        params.append(destination.strip().upper())

    if status_filter:
        where_clauses.append("UPPER(status) = ?")
        params.append(status_filter.strip().upper())

    if date_from:
        where_clauses.append("planned_departure >= ?")
        params.append(date_from.strip())

    if date_to:
        where_clauses.append("planned_departure <= ?")
        params.append(date_to.strip())

    where_sql = " AND ".join(where_clauses)

    with get_read_connection() as conn:
        count_res = conn.execute(
            f"SELECT COUNT(*) FROM curated_shipments WHERE {where_sql};",
            params,
        ).fetchone()
        total_count = count_res[0] if count_res else 0

        query_sql = f"""
            SELECT
                shipment_id, customer_id, origin_port, destination_port, route_key,
                vessel_id, CAST(booking_date AS VARCHAR) AS booking_date,
                CAST(planned_departure AS VARCHAR) AS planned_departure,
                CAST(actual_departure AS VARCHAR) AS actual_departure,
                CAST(planned_arrival AS VARCHAR) AS planned_arrival,
                CAST(actual_arrival AS VARCHAR) AS actual_arrival,
                container_count, cargo_type, weight_tons, status,
                actual_delay_hours, on_time_flag, transit_days_planned,
                transit_days_actual, is_quarantined, quarantine_reasons
            FROM curated_shipments
            WHERE {where_sql}
            ORDER BY booking_date DESC
            LIMIT ? OFFSET ?;
        """
        rows = conn.execute(query_sql, params + [actual_limit, actual_offset]).fetchall()

    columns = [
        "shipment_id", "customer_id", "origin_port", "destination_port", "route_key",
        "vessel_id", "booking_date", "planned_departure", "actual_departure",
        "planned_arrival", "actual_arrival", "container_count", "cargo_type",
        "weight_tons", "status", "actual_delay_hours", "on_time_flag",
        "transit_days_planned", "transit_days_actual", "is_quarantined", "quarantine_reasons"
    ]
    items = [ShipmentDetail(**dict(zip(columns, r))) for r in rows]
    total_pages = (total_count + actual_limit - 1) // actual_limit if total_count > 0 else 0

    return PaginatedShipmentsResponse(
        items=items,
        total=total_count,
        page=current_page,
        page_size=actual_limit,
        total_pages=total_pages,
    )


@router.get(
    "/routes/{origin}/{destination}/stats",
    response_model=RouteStatsResponse,
    summary="Get aggregated delay and on-time statistics for a route",
    tags=["Routes"],
)
def get_route_stats(origin: str, destination: str) -> RouteStatsResponse:
    """Retrieve route performance metrics: average delay, on-time rate, shipment count."""
    clean_origin = origin.strip().upper()
    clean_dest = destination.strip().upper()
    route_key = f"{clean_origin} → {clean_dest}"

    with get_read_connection() as conn:
        res = conn.execute(
            """
            SELECT
                shipment_count,
                avg_delay_hours,
                on_time_rate,
                avg_transit_days_actual,
                min_delay_hours,
                max_delay_hours
            FROM routes_summary
            WHERE origin_port = ? AND destination_port = ?;
            """,
            [clean_origin, clean_dest],
        ).fetchone()

    if not res or res[0] == 0:
        return RouteStatsResponse(
            origin=clean_origin,
            destination=clean_dest,
            route_key=route_key,
            shipment_count=0,
            average_delay_hours=0.0,
            on_time_rate=0.0,
            avg_transit_days_actual=None,
            min_delay_hours=None,
            max_delay_hours=None,
        )

    return RouteStatsResponse(
        origin=clean_origin,
        destination=clean_dest,
        route_key=route_key,
        shipment_count=res[0],
        average_delay_hours=round(float(res[1]), 2),
        on_time_rate=round(float(res[2]), 4),
        avg_transit_days_actual=round(float(res[3]), 2) if res[3] is not None else None,
        min_delay_hours=round(float(res[4]), 2) if res[4] is not None else None,
        max_delay_hours=round(float(res[5]), 2) if res[5] is not None else None,
    )


@router.post(
    "/predict-delay",
    response_model=PredictDelayResponse,
    summary="Predict shipment delay risk at booking time",
    tags=["Machine Learning"],
)
def predict_delay(req: PredictDelayRequest) -> PredictDelayResponse:
    """
    Call ML delay prediction model.
    Accepts either an existing shipment_id or booking-time feature attributes.
    """
    predictor = get_predictor()

    booking_payload: Dict[str, Any] = {}

    if req.shipment_id:
        clean_id = req.shipment_id.strip()
        with get_read_connection() as conn:
            row = conn.execute(
                """
                SELECT
                    shipment_id, origin_port, destination_port, cargo_type,
                    container_count, weight_tons,
                    CAST(booking_date AS VARCHAR) AS booking_date,
                    CAST(planned_departure AS VARCHAR) AS planned_departure,
                    CAST(planned_arrival AS VARCHAR) AS planned_arrival
                FROM curated_shipments
                WHERE shipment_id = ?;
                """,
                [clean_id],
            ).fetchone()

        if not row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Shipment '{clean_id}' not found to extract booking features.",
            )

        cols = [
            "shipment_id", "origin_port", "destination_port", "cargo_type",
            "container_count", "weight_tons", "booking_date",
            "planned_departure", "planned_arrival"
        ]
        booking_payload = dict(zip(cols, row))
    else:
        # Validate that minimal origin and destination are provided
        if not req.origin_port or not req.destination_port:
            raise HTTPException(
                status_code=422,
                detail="Either 'shipment_id' or both 'origin_port' and 'destination_port' must be provided.",
            )

        booking_payload = {
            "origin_port": req.origin_port,
            "destination_port": req.destination_port,
            "cargo_type": req.cargo_type or "General Cargo",
            "container_count": req.container_count or 20,
            "weight_tons": req.weight_tons or 500.0,
            "booking_date": req.booking_date or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "planned_departure": req.planned_departure or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "planned_arrival": req.planned_arrival or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        }

    prediction = predictor.predict(booking_payload)
    if req.shipment_id:
        prediction["shipment_id"] = req.shipment_id

    return PredictDelayResponse(**prediction)


@router.get(
    "/data-quality/report",
    summary="Get standalone Data Quality profiling report",
    tags=["Data Quality"],
)
def get_data_quality_report() -> Dict[str, Any]:
    """Return JSON report of all data quality checks performed on raw data."""
    if REPORT_PATH.exists():
        with open(REPORT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    # If report file does not exist, run checks and generate
    report = run_dq_checks("./data/raw", REPORT_PATH)
    return report.to_dict()
