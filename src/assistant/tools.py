"""
Tool execution functions and JSON schemas for assistant tool calling.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.data.db import get_read_connection
from src.ml.predict import get_predictor


# Port name to UN/LOCODE code mapper for conversational queries
PORT_NAME_TO_CODE = {
    "shanghai": "CNSHA",
    "rotterdam": "NLRTM",
    "singapore": "SGSIN",
    "shenzhen": "CNSZX",
    "ningbo": "CNNGB",
    "hong kong": "HKHKG",
    "los angeles": "USLAX",
    "long beach": "USLGB",
    "new york": "USNYC",
    "savannah": "USSAV",
    "antwerp": "BEANR",
    "hamburg": "DEHAM",
    "valencia": "ESVLC",
    "felixstowe": "GBFXT",
    "dubai": "AEDXB",
    "jebel ali": "AEDXB",
    "karachi": "PKKAR",
    "nhava sheva": "INNSA",
    "mumbai": "INNSA",
    "mundra": "INMUN",
    "port said": "EGPSD",
    "busan": "KRPUS",
    "yokohama": "JPYOK",
    "bangkok": "THBKK",
    "port klang": "MYPKG",
    "durban": "ZADUR",
    "santos": "BRSSZ",
}


def resolve_port_code(name_or_code: str) -> str:
    """Normalize user port input (e.g. 'Shanghai' or 'CNSHA') to uppercase 5-letter code."""
    cleaned = name_or_code.strip()
    if len(cleaned) == 5 and cleaned.isupper():
        return cleaned
    lookup = cleaned.lower()
    return PORT_NAME_TO_CODE.get(lookup, cleaned.upper())


def query_shipments(filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Query shipments from the curated data product.
    Supports filtering by origin, destination, status, cargo_type, and limit.
    """
    filters = filters or {}
    where_clauses = ["1=1"]
    params: List[Any] = []

    origin = filters.get("origin") or filters.get("origin_port")
    if origin:
        where_clauses.append("origin_port = ?")
        params.append(resolve_port_code(str(origin)))

    dest = filters.get("destination") or filters.get("destination_port")
    if dest:
        where_clauses.append("destination_port = ?")
        params.append(resolve_port_code(str(dest)))

    status = filters.get("status")
    if status:
        where_clauses.append("UPPER(status) = ?")
        params.append(str(status).strip().upper())

    cargo = filters.get("cargo_type")
    if cargo:
        where_clauses.append("LOWER(cargo_type) LIKE ?")
        params.append(f"%{str(cargo).strip().lower()}%")

    limit = int(filters.get("limit", 10))
    limit = min(max(limit, 1), 50)

    where_sql = " AND ".join(where_clauses)
    query_sql = f"""
        SELECT
            shipment_id, route_key, cargo_type, container_count, weight_tons,
            status, actual_delay_hours, on_time_flag,
            CAST(booking_date AS VARCHAR) AS booking_date,
            CAST(planned_departure AS VARCHAR) AS planned_departure,
            CAST(planned_arrival AS VARCHAR) AS planned_arrival,
            CAST(actual_arrival AS VARCHAR) AS actual_arrival
        FROM curated_shipments
        WHERE {where_sql}
        ORDER BY booking_date DESC
        LIMIT ?;
    """

    with get_read_connection() as conn:
        rows = conn.execute(query_sql, params + [limit]).fetchall()
        cols = [
            "shipment_id", "route_key", "cargo_type", "container_count", "weight_tons",
            "status", "actual_delay_hours", "on_time_flag", "booking_date",
            "planned_departure", "planned_arrival", "actual_arrival"
        ]
        items = [dict(zip(cols, r)) for r in rows]

    return {
        "count": len(items),
        "shipments": items,
        "applied_filters": filters,
    }


def get_route_stats(origin: str, destination: str) -> Dict[str, Any]:
    """
    Get aggregated delay and on-time metrics for a specific route.
    """
    code_orig = resolve_port_code(origin)
    code_dest = resolve_port_code(destination)
    route_key = f"{code_orig} → {code_dest}"

    with get_read_connection() as conn:
        row = conn.execute(
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
            [code_orig, code_dest],
        ).fetchone()

    if not row or row[0] == 0:
        return {
            "origin": code_orig,
            "destination": code_dest,
            "route_key": route_key,
            "shipment_count": 0,
            "average_delay_hours": 0.0,
            "on_time_rate": 0.0,
            "message": f"No completed shipment records found for route {route_key}.",
        }

    return {
        "origin": code_orig,
        "destination": code_dest,
        "route_key": route_key,
        "shipment_count": row[0],
        "average_delay_hours": round(float(row[1]), 2),
        "on_time_rate": round(float(row[2]), 4),
        "avg_transit_days_actual": round(float(row[3]), 2) if row[3] is not None else None,
        "min_delay_hours": round(float(row[4]), 2) if row[4] is not None else None,
        "max_delay_hours": round(float(row[5]), 2) if row[5] is not None else None,
    }


def get_top_delayed_routes(limit: int = 5) -> Dict[str, Any]:
    """
    Get routes with highest average delay.
    """
    with get_read_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                route_key,
                origin_port,
                destination_port,
                shipment_count,
                avg_delay_hours,
                on_time_rate
            FROM routes_summary
            WHERE shipment_count >= 5
            ORDER BY avg_delay_hours DESC
            LIMIT ?;
            """,
            [limit],
        ).fetchall()

    cols = ["route_key", "origin", "destination", "shipment_count", "avg_delay_hours", "on_time_rate"]
    routes = [dict(zip(cols, r)) for r in rows]
    return {
        "top_delayed_routes": routes,
        "criteria": "Routes with >= 5 shipments sorted by average delay hours descending",
    }


def predict_delay(shipment_id: str) -> Dict[str, Any]:
    """
    Call ML delay prediction model for a specific shipment ID.
    """
    clean_id = shipment_id.strip()
    with get_read_connection() as conn:
        row = conn.execute(
            """
            SELECT
                shipment_id, origin_port, destination_port, cargo_type,
                container_count, weight_tons,
                CAST(booking_date AS VARCHAR) AS booking_date,
                CAST(planned_departure AS VARCHAR) AS planned_departure,
                CAST(planned_arrival AS VARCHAR) AS planned_arrival,
                status
            FROM curated_shipments
            WHERE shipment_id = ?;
            """,
            [clean_id],
        ).fetchone()

    if not row:
        return {
            "error": f"Shipment '{clean_id}' was not found in the database.",
            "shipment_id": clean_id,
        }

    cols = [
        "shipment_id", "origin_port", "destination_port", "cargo_type",
        "container_count", "weight_tons", "booking_date",
        "planned_departure", "planned_arrival", "status"
    ]
    payload = dict(zip(cols, row))

    predictor = get_predictor()
    pred = predictor.predict(payload)

    return {
        "shipment_id": clean_id,
        "origin_port": payload["origin_port"],
        "destination_port": payload["destination_port"],
        "cargo_type": payload["cargo_type"],
        "delay_probability": pred["delay_probability"],
        "predicted_delayed": pred["predicted_delayed"],
        "risk_level": pred["risk_level"],
        "current_status": payload["status"],
        "model_version": pred["model_version"],
    }


# Tool catalog mapping
TOOL_MAP = {
    "query_shipments": query_shipments,
    "get_route_stats": get_route_stats,
    "get_top_delayed_routes": get_top_delayed_routes,
    "predict_delay": predict_delay,
}


# Standard JSON Schemas for LLM Tool Calling (OpenAI format)
OPENAI_TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "query_shipments",
            "description": "Reads and queries shipments from the curated data product with optional filters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string", "description": "Origin port name or code (e.g. 'Shanghai' or 'CNSHA')"},
                    "destination": {"type": "string", "description": "Destination port name or code (e.g. 'Rotterdam' or 'NLRTM')"},
                    "status": {"type": "string", "enum": ["DELIVERED", "DELAYED", "CANCELLED"]},
                    "cargo_type": {"type": "string", "description": "Type of cargo (e.g. Electronics, Chemicals)"},
                    "limit": {"type": "integer", "description": "Max rows to return (default: 10)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_route_stats",
            "description": "Returns aggregated historical delay hours, on-time rate, and shipment counts for a route.",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string", "description": "Origin port name or code (e.g. Shanghai or CNSHA)"},
                    "destination": {"type": "string", "description": "Destination port name or code (e.g. Rotterdam or NLRTM)"},
                },
                "required": ["origin", "destination"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_delayed_routes",
            "description": "Returns the routes with the highest average delay hours across the network.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Number of top delayed routes to return (default 5)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_delay",
            "description": "Calls the ML delay risk prediction model to assess booking-time delay probability for a shipment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "shipment_id": {"type": "string", "description": "The shipment ID (e.g. 'SHP-00421')"},
                },
                "required": ["shipment_id"],
            },
        },
    },
]
