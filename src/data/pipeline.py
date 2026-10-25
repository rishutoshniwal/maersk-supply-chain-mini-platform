"""
Layered, idempotent ETL pipeline: Raw -> Curated -> Serving.
Transforms raw CSVs into an analytics-ready DuckDB data product.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from src.data.db import get_write_connection
from src.data.dq_check import resolve_raw_files


def run_pipeline(
    raw_dir: str | Path = "./data/raw",
    db_path: Optional[str | Path] = None,
) -> dict:
    """
    Execute layered ETL pipeline.
    Idempotent: Re-running drops and recreates tables deterministically without duplicates.
    """
    ports_csv, shipments_csv, events_csv = resolve_raw_files(raw_dir)
    target_db = Path(db_path) if db_path else None

    with get_write_connection(target_db) as conn:
        print("[*] Starting Data Pipeline Ingestion...")

        # -------------------------------------------------------------
        # 1. RAW LAYER: Ingest raw CSVs as staging tables
        # -------------------------------------------------------------
        conn.execute("BEGIN TRANSACTION;")

        conn.execute(f"""
            CREATE OR REPLACE TABLE raw_ports AS
            SELECT * FROM read_csv_auto('{ports_csv}', header=True);
        """)

        conn.execute(f"""
            CREATE OR REPLACE TABLE raw_shipments AS
            SELECT * FROM read_csv_auto('{shipments_csv}', header=True);
        """)

        conn.execute(f"""
            CREATE OR REPLACE TABLE raw_port_events AS
            SELECT * FROM read_csv_auto('{events_csv}', header=True);
        """)

        # -------------------------------------------------------------
        # 2. CURATED LAYER: Cleaned, Deduplicated, Validated & Enriched
        # -------------------------------------------------------------

        # 2a. Curated Ports: fix missing country (Antwerp -> Belgium)
        conn.execute("""
            CREATE OR REPLACE TABLE curated_ports AS
            SELECT
                TRIM(port_code) AS port_code,
                TRIM(port_name) AS port_name,
                CASE
                    WHEN TRIM(port_code) = 'BEANR' AND (country IS NULL OR TRIM(country) = '') THEN 'Belgium'
                    ELSE TRIM(country)
                END AS country,
                TRIM(region) AS region,
                TRIM(timezone) AS timezone,
                CAST(avg_congestion_score AS DOUBLE) AS avg_congestion_score
            FROM raw_ports;
        """)

        # 2b. Curated Port Events: resolve duplicate event_ids with surrogate key, impute missing event_type & notes
        conn.execute("""
            CREATE OR REPLACE TABLE curated_port_events AS
            SELECT
                TRIM(event_id) AS event_id,
                TRIM(event_id) || '_' || MD5(
                    COALESCE(event_timestamp, '') || '_' ||
                    COALESCE(port_code, '') || '_' ||
                    COALESCE(vessel_id, '') || '_' ||
                    COALESCE(notes, '')
                ) AS surrogate_id,
                CAST(event_timestamp AS TIMESTAMP) AS event_timestamp,
                TRIM(port_code) AS port_code,
                TRIM(vessel_id) AS vessel_id,
                CASE
                    WHEN event_type IS NOT NULL AND TRIM(event_type) != '' THEN TRIM(event_type)
                    WHEN LOWER(notes) LIKE '%weather%' OR LOWER(notes) LIKE '%congestion%'
                         OR LOWER(notes) LIKE '%strike%' OR LOWER(notes) LIKE '%fault%'
                         OR LOWER(notes) LIKE '%missed%' THEN 'DELAYED'
                    WHEN LOWER(notes) LIKE '%berth%' THEN 'BERTHED'
                    WHEN LOWER(notes) LIKE '%customs%' THEN 'CUSTOMS_CLEARED'
                    ELSE 'UNKNOWN'
                END AS event_type,
                CAST(COALESCE(delay_minutes, 0) AS INTEGER) AS delay_minutes,
                COALESCE(TRIM(notes), '') AS notes
            FROM raw_port_events;
        """)

        # Calculate median weights per cargo_type to impute null weights
        conn.execute("""
            CREATE OR REPLACE TEMP TABLE _cargo_weight_medians AS
            SELECT
                CASE
                    WHEN LOWER(TRIM(cargo_type)) = 'chemicals' THEN 'Chemicals'
                    WHEN LOWER(TRIM(cargo_type)) = 'furniture' THEN 'Furniture'
                    WHEN LOWER(TRIM(cargo_type)) = 'machinery' THEN 'Machinery'
                    WHEN LOWER(TRIM(cargo_type)) = 'pharmaceuticals' THEN 'Pharmaceuticals'
                    WHEN LOWER(TRIM(cargo_type)) = 'electronics' THEN 'Electronics'
                    WHEN LOWER(TRIM(cargo_type)) = 'consumer goods' THEN 'Consumer Goods'
                    WHEN LOWER(TRIM(cargo_type)) = 'apparel' THEN 'Apparel'
                    WHEN LOWER(TRIM(cargo_type)) = 'food & beverage' THEN 'Food & Beverage'
                    WHEN LOWER(TRIM(cargo_type)) = 'raw materials' THEN 'Raw Materials'
                    WHEN LOWER(TRIM(cargo_type)) = 'automotive parts' THEN 'Automotive Parts'
                    ELSE 'General Cargo'
                END AS clean_cargo_type,
                MEDIAN(CAST(weight_tons AS DOUBLE)) AS median_weight
            FROM raw_shipments
            WHERE weight_tons IS NOT NULL AND CAST(weight_tons AS DOUBLE) > 0
            GROUP BY 1;
        """)

        # 2c. Curated Shipments: deduplicate, clean strings, derive required fields, flag quarantine
        conn.execute("""
            CREATE OR REPLACE TABLE curated_shipments AS
            WITH ranked_shipments AS (
                SELECT
                    TRIM(shipment_id) AS shipment_id,
                    TRIM(customer_id) AS customer_id,
                    TRIM(origin_port) AS origin_port,
                    TRIM(destination_port) AS destination_port,
                    TRIM(vessel_id) AS vessel_id,
                    CAST(booking_date AS TIMESTAMP) AS booking_date,
                    CAST(planned_departure AS TIMESTAMP) AS planned_departure,
                    CAST(actual_departure AS TIMESTAMP) AS actual_departure,
                    CAST(planned_arrival AS TIMESTAMP) AS planned_arrival,
                    CAST(actual_arrival AS TIMESTAMP) AS actual_arrival,
                    CAST(container_count AS INTEGER) AS container_count,
                    CASE
                        WHEN cargo_type IS NULL OR TRIM(cargo_type) = '' THEN 'Unknown Cargo'
                        WHEN LOWER(TRIM(cargo_type)) = 'chemicals' THEN 'Chemicals'
                        WHEN LOWER(TRIM(cargo_type)) = 'furniture' THEN 'Furniture'
                        WHEN LOWER(TRIM(cargo_type)) = 'machinery' THEN 'Machinery'
                        WHEN LOWER(TRIM(cargo_type)) = 'pharmaceuticals' THEN 'Pharmaceuticals'
                        WHEN LOWER(TRIM(cargo_type)) = 'electronics' THEN 'Electronics'
                        WHEN LOWER(TRIM(cargo_type)) = 'consumer goods' THEN 'Consumer Goods'
                        WHEN LOWER(TRIM(cargo_type)) = 'apparel' THEN 'Apparel'
                        WHEN LOWER(TRIM(cargo_type)) = 'food & beverage' THEN 'Food & Beverage'
                        WHEN LOWER(TRIM(cargo_type)) = 'raw materials' THEN 'Raw Materials'
                        WHEN LOWER(TRIM(cargo_type)) = 'automotive parts' THEN 'Automotive Parts'
                        ELSE TRIM(cargo_type)
                    END AS cargo_type,
                    CAST(weight_tons AS DOUBLE) AS raw_weight_tons,
                    CASE
                        WHEN status IS NULL OR TRIM(status) = '' THEN
                            CASE
                                WHEN actual_arrival IS NOT NULL THEN
                                    CASE
                                        WHEN (EPOCH(CAST(actual_arrival AS TIMESTAMP)) - EPOCH(CAST(planned_arrival AS TIMESTAMP))) / 3600.0 <= 24.0 THEN 'DELIVERED'
                                        ELSE 'DELAYED'
                                    END
                                ELSE 'UNKNOWN'
                            END
                        WHEN UPPER(TRIM(status)) IN ('DELIVERED', 'COMPLETED', 'COMPLETE') THEN 'DELIVERED'
                        WHEN UPPER(TRIM(status)) = 'DELAYED' THEN 'DELAYED'
                        WHEN UPPER(TRIM(status)) = 'CANCELLED' THEN 'CANCELLED'
                        ELSE UPPER(TRIM(status))
                    END AS status,
                    ROW_NUMBER() OVER (
                        PARTITION BY TRIM(shipment_id)
                        ORDER BY CAST(booking_date AS TIMESTAMP) DESC
                    ) AS rn
                FROM raw_shipments
            ),
            deduped_shipments AS (
                SELECT * EXCLUDE (rn)
                FROM ranked_shipments
                WHERE rn = 1
            ),
            valid_ports_list AS (
                SELECT port_code FROM curated_ports
            )
            SELECT
                s.shipment_id,
                s.customer_id,
                s.origin_port,
                s.destination_port,
                s.origin_port || ' → ' || s.destination_port AS route_key,
                s.vessel_id,
                s.booking_date,
                s.planned_departure,
                s.actual_departure,
                s.planned_arrival,
                s.actual_arrival,
                s.container_count,
                s.cargo_type,
                COALESCE(
                    CASE WHEN s.raw_weight_tons > 0 THEN s.raw_weight_tons ELSE NULL END,
                    m.median_weight,
                    500.0
                ) AS weight_tons,
                s.status,

                -- Required derived fields
                -- 1. actual_delay_hours (actual_arrival - planned_arrival, null-safe)
                CASE
                    WHEN s.actual_arrival IS NOT NULL AND s.planned_arrival IS NOT NULL
                    THEN ROUND((EPOCH(s.actual_arrival) - EPOCH(s.planned_arrival)) / 3600.0, 2)
                    ELSE NULL
                END AS actual_delay_hours,

                -- 2. on_time_flag (true when delay <= 24 hours, null-safe)
                CASE
                    WHEN s.actual_arrival IS NULL THEN NULL
                    WHEN (EPOCH(s.actual_arrival) - EPOCH(s.planned_arrival)) / 3600.0 <= 24.0 THEN TRUE
                    ELSE FALSE
                END AS on_time_flag,

                -- 3. transit_days_planned
                ROUND((EPOCH(s.planned_arrival) - EPOCH(s.planned_departure)) / 86400.0, 2) AS transit_days_planned,

                -- 4. transit_days_actual
                CASE
                    WHEN s.actual_arrival IS NOT NULL AND s.actual_departure IS NOT NULL
                    THEN ROUND((EPOCH(s.actual_arrival) - EPOCH(s.actual_departure)) / 86400.0, 2)
                    ELSE NULL
                END AS transit_days_actual,

                -- Quarantine detection
                CASE
                    WHEN s.origin_port NOT IN (SELECT port_code FROM valid_ports_list)
                         OR s.destination_port NOT IN (SELECT port_code FROM valid_ports_list)
                         OR s.container_count <= 0
                         OR s.raw_weight_tons <= 0
                         OR (s.actual_departure IS NOT NULL AND s.actual_departure < s.booking_date)
                    THEN TRUE
                    ELSE FALSE
                END AS is_quarantined,

                CONCAT_WS(', ',
                    CASE WHEN s.origin_port NOT IN (SELECT port_code FROM valid_ports_list) THEN 'invalid_origin_port' END,
                    CASE WHEN s.destination_port NOT IN (SELECT port_code FROM valid_ports_list) THEN 'invalid_destination_port' END,
                    CASE WHEN s.container_count <= 0 THEN 'invalid_container_count' END,
                    CASE WHEN s.raw_weight_tons <= 0 THEN 'negative_or_zero_weight' END,
                    CASE WHEN s.actual_departure IS NOT NULL AND s.actual_departure < s.booking_date THEN 'departure_before_booking' END
                ) AS quarantine_reasons
            FROM deduped_shipments s
            LEFT JOIN _cargo_weight_medians m ON s.cargo_type = m.clean_cargo_type;
        """)

        # -------------------------------------------------------------
        # 3. SERVING LAYER: Pre-aggregated metrics for high-speed API
        # -------------------------------------------------------------
        conn.execute("""
            CREATE OR REPLACE TABLE routes_summary AS
            SELECT
                origin_port,
                destination_port,
                route_key,
                COUNT(*) AS shipment_count,
                ROUND(AVG(actual_delay_hours), 2) AS avg_delay_hours,
                ROUND(AVG(CASE WHEN on_time_flag THEN 1.0 ELSE 0.0 END), 4) AS on_time_rate,
                ROUND(AVG(transit_days_actual), 2) AS avg_transit_days_actual,
                MIN(actual_delay_hours) AS min_delay_hours,
                MAX(actual_delay_hours) AS max_delay_hours
            FROM curated_shipments
            WHERE NOT is_quarantined
              AND actual_arrival IS NOT NULL
            GROUP BY origin_port, destination_port, route_key;
        """)

        # Create indexes on curated tables for instant queries
        conn.execute("CREATE INDEX IF NOT EXISTS idx_shipments_id ON curated_shipments(shipment_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_shipments_route ON curated_shipments(origin_port, destination_port);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_shipments_status ON curated_shipments(status);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_routes_summary ON routes_summary(origin_port, destination_port);")

        conn.execute("COMMIT;")

        # Gather summary counts
        ports_count = conn.execute("SELECT COUNT(*) FROM curated_ports;").fetchone()[0]
        shipments_count = conn.execute("SELECT COUNT(*) FROM curated_shipments;").fetchone()[0]
        quarantined_count = conn.execute("SELECT COUNT(*) FROM curated_shipments WHERE is_quarantined;").fetchone()[0]
        events_count = conn.execute("SELECT COUNT(*) FROM curated_port_events;").fetchone()[0]
        routes_count = conn.execute("SELECT COUNT(*) FROM routes_summary;").fetchone()[0]

        summary = {
            "curated_ports": ports_count,
            "curated_shipments": shipments_count,
            "quarantined_shipments": quarantined_count,
            "curated_port_events": events_count,
            "active_routes_summarized": routes_count,
        }

        print("[+] Layered Pipeline executed successfully!")
        for k, v in summary.items():
            print(f"    - {k}: {v:,}")

        return summary


def main():
    parser = argparse.ArgumentParser(description="Execute layered supply chain ETL pipeline.")
    parser.add_argument("--raw-dir", "-r", default="./data/raw", help="Path to raw CSV directory")
    parser.add_argument("--db-path", "-d", default="./data/curated/supply_chain.duckdb", help="DuckDB output database path")
    args = parser.parse_args()

    run_pipeline(args.raw_dir, args.db_path)


if __name__ == "__main__":
    main()
