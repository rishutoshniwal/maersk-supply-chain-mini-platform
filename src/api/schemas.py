"""
Pydantic schemas for REST API requests and responses.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., json_schema_extra={"example": "healthy"})
    version: str = Field(..., json_schema_extra={"example": "1.0.0"})
    service: str = Field(..., json_schema_extra={"example": "supply-chain-intelligence-api"})
    database: str = Field(..., json_schema_extra={"example": "connected"})
    model_loaded: bool = Field(..., json_schema_extra={"example": True})
    timestamp: str


class ShipmentDetail(BaseModel):
    shipment_id: str
    customer_id: str
    origin_port: str
    destination_port: str
    route_key: str
    vessel_id: str
    booking_date: Optional[str] = None
    planned_departure: Optional[str] = None
    actual_departure: Optional[str] = None
    planned_arrival: Optional[str] = None
    actual_arrival: Optional[str] = None
    container_count: int
    cargo_type: str
    weight_tons: float
    status: str
    actual_delay_hours: Optional[float] = None
    on_time_flag: Optional[bool] = None
    transit_days_planned: Optional[float] = None
    transit_days_actual: Optional[float] = None
    is_quarantined: bool = False
    quarantine_reasons: Optional[str] = None


class PaginatedShipmentsResponse(BaseModel):
    items: List[ShipmentDetail]
    total: int
    page: int
    page_size: int
    total_pages: int


class RouteStatsResponse(BaseModel):
    origin: str
    destination: str
    route_key: str
    shipment_count: int
    average_delay_hours: float
    on_time_rate: float
    avg_transit_days_actual: Optional[float] = None
    min_delay_hours: Optional[float] = None
    max_delay_hours: Optional[float] = None


class PredictDelayRequest(BaseModel):
    shipment_id: Optional[str] = Field(None, json_schema_extra={"example": "SHP-00421"})
    origin_port: Optional[str] = Field(None, json_schema_extra={"example": "CNSHA"})
    destination_port: Optional[str] = Field(None, json_schema_extra={"example": "NLRTM"})
    cargo_type: Optional[str] = Field(None, json_schema_extra={"example": "Electronics"})
    container_count: Optional[int] = Field(None, ge=1, json_schema_extra={"example": 35})
    weight_tons: Optional[float] = Field(None, gt=0, json_schema_extra={"example": 850.0})
    booking_date: Optional[str] = Field(None, json_schema_extra={"example": "2025-06-01 10:00:00"})
    planned_departure: Optional[str] = Field(None, json_schema_extra={"example": "2025-06-08 10:00:00"})
    planned_arrival: Optional[str] = Field(None, json_schema_extra={"example": "2025-07-05 10:00:00"})


class PredictDelayResponse(BaseModel):
    shipment_id: Optional[str] = None
    delay_probability: float
    predicted_delayed: bool
    delay_threshold_hours: float = 24.0
    risk_level: str
    model_version: str
    model_type: str


class ErrorDetail(BaseModel):
    detail: str
    error_code: str
    status_code: int
    timestamp: str
    path: Optional[str] = None
