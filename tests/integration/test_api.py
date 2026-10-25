"""
Integration / End-to-End tests exercising real HTTP endpoints on the FastAPI service.
"""

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "1.0.0"
    assert data["database"] == "connected"
    assert data["model_loaded"] is True
    assert "timestamp" in data
    # Verify structured logging headers
    assert "x-request-id" in response.headers
    assert "x-response-time-ms" in response.headers


def test_shipment_list_and_filters():
    # 1. Base list
    resp = client.get("/shipments?page=1&page_size=5")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] > 0
    assert len(data["items"]) == 5
    assert data["page"] == 1
    assert data["page_size"] == 5

    # 2. Filter by status
    resp_delayed = client.get("/shipments?status=DELAYED&page_size=5")
    assert resp_delayed.status_code == 200
    delayed_items = resp_delayed.json()["items"]
    for item in delayed_items:
        assert item["status"] == "DELAYED"


def test_shipment_successful_lookup():
    # First get a valid ID from the list
    list_resp = client.get("/shipments?page_size=1")
    assert list_resp.status_code == 200
    valid_id = list_resp.json()["items"][0]["shipment_id"]

    # Lookup by ID
    resp = client.get(f"/shipments/{valid_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["shipment_id"] == valid_id
    assert "origin_port" in data
    assert "destination_port" in data
    assert "route_key" in data
    assert "status" in data
    assert "weight_tons" in data
    assert "container_count" in data


def test_shipment_unknown_id_returns_404():
    unknown_id = "SHP-NONEXISTENT-99999"
    resp = client.get(f"/shipments/{unknown_id}")
    assert resp.status_code == 404
    data = resp.json()
    assert data["error_code"] == "RESOURCE_NOT_FOUND"
    assert "not found" in data["detail"].lower()
    assert data["status_code"] == 404
    assert "timestamp" in data
    assert data["path"] == f"/shipments/{unknown_id}"


def test_route_stats_endpoint():
    resp = client.get("/routes/USLAX/PKKAR/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["origin"] == "USLAX"
    assert data["destination"] == "PKKAR"
    assert data["route_key"] == "USLAX → PKKAR"
    assert data["shipment_count"] > 0
    assert isinstance(data["average_delay_hours"], float)
    assert 0.0 <= data["on_time_rate"] <= 1.0


def test_predict_delay_by_shipment_id():
    list_resp = client.get("/shipments?page_size=1")
    valid_id = list_resp.json()["items"][0]["shipment_id"]

    resp = client.post("/predict-delay", json={"shipment_id": valid_id})
    assert resp.status_code == 200
    data = resp.json()
    assert data["shipment_id"] == valid_id
    assert 0.0 <= data["delay_probability"] <= 1.0
    assert isinstance(data["predicted_delayed"], bool)
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert data["model_version"] == "1.0.0"


def test_predict_delay_by_booking_payload():
    payload = {
        "origin_port": "CNSHA",
        "destination_port": "NLRTM",
        "cargo_type": "Electronics",
        "container_count": 45,
        "weight_tons": 1100.0,
        "booking_date": "2025-06-01 10:00:00",
        "planned_departure": "2025-06-08 10:00:00",
        "planned_arrival": "2025-07-05 10:00:00",
    }
    resp = client.post("/predict-delay", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert 0.0 <= data["delay_probability"] <= 1.0
    assert isinstance(data["predicted_delayed"], bool)
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]


def test_predict_delay_invalid_payload():
    # Neither shipment_id nor origin/destination provided
    resp = client.post("/predict-delay", json={})
    assert resp.status_code == 422


def test_data_quality_report_endpoint():
    resp = client.get("/data-quality/report")
    assert resp.status_code == 200
    data = resp.json()
    assert "summary" in data
    assert "checks" in data
    assert data["summary"]["total_checks"] == 16
    assert data["summary"]["critical_issues"] > 0
