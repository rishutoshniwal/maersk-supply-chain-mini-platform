"""
Unit tests for core business metrics and delay calculations.
"""

from datetime import datetime


def calculate_delay_hours(planned_arrival_str: str, actual_arrival_str: str | None) -> float | None:
    if not actual_arrival_str:
        return None
    fmt = "%Y-%m-%d %H:%M:%S"
    planned = datetime.strptime(planned_arrival_str, fmt)
    actual = datetime.strptime(actual_arrival_str, fmt)
    return round((actual - planned).total_seconds() / 3600.0, 2)


def is_on_time(actual_delay_hours: float | None) -> bool | None:
    if actual_delay_hours is None:
        return None
    return actual_delay_hours <= 24.0


def calculate_transit_days(departure_str: str, arrival_str: str | None) -> float | None:
    if not arrival_str or not departure_str:
        return None
    fmt = "%Y-%m-%d %H:%M:%S"
    dep = datetime.strptime(departure_str, fmt)
    arr = datetime.strptime(arrival_str, fmt)
    return round((arr - dep).total_seconds() / 86400.0, 2)


def test_on_time_flag_logic():
    # Exactly on-time
    assert is_on_time(0.0) is True
    # Arrived earlier than planned
    assert is_on_time(-5.0) is True
    # Arrived 24 hours late (within threshold)
    assert is_on_time(24.0) is True
    # Arrived 24.1 hours late (exceeds threshold)
    assert is_on_time(24.1) is False
    # Severely delayed
    assert is_on_time(96.0) is False
    # Unarrived / cancelled shipment (null-safe)
    assert is_on_time(None) is None


def test_delay_calculation():
    planned = "2024-07-10 12:00:00"
    actual_on_time = "2024-07-11 10:00:00"  # 22 hours later
    actual_delayed = "2024-07-12 14:00:00"  # 50 hours later

    delay_1 = calculate_delay_hours(planned, actual_on_time)
    assert delay_1 == 22.0
    assert is_on_time(delay_1) is True

    delay_2 = calculate_delay_hours(planned, actual_delayed)
    assert delay_2 == 50.0
    assert is_on_time(delay_2) is False

    assert calculate_delay_hours(planned, None) is None


def test_transit_days():
    dep = "2024-07-01 10:00:00"
    arr = "2024-07-15 10:00:00"
    assert calculate_transit_days(dep, arr) == 14.0
    assert calculate_transit_days(dep, None) is None
