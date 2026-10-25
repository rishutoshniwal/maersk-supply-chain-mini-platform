"""
Integration tests for the GenAI Assistant end-to-end question answering and audit logging.
"""

from src.assistant.agent import SupplyChainAssistant


def test_assistant_on_time_rate_query(tmp_path):
    audit_log = tmp_path / "test_audit.jsonl"
    assistant = SupplyChainAssistant(audit_log_path=audit_log)

    res = assistant.ask("What is the on-time rate for shipments from Shanghai to Rotterdam?")
    assert not res["guardrail_blocked"]
    assert len(res["tools_called"]) == 1
    assert res["tools_called"][0]["tool_name"] == "get_route_stats"
    assert "CNSHA → NLRTM" in res["response"]
    assert "[Source: get_route_stats" in res["response"]
    assert audit_log.exists()


def test_assistant_top_delayed_routes_query(tmp_path):
    audit_log = tmp_path / "test_audit.jsonl"
    assistant = SupplyChainAssistant(audit_log_path=audit_log)

    res = assistant.ask("Which routes have had the highest average delay this quarter?")
    assert not res["guardrail_blocked"]
    assert len(res["tools_called"]) == 1
    assert res["tools_called"][0]["tool_name"] == "get_top_delayed_routes"
    assert "highest average delay" in res["response"]
    assert "[Source: get_top_delayed_routes" in res["response"]


def test_assistant_predict_delay_query(tmp_path):
    audit_log = tmp_path / "test_audit.jsonl"
    assistant = SupplyChainAssistant(audit_log_path=audit_log)

    res = assistant.ask("What is the delay risk for shipment SHP-00421?")
    assert not res["guardrail_blocked"]
    assert len(res["tools_called"]) == 1
    assert res["tools_called"][0]["tool_name"] == "predict_delay"
    assert "SHP-00421" in res["response"]
    assert "[Source: predict_delay(shipment_id='SHP-00421')]" in res["response"]


def test_assistant_blocks_off_domain(tmp_path):
    audit_log = tmp_path / "test_audit.jsonl"
    assistant = SupplyChainAssistant(audit_log_path=audit_log)

    res = assistant.ask("Can you share a recipe for chocolate cake?")
    assert res["guardrail_blocked"] is True
    assert len(res["tools_called"]) == 0
    assert "restricted to answering questions regarding shipments" in res["response"]
