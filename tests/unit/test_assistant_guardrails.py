"""
Unit tests for GenAI Assistant guardrails and citation enforcement.
"""

from src.assistant.guardrails import (
    CitationGuardrail,
    DomainGuardrail,
    ToolCallBudgetGuardrail,
)


def test_domain_guardrail_allows_supply_chain_queries():
    guard = DomainGuardrail()

    valid_queries = [
        "Which routes have had the highest average delay this quarter?",
        "What is the on-time rate for shipments from Shanghai to Rotterdam?",
        "What is the delay risk for shipment SHP-00421?",
        "List all delayed shipments carrying electronics to Antwerp.",
    ]

    for q in valid_queries:
        res = guard.check(q)
        assert res.allowed is True, f"Valid query was falsely blocked: {q}"


def test_domain_guardrail_blocks_off_domain_queries():
    guard = DomainGuardrail()

    invalid_queries = [
        "Can you share a recipe for chocolate cake?",
        "Write a poem about the sea and sailors.",
        "Who won the presidential election?",
        "Write a python script to scrape twitter.",
    ]

    for q in invalid_queries:
        res = guard.check(q)
        assert res.allowed is False, f"Off-domain query was not blocked: {q}"
        assert res.refusal_message is not None


def test_tool_call_budget_guardrail():
    budget = ToolCallBudgetGuardrail(max_calls_per_turn=3)

    assert budget.check(0).allowed is True
    assert budget.check(2).allowed is True
    assert budget.check(3).allowed is False
    assert budget.check(4).allowed is False


def test_citation_guardrail_appends_sources():
    tools = [
        {"tool_name": "get_route_stats", "arguments": {"origin": "Shanghai", "destination": "Rotterdam"}},
    ]
    raw_response = "The on-time rate for Shanghai to Rotterdam is 75.0%."

    cited = CitationGuardrail.enforce_citations(raw_response, tools)
    assert "[Source: get_route_stats(origin='Shanghai', destination='Rotterdam')]" in cited

    # Test idempotency (doesn't duplicate existing citation)
    cited_again = CitationGuardrail.enforce_citations(cited, tools)
    assert cited_again == cited
