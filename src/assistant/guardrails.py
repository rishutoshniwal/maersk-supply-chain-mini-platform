"""
Safety, domain validation, and citation guardrails for the GenAI Assistant.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class GuardrailResult:
    allowed: bool
    reason: Optional[str] = None
    refusal_message: Optional[str] = None


class DomainGuardrail:
    """Refuses queries outside the supply chain intelligence domain."""

    SUPPLY_CHAIN_KEYWORDS = {
        "shipment", "shipments", "cargo", "port", "ports", "route", "routes",
        "delay", "delays", "on-time", "ontime", "vessel", "vessels", "container",
        "containers", "transit", "arrival", "departure", "booking", "logistics",
        "freight", "teu", "tons", "weight", "shanghai", "rotterdam", "singapore",
        "los angeles", "long beach", "antwerp", "hamburg", "felixstowe",
        "karachi", "nhava sheva", "busan", "yokohama", "dubai", "durban",
        "santos", "cnsha", "nlrtm", "sgsin", "uslax", "uslgb", "beanr", "deham",
        "shp-", "evt-", "vsl-", "cust-", "risk", "predict", "stats", "quality"
    }

    OFF_DOMAIN_TRIGGERS = [
        r"\b(recipe|cook|bake|cake|food recipe|cocktail)\b",
        r"\b(poem|poetry|song lyrics|write a story|joke)\b",
        r"\b(politics|president|election|senate|prime minister)\b",
        r"\b(python|javascript|react|scrape|twitter|script to|code to)\b",
        r"\b(who won the world cup|capital of france|weather in paris)\b",
    ]

    def check(self, prompt: str) -> GuardrailResult:
        cleaned = prompt.strip().lower()

        # Check explicit off-domain patterns
        for pattern in self.OFF_DOMAIN_TRIGGERS:
            if re.search(pattern, cleaned):
                return GuardrailResult(
                    allowed=False,
                    reason="Explicit off-domain query detected.",
                    refusal_message=(
                        "I am Maersk's Supply Chain Intelligence Assistant. I am restricted to answering questions "
                        "regarding shipments, port routes, on-time performance, and delay risk predictions. "
                        "Please submit a supply-chain or logistics query."
                    ),
                )

        # Check keyword presence for short ambiguous queries
        words = set(re.findall(r"\w+", cleaned))
        has_supply_chain_terms = any(kw in cleaned for kw in self.SUPPLY_CHAIN_KEYWORDS)
        matches_shipment_id = bool(re.search(r"shp-\d{5}", cleaned, re.IGNORECASE))

        if not has_supply_chain_terms and not matches_shipment_id and len(words) < 3:
            return GuardrailResult(
                allowed=False,
                reason="Query lacks supply-chain context.",
                refusal_message=(
                    "Could you please specify your question in the context of shipments, routes, or delay predictions?"
                ),
            )

        return GuardrailResult(allowed=True)


class ToolCallBudgetGuardrail:
    """Enforces a strict upper bound on tool calls per session / turn to prevent runaway loops."""

    def __init__(self, max_calls_per_turn: int = 5):
        self.max_calls = max_calls_per_turn

    def check(self, current_call_count: int) -> GuardrailResult:
        if current_call_count >= self.max_calls:
            return GuardrailResult(
                allowed=False,
                reason=f"Exceeded maximum tool call budget ({self.max_calls} calls).",
                refusal_message="Tool invocation budget exceeded for this request. Returning consolidated response.",
            )
        return GuardrailResult(allowed=True)


class CitationGuardrail:
    """Requires the assistant to explicitly cite the tool source for any factual claims."""

    @staticmethod
    def enforce_citations(response_text: str, tools_invoked: List[Dict[str, Any]]) -> str:
        """Append explicit citations if not already formatted in the response."""
        if not tools_invoked:
            return response_text

        # Check if text already contains source tag
        if "[Source:" in response_text or "Source:" in response_text:
            return response_text

        # Append source citation block
        citations = []
        for call in tools_invoked:
            name = call.get("tool_name")
            args = call.get("arguments", {})
            args_str = ", ".join(f"{k}='{v}'" for k, v in args.items())
            citations.append(f"[Source: {name}({args_str})]")

        citation_block = "\n\n" + " ".join(citations)
        return response_text + citation_block
