"""
GenAI Assistant Agent with Tool Calling, Guardrails, and Production Observability Logging.
Supports both OpenAI API function-calling and deterministic local fallback routing.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.assistant.guardrails import (
    CitationGuardrail,
    DomainGuardrail,
    ToolCallBudgetGuardrail,
)
from src.assistant.tools import (
    OPENAI_TOOL_DEFINITIONS,
    TOOL_MAP,
)

AUDIT_LOG_PATH = Path("./logs/assistant_audit.jsonl")


class SupplyChainAssistant:
    """Agent orchestrating LLM tool calling, guardrails, and audit logging."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        audit_log_path: Path = AUDIT_LOG_PATH,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name
        self.audit_log_path = Path(audit_log_path)
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)

        self.domain_guardrail = DomainGuardrail()
        self.budget_guardrail = ToolCallBudgetGuardrail(max_calls_per_turn=5)

    def _log_audit_event(self, event: Dict[str, Any]) -> None:
        """Write structured audit log entry for production observability."""
        with open(self.audit_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

    def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        """Execute tool and measure execution latency."""
        tool_fn = TOOL_MAP.get(tool_name)
        if not tool_fn:
            return {"error": f"Tool '{tool_name}' not found."}, 0.0

        start = time.perf_counter()
        try:
            res = tool_fn(**args)
        except Exception as e:
            res = {"error": f"Tool execution failed: {str(e)}"}
        latency_ms = round((time.perf_counter() - start) * 1000.0, 2)
        return res, latency_ms

    def _deterministic_router(self, user_prompt: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Deterministic, zero-dependency semantic tool-calling engine.
        Enables instant out-of-the-box local testing without third-party API keys.
        """
        prompt = user_prompt.strip()
        lower = prompt.lower()
        tools_called: List[Dict[str, Any]] = []

        # 1. Delay risk prediction query (e.g. "What is the delay risk for shipment SHP-00421?")
        shp_match = re.search(r"SHP-\d{5}", prompt, re.IGNORECASE)
        if shp_match and any(w in lower for w in ["delay", "risk", "predict", "late", "status", "forecast"]):
            shipment_id = shp_match.group(0).upper()
            tool_args = {"shipment_id": shipment_id}
            tool_res, latency = self._execute_tool("predict_delay", tool_args)
            tools_called.append({
                "tool_name": "predict_delay",
                "arguments": tool_args,
                "response": tool_res,
                "latency_ms": latency,
            })

            if "error" in tool_res:
                ans = f"Could not assess delay risk: {tool_res['error']}"
            else:
                prob = tool_res["delay_probability"]
                risk = tool_res["risk_level"]
                flag = tool_res["predicted_delayed"]
                cargo = tool_res.get("cargo_type", "General")
                orig = tool_res.get("origin_port", "Unknown")
                dest = tool_res.get("destination_port", "Unknown")
                ans = (
                    f"Shipment **{shipment_id}** ({orig} → {dest}, Cargo: {cargo}) has a delay risk assessed as **{risk}** "
                    f"with a predicted delay probability of **{prob:.1%}**. "
                    f"Predicted to arrive > 24 hours late: **{'Yes' if flag else 'No'}**."
                )
            return ans, tools_called

        # 2. Route metrics / On-time rate query (e.g. "What is the on-time rate for shipments from Shanghai to Rotterdam?")
        route_patterns = [
            r"from\s+([A-Za-z\s]+?)\s+to\s+([A-Za-z\s]+)",
            r"between\s+([A-Za-z\s]+?)\s+and\s+([A-Za-z\s]+)",
            r"route\s+([A-Za-z\s]+?)\s*(?:->|→|to)\s*([A-Za-z\s]+)",
        ]
        matched_route = None
        for pat in route_patterns:
            m = re.search(pat, prompt, re.IGNORECASE)
            if m:
                matched_route = (m.group(1).strip(" ?.,"), m.group(2).strip(" ?.,"))
                break

        if matched_route and any(w in lower for w in ["on-time", "ontime", "rate", "stats", "performance", "average delay", "how many"]):
            orig_name, dest_name = matched_route
            tool_args = {"origin": orig_name, "destination": dest_name}
            tool_res, latency = self._execute_tool("get_route_stats", tool_args)
            tools_called.append({
                "tool_name": "get_route_stats",
                "arguments": tool_args,
                "response": tool_res,
                "latency_ms": latency,
            })

            count = tool_res.get("shipment_count", 0)
            if count == 0:
                ans = f"No completed shipment records were found for route **{orig_name} → {dest_name}**."
            else:
                ot_rate = tool_res.get("on_time_rate", 0.0)
                avg_delay = tool_res.get("average_delay_hours", 0.0)
                route_key = tool_res.get("route_key", f"{orig_name} → {dest_name}")
                ans = (
                    f"For route **{route_key}** (based on {count} completed shipments):\n"
                    f"- **On-Time Rate**: **{ot_rate:.1%}**\n"
                    f"- **Average Delay**: **{avg_delay} hours**\n"
                    f"- **Min / Max Delay**: {tool_res.get('min_delay_hours')}h / {tool_res.get('max_delay_hours')}h"
                )
            return ans, tools_called

        # 3. Top delayed routes query (e.g. "Which routes have had the highest average delay this quarter?")
        if any(phrase in lower for phrase in ["highest average delay", "top delayed", "worst routes", "most delayed routes", "highest delay"]):
            tool_args = {"limit": 5}
            tool_res, latency = self._execute_tool("get_top_delayed_routes", tool_args)
            tools_called.append({
                "tool_name": "get_top_delayed_routes",
                "arguments": tool_args,
                "response": tool_res,
                "latency_ms": latency,
            })

            routes = tool_res.get("top_delayed_routes", [])
            lines = ["Here are the top routes with the highest average delay across the network:"]
            for i, r in enumerate(routes, 1):
                lines.append(
                    f"{i}. **{r['route_key']}**: Avg Delay **{r['avg_delay_hours']} hrs** "
                    f"(On-time rate: {r['on_time_rate']:.1%}, Completed shipments: {r['shipment_count']})"
                )
            ans = "\n".join(lines)
            return ans, tools_called

        # 4. General query on shipments
        tool_args = {"limit": 5}
        if "delayed" in lower:
            tool_args["status"] = "DELAYED"
        elif "delivered" in lower:
            tool_args["status"] = "DELIVERED"
        if "electronics" in lower:
            tool_args["cargo_type"] = "Electronics"

        tool_res, latency = self._execute_tool("query_shipments", tool_args)
        tools_called.append({
            "tool_name": "query_shipments",
            "arguments": tool_args,
            "response": tool_res,
            "latency_ms": latency,
        })
        items = tool_res.get("shipments", [])
        if not items:
            ans = "I queried the curated shipment records, but found no matching entries for your criteria."
        else:
            lines = [f"Retrieved {len(items)} shipment records from the curated data product:"]
            for s in items[:5]:
                lines.append(
                    f"- **{s['shipment_id']}** ({s['route_key']}): Status **{s['status']}**, "
                    f"Delay: {s.get('actual_delay_hours', 'N/A')}h, Cargo: {s.get('cargo_type')}"
                )
            ans = "\n".join(lines)
        return ans, tools_called

    def _call_openai_with_tools(self, user_prompt: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Execute LLM tool calling using OpenAI API."""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
        except Exception:
            # Fall back to deterministic router if OpenAI library fails
            return self._deterministic_router(user_prompt)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are Maersk's Supply Chain Intelligence Assistant. You answer questions about container shipments, "
                    "route analytics, port operations, and delay risks. You MUST call tools to retrieve accurate factual data. "
                    "Always cite the tool and parameters used."
                ),
            },
            {"role": "user", "content": user_prompt},
        ]

        tools_called: List[Dict[str, Any]] = []
        call_count = 0

        # Initial call
        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=OPENAI_TOOL_DEFINITIONS,
            tool_choice="auto",
        )
        msg = response.choices[0].message

        # Handle tool calls in a loop with budget guardrail
        while msg.tool_calls and call_count < self.budget_guardrail.max_calls:
            call_count += len(msg.tool_calls)
            messages.append(msg)

            for t_call in msg.tool_calls:
                name = t_call.function.name
                args = json.loads(t_call.function.arguments) if t_call.function.arguments else {}
                res, latency = self._execute_tool(name, args)
                tools_called.append({
                    "tool_name": name,
                    "arguments": args,
                    "response": res,
                    "latency_ms": latency,
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": t_call.id,
                    "content": json.dumps(res),
                })

            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
            )
            msg = response.choices[0].message

        final_content = msg.content or "No response generated."
        return final_content, tools_called

    def ask(self, user_prompt: str) -> Dict[str, Any]:
        """
        Process user query end-to-end:
        1. Check Domain Guardrail.
        2. Execute tool calling (OpenAI API or Deterministic Engine).
        3. Enforce Citation Guardrail.
        4. Write audit log for production observability.
        """
        session_id = str(uuid.uuid4())
        start_time = time.perf_counter()

        # Step 1: Guardrail Check
        guardrail_res = self.domain_guardrail.check(user_prompt)
        if not guardrail_res.allowed:
            total_latency = round((time.perf_counter() - start_time) * 1000.0, 2)
            audit_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "session_id": session_id,
                "prompt": user_prompt,
                "guardrail_status": "BLOCKED",
                "guardrail_reason": guardrail_res.reason,
                "tools_invoked": [],
                "final_response": guardrail_res.refusal_message,
                "latency_ms": total_latency,
            }
            self._log_audit_event(audit_entry)
            return {
                "session_id": session_id,
                "prompt": user_prompt,
                "response": guardrail_res.refusal_message,
                "tools_called": [],
                "guardrail_blocked": True,
                "latency_ms": total_latency,
            }

        # Step 2: Tool Execution & Generation
        if self.api_key:
            try:
                response_text, tools_called = self._call_openai_with_tools(user_prompt)
            except Exception:
                response_text, tools_called = self._deterministic_router(user_prompt)
        else:
            response_text, tools_called = self._deterministic_router(user_prompt)

        # Step 3: Citation Enforcement
        grounded_response = CitationGuardrail.enforce_citations(response_text, tools_called)
        total_latency = round((time.perf_counter() - start_time) * 1000.0, 2)

        # Step 4: Audit Observability Logging
        audit_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "prompt": user_prompt,
            "guardrail_status": "PASSED",
            "guardrail_reason": None,
            "tools_invoked": tools_called,
            "final_response": grounded_response,
            "latency_ms": total_latency,
        }
        self._log_audit_event(audit_entry)

        return {
            "session_id": session_id,
            "prompt": user_prompt,
            "response": grounded_response,
            "tools_called": tools_called,
            "guardrail_blocked": False,
            "latency_ms": total_latency,
        }
