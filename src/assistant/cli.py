"""
Command-line interface (CLI) for the Supply Chain GenAI Assistant.
"""

from __future__ import annotations

import argparse

from src.assistant.agent import SupplyChainAssistant


def run_demo(assistant: SupplyChainAssistant) -> None:
    """Run demonstration with the assignment prompt questions and guardrail check."""
    questions = [
        "Which routes have had the highest average delay this quarter?",
        "What is the on-time rate for shipments from Shanghai to Rotterdam?",
        "What is the delay risk for shipment SHP-00421?",
        "Can you share a recipe for chocolate cake?",  # Guardrail test
    ]

    print("\n" + "=" * 90)
    print("  MAERSK SUPPLY CHAIN INTELLIGENCE ASSISTANT — DEMO WALKTHROUGH")
    print("=" * 90)

    for i, q in enumerate(questions, 1):
        print(f"\n[Question {i}]: \"{q}\"")
        res = assistant.ask(q)
        tools_used = [t["tool_name"] for t in res["tools_called"]]
        print(f"Tools Invoked: {tools_used if tools_used else 'None (Guardrail Blocked)'} | Latency: {res['latency_ms']}ms")
        print("-" * 90)
        print(res["response"])
        print("-" * 90)

    print("\n[+] Demo completed! Structured audit log recorded at: ./logs/assistant_audit.jsonl\n")


def interactive_session(assistant: SupplyChainAssistant) -> None:
    """Run interactive REPL session."""
    print("\n" + "=" * 80)
    print("  MAERSK SUPPLY CHAIN INTELLIGENCE ASSISTANT (CLI REPL)")
    print("  Type your questions below. Type 'exit' or 'quit' to end.")
    print("=" * 80 + "\n")

    while True:
        try:
            prompt = input("SupplyChainAgent > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not prompt:
            continue
        if prompt.lower() in ["exit", "quit", "q"]:
            print("Goodbye.")
            break

        res = assistant.ask(prompt)
        tools_used = [t["tool_name"] for t in res["tools_called"]]
        print(f"\n[Tools: {tools_used or 'None'}] [Latency: {res['latency_ms']}ms]")
        print(res["response"])
        print()


def main():
    parser = argparse.ArgumentParser(description="Supply Chain GenAI Assistant CLI")
    parser.add_argument("--question", "-q", type=str, help="Single question to answer")
    parser.add_argument("--demo", action="store_true", help="Run assignment questions demo")
    parser.add_argument("--interactive", "-i", action="store_true", help="Launch interactive REPL")
    args = parser.parse_args()

    assistant = SupplyChainAssistant()

    if args.question:
        res = assistant.ask(args.question)
        print(res["response"])
    elif args.demo:
        run_demo(assistant)
    elif args.interactive:
        interactive_session(assistant)
    else:
        # Default to running demo if no args provided
        run_demo(assistant)


if __name__ == "__main__":
    main()
