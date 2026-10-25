#!/usr/bin/env python3
"""
CLI entry point for running the Supply Chain GenAI Assistant.
Usage:
  python run_assistant.py --demo
  python run_assistant.py --question "What is the on-time rate for shipments from Shanghai to Rotterdam?"
  python run_assistant.py --interactive
"""

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.assistant.cli import main

if __name__ == "__main__":
    main()
