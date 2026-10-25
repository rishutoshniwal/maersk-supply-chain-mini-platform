#!/usr/bin/env python3
"""
CLI entry point for running data quality checks standalone.
Usage: python dq_check.py --input ./data/raw --output ./reports/dq_report.json
"""

import sys
from pathlib import Path

# Ensure repo root is on python path
repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.data.dq_check import main

if __name__ == "__main__":
    main()
