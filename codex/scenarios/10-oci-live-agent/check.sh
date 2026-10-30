#!/bin/bash
# Checks for this scenario. Runs from the project directory Codex built, after the run.
# Exit 0 means pass. Every line that starts with "ok:" is reported in the summary.
set -euo pipefail
PY=.venv/bin/python
$PY -m pytest -q test_run.py
OUT=$($PY run.py 2>/dev/null)
echo "$OUT" | grep -qi "ORD-7842" && echo "ok: the answer names the order"
echo "$OUT" | grep -qi "delivered" && echo "ok: the answer carries the tool's status"
