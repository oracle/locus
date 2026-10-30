#!/bin/bash
# Checks for this scenario. Runs from the project directory Codex built, after the run.
# Exit 0 means pass. Every line that starts with "ok:" is reported in the summary.
set -euo pipefail
PY=.venv/bin/python
$PY -m pytest -q test_flow.py
rm -rf approvals.json checkpoints
OUT=$($PY flow.py)
echo "$OUT" | grep -q "^HELD " && echo "ok: the run paused on the hold"
echo "$OUT" | grep -q "^REFUNDS 1$" && echo "ok: the refund ran exactly once after approval"
echo "$OUT" | grep -q "^TRAIL True$" && echo "ok: the audit chain verifies"
