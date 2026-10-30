#!/bin/bash
# Checks for this scenario. Runs from the project directory Codex built, after the run.
# Exit 0 means pass. Every line that starts with "ok:" is reported in the summary.
set -euo pipefail
PY=.venv/bin/python
$PY -m pytest -q test_triage_graph.py
OUT=$($PY triage_graph.py)
echo "$OUT" | grep -q "^REPORT 3" && echo "ok: three fan-out results merged"
echo "$OUT" | grep -q "merge" && echo "ok: mermaid names the merge node"
