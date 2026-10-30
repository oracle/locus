#!/bin/bash
# Checks for this scenario. Runs from the project directory Codex built, after the run.
# Exit 0 means pass. Every line that starts with "ok:" is reported in the summary.
set -euo pipefail
PY=.venv/bin/python
$PY -m pytest -q test_triage.py
grep -q "output_schema" triage.py && echo "ok: typed answer configured"
grep -Eq "MaxIterations.*\|.*ToolCalled|ToolCalled.*\|.*MaxIterations" triage.py && echo "ok: termination composed"
