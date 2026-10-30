#!/bin/bash
# Checks for this scenario. Runs from the project directory Codex built, after the run.
# Exit 0 means pass. Every line that starts with "ok:" is reported in the summary.
set -euo pipefail
PY=.venv/bin/python
$PY -m pytest -q test_incident_commander.py
$PY - <<'EOF'
import asyncio, inspect
from incident_commander import run
result = run("Why are checkout requests failing?")
if inspect.isawaitable(result):
    result = asyncio.run(result)
types = sorted(r.specialist_type for r in result.specialist_results.values())
assert types == ["log_analyst", "metrics_analyst"], types
print("ok: both specialists ran:", types)
EOF
