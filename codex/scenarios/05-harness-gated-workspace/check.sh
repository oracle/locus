#!/bin/bash
# Checks for this scenario. Runs from the project directory Codex built, after the run.
# Exit 0 means pass. Every line that starts with "ok:" is reported in the summary.
set -euo pipefail
PY=.venv/bin/python
$PY -m pytest -q test_workspace_tools.py
$PY - <<'EOF'
import asyncio
from workspace_tools import build
harness, trail = build({"app.py": "print('hi')\n"})
out = asyncio.run(harness.tool("write").execute(path="new.py", content="x = 1\n"))
assert "held_for_approval" in str(out), out
assert trail.verify()
print("ok: writes are held, reads pass, chain verifies")
EOF
