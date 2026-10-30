#!/bin/bash
# Checks for this scenario. Runs from the project directory Codex built, after the run.
# Exit 0 means pass. Every line that starts with "ok:" is reported in the summary.
set -euo pipefail
PY=.venv/bin/python
$PY -m pytest -q test_memory_agent.py
$PY memory_agent.py | grep -q "^TURNS 1 2 1" && echo "ok: history reloads per thread"
