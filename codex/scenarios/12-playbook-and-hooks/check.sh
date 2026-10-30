#!/bin/bash
# Checks for this scenario. Runs from the project directory Codex built, after the run.
# Exit 0 means pass. Every line that starts with "ok:" is reported in the summary.
set -euo pipefail
PY=.venv/bin/python
$PY -m pytest -q test_refund_playbook.py
grep -q "PlaybookEnforcerHook" refund_playbook.py && echo "ok: playbook enforced"
grep -q "HookProvider" refund_playbook.py && echo "ok: custom hook present"
