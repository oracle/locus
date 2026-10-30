#!/bin/bash
# Checks for this scenario. Runs from the project directory Codex built, after the run.
# Exit 0 means pass. Every line that starts with "ok:" is reported in the summary.
set -euo pipefail
PY=.venv/bin/python
$PY -m pytest -q test_support_agent.py
grep -q "gate_tool" support_agent.py && echo "ok: refund tool is gated"
grep -q "AuditTrail" support_agent.py && echo "ok: audit trail present"
$PY - <<'EOF'
from locus.testing import ScriptedModel, text, tool_call
import support_agent
model = ScriptedModel([tool_call("issue_refund", order_id="ORD-7842", amount_usd=500.0), text("Done.")])
support_agent.build_agent(model).run_sync("Refund ORD-7842 for $500.")
trails = [v for v in vars(support_agent).values() if type(v).__name__ == "AuditTrail"]
assert trails, "a module-level AuditTrail"
outcome = trails[0].records()[-1].payload["outcome"]
assert outcome == "require_human", outcome
assert trails[0].verify()
print("ok: a 500 refund is held and the chain verifies")
EOF
