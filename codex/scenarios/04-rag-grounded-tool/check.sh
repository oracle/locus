#!/bin/bash
# Checks for this scenario. Runs from the project directory Codex built, after the run.
# Exit 0 means pass. Every line that starts with "ok:" is reported in the summary.
set -euo pipefail
PY=.venv/bin/python
$PY -m pytest -q test_kb_agent.py
grep -Eq "create_rag_tool|as_tool" kb_agent.py && echo "ok: retrieval is exposed as a tool"
$PY - <<'EOF'
import asyncio
from kb_agent import make_retriever
hits = asyncio.run(make_retriever().retrieve("refund approval threshold", limit=1))
assert "approval" in hits.documents[0].document.content.lower()
print("ok: the refund policy ranks first")
EOF
