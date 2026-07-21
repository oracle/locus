#!/usr/bin/env python3
# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Notebook 74: SOC playbooks over the first-class security toolset.

The SDK ships curated IR/SOC playbooks (``tulip.security.phishing_triage``,
``nist_800_61_ir``, ``ransomware_containment``, ``cloud_posture_audit``) and
the agent-ready security adapters they drive (``security_toolset()`` —
IOC enrichment, SIEM search, EDR forensics, vuln/posture scanning, inference
fingerprinting). A playbook pins an investigation to its steps in order via
the ``PlaybookEnforcer``; the toolset gives the agent the tools each step
names.

This notebook:
  1. lists the bundled playbooks,
  2. wires one onto an Agent with the security toolset and runs it,
  3. walks the PlaybookEnforcer deterministically to show the step gate.

Runs offline on the bundled mock model — no credentials, no network. Set
``TULIP_MODEL_PROVIDER`` (+ key) for a live provider.

Run:
    python examples/notebook_74_security_playbooks.py
    TULIP_MODEL_PROVIDER=mock python examples/notebook_74_security_playbooks.py
"""

from __future__ import annotations

import asyncio

from config import get_model

from tulip.agent import Agent
from tulip.playbooks import PlaybookEnforcer
from tulip.security import all_playbooks, phishing_triage, security_toolset


async def main() -> int:
    # 1. The bundled playbooks.
    print("== Bundled security playbooks ==")
    for pid, pb in all_playbooks().items():
        steps = " -> ".join(s.id for s in pb.steps)
        print(f"  {pid:24s} {steps}")
    print()

    # 2. Wire a playbook + the read-only triage toolset onto an agent.
    playbook = phishing_triage()
    tools = security_toolset(allow_containment=True)  # triage loop + isolate_host
    print(f"== Running '{playbook.name}' with {len(tools)} security tools (mock model) ==")
    agent = Agent(
        model=get_model(),
        tools=tools,
        playbook=playbook,
        system_prompt=(
            "You are a SOC analyst. Follow the playbook steps in order: gather "
            "the events, enrich the indicators, scope the affected hosts, and "
            "contain only on confirmation. Cite the evidence behind every step."
        ),
        max_iterations=8,
    )
    result = await agent.arun(
        "A user reported a phishing email linking to login.phish.example.net. Triage it."
    )
    print(f"  agent result (truncated): {str(result.text)[:160]}")
    print()

    # 3. The step gate, deterministically (independent of what the mock chose
    #    to call): walk each step's expected tools and watch progress advance.
    print("== PlaybookEnforcer step gate ==")
    enforcer = PlaybookEnforcer.from_playbook(playbook, block_violations=True)
    for step in playbook.steps:
        if not step.expected_tools:
            enforcer.complete_current_step()
            continue
        for tool_name in step.expected_tools:
            verdict = enforcer.validate_tool_call(tool_name)
            mark = "ok" if verdict.allowed else "blocked"
            print(f"  [{step.id}] {tool_name:20s} -> {mark}")
            if verdict.allowed:
                enforcer.record_tool_call(tool_name)
        enforcer.complete_current_step()
    print(f"  progress: {enforcer.progress:.0%}  violations: {len(enforcer.violations)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
