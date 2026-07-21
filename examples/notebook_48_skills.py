# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""Notebook 48: skills — a payments-ops skill library with progressive disclosure.

A Skill (the AgentSkills.io shape) bundles a name, a description, and
a block of instructions — think of each one as a vetted operations
procedure: dispute triage, refund-policy review, safe queries against
the ledger. The ``SkillsPlugin`` exposes a catalog of skills to the
agent and only injects the full instructions once the agent activates a
specific one. Progressive disclosure keeps the system prompt small and
the payments agent focused on the procedure that matters right now.

- ``Skill`` — built in code or loaded from a ``SKILL.md`` file with
  YAML front-matter.
- ``Skill.from_directory(path)`` — load every skill under a directory
  (your team's reviewed, versioned procedure library).
- ``Agent(skills=[...])`` — wires up the SkillsPlugin and the
  ``skills`` tool that the agent calls to activate one.
- The ``SKILL.md`` format itself — front-matter for ``name``,
  ``description``, ``allowed-tools``, ``metadata``, plus the
  instruction body and optional ``scripts/``, ``references/``,
  ``assets/`` directories.

Run it:
    # The bundled mock model is the default; set TULIP_MODEL_PROVIDER for a live provider.
    TULIP_MODEL_ID=openai.gpt-4.1 python examples/notebook_48_skills.py

    # Offline:
    TULIP_MODEL_PROVIDER=mock python examples/notebook_48_skills.py

Prerequisites:
- An OpenAI or Anthropic API key, or set ``TULIP_MODEL_PROVIDER`` to
  ``openai`` / ``anthropic`` / ``mock``.
- Optional: an ``examples/skills/`` directory with one or more
  ``SKILL.md`` files for Part 2 to find (the bundled ones include
  dispute-triage and a refund-policy review checklist).
"""

import asyncio
from pathlib import Path

from config import get_model

from tulip.agent import Agent, AgentConfig
from tulip.skills import Skill


# =============================================================================
# Part 1: Build a Skill in code — no SKILL.md file required.
# =============================================================================


async def example_programmatic():
    print("=== Part 1: Programmatic Skills ===\n")

    model = get_model()

    dispute_triage = Skill(
        name="dispute-triage",
        description="Use when triaging a card dispute to decide valid/invalid chargeback.",
        instructions=(
            "# Dispute Triage Checklist\n"
            "1. Check the cardholder's dispute reason code and history\n"
            "2. Check whether the transaction matches the account's spend baseline\n"
            "3. Classify as VALID CHARGEBACK or INVALID CHARGEBACK with confidence\n"
            "4. Report findings as: FINDING: <description>"
        ),
    )

    agent = Agent(
        config=AgentConfig(
            system_prompt="You are a payments operations analyst. Use available skills.",
            max_iterations=5,
            model=model,
            skills=[dispute_triage],
        )
    )

    result = await agent.arun(
        "Triage: $420 chargeback on card ending 4242 at merchant 'Acme', "
        "cardholder claims 'item not received' but tracking shows delivered at 03:14Z"
    )
    print(f"Response: {result.message[:200]}...")

    skills_used = [te for te in result.tool_executions if te.tool_name == "skills"]
    print(f"Skills activated: {len(skills_used)}")


# =============================================================================
# Part 2: Load every SKILL.md under a directory — the team's vetted
#         procedure library (dispute-triage, refund-policy, fraud-scoring, ledger-query, …).
# =============================================================================


async def example_filesystem():
    print("\n=== Part 2: Filesystem Skills ===\n")

    skills_dir = Path(__file__).parent / "skills"
    if skills_dir.exists():
        skills = Skill.from_directory(skills_dir)
        print(f"Loaded {len(skills)} skills:")
        for s in skills:
            print(f"  - {s.name}: {s.description[:60]}...")
    else:
        print("No skills directory found. Create examples/skills/my-skill/SKILL.md")

    import time as _t

    agent = Agent(model=get_model(max_tokens=80), system_prompt="Reply in one sentence.")
    t0 = _t.perf_counter()
    res = await agent.arun(
        "In one sentence, why should a payments team version its operations procedures as "
        "reviewable SKILL.md files instead of hard-coding prompts in source?"
    )
    dt = _t.perf_counter() - t0
    print(
        f"  [model call: {dt:.2f}s · {res.metrics.prompt_tokens}→{res.metrics.completion_tokens} tokens]"
    )
    print(f"  AI rationale: {res.message.strip()}")


# =============================================================================
# Part 3: SKILL.md file format — YAML front-matter + instruction body.
# =============================================================================


async def example_format():
    print("\n=== Part 3: SKILL.md Format ===\n")

    print("""
---
name: bin-enrichment
description: Use when the user provides a card indicator (BIN, PAN prefix, issuer) to enrich.
allowed-tools: lookup_bin fetch_issuer
metadata:
  author: payments-team
  version: "1.0"
---

# Instructions for the Agent

1. First, identify the indicator type
2. Then, enrich it with the allowed tools
3. Finally, summarize the verdict with evidence

## Resource Files
Place additional files in:
- scripts/   — executable code
- references/ — documentation
- assets/    — templates, data
    """)

    import time as _t

    agent = Agent(model=get_model(max_tokens=120), system_prompt="Reply in one short paragraph.")
    t0 = _t.perf_counter()
    res = await agent.arun(
        "Write a one-paragraph SKILL.md description for a skill named "
        "'ledger-forensics' that helps an agent reconstruct a transaction's "
        "settlement timeline from authorization and clearing logs."
    )
    dt = _t.perf_counter() - t0
    print(
        f"  [model call: {dt:.2f}s · {res.metrics.prompt_tokens}→{res.metrics.completion_tokens} tokens]"
    )
    print(f"  AI-authored sample description:\n  {res.message.strip()}")


async def main() -> None:
    await example_programmatic()
    await example_filesystem()
    await example_format()


if __name__ == "__main__":
    asyncio.run(main())
