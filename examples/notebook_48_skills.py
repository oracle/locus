# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
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
- ``Skill.from_file(path)`` — load one skill directory or ``SKILL.md``.
- ``Agent(skills=[...])`` — wires up the SkillsPlugin and the
  ``skills`` tool that the agent calls to activate one.
- The ``SKILL.md`` format itself — front-matter for ``name``,
  ``description``, ``allowed-tools``, ``metadata``, plus the
  instruction body and optional ``scripts/``, ``references/``,
  ``assets/`` directories.

Run it:
    # Default: OCI GenAI if ~/.oci/config has a usable profile, else the bundled mock model
    LOCUS_MODEL_PROVIDER=openai LOCUS_MODEL_ID=gpt-4o python examples/notebook_48_skills.py

    # Offline:
    LOCUS_MODEL_PROVIDER=mock python examples/notebook_48_skills.py

Prerequisites:
- An OpenAI or Anthropic API key, or set ``LOCUS_MODEL_PROVIDER`` to
  ``openai`` / ``anthropic`` / ``mock``.
- Optional: an ``examples/skills/`` directory with one or more
  ``SKILL.md`` files for Part 2 to find. The bundled ones are
  ``code-review``, ``incident-triage``, ``ioc-enrichment`` and
  ``siem-query``.
"""

import asyncio
import tempfile
from pathlib import Path

from config import get_model

from locus.agent import Agent, AgentConfig
from locus.skills import Skill


def _banner(result) -> None:
    m = result.metrics
    print(
        f"  [model call: {m.duration_ms / 1000.0:.2f}s · {m.prompt_tokens}→{m.completion_tokens} tokens]"
    )


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
#         procedure library (code-review, incident-triage, ioc-enrichment,
#         siem-query) — and hand the whole catalog to one agent.
# =============================================================================


async def example_filesystem():
    print("\n=== Part 2: Filesystem Skills ===\n")

    skills_dir = Path(__file__).parent / "skills"
    if not skills_dir.exists():
        print("No skills directory found. Create examples/skills/my-skill/SKILL.md")
        return

    skills = Skill.from_directory(skills_dir)
    print(f"Loaded {len(skills)} skills:")
    for s in skills:
        print(f"  - {s.name}: {s.description[:60]}...")

    # The agent sees only the catalog (names + descriptions) up front; the
    # full instructions arrive when it activates one — so four procedures
    # cost the system prompt four lines, not four runbooks.
    agent = Agent(
        model=get_model(max_tokens=200),
        system_prompt="You are the on-call operations agent. Use available skills.",
        skills=skills,
        max_iterations=4,
    )
    result = await agent.arun(
        "Diagnose: the checkout-api p99 latency alert has been firing for 10 minutes "
        "in us-ashburn-1 and error rate is climbing."
    )
    _banner(result)
    print(f"Response: {result.message[:200]}...")
    skills_used = [te for te in result.tool_executions if te.tool_name == "skills"]
    print(f"Skills activated: {len(skills_used)}")


# =============================================================================
# Part 3: SKILL.md file format — YAML front-matter + instruction body,
#         written to disk and read back through the SDK's loader.
# =============================================================================


SKILL_MD = """\
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
"""


async def example_format():
    print("\n=== Part 3: SKILL.md Format ===\n")

    with tempfile.TemporaryDirectory() as tmp:
        skill_dir = Path(tmp) / "bin-enrichment"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")
        (skill_dir / "references").mkdir()
        (skill_dir / "references" / "bin-ranges.md").write_text("# BIN ranges\n")

        # Same parser the directory loader uses: front-matter becomes typed
        # fields, the body becomes the instructions injected on activation.
        skill = Skill.from_file(skill_dir)
        print(f"name:          {skill.name}")
        print(f"description:   {skill.description}")
        print(f"allowed_tools: {skill.allowed_tools}")
        print(f"metadata:      {skill.metadata}")
        print(f"resources:     {skill.list_resources()}")
        print(f"instructions:  {skill.instructions.splitlines()[0]} …")


async def main() -> None:
    await example_programmatic()
    await example_filesystem()
    await example_format()


if __name__ == "__main__":
    asyncio.run(main())
