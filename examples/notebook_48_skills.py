# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""Notebook 45: skills — packaged instruction bundles with progressive disclosure.

A Skill (the AgentSkills.io shape) bundles a name, a description, and
a block of instructions. The ``SkillsPlugin`` exposes a catalog of
skills to the agent and only injects the full instructions once the
agent activates a specific one. Progressive disclosure keeps the
system prompt small and the agent focused.

- ``Skill`` — built in code or loaded from a ``SKILL.md`` file with
  YAML front-matter.
- ``Skill.from_directory(path)`` — load every skill under a directory.
- ``Agent(skills=[...])`` — wires up the SkillsPlugin and the
  ``skills`` tool that the agent calls to activate one.
- The ``SKILL.md`` format itself — front-matter for ``name``,
  ``description``, ``allowed-tools``, ``metadata``, plus the
  instruction body and optional ``scripts/``, ``references/``,
  ``assets/`` directories.

Run it:
    # The bundled mock model is the default; set TULIP_MODEL_PROVIDER for a live provider.
    TULIP_MODEL_ID=openai.gpt-4.1 python examples/notebook_50_skills.py

    # Offline:
    TULIP_MODEL_PROVIDER=mock python examples/notebook_50_skills.py

Prerequisites:
- An OpenAI or Anthropic API key, or set ``TULIP_MODEL_PROVIDER`` to
  ``openai`` / ``anthropic`` / ``mock``.
- Optional: an ``examples/skills/`` directory with one or more
  ``SKILL.md`` files for Part 2 to find.
"""

from pathlib import Path

from config import get_model

from tulip.agent import Agent, AgentConfig
from tulip.skills import Skill


# =============================================================================
# Part 1: Build a Skill in code — no SKILL.md file required.
# =============================================================================


def example_programmatic():
    print("=== Part 1: Programmatic Skills ===\n")

    model = get_model()

    code_review = Skill(
        name="code-review",
        description="Use when reviewing code for bugs and security issues.",
        instructions=(
            "# Code Review Checklist\n"
            "1. Check for SQL injection\n"
            "2. Check for hardcoded credentials\n"
            "3. Check error handling\n"
            "4. Report findings as: FINDING: <description>"
        ),
    )

    agent = Agent(
        config=AgentConfig(
            system_prompt="You are a security reviewer. Use available skills.",
            max_iterations=5,
            model=model,
            skills=[code_review],
        )
    )

    result = agent.run_sync(
        "Review: def login(u,p): return db.query(f'SELECT * FROM users WHERE name={u}')"
    )
    print(f"Response: {result.message[:200]}...")

    skills_used = [te for te in result.tool_executions if te.tool_name == "skills"]
    print(f"Skills activated: {len(skills_used)}")


# =============================================================================
# Part 2: Load every SKILL.md under a directory.
# =============================================================================


def example_filesystem():
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
    res = agent.run_sync(
        "In one sentence, why is loading skills from SKILL.md files better than "
        "hard-coding system prompts in source?"
    )
    dt = _t.perf_counter() - t0
    print(
        f"  [model call: {dt:.2f}s · {res.metrics.prompt_tokens}→{res.metrics.completion_tokens} tokens]"
    )
    print(f"  AI rationale: {res.message.strip()}")


# =============================================================================
# Part 3: SKILL.md file format — YAML front-matter + instruction body.
# =============================================================================


def example_format():
    print("\n=== Part 3: SKILL.md Format ===\n")

    print("""
---
name: my-skill
description: Use when the user asks about X.
allowed-tools: search analyze
metadata:
  author: your-name
  version: "1.0"
---

# Instructions for the Agent

1. First, do this
2. Then, do that
3. Finally, summarize

## Resource Files
Place additional files in:
- scripts/   — executable code
- references/ — documentation
- assets/    — templates, data
    """)

    import time as _t

    agent = Agent(model=get_model(max_tokens=120), system_prompt="Reply in one short paragraph.")
    t0 = _t.perf_counter()
    res = agent.run_sync(
        "Write a one-paragraph SKILL.md description for a skill named "
        "'sql-debug' that helps an agent diagnose slow SQL queries."
    )
    dt = _t.perf_counter() - t0
    print(
        f"  [model call: {dt:.2f}s · {res.metrics.prompt_tokens}→{res.metrics.completion_tokens} tokens]"
    )
    print(f"  AI-authored sample description:\n  {res.message.strip()}")


if __name__ == "__main__":
    example_programmatic()
    example_filesystem()
    example_format()
