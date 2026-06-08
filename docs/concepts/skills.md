# Skills

A skill is a reusable capability bundle the agent loads only when it
needs it. You give the agent fifty skills; it sees fifty one-line
descriptions in its system prompt and pays the cost of one. When the
model decides a skill is relevant to the current task, the full
instructions for *that one skill* enter the conversation. This is the
[AgentSkills.io](https://agentskills.io) spec — **progressive
disclosure** — and it's how you compose **broad agents** (one model,
many domain skills) without blowing the context budget on capabilities
the run won't use.

Each skill is a folder with a `SKILL.md`. Point your agent at the
parent directory and Tulip
handles the disclosure tiers:

- **L1 — catalog.** Names + one-line descriptions live in the system
  prompt. Cheap, always loaded.
- **L2 — instructions.** When the model decides a skill is relevant,
  the full `SKILL.md` body loads into the conversation.
- **L3 — resources.** Scripts, references, and assets in
  `scripts/`, `references/`, `assets/` subfolders only enter context
  when the agent reaches for them.

```python
from tulip.agent import Agent
from tulip.agent import AgentConfig
from tulip.skills import Skill

skill = Skill(
    name="code-review",
    description="Use when reviewing code for bugs and security issues.",
    instructions=(
        "# Code Review Checklist\n"
        "1. Check for SQL injection\n"
        "2. Check for hardcoded credentials\n"
        "3. Check error handling\n"
        "Report findings as: FINDING: <description>"
    ),
)

agent = Agent(config=AgentConfig(
    model="anthropic:claude-sonnet-4-6",
    system_prompt="You are a security reviewer. Use available skills.",
    skills=[skill],
))
```

## When to reach for skills

| Situation | Skills? |
|---|---|
| One agent that handles many domains (research / coding / triage) — context budget would explode if every domain's prompt is always loaded | **yes — progressive disclosure earns its keep here** |
| Capability written and edited by non-engineers (markdown, not code) | **yes** |
| Reusable across agents and projects (clone the skill folder) | **yes** |
| Single-domain agent with a fixed system prompt | no — just put the prompt in `system_prompt=` |
| Strict compliance workflow with audit-able steps | use [Playbooks](playbooks.md) instead — skills are *recommendations*, playbooks *enforce* |

## Getting started

### Programmatic — define a skill in code

```python
from tulip.skills import Skill

researcher = Skill(
    name="vendor-research",
    description="Use when the task is a sourcing decision (vendor, price, RFP).",
    instructions=(
        "# Vendor Research\n\n"
        "1. Look up vendors with `vendor_lookup`.\n"
        "2. Quote each option with `quote_price`.\n"
        "3. Compare on (price, lead-time, vendor-rating).\n"
        "4. Return a recommendation with reasoning.\n"
    ),
    allowed_tools=["vendor_lookup", "quote_price"],
)
```

`allowed_tools` scopes which tools the skill may invoke when active —
enforced at the loop level. A skill with `allowed_tools=None` can use
any tool registered with the agent.

### Filesystem — drop a `SKILL.md`

```text
skills/vendor-research/
├── SKILL.md
├── scripts/
│   └── compare.py
└── references/
    └── pricing-tiers.md
```

```markdown
---
name: vendor-research
description: Use when the task is a sourcing decision (vendor, price, RFP).
allowed-tools: vendor_lookup quote_price
metadata:
  author: ops-team
  version: 1.0
---

# Vendor Research

Look up vendors, quote each, compare on price / lead-time /
vendor-rating. Reference `references/pricing-tiers.md` for the
internal tier-to-discount mapping. Use `scripts/compare.py` if you
need a structured comparison spreadsheet.
```

### Load and attach

```python
from pathlib import Path
from tulip.skills import Skill

skills = Skill.from_directory(Path("./skills"))   # all SKILL.md folders
# …or one at a time:
single = Skill.from_file("./skills/vendor-research")

agent = Agent(config=AgentConfig(model=..., skills=skills))
```

## Why progressive disclosure earns its keep

A naive "stuff every capability into the system prompt" approach
costs you tokens on every turn for skills the run never uses. With
progressive disclosure:

- The catalog is ~1 line per skill — fits 50+ skills in a few hundred
  tokens.
- The full instructions only load when the model decides the skill is
  relevant.
- Resource files (`scripts/`, `references/`, `assets/`) load only
  when the agent explicitly opens them — typically once or twice per
  run, not every turn.

For an agent with 30 skills, that's the difference between **30k
tokens of system prompt every turn** and **~600 tokens catalog +
2-3k of one skill's instructions when it's the right call**.

## Skill vs Playbook vs Tool

Easy to confuse. Quick disambiguation:

| Primitive | What it is | When to use |
|---|---|---|
| **Tool** | A typed function the model can call | The atomic unit — every primitive bottoms out in tools |
| **Skill** | A markdown bundle the model loads when relevant | Reusable capability with prose instructions |
| **Playbook** | An ordered, enforced execution plan | Compliance / audit / exact-sequence requirements |

A skill *suggests*; a playbook *enforces*. A tool is the verb both
of them call.

## Common gotchas

| Symptom | Likely cause |
|---|---|
| Skill never activates | `description` doesn't match how the user phrases the request. Rewrite it as a "use when…" sentence with the user's vocabulary. |
| All skills load every turn | Progressive disclosure only kicks in if `skills=[...]` is set — passing skills as raw text in `system_prompt=` defeats it. |
| `allowed_tools` is silently ignored | Tools must also be registered on the agent (`tools=[...]`). The skill's `allowed_tools` is a *subset* filter, not a registration. |
| Skill resource file isn't read | The model has to ask for it. If a reference is mandatory, inline its key bullets in `instructions=` instead. |

## Source and notebook

- [`notebook_48_skills.py`](https://github.com/tuliplabs-ai/sdk-python/blob/main/examples/notebook_48_skills.py) — programmatic and filesystem-loaded skills end-to-end.
- [`tulip.skills`](https://github.com/tuliplabs-ai/sdk-python/tree/main/src/tulip/skills) — `Skill`, `SkillsPlugin`.
- [AgentSkills.io specification](https://agentskills.io) — the format the SDK implements.

## See also

- [Playbooks](playbooks.md) — ordered, enforced plans (compliance-grade).
- [Tools](tools.md) — what skills ultimately call.
- [Prompts](prompts.md) — for single-domain agents, a system prompt is simpler.
