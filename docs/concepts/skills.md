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
parent directory and Locus
handles the disclosure tiers:

- **L1 — catalog.** Names + one-line descriptions live in the system
  prompt. Cheap, always loaded.
- **L2 — instructions.** When the model decides a skill is relevant,
  the full `SKILL.md` body loads into the conversation.
- **L3 — resources.** Activation appends a *listing* of the files in
  `scripts/`, `references/` and `assets/` (the first 20 files;
  `AgentConfig(skills=…)` has no setting to change that cap; a skill
  with no folder, such as a `Skill(...)` defined in code, has none). The skills plugin registers
  one tool — the `skills` activation tool — and no file reader, so the
  agent can only open a listed file if you register a filesystem or MCP
  read tool alongside the skill.

```python
from locus.agent import Agent
from locus.agent import AgentConfig
from locus.skills import Skill

skill = Skill(
    name="release-notes",
    description="Use when drafting release notes from a list of merged changes.",
    instructions=(
        "# Release Notes Checklist\n"
        "1. Group changes: features, fixes, breaking\n"
        "2. Lead with user-visible impact, not internals\n"
        "3. Call out migration steps for breaking changes\n"
        "Report each entry as: NOTE: <description>"
    ),
)

agent = Agent(config=AgentConfig(
    model="{{ locus_example_model }}",
    system_prompt="You are a release manager. Use available skills.",
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
from locus.skills import Skill

refund_triage = Skill(
    name="refund-triage",
    description="Use when a customer asks for a refund or disputes a charge.",
    instructions=(
        "# Refund Triage\n\n"
        "1. Pull the order with `lookup_order`; check the payment history.\n"
        "2. Check eligibility with `check_refund_policy` — window, amount cap, prior refunds.\n"
        "3. Weigh order value, account age, and dispute history.\n"
        "4. Return a recommendation (approve / deny / escalate) with the evidence.\n"
        "5. No corroborating evidence? Escalate — never guess.\n"
    ),
    allowed_tools=["lookup_order", "check_refund_policy"],
)
```

On its own, `allowed_tools` is an **advisory hint**: when the skill
activates, its allowed-tools list is surfaced to the model as an
`Allowed tools: …` line appended to the skill instructions. It is not
a hard loop-level filter — the model still sees every tool registered
on the agent, so treat `allowed_tools` as guidance, not enforcement.
When you need a tool to be genuinely unreachable, don't register it on
the agent (or gate it with a [hook](hooks.md)). A skill with
`allowed_tools=None` adds no such line.

The one place the list *is* enforced is inside a
[playbook](playbooks.md) step that names the skill. Give the agent the
skill as a `Skill` object (`skills=[refund_triage]`, or the list
`Skill.from_directory` returns; a directory path in `skills=` does not
work here — a string path fails at construction once `playbook=` is
set, and a `Path` leaves `uses` resolving to nothing) and a playbook
(`playbook=...`) whose step lists it in `uses` — `PlaybookStep(id="triage", description="Triage the
refund request", uses=["refund-triage"])` — and while that step is
current, any call outside the union of its skills' `allowed_tools` is
blocked as a `tool_outside_skill` violation, as long as the playbook
keeps `allow_extra_tools=False` and the enforcer `block_violations=True`
(both defaults). That includes the `skills` activation tool itself
unless you list it. The step also inherits the skill's
`required_probes` (a list of `RequiredProbe` from
`locus.playbooks.models`); each probe's `match` is sought,
case-insensitively, in the tool name and serialised arguments of the
step's calls, not in their results. `Skill` also accepts
`min_tool_calls` / `max_tool_calls`, but in {{ locus_sdk_version }} a
step that uses the skill does not apply them — declare those bounds on
the step itself (see the
[step fields](playbooks.md#1-build-a-playbook-in-python)). Outside such
a step, `allowed_tools` constrains nothing.

### Filesystem — drop a `SKILL.md`

```text
skills/refund-triage/
├── SKILL.md
├── scripts/
│   └── payment_history.py
└── references/
    └── refund-policy.md
```

```markdown
---
name: refund-triage
description: Use when a customer asks for a refund or disputes a charge.
allowed-tools: lookup_order check_refund_policy
metadata:
  author: support-team
  version: 1.0
---

# Refund Triage

Pull the order, check eligibility, and weigh order value, account age,
and dispute history. Escalate when nothing corroborates the request.
Reference `references/refund-policy.md` for the caps and windows. Use
`scripts/payment_history.py` to pull related charges.
```

### Load and attach

```python
from pathlib import Path
from locus.skills import Skill

skills = Skill.from_directory(Path("./skills"))   # all SKILL.md folders
# …or one at a time:
single = Skill.from_file("./skills/refund-triage")

agent = Agent(config=AgentConfig(model=..., skills=skills))
```

### Worked example — a contained refund-triage skill

Refund triage, scoped so the loaded skill is *steered* to read and
recommend rather than pay out — the refund itself stays a deliberate,
separately-authorised step. (Because `allowed_tools` is advisory
outside a playbook step, the hard guarantee comes from how you register
tools, not the skill — see below.)

```python
from locus.agent import Agent, AgentConfig
from locus.skills import Skill

refund_triage = Skill(
    name="refund-triage",
    description="Use when a customer asks for a refund or disputes a charge.",
    instructions=(
        "# Refund triage — verify & recommend\n\n"
        "1. Pull the order with `lookup_order`; correlate charges "
        "with `lookup_customer`.\n"
        "2. Check eligibility with `check_refund_policy` — window, "
        "amount cap, prior refunds.\n"
        "3. Ground each conclusion: only report what a tool returned. "
        "No evidence -> ESCALATE, do not speculate.\n"
        "4. Recommend the refund in prose. Do NOT call `issue_refund` "
        "— that is the approver's call.\n"
    ),
    # read + check only; steers the model away from the write tool
    allowed_tools=["lookup_order", "lookup_customer", "check_refund_policy"],
)

agent = Agent(config=AgentConfig(
    model="{{ locus_example_model }}",
    system_prompt="You are a support agent. Cite evidence; escalate without it.",
    # full toolset registered; the skill only steers away from issue_refund
    tools=[lookup_order, lookup_customer, check_refund_policy, issue_refund],
    skills=[refund_triage],
))

result = agent.run_sync("Customer on ord-4821 reports a duplicate charge — triage it.")
```

`issue_refund` is registered on the agent, and the skill's
instructions steer the model away from it while `refund-triage` is
loaded. Because `allowed_tools` is advisory here (not a hard filter),
if you need the write to be *unreachable* during triage, don't register
`issue_refund` at all — or run triage as a playbook step that `uses`
the skill, so the call is blocked (see above). Either way, a human (or
a separate, audited [playbook](playbooks.md)) pulls the trigger.

The security-flavored variant is the same lesson. A phishing-triage
skill steers the model to `fetch_alert` / `query_siem` /
`enrich_indicator` and ends with *"Recommend containment in prose. Do
NOT call `isolate_host` — that is the responder's call"*, over
`tools=security_toolset(allow_containment=True)` — and gates the final
verdict through `ground_finding` so an unsupported compromise claim
abstains instead of shipping. See [Security](security.md) and
[GSAR](gsar.md).

## Why progressive disclosure earns its keep

A naive "stuff every capability into the system prompt" approach
costs you tokens on every turn for skills the run never uses. With
progressive disclosure:

- The catalog is ~1 line per skill — fits 50+ skills in a few hundred
  tokens.
- The full instructions only load when the model decides the skill is
  relevant.
- Resource files (`scripts/`, `references/`, `assets/`) never load on
  their own — activation lists their names, and a file's contents enter
  context only if the agent reads it with a file tool you registered.

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
| A tool in `allowed_tools` is never called | `allowed_tools` doesn't register tools — on its own it adds an advisory `Allowed tools:` hint to the skill. The tool must still be registered on the agent (`tools=[...]`) for the model to call it. |
| The model called a tool *not* in `allowed_tools` | Expected outside a playbook — on its own `allowed_tools` is advisory. Name the skill in a playbook step's `uses` to have calls outside the list blocked, or don't register the tool (or gate it with a hook). |
| Skill resource file isn't read | Unless you registered a file-reading tool, nothing can read it: the skills plugin lists resource files but registers no reader. Register one (see `locus.tools.path_safety` for safe path resolution). Even then the model has to choose to open the file, so if a reference is mandatory, inline its key bullets in `instructions=`. |

## Source and notebook

- [`notebook_48_skills.py`](https://github.com/oracle/locus/blob/main/examples/notebook_48_skills.py) — programmatic and filesystem-loaded skills end-to-end.
- [`locus.skills`](https://github.com/oracle/locus/tree/main/src/locus/skills) — `Skill`, `SkillsPlugin`.
- [AgentSkills.io specification](https://agentskills.io) — the format the SDK implements.

## See also

- [Playbooks](playbooks.md) — ordered, enforced plans (compliance-grade).
- [Tools](tools.md) — what skills ultimately call.
- [Prompts](prompts.md) — for single-domain agents, a system prompt is simpler.
