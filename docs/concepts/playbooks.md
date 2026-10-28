# Playbooks

A playbook is a **declarative execution plan** — an ordered list of
steps, each with a description, expected tools, hints, and validation
criteria. The `PlaybookEnforcer` checks that the agent runs the right
tools in the right order and reports any deviation.

If your agent issues refunds, rolls out changes, or isolates a host —
anything an auditor will review — you want a playbook. The model still
picks the wording; the *side effects* follow the plan.

```python
from locus.playbooks import Playbook, PlaybookStep

# A refund-escalation runbook: verify, check eligibility, then pay out.
refund_escalation = Playbook(
    id="refund-escalation",
    name="Refund escalation",
    steps=[
        PlaybookStep(
            id="verify_order",
            description="Pull the order record and its payment history.",
            expected_tools=["lookup_order", "lookup_customer"],
            hints=["Start with the most recent charge", "Check for duplicates first"],
            max_tool_calls=5,
        ),
        PlaybookStep(
            id="check_eligibility",
            description="Check the refund against policy — window, amount cap, prior refunds.",
            expected_tools=["check_refund_policy"],
        ),
        PlaybookStep(
            id="issue_refund",
            description="Issue the refund and write a one-paragraph resolution note.",
            expected_tools=["issue_refund"],
        ),
    ],
)

agent = Agent(
    model="{{ locus_example_model }}",
    tools=[lookup_order, lookup_customer, check_refund_policy, issue_refund],
    playbook=refund_escalation,   # installs the PlaybookEnforcerHook for you
)
```

## When to reach for a playbook

| Situation | Playbook? |
|---|---|
| Regulated workflow (refund approval chains, change management, incident response) | **yes** |
| Multi-step process where order matters | **yes** |
| Repeatable runbook the team executes manually today | **yes — encode it** |
| Audit-trail requirement: "every refund follows the same sequence" | **yes — the enforcer's in-memory execution log captures the sequence** (persist it, or pair it with [`AuditTrail`](agentic-ai-security.md), for a durable record) |
| One-shot exploration, freeform Q&A | no — overhead's not worth it |
| You want the model to choose tools freely | no — that's what `Agent(tools=[...])` already gives you |

## Getting started

### 1. Build a `Playbook` in Python

`refund_escalation` above is the whole shape: a `Playbook` of ordered
`PlaybookStep`s. The same shape carries a security-flavored example — a
NIST SP 800-61 flow (the standard incident-response process),
**detection → analysis → containment**, where containment never fires
before `enrich_indicator` has run — shown as YAML in the next step.

`PlaybookStep` fields:

| Field | Meaning |
|---|---|
| `id` | Unique step identifier. |
| `description` | Human-readable, for whoever reads the playbook — the SDK never sends it to the model. |
| `expected_tools` | Tools the agent is supposed to call during this step. |
| `hints` | Steering text. The model sees it only when a call in this step is rejected as the wrong tool. |
| `required` | If `False`, the step can be skipped. |
| `uses` | Skills the step is carried out with, by name. The step inherits each skill's `required_probes`, and while it is the current step a skill's `allowed_tools` bounds what may be called. See [Steps that use skills](#steps-that-use-skills). |
| `required_probes` | Evidence the step must gather: `RequiredProbe(name=..., match=...)` (from `locus.playbooks.models`), where `match` is a case-insensitive substring sought in the step's tool calls — the tool name and its serialised arguments, not its result. Completing the step with a probe unmatched records an `evidence_incomplete` violation. |
| `min_tool_calls` | Floor on effort. The hook still closes the step once its `expected_tools` have all fired; if that took fewer calls, an `insufficient_effort` violation is recorded. |
| `max_tool_calls` | Hard cap on tool calls for this step. |
| `validation` | Optional dict of post-step criteria; carried on the step but not evaluated by the enforcer in {{ locus_sdk_version }}. |
| `timeout_seconds` | Optional per-step timeout; carried on the step but not enforced in {{ locus_sdk_version }}. |
| `metadata` | Arbitrary dict for your own use. |

### 2. Load from YAML or JSON

For checked-in playbooks, use the loader:

```python
from locus.playbooks import load_playbook

nist_ir = load_playbook("playbooks/nist_800_61_ir.yaml")
```

```yaml
# playbooks/nist_800_61_ir.yaml
id: nist-800-61-ir
name: NIST 800-61 incident response
description: Detect from the SIEM, analyze the IOCs, then contain.
allow_extra_tools: false
steps:
  - id: detection
    description: Pull the alert and its correlated events from the SIEM.
    expected_tools: [query_siem]
  - id: analysis
    description: Enrich every IOC from detection; confirm it's malicious.
    expected_tools: [enrich_indicator]
  - id: containment
    description: Isolate the affected host from the network.
    expected_tools: [isolate_host]
```

### 3. Wire the enforcer

`Agent(playbook=...)` installs the `PlaybookEnforcerHook` for you.
Construct it yourself when you want a handle on the enforcer — to
[inspect the plan](#inspecting-execution) after the run — or when steps
`uses` skills you resolve explicitly:

```python
from locus.playbooks.hook import PlaybookEnforcerHook

enforcer_hook = PlaybookEnforcerHook(playbook=refund_escalation)
agent = Agent(
    model="{{ locus_example_model }}",
    tools=[lookup_order, lookup_customer, check_refund_policy, issue_refund],
    hooks=[enforcer_hook],
)

result = agent.run_sync("Customer on ord-4821 reports a duplicate charge — refund if eligible.")
```

The hook validates each tool call against the current step's
`expected_tools` and records the executions. It does not add the plan
to the prompt: a step's `hints` reach the model only when a call is
rejected (with the hook's default `block_violations=True`), as part of
the `PlaybookEnforcer blocked: … | Hints: …` tool result the model then
has to recover from; `description` is never sent. If you want the
model steered up front, put the procedure in `system_prompt=` as well.
The verify → check-eligibility → refund order is enforced: an
`issue_refund` call during the `verify_order` step is rejected, so no
refund fires before `check_refund_policy` has run.

## Strict vs lenient enforcement

| Setting | Effect |
|---|---|
| `allow_extra_tools=False` (default) | During a step, a call must be in the step's `expected_tools` (a step with an empty `expected_tools` list constrains nothing here) and, when the step `uses` a skill that declares `allowed_tools`, in that allow-list too. Once the last step completes, any further tool call is rejected. |
| `allow_extra_tools=True` | None of those three checks apply — any registered tool may fire; the playbook is a recommendation, not a contract. |

`strict_sequence` is carried on `Playbook` for authoring and
serialisation but is not consulted by `PlaybookEnforcer` or
`PlaybookEnforcerHook` in {{ locus_sdk_version }}; ordering comes from
the plan's current-step pointer, which the hook advances once the
current step's `expected_tools` have all fired (or its
`max_tool_calls` is reached).

For compliance-grade workflows, keep `allow_extra_tools` at its
default. For "loose runbook" guidance, set `allow_extra_tools=True`.

### Steps that use skills

A step can name the [skills](skills.md) it is carried out with, so the
procedure is written at the level of the job rather than the tool:

```python
from locus.playbooks import Playbook, PlaybookStep
from locus.playbooks.hook import PlaybookEnforcerHook
from locus.skills import Skill

refund_triage = Skill(
    name="refund-triage",
    description="Use when a customer asks for a refund or disputes a charge.",
    instructions="Pull the order, then check it against the refund policy.",
    allowed_tools=["lookup_order", "check_refund_policy"],
)

refund_intake = Playbook(
    id="refund-intake",
    name="Refund intake",
    steps=[
        PlaybookStep(
            id="triage",
            description="Triage the refund request.",
            uses=["refund-triage"],
            max_tool_calls=6,
        ),
    ],
)

intake_hook = PlaybookEnforcerHook(playbook=refund_intake, skills={"refund-triage": refund_triage})
```

- **Resolution.** Names resolve against the `skills=` mapping given to
  `PlaybookEnforcerHook`, or, when the playbook is passed as
  `Agent(playbook=...)`, against the agent's `skills=`. In
  {{ locus_sdk_version }} that works only for `Skill` objects: a string
  path there makes `Agent(...)` fail at construction (`AttributeError`),
  and a `Path` leaves the name inert (no allow-list, no probes), so load
  it with `Skill.from_file(...)` first. A name that resolves to nothing
  is inert too: no error, no constraint.
- **Allow-list.** While the step is current, the union of the
  allow-lists its skills declare bounds what may be called. Any other
  call is recorded as a `tool_outside_skill` violation and, with
  `block_violations=True` (the default), blocked. This applies only
  while `allow_extra_tools=False` (the default); with
  `allow_extra_tools=True` the call goes through and no violation is
  recorded. A skill without an allow-list constrains nothing, and
  outside a step that names it, a skill's list constrains nothing
  either. The step's own `expected_tools` check still applies on top.
- **Evidence.** The step inherits each skill's `required_probes`; a
  probe the step declares under the same name wins. A skill's own
  `min_tool_calls` / `max_tool_calls` are not applied to the step in
  {{ locus_sdk_version }} — declare them on the step.
- **Advancing.** The hook closes a step when its `expected_tools` have
  all fired or its `max_tool_calls` is reached, never on `uses` alone,
  so a step that names only skills needs a `max_tool_calls` to move on
  (the `max_tool_calls=6` above).
- **The activation tool.** The `skills` tool that `Agent(skills=[...])`
  registers is checked like any other call: while such a step is
  current it is blocked unless it is on the step's skill allow-list
  (when one is declared) and in the step's `expected_tools` (when those
  are listed).

## Inspecting execution

The enforcer maintains a `PlaybookPlan` — an in-memory record of
every step's status, tool calls, and counts. It lives on the hook's
`enforcer` (not on the result), so read it off the hook after the run:

```python
plan = enforcer_hook.enforcer.plan
for execution in plan.step_executions.values():
    print(f"{execution.step_id}: {execution.status.value} "
          f"({len(execution.tool_calls)} tool calls)")
```

`plan.step_executions` is a `dict[str, StepExecution]`; `StepStatus` is one of
`pending`, `in_progress`, `completed`, `skipped`, `failed`. The plan is an
ephemeral Pydantic object — serialize it (or wire an [`AuditTrail`](agentic-ai-security.md))
if you need a record that outlives the process.

### Evidence, not just sequence

`expected_tools` asks whether the right tool was called;
`required_probes` asks whether the right thing was looked at. Declare
the evidence on the step, then read the violations off the enforcer
after the run:

```python
from locus.playbooks import PlaybookStep
from locus.playbooks.models import RequiredProbe

# The refund playbook's first step, now with evidence it must gather:
PlaybookStep(
    id="verify_order",
    description="Pull the order record and its payment history.",
    expected_tools=["lookup_order", "lookup_customer"],
    required_probes=[RequiredProbe(name="charge_history", match="include_charges")],
)

# ... run the agent with hooks=[enforcer_hook] as in step 3 ...

for violation in enforcer_hook.enforcer.violations:
    print(f"{violation.violation_type} @ {violation.step_id}: {violation.message}")

print(f"probe coverage (steps reached): {enforcer_hook.enforcer.adherence_score():.2f}")
```

If the agent's `lookup_order` arguments never mention `include_charges`,
the `verify_order` step still closes — probes are recorded, they don't
block — and the violation list carries an `evidence_incomplete` entry
naming `charge_history`. `adherence_score()` averages probe coverage
over the steps the run reached, not the whole playbook: a run that
closes `verify_order` with its probe matched and then stops scores
1.00, so read it alongside
`enforcer_hook.enforcer.plan.unresolved_required_steps()`. A substring match
shows the string was asked for, not that the evidence was examined, so
a low score is worth investigating and a high one proves little. The
violation list is the thing to show a reviewer.

## Common gotchas

| Symptom | Likely cause |
|---|---|
| Agent skips a step it shouldn't | Nothing up front tells the model the procedure: it never sees a step's `description`, and sees `hints` only after a rejected call. Put the procedure in `system_prompt=` too, and check `enforcer_hook.enforcer.plan.unresolved_required_steps()` after the run — a model that answers without calling a step's tools leaves that step unresolved rather than tripping a rejection. |
| Enforcer rejects a tool that *should* be allowed | The tool isn't in `expected_tools` for the current step. Add it, or set `allow_extra_tools=True` if the policy allows. |
| `max_tool_calls` exhausts mid-step | Bump the limit or split the step in two — the model may need search-and-refine cycles. |
| YAML loads but the agent doesn't follow it | Pass it as `Agent(playbook=...)` or through `PlaybookEnforcerHook(...)` — `Playbook` alone is just data. |

## Source and notebook

- [`notebook_49_playbooks.py`](https://github.com/oracle/locus/blob/main/examples/notebook_49_playbooks.py) — runnable end-to-end with execution tracking.
- [`locus.playbooks`](https://github.com/oracle/locus/tree/main/src/locus/playbooks) — `Playbook`, `PlaybookStep`, `PlaybookEnforcerHook`, `load_playbook`.

## See also

- [Skills](skills.md) — the natural-language analogue: filesystem-first capability bundles.
- [Hooks](hooks.md) — `PlaybookEnforcerHook` is a normal hook; you can add it alongside guardrails / steering / telemetry.
- [Tools](tools.md) — playbook steps reference the tools you registered with `@tool`.
