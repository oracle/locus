# Playbooks

A playbook is a **declarative execution plan** — an ordered list of
steps, each with a description, expected tools, hints, and validation
criteria. The `PlaybookEnforcer` checks that the agent runs the right
tools in the right order and reports any deviation.

If your agent ships customer money, files an SR, or touches anything
regulated, you want a playbook. The model still picks the wording;
the *side effects* follow the plan.

```python
from tulip.playbooks import Playbook, PlaybookStep
from tulip.playbooks.hook import PlaybookEnforcerHook

incident_triage = Playbook(
    id="incident-triage",
    name="Incident triage",
    steps=[
        PlaybookStep(
            id="gather_logs",
            description="Collect logs from affected services.",
            expected_tools=["read_file", "search_logs"],
            hints=["Start with the most recent", "ERROR / WARN levels first"],
            max_tool_calls=5,
        ),
        PlaybookStep(
            id="analyze_errors",
            description="Group errors by type, note timestamps.",
            expected_tools=["analyze_logs", "count_errors"],
        ),
        PlaybookStep(
            id="summarize_findings",
            description="Write a one-paragraph root-cause summary.",
            expected_tools=[],
        ),
    ],
    strict_sequence=True,
)

agent = Agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[read_file, search_logs, analyze_logs, count_errors],
    hooks=[PlaybookEnforcerHook(playbook=incident_triage)],
)
```

## When to reach for a playbook

| Situation | Playbook? |
|---|---|
| Regulated workflow (KYC, refunds, account changes) | **yes** |
| Multi-step process where order matters | **yes** |
| Repeatable runbook the team executes manually today | **yes — encode it** |
| Audit-trail requirement: "every refund follows the same sequence" | **yes — the execution log *is* the audit trail** |
| One-shot exploration, freeform Q&A | no — overhead's not worth it |
| You want the model to choose tools freely | no — that's what `Agent(tools=[...])` already gives you |

## Getting started

### 1. Build a `Playbook` in Python

```python
from tulip.playbooks import Playbook, PlaybookStep

refund = Playbook(
    id="refund-flow",
    name="Refund flow",
    description="Issue a refund only after verifying customer and order.",
    steps=[
        PlaybookStep(
            id="verify_customer",
            description="Look up the customer and confirm they're active.",
            expected_tools=["lookup_customer"],
            required=True,
        ),
        PlaybookStep(
            id="verify_order",
            description="Look up the order and confirm it belongs to the customer.",
            expected_tools=["lookup_order"],
            required=True,
        ),
        PlaybookStep(
            id="issue_refund",
            description="Refund the order amount.",
            expected_tools=["refund"],
            required=True,
        ),
    ],
    strict_sequence=True,
    allow_extra_tools=False,
)
```

`PlaybookStep` fields:

| Field | Meaning |
|---|---|
| `id` | Unique step identifier. |
| `description` | Human-readable; the agent sees this as a hint. |
| `expected_tools` | Tools the agent is supposed to call during this step. |
| `hints` | Extra steering text. |
| `required` | If `False`, the step can be skipped. |
| `max_tool_calls` | Hard cap on tool calls for this step. |
| `validation` | Optional dict of post-step checks. |

### 2. Load from YAML or JSON

For checked-in playbooks, use the loader:

```python
from tulip.playbooks import load_playbook

refund = load_playbook("playbooks/refund.yaml")
```

```yaml
# playbooks/refund.yaml
id: refund-flow
name: Refund flow
description: Issue a refund only after verifying customer and order.
strict_sequence: true
allow_extra_tools: false
steps:
  - id: verify_customer
    description: Look up the customer and confirm they're active.
    expected_tools: [lookup_customer]
  - id: verify_order
    description: Look up the order and confirm it belongs to the customer.
    expected_tools: [lookup_order]
  - id: issue_refund
    description: Refund the order amount.
    expected_tools: [refund]
```

### 3. Wire the enforcer

```python
from tulip.playbooks.hook import PlaybookEnforcerHook

agent = Agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[lookup_customer, lookup_order, refund],
    hooks=[PlaybookEnforcerHook(playbook=refund)],
)

result = agent.run_sync("Refund order ORD-42 for customer C-7.")
```

The hook injects step descriptions and hints into the agent's
context, validates each tool call against the current step, and
records the executions. If the agent tries to skip ahead or call a
tool not in `expected_tools` while `allow_extra_tools=False`, the
hook rejects the call.

## Strict vs lenient enforcement

| Setting | Effect |
|---|---|
| `strict_sequence=True` (default) | Steps must run in order; skipping ahead rejects the call. |
| `strict_sequence=False` | Steps can run in any order, but each must complete. |
| `allow_extra_tools=False` (default) | Only `expected_tools` may fire during a step. |
| `allow_extra_tools=True` | Any registered tool may fire — playbook is a recommendation, not a contract. |

For compliance-grade workflows, keep both at their defaults. For
"loose runbook" guidance, flip them.

## Inspecting execution

The enforcer maintains a `PlaybookPlan` — an audit-grade record of
every step's status, tool calls, and timestamps. Read it after the
run:

```python
plan = result.playbook_plan
for execution in plan.executions:
    print(f"{execution.step_id}: {execution.status.value} "
          f"({len(execution.tool_calls)} tool calls)")
```

`StepStatus` is one of `pending`, `in_progress`, `completed`,
`skipped`, `failed`.

## Common gotchas

| Symptom | Likely cause |
|---|---|
| Agent skips a step it shouldn't | The current step's `description` isn't specific enough — the model is interpreting the user's request as already satisfying the step. Sharpen the description. |
| Enforcer rejects a tool that *should* be allowed | The tool isn't in `expected_tools` for the current step. Add it, or set `allow_extra_tools=True` if the policy allows. |
| `max_tool_calls` exhausts mid-step | Bump the limit or split the step in two — the model may need search-and-refine cycles. |
| YAML loads but the agent doesn't follow it | Pass it through `PlaybookEnforcerHook(...)` — `Playbook` alone is just data. |

## Source and notebook

- [`notebook_46_playbooks.py`](https://github.com/tuliplabs-ai/sdk-python/blob/main/examples/notebook_46_playbooks.py) — runnable end-to-end with execution tracking.
- [`tulip.playbooks`](https://github.com/tuliplabs-ai/sdk-python/tree/main/src/tulip/playbooks) — `Playbook`, `PlaybookStep`, `PlaybookEnforcerHook`, `load_playbook`.

## See also

- [Skills](skills.md) — the natural-language analogue: filesystem-first capability bundles.
- [Hooks](hooks.md) — `PlaybookEnforcerHook` is a normal hook; you can add it alongside guardrails / steering / telemetry.
- [Tools](tools.md) — playbook steps reference the tools you registered with `@tool`.
