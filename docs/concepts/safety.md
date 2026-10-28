# Safety, guardrails, and steering

Three layers cooperate inside an agent run:

1. **Validation** — typed tool arguments are JSON-schema-checked before
   the call lands. No opt-in needed.
2. **Guardrails** — content policy, PII redaction, dangerous-tool
   blocking, prompt/result length caps. Runs as a hook at three points:
   on the incoming prompt, before each tool call, and on each tool
   result. To filter the model's own output, add `OutputFilterHook`.
3. **Steering** — a second model evaluates every tool call before it
   fires. The judge sees the recent tool-call activity and your policy,
   and emits one of *PROCEED / GUIDE / INTERRUPT* (allow, cancel with
   feedback, or pause for a human).

Each layer plugs in independently. You can turn one on without the
others.

## When to reach for which layer

| Situation | Layer |
|---|---|
| Tool args from the model are sometimes malformed | Validation — already on; nothing to do |
| Public-facing agent — block prompt injection, SQL/command/path-traversal patterns, cap input length | `GuardrailsHook` with the default `GuardrailConfig` |
| Customer-facing answer where leaking PII (emails, SSN, credit cards, IPs) is a compliance issue | `GuardrailsHook` with PII patterns enabled |
| High-stakes tools (`issue_refund`, `delete_records`, `isolate_host`) — the action must not happen unless a policy allows it | [`gate_tool`](../api/control.md) — a check in real code, which the model cannot talk around |
| The same tools, where you want a second model's opinion *as well* | `SteeringHook` with a steering `model=` and a policy string — advisory, see below |
| Domain restriction — *"the user came in to resolve this ticket, reject anything else"* | `SteeringHook` with that policy verbatim |
| Internal-only agent, trusted prompts, low-stakes tools | none of the above; default validation is enough |

## Getting started

### Guardrails — block dangerous tools and redact PII

```python
from locus.agent import Agent
from locus.hooks.builtin.guardrails import (
    GuardrailsHook, GuardrailConfig, GuardrailAction,
)

config = GuardrailConfig(
    block_dangerous_tools=frozenset({"shell", "exec", "rm", "drop"}),
    max_prompt_length=50_000,
    default_action=GuardrailAction.BLOCK,
)

agent = Agent(
    model="{{ locus_example_model }}",
    tools=[query_siem, summarise],
    hooks=[GuardrailsHook(config=config)],
)
```

`GuardrailsHook` ships with sensible defaults — the empty
`GuardrailConfig()` already blocks `eval`, `exec`, `system`, `shell`,
`rm`, `delete`, `drop`, `truncate`; detects email / phone / SSN /
credit-card / IP patterns; and watches for SQL-injection,
path-traversal, and command-injection shapes in tool inputs.

### Topic and content policies — domain restriction

`TopicPolicy` and `ContentPolicy` are enforced by `OutputFilterHook`,
which scans the model's **output** (not tool inputs — that's
`GuardrailsHook`'s job):

```python
from locus.hooks.builtin.guardrails import (
    OutputFilterHook, TopicPolicy, ContentPolicy,
)

topic_policy = TopicPolicy(
    blocked_topics={"weapons", "hacking"},
    keywords={
        "weapons": ["gun", "rifle", "ammunition"],
        "hacking": ["exploit", "zero-day", "rootkit"],
    },
)

content_policy = ContentPolicy(
    enabled_categories={"hate_speech", "self_harm", "illegal_activity"},
)

agent = Agent(
    model="{{ locus_example_model }}",
    tools=[...],
    hooks=[OutputFilterHook(
        topic_policy=topic_policy,
        content_policy=content_policy,
    )],
)
```

Both policies are simple keyword classifiers — fast, predictable,
auditable. For production-grade content moderation, swap in an
ML-backed policy (OpenAI Moderation, etc.)
behind the same `Policy.check(text) -> str | None` shape.

### Steering — a second model judges every tool call

!!! warning "Steering is advisory, and can fail open"

    A steering policy is a second model asked to judge the first, so a judge
    that does not intervene is indistinguishable from one that approved.

    Measured against a self-hosted Qwen3.6-35B with
    `policy="Never allow delete or destructive operations."` — the judge did
    not intervene, and the agent reported:

    > the `users` table has been successfully deleted.

    Under the **same model and the same prompt**, a tool whose body goes
    through [`admit()`](../api/control.md) held: the model still called the
    destructive tool, admission refused it, the side effect never ran, and the
    refusal landed on a chain that verifies.

    Use steering to *steer*. For an action that must not happen, use
    [`gate_tool`](../api/control.md) — or `interrupt_tools` below, which is a
    set-membership check that never consults the judge and so cannot fail open.

```python
from locus.models import AnthropicModel
from locus.hooks.builtin import SteeringHook

# The steering model is a model instance — it can be a smaller/cheaper
# model than the agent's main one.
judge = AnthropicModel(model="claude-sonnet-5-5")

agent = Agent(
    model="{{ locus_example_model }}",
    tools=[query_siem, block_indicator, isolate_host],
    hooks=[
        SteeringHook(
            model=judge,
            policy=(
                "The user came in to triage alert A-4271. "
                "Block any tool call unrelated to that alert."
            ),
        ),
    ],
)
```

Before `block_indicator` or `isolate_host` fires, the judge sees the
recent tool-call activity and your policy, then emits one of three
decisions:

- **PROCEED** — the call goes through.
- **GUIDE** — the call is cancelled and corrective feedback is fed back
  to the model, triggering a re-plan.
- **INTERRUPT** — the run pauses for human approval (also used for any
  tool listed in `interrupt_tools=`).

Use the smallest model that gives reliable decisions — a `mini` /
`flash` / `haiku` is usually enough.

## Validation (you don't have to do anything)

The `@tool` decorator builds a JSON schema from the function's typed
signature. Every model tool call goes through that schema before the
function body runs. Schema violations come back to the model as a
tool error so it can retry with corrected arguments — you don't have
to write any of that defensively.

```python
@tool
def isolate_host(host_id: str, incident_id: str, mode: Literal["full", "soft", "audit"]) -> dict:
    ...
```

A model call with `mode="quarantine"` is rejected before the body
runs; the model sees the typed-error message and retries with `"full"`.

## Structural guardrails — restrict the tool, not just the call

The strongest guardrail is a tool that *cannot* perform the dangerous action
in the first place. Hooks and steering inspect a call and decide; a
capability-restricted tool removes the capability, so there is nothing to
decide. The two compose — but when you can constrain the tool, do that first.

The [cloud-posture agent](cloud-posture.md) is the worked example. Its
`use_aws` tool admits an operation only if its name starts with a read verb
(`Describe` / `List` / `Get` / …), and it checks **before** any API call:

```python
from locus_security import use_aws

use_aws("iam", "CreateUser", {"UserName": "x"})
# PermissionError: refused: iam:CreateUser is not a read-only operation.
```

The principle is **defense in depth across a trust boundary**: pair an
in-process gate (fast, explicit intent) with an out-of-process backstop that
doesn't trust your code at all —

| In-process gate | Out-of-process backstop |
|---|---|
| read-verb allowlist in `use_aws` | read-only IAM identity (`SecurityAudit` + `ViewOnlyAccess`) |
| `block_dangerous_tools` in `GuardrailsHook` | container without the binary; seccomp profile |
| SQL-shape detection in tool inputs | database role with `SELECT`-only grants |

If the gate is ever bypassed, the backstop still refuses. Neither layer trusts
the model — and the backstop doesn't trust the gate either.

## Common gotchas

| Symptom | Likely cause |
|---|---|
| PII redaction over-aggressive | The default IP regex matches version strings too. Drop `ip_address` from `pii_patterns` or tighten to a CIDR-aware pattern. |
| Steering rejects almost everything | Judge model is too strict. Tune the policy or move to a stronger model — a `nano` is often too small for nuanced judgement. |
| `GuardrailsHook` blocks a legitimate message | Inspect `hook._violations` after the run for the violation type, then add an action override (`action_overrides={"sql_injection": ALLOW}`) or trim the regex. |
| Validation error swallows a tool-arg bug | The error came back to the model — it's in the trace, look for `ToolCompleteEvent.error`. |

## Source and notebooks

- [`notebook_69_guardrails_security.py`](https://github.com/oracle/locus/blob/main/examples/notebook_69_guardrails_security.py) — basic guardrails.
- [`notebook_70_guardrails_advanced.py`](https://github.com/oracle/locus/blob/main/examples/notebook_70_guardrails_advanced.py) — topic + content + PII layered.
- [`notebook_68_steering.py`](https://github.com/oracle/locus/blob/main/examples/notebook_68_steering.py) — judge-model approval.
- [`locus.hooks.builtin.guardrails`](https://github.com/oracle/locus/blob/main/src/locus/hooks/builtin/guardrails.py)
- [`locus.hooks.builtin.steering`](https://github.com/oracle/locus/blob/main/src/locus/hooks/builtin/steering.py)

## See also

- [Hooks](hooks.md) — how `GuardrailsHook` and `SteeringHook` plug into the lifecycle.
- [Tools](tools.md) — the `@tool` decorator and its schema validation.
- [Reasoning: grounding](reasoning.md#grounding) — the answer-side analogue, claim-by-claim.
- [Cloud-posture agent](cloud-posture.md) — read-only-by-construction tools in practice.
- [Threat scenarios](threat-scenarios.md) — security mapping: which guardrail covers which OWASP / ATLAS item.
