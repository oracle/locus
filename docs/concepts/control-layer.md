---
title: Control-layer architecture
description: How the SDK's agent loop, grounding, admission policy, approvals, persistence, and audit records fit together—and where each guarantee stops.
---

# Control-layer architecture

The SDK separates deciding what to do from deciding whether a consequential
action may execute. This page is the domain-neutral map. `SecurityContext` is
one worked application of the same control primitives.

{{ locus_diagram control-layer-flow }}

A hold is `require_human` in the API: `ApprovalDecision.outcome` is
`ApprovalOutcome.REQUIRE_HUMAN`. These pages use the words allow, hold and
deny; the API strings are secondary detail.

## Responsibilities by layer

| Layer | Runtime responsibility | Developer responsibility | Model-dependent part |
|---|---|---|---|
| Agent loop | Runs the think/tool/observe cycle and emits typed events | Set tools, budgets, termination, and provider configuration | Tool choice and generated arguments |
| Grounding | Applies configured thresholds to a typed evidence partition | Enable it, supply relevant evidence, calibrate thresholds, choose fail behavior | Claim extraction and evidence classification |
| Action classification | Carries explicit environment, kind, scope, and tags | Derive complete, accurate labels from the actual call | Avoid using the model as the sole classifier for critical labels |
| Admission | Calls the side effect only on `allow`; returns/raises on hold or deny | Route every consequential path through `admit()` or `gate_tool()` | None in `ControlPolicy` evaluation itself |
| Approval | Persists a pending decision through a configured store | Provide a store that survives a restart (such as `FileApprovals`), an authenticated decision channel, and a resume flow | None required for the store state machine |
| Persistence | Restores supported run state through a checkpointer | Choose and operate a backend with suitable durability and isolation | None |
| Audit | Hash-chains records supplied to an `AuditTrail`; `OracleAuditTrail` makes them rows of an immutable table, `anchor_trail()` keeps the head in object storage | Choose the [backend](observability.md#where-the-audit-trail-lives), its retention window, and the bucket's retention rule | None |

## The enforcement boundary

`admit(action, perform, ...)` owns one concrete boundary: it does not call
`perform` unless policy returns `allow` (or a previously held call is resumed
with a valid approval). A deny or hold is recorded when a trail is supplied.

That guarantee applies only to the callable passed as `perform`. It does not
discover an alternate direct call elsewhere in your application. Prefer
`gate_tool()` when exposing a side-effecting tool to an agent, keep the
ungated implementation private, and test that every registered mutating tool
is wrapped.

### What the runtime enforces: tool-call integrity

The admission gate decides whether an action may run. A second check, inside
the runtime, decides whether the call that reaches the executor is the call
the model produced. With `Agent(tool_call_integrity=True)` the runtime signs
every `ToolCall` as the model's response is parsed, streamed or not, over a
canonical form of its id, name and arguments (sorted keys, no whitespace)
with HMAC-SHA256 under a random per-run key that lives only in memory. The
executor verifies each call against that record before it runs it.

A call that no longer matches is refused: the tool does not run, the model
reads a tool error that says so, and the run continues. The refusal lands on
the agent's `AuditTrail` when it has one (`governed_agent()`, or an
`AuditHook` among its hooks) with the signed and received digests and the
hook the change is attributed to, and `tool.integrity.refused` is published
on the event bus. `tool_call_integrity="abort"` ends the run with the typed
`ToolCallIntegrityError` instead, once the batch of tools has finished.

Hooks that legitimately rewrite arguments re-sign the call under their own
name, so the chain of custody stays explicit:

```python
from locus.hooks.provider import BeforeToolCallEvent, HookProvider


class RedactingHook(HookProvider):
    async def on_before_tool_call(self, event: BeforeToolCallEvent) -> None:
        event.arguments = {k: str(v).replace("@", " at ") for k, v in event.arguments.items()}
        event.resign(by="RedactingHook")  # chain of custody: ("model", "RedactingHook")
```

`event.resign()` is a no-op when the feature is off, so a hook can call it
unconditionally. The built-in `GuardrailsHook` (PII redaction) and
`ExternalHooks` (`PreToolUse` rewrites) already do. A hook that changes the
arguments without re-signing taints the record: the executor refuses the
call and the audit entry names that hook. A later re-sign does not clear the
taint, so one hook cannot launder another's silent change. The same holds
for anything else that changes a call after the model produced it, such as a
consumer mutating the `ToolCall` it was handed on a `ThinkEvent`.

Scope. The signature is kept on the run, keyed by call id, not on the
`ToolCall`, so message types, checkpoints and wire formats are unchanged. The
key is per run and never persisted. A held call that `resume(...,
perform_dangling=True)` re-issues is signed again under the new run's key
before it enters the hook seam; a call that reaches the executor without any
record in the current run, because code outside the runtime issued it, is
unsigned and runs unverified. The guarantee is that once the runtime has
signed a call, nothing may change it unnamed. Off by default in 5.x, on by
default from 6.0.

## Failure behavior

| Failure | Behavior to design for |
|---|---|
| Policy returns hold or deny | Side effect is not called; handle the refusal or interrupt explicitly |
| Process stops while held | Resume requires an `ApprovalStore` that survives a restart, such as [`FileApprovals`](../api/control.md#pausing-until-someone-decides), and a checkpointer; `InMemoryApprovals` is for tests and is lost with the process |
| External API succeeds, process stops before local result is recorded | A replay may occur; pass a stable idempotency key to the external API |
| Audit records are changed | `AuditTrail.verify()` detects a broken local chain; `OracleAuditTrail` keeps the rows where the database refuses the change, and `verify_anchor()` catches a chain cut short of its anchored head |
| Action is mislabeled or a route skips the gate | Policy cannot protect what it cannot see; coverage tests and code review are required |
| Grounding judge is wrong | Typed scores remain an assessment, not a truth oracle; evaluate and calibrate the judge for the deployment |

## Recommended production composition

1. Define a typed tool for the side effect and an argument-derived `Action`.
2. Wrap it with `gate_tool()` and a reviewed `ControlPolicy`. Keep the policy
   as a YAML or JSON file the owning team reviews, lint it in CI with
   `locus policy lint`, and load it with `ControlPolicy.load()` or a
   `PolicyWatcher` for reload without a deploy; see
   [the policy as a file](policy-authoring.md#the-policy-as-a-file). Every
   audit entry carries the hash of the rules it was decided under.
3. Use an `ApprovalStore` that survives a restart, such as `FileApprovals`, for
   human decisions (see
   [Pausing until someone decides](../api/control.md#pausing-until-someone-decides)).
   `gate_tool(..., on_refusal="interrupt", approval=store)` pauses the run on
   a hold, and `agent.resume(..., thread_id=..., perform_dangling=True)`
   re-issues the held call once a person has decided with `store.decide(...)`.
   `FileApprovals` keeps held calls in one JSON file across a restart but
   serialises writers within one process only; `InMemoryApprovals` does not
   survive a restart. An `ApprovalAuthority` limits which principals may
   decide; authenticating the person named in `by=` is your decision
   channel's job.
4. Use a checkpointer for resumable agent state.
5. Send a stable operation key to the downstream service.
6. Keep audit records where the agent cannot rewrite them: an `OracleAuditTrail`
   on an immutable table, or a trail whose head is anchored with `anchor_trail()`
   into a bucket with a retention rule (see
   [where the audit trail lives](observability.md#where-the-audit-trail-lives)).
7. Test allow, hold, deny, restart, duplicate-call, and crash-recovery paths.

See [First controlled action](../how-to/first-controlled-action.md) for a small
offline run, [policy authoring](policy-authoring.md) for the file form, lint
and classification coverage, [`gate_tool`](../api/control.md#locus.control.gate.gate_tool) for the
interrupt-and-resume parameters, [interrupts](interrupts.md) for
`agent.resume(...)`.
