---
title: Control-layer architecture
description: How the SDK's agent loop, grounding, admission policy, approvals, persistence, and audit records fit together—and where each guarantee stops.
---

# Control-layer architecture

The SDK separates deciding what to do from deciding whether a consequential
action may execute. This page is the domain-neutral map.

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
| Approval | Persists a pending decision through a configured store, and writes the hold into the thread's checkpoint | Provide a store that survives a restart (`FileApprovals`, or `PostgresApprovals` / `RedisApprovals` / `OracleApprovals` when replicas share it), an authenticated decision channel, and a resume flow | None required for the store state machine |
| Persistence | Restores supported run state through a checkpointer | Choose and operate a backend with suitable durability and isolation | None |
| Audit | Hash-chains records supplied to an `AuditTrail`; `OracleAuditTrail` makes them rows of an immutable table, `anchor_trail()` keeps the head in object storage | Choose the [backend](observability.md#where-the-audit-trail-lives), its retention window, and the bucket's retention rule | None |
| Judge (optional) | Folds a typed allow / hold / deny verdict into the admission decision; the stricter side wins | Write the intent and per-tool guidance, choose and evaluate the judge model, set a budget | The verdict |

## The enforcement boundary

`admit(action, perform, ...)` owns one concrete boundary: it does not call
`perform` unless policy returns `allow` (or a previously held call is resumed
with a valid approval). A deny or hold is recorded when a trail is supplied.

That guarantee applies only to the callable passed as `perform`. It does not
discover an alternate direct call elsewhere in your application. Prefer
`gate_tool()` when exposing a side-effecting tool to an agent, keep the
ungated implementation private, and test that every registered mutating tool
is wrapped.

## A second check that reads the call

The policy weighs the labels an `Action` carries, and labels are all it can
see: a staging rollout pinned to `1.8.2` and one to the floating `latest` tag
are the same action to it. `ToolCallJudge` is an optional, model-based check
that reads the proposed call itself, the tool's name and the arguments as the
model sent them, against the policy's intent in plain words and optional
per-tool guidance, and returns a typed verdict: allow, hold or deny, with a
one-line reason. The reply is parsed through the structured-output path, so it
is a value of a known schema or it is no verdict.

Hand it to `gate_tool(..., judge=judge)` and the verdict reaches `admit()` as
evidence, folded into the decision by `tighten()`: the stricter of the
policy's outcome and the judge's wins, so the judge can turn an allow into a
hold or a deny and can never do the reverse. Install it as a hook
(`hooks=[judge]`) and it judges every call the agent makes, cancelling a held
or denied one with a refusal the model can read. Either way the verdict is on
the audit record (judge model, verdict, reason, whether it came from the
cache) and emitted as `control.judge.decided`.

What it adds: judgement over the arguments, which no label set carries, and a
reading of intent that can catch the call the risk list never named (see
[guidance for the judge](policy-authoring.md#guidance-for-the-judge)).

Where it stops:

- It is a model. A wrong verdict is possible in both directions; the fold
  bounds the damage of a wrong *allow*, because the policy's own hold or deny
  stands whatever the judge says. A wrong hold costs a human review.
- It fails closed. A judge model that errors, replies with something that is
  not a verdict, or has spent its per-run `budget` returns a hold, with the
  failure on the audit entry under `judge.error`. An unavailable judge turns
  an allow into a hold, never into a silent allow.
- It costs a model call per distinct call. Tools declared
  `@tool(read_only=True)` are not judged, verdicts are cached per (tool,
  canonical arguments) within a run, and `budget=` caps calls per run.
- A judge given only to `gate_tool` sees only gated tools. As a hook it sees
  every call, but a hook hold stops the call before the gate, so the trail
  then carries a `tool-call-judged` record for it rather than an
  `action-admission` one.

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
from locus.hooks.provider import BeforeToolCallEvent, HookPriority, HookProvider


class RedactingHook(HookProvider):
    priority = HookPriority.SECURITY_DEFAULT

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
| Process stops while held | Resume requires an `ApprovalStore` that survives a restart ([`FileApprovals`, `PostgresApprovals`, `RedisApprovals` or `OracleApprovals`](../api/control.md#pausing-until-someone-decides)) and a checkpointer; the checkpoint carries the hold (`pending_holds()`), so the resuming process knows which approval the thread waits on and in which store; `InMemoryApprovals` is for tests and is lost with the process |
| External API succeeds, process stops before local result is recorded | A replay may occur; pass a stable idempotency key to the external API |
| Audit records are changed | `AuditTrail.verify()` detects a broken local chain; `OracleAuditTrail` keeps the rows where the database refuses the change, and `verify_anchor()` catches a chain cut short of its anchored head |
| Action is mislabeled or a route skips the gate | Policy cannot protect what it cannot see; coverage tests and code review are required |
| Grounding judge is wrong | Typed scores remain an assessment, not a truth oracle; evaluate and calibrate the judge for the deployment |
| Tool-call judge errors, replies off-schema, or its budget is spent | The call is held, not allowed; the failure is on the audit entry under `judge.error` |
| Tool-call judge verdict is wrong | A wrong hold costs a review; a wrong allow changes nothing the policy holds or denies, because the fold is one-directional |

## Recommended production composition

1. Define a typed tool for the side effect and an argument-derived `Action`.
2. Wrap it with `gate_tool()` and a reviewed `ControlPolicy`. Keep the policy
   as a YAML or JSON file the owning team reviews, lint it in CI with
   `locus policy lint`, and load it with `ControlPolicy.load()` or a
   `PolicyWatcher` for reload without a deploy; see
   [the policy as a file](policy-authoring.md#the-policy-as-a-file). Every
   audit entry carries the hash of the rules it was decided under.
3. Use an `ApprovalStore` that survives a restart for human decisions (see
   [Pausing until someone decides](../api/control.md#pausing-until-someone-decides)).
   `gate_tool(..., on_refusal="interrupt", approval=store)` pauses the run on
   a hold, and `agent.resume(..., thread_id=..., perform_dangling=True)`
   re-issues the held call once a person has decided with `store.decide(...)`.
   The stores:

    | Store | Where held calls live | Use it when |
    |---|---|---|
    | `InMemoryApprovals` | This process | Tests and demos; gone on restart |
    | `FileApprovals` | One JSON file, written atomically | One agent process and the operators that can open its file; writers are serialised within one process only |
    | `PostgresApprovals` | One table, each decision under a row lock | Several replicas share the store; the same `asyncpg` conventions as the PostgreSQL checkpointer |
    | `RedisApprovals` | Keys under a prefix, changed under a lock | Replicas that already share a Redis |
    | `OracleApprovals` | One table in Oracle AI Database, each decision under a row lock | Replicas on Autonomous Database; the same wallet and pool conventions as the Oracle checkpointer |

    Every store names itself with a spec (`store.spec`: `file://…`,
    `postgresql://…`, `redis://…`, `oracle://…`), `approval_store(spec)` opens
    one, and `locus approvals --store <spec> list|approve|reject --by <who>
    --reason <why>` decides from the command line. An `ApprovalAuthority`
    limits which principals may decide; authenticating the person named in
    `by=` is your decision channel's job.

    **Across a restart.** When the gate holds a call during a run with a
    checkpointer, the hold (approval id, action, asset, principal, arguments,
    store spec) is written into the thread's checkpoint. A new process with
    the same checkpointer reads it with `pending_holds(checkpointer,
    thread_id)`, re-checks it against the store (`hold.state(store)`), and
    `agent.resume(..., thread_id=..., perform_dangling=True)` performs the
    approved call once; the hold leaves the checkpoint when the call is
    answered. Checkpoints written before holds existed carry none, and resume
    falls back to the dangling tool call as before.
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

Notebooks [71](../notebooks/notebook_71_payment_refund_gate.md) (payment
refund), [72](../notebooks/notebook_72_infra_deploy_gate.md) (infrastructure
deploy), [73](../notebooks/notebook_73_support_account_gate.md) (support
account), [74](../notebooks/notebook_74_data_deletion_gate.md) (data deletion)
and [75](../notebooks/notebook_75_cloud_resource_gate.md) (cloud resource) run
this composition offline, one gate per domain.

## Conformance fixtures

A port of the governance core to another language has to reproduce the same
bytes: the canonical JSON the hashes are taken over, the audit record fields
and the sentence `check_jsonl()` returns for each tamper class, the policy
hash, lint findings and decisions, approval digests and ids, and the
`action-admission` line an admission writes. The repository exports those as
test vectors under
[`conformance/v1/`](https://github.com/oracle/locus/tree/main/conformance/v1),
generated by `scripts/conformance_export.py` from a fixed clock, fixed inputs
and fixture signing keys. The README there explains each directory and the
canonicalisation rules in prose, and a dependency-free Node script under
`conformance/v1/tools/` recomputes the digests and verifies the signatures
from the other side.

The fixtures are a contract: `v1` never changes once published. A unit test
regenerates them and fails on any difference, so a change to a format is a
deliberate `conformance/v2` next to it, not an edit of `v1`. A port that
passes every `v1` vector interoperates with this SDK: a trail it signs
verifies with `locus audit verify --key --head`, and the reverse.
