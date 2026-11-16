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
