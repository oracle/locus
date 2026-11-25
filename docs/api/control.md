# Control

The admission gate: decide whether a consequential action may run, and
record the decision either way.

`locus.control` is the domain-neutral surface, and the import path to use.

For the concepts, start with [The control layer](../concepts/control-layer.md)
and [Writing a policy that holds](../concepts/policy-authoring.md).

## Admitting an action

`admit()` evaluates the policy, records the decision on the audit trail, and
runs the action only if it was allowed. A held or denied action raises
`AdmissionError` carrying the `ApprovalDecision` that explains why.

::: locus.control.admission.admit
::: locus.control.admission.AdmissionError

## Deciding

`approve()` is the pure decision function — no I/O, no side effects. It takes
an action and a policy and returns the outcome. Rules combine by taking the
strongest result, so deny beats hold beats allow (in the API,
`ApprovalOutcome.DENY` > `REQUIRE_HUMAN` > `ALLOW`).

::: locus.control.policy.approve
::: locus.control.policy.ControlPolicy
::: locus.control.policy.ApprovalDecision
::: locus.control.policy.ApprovalOutcome

### The policy as a file

`ControlPolicy.load()` reads a policy from a YAML or JSON file and
`policy.dump()` writes one; every key is a field of the same name, unknown
keys are errors that name the key, and the two forms round-trip.
`policy.content_hash()` names the rules a decision was made under; `admit()`
records it as `policy_hash` on every audit entry. `PolicyWatcher` follows a
file and hands `admit()` or `gate_tool()` the current policy, one per
decision. The [policy authoring](../concepts/policy-authoring.md#the-policy-as-a-file)
page shows the file form and the `locus policy lint` and `locus policy
explain` commands that read it.

::: locus.control.policy_file.PolicyWatcher
::: locus.control.policy_file.PolicyDocument
::: locus.control.policy_file.PolicyFileError
::: locus.control.policy_file.policy_hash
::: locus.control.policy.PolicySource

### Linting and explaining a policy

What `locus policy lint` and `locus policy explain` do, as calls.

::: locus.control.policy_lint.lint_policy
::: locus.control.policy_lint.LintReport
::: locus.control.policy_lint.LintFinding
::: locus.control.policy_lint.explain
::: locus.control.policy_lint.Explanation
::: locus.control.policy.evaluate_rules
::: locus.control.policy.RuleCheck

## Describing an action

A policy matches on what an action *is* — its environment, kind, blast
radius, and tags — never on the name of the tool performing it.

::: locus.control.policy.Action

## Gating a tool

`gate_tool` puts the gate in front of a tool the agent already has. The
returned tool keeps the original's name, description and parameter schema, so
the model sees no difference and nothing else in the agent changes — which is
what makes the control structural rather than advisory. There is nothing to
notice, so nothing to talk around.

```python
from locus.control import ControlPolicy, gate_tool

agent = Agent(model=model, tools=[
    lookup_order,                                     # read-only, ungated
    gate_tool(issue_refund, policy=ControlPolicy()),  # gated
])
```

A refusal comes back to the model as a readable result naming the outcome and
the reason, so the agent can explain the hold rather than the run ending in a
traceback. That is `on_refusal="return"`, the default. Pass
`on_refusal="raise"` for a caller that would rather stop, or
`on_refusal="interrupt"` to pause the run on a hold until someone decides; a
denial still comes back as a refusal (see
[Pausing until someone decides](#pausing-until-someone-decides)).

Gating a **sandboxed** tool composes rather than replacing it: the gate
decides, and only an admitted call reaches the sandbox.

::: locus.control.gate.gate_tool

### Holding an action for a human

A hold is only useful if the agent can find out what happened next. Give
`gate_tool` an `approval` bridge and a held refusal carries an `approval_id` the
agent can poll, while a human decides on a channel the agent cannot reach:

```json
{"status": "held_for_approval", "outcome": "require_human",
 "action": "issue_refund", "asset": "ord-4821", "reason": "...",
 "approval_id": "appr-77",
 "next": "call approval_status(approval_id) once a human decides"}
```

A **denial** deliberately gets no id. It is final, and offering one would invite
the agent to wait for a decision that is not coming.

`ApprovalBridge` is a structural `Protocol` with no import-time dependency, so
any approval queue with `submit` and `state` methods satisfies it. The stores
the SDK ships, `InMemoryApprovals`, `FileApprovals`, `PostgresApprovals`,
`RedisApprovals` and `OracleApprovals`, satisfy it too; see
[Pausing until someone decides](#pausing-until-someone-decides).

::: locus.control.gate.ApprovalBridge

### What the user hears when an action is refused

The `reason` in that payload is, by default, the policy's own — a join of the
checks that fired:

```
"blast radius 3 exceeds the maximum 1; labels ['large_refund'] are denied by policy"
```

That is the right level of detail for the audit trail and for a developer
reading a log. It is also control-plane vocabulary, and a model handed it
repeats it verbatim. Run against a live model, the refusal above reached the
customer as *"the blast radius (3) exceeds the maximum 1"* and *"it's
classified as a `large_refund`"*.

`refusal_reason` gives the model the sentence you want the user to hear
instead. Pass a string, or `(decision) -> str` to vary it by outcome:

```python
gate_tool(
    issue_refund,
    policy=policy,
    trail=trail,
    refusal_reason=lambda d: (
        "We can't refund this amount automatically."
        if d.outcome == "deny"
        else "This refund is waiting on a manager."
    ),
)
```

The full policy reason still goes to the trail — a friendlier sentence for the
customer must not shrink the record. Added in 2.10.0.

### Pausing until someone decides

A polled id keeps the run going while a human decides. With
`on_refusal="interrupt"` and an approval store, a hold pauses the run instead:
the agent yields an `InterruptEvent` whose `metadata` carries the
`approval_id`, the checkpointer keeps the conversation, and the store keeps the
pending approval. A person decides with `decide()`, and
`agent.resume(..., thread_id=..., perform_dangling=True)` re-issues the held
call, which finds the decision:

```python
from locus.control import FileApprovals, gate_tool

store = FileApprovals("approvals.json")
refund = gate_tool(
    issue_refund,
    policy=policy,
    approval=store,
    on_refusal="interrupt",
    trail=trail,
)
agent = Agent(model=model, tools=[refund], checkpointer=FileCheckpointer("checkpoints"))

async for event in agent.run("refund order 4821", thread_id="t1"):
    if isinstance(event, InterruptEvent):
        approval_id = event.metadata["approval_id"]   # the run is parked

# Later, from any process that can open the same file:
store.decide(approval_id, "approved", by="alice@example.com")
async for event in agent.resume("approved", thread_id="t1", perform_dangling=True):
    ...
```

An approval names one call. Its id is derived from a digest of the principal, the tool, the arguments, `policy.version` when set, and any `approval_context`, so a call with different arguments waits for its own decision.

An approved call is weighed against the policy again when it is redeemed, so a `deny` such as a spend limit still refuses it, and it runs at most once as long as the store's writes do not race (see the `FileApprovals` limit below): the approval is consumed immediately before the side effect, so a repeated call holds again instead of riding an old yes.

A denied call returns a refusal. A policy `deny` never pauses. `on_refusal="interrupt"` needs an `ApprovalStore`, not a bare `ApprovalBridge`, and raises `TypeError` without one.

`InMemoryApprovals` lives in one process and is gone on restart; it is for tests and demos. `FileApprovals` keeps every record in one JSON file, re-read on every call and written atomically, so a decision written by another process is seen by the next call.

Writes are serialised only through one `FileApprovals` object in one process, and the agent's own submits and consumes are writes too. A decision saved from another process, or through another `FileApprovals` object, at the same moment can overwrite one of them. In the worst case a consumed approval goes back to `approved` and can be redeemed again. Where that matters, use one of the database-backed stores.

**Database-backed stores.** `PostgresApprovals`, `RedisApprovals` and `OracleApprovals` keep one record per held call and make every submit, decision and consumption under the database's own lock (`SELECT ... FOR UPDATE` in PostgreSQL and Oracle, a lock key in Redis), so replicas that share the store never both redeem one approval.

They follow the connection conventions of the matching checkpointer backend:

- PostgreSQL: a libpq DSN or host / port / database / user / password
- Redis: a `redis://` URL and key prefix
- Oracle: a TNS alias or host / port / service name with an optional `config_dir` (the `tnsnames.ora` directory, for a TLS-only database) or wallet (mutual TLS)

The table is created on first use only when it is missing, so a role without `CREATE` works against a table a migration made. Each store owns one event loop on a background thread, where its pool lives; the `ApprovalStore` methods block on it, and `asubmit`, `aget`, `adecide`, `aconsume`, `apending` await it from your own loop. Call `close()` (or `await aclose()`) when the application shuts down.

```python
from locus.control import OracleApprovals, PostgresApprovals, RedisApprovals

store = PostgresApprovals(dsn="postgresql://locus@db.internal:5432/locus")
store = RedisApprovals("redis://cache.internal:6379", prefix="billing:approvals:")
store = OracleApprovals(dsn="locusdb_low", user="LOCUS", password=pw, wallet_location="/opt/wallet")
```

**Naming a store.** Every store has a `spec` with any password removed
(`memory://`, `file://approvals.json`, `postgresql://locus@db.internal:5432/locus?table=public.locus_approvals`,
`redis://cache.internal:6379?prefix=billing:approvals:`, `oracle://LOCUS@/?dsn=locusdb_low&wallet=/opt/wallet&table=locus_approvals`),
and `approval_store(spec)` opens one from it. A database spec with nothing
after the scheme reads the environment the integration tests use
(`POSTGRES_HOST` and friends, `REDIS_URL`, `ORACLE_DSN` / `ORACLE_USER` /
`ORACLE_PASSWORD` / `ORACLE_CONFIG_DIR` / `ORACLE_WALLET`). The `locus approvals`
command takes the
same spec:

```bash
locus approvals --store postgresql://locus@db.internal:5432/locus list
locus approvals --store postgresql://locus@db.internal:5432/locus approve appr-… --by alice --reason "ticket 4821"
locus approvals --store redis://cache.internal:6379 reject appr-… --by bob
```

`--store` defaults to `LOCUS_APPROVALS_STORE` and `--by` to `LOCUS_APPROVER`;
`--authority MODULE:ATTR` checks every decision against an `ApprovalAuthority`.
The decision records who decided and the `--reason` given.

**The hold in the checkpoint.** When the gate holds a call during a run with a
checkpointer, the runtime writes the hold into the thread's checkpoint
(`state.metadata["_locus_pending_holds"]`): the approval id, tool, action,
asset, principal, reason, arguments, approvers so far, context and the store's
spec. A process that finds the thread later reads them with
`pending_holds(checkpointer, thread_id)` and re-checks each against its store:

```python
from locus.control import approval_store, pending_holds


async def show_holds(checkpointer):
    for hold in await pending_holds(checkpointer, "t1"):
        store = approval_store(hold.store)
        print(hold.approval_id, hold.action, hold.asset, hold.state(store))
```

`agent.resume(..., thread_id=..., perform_dangling=True)` then performs the
approved call once (or raises `ApprovalPendingError` while it is still
pending), and the hold leaves the checkpoint when the call is answered. A
checkpoint written before holds existed carries none; resume falls back to the
dangling tool call, as before.

**Who may decide.** Without an `ApprovalAuthority`, any named principal can
decide. With one, every decision is checked when it is made:

```python
from locus.control import ApprovalAuthority, ApproverRule, FileApprovals

authority = ApprovalAuthority(
    rules=(
        ApproverRule(labels=frozenset({"payment"}), roles=frozenset({"finance"})),
        ApproverRule(
            labels=frozenset({"production"}),
            approvers=frozenset({"olga", "sam"}),
            quorum=2,
        ),
    ),
    roles_of=directory.roles_for,
)
store = FileApprovals("approvals.json", authority=authority)
```

Every rule that matches the held action's labels must reach its quorum of distinct authorised approvers before the call is approved; one authorised denial ends it. The principal that requested the action cannot approve it, directly or through a delegation it granted, unless every matching rule sets `allow_self_approval`.

A `Delegation` lends an approver's authority to someone else until a deadline. A decision by someone without authority raises `ApprovalAuthorityError` and is kept on the record as a rejection. An action no rule covers cannot be approved by anyone: the authority fails closed.

::: locus.control.approvals.ApprovalStore
::: locus.control.approvals.InMemoryApprovals
::: locus.control.approvals.FileApprovals
::: locus.control.approvals_postgres.PostgresApprovals
::: locus.control.approvals_redis.RedisApprovals
::: locus.control.approvals_oracle.OracleApprovals
::: locus.control.approvals_spec.approval_store
::: locus.control.approvals_spec.store_spec
::: locus.control.holds.PendingHold
::: locus.control.holds.pending_holds
::: locus.control.holds.holds_of
::: locus.control.approvals.ApprovalRecord
::: locus.control.approvals.call_digest
::: locus.control.approvals.ApprovalAuthority
::: locus.control.approvals.ApproverRule
::: locus.control.approvals.Delegation
::: locus.control.approvals.ApprovalAuthorityError

### Capping spend

`Action.cost_usd` says what an action spends. `ControlPolicy` can hold an
action above a per-action cost (`require_human_over_usd`) and deny one that
would take its scope's cumulative spend past a limit (`spend_limit_usd`); no
approval overrides that denial. A spend ledger supplies the cumulative figure:

```python
from locus.control import Action, ControlPolicy, FileSpendLedger, gate_tool

refund = gate_tool(
    issue_refund,
    policy=ControlPolicy(spend_limit_usd=5_000, require_human_over_usd=500),
    action=lambda name, args: Action(
        name=name, asset=args["order_id"], cost_usd=args["amount_usd"]
    ),
    ledger=FileSpendLedger("spend.json"),
    spend_scope=lambda name, args: f"customer:{args['customer_id']}",
)
```

A scope is any string: a thread, a customer, a tenant, a month. Spend is recorded only after the action has run, so a refused or failed action costs nothing.

The check and the record are not one transaction. Two calls against the same scope that overlap can each pass the check before either records, so the cap can be exceeded. That includes two processes, and also two tool calls the agent runs together in one turn (the default is `tool_execution="concurrent"`). `SpendLedger` reads and records in separate calls, before and after the action, so a ledger cannot make the check atomic by itself.

Where the cap must hold, run the agent that has the spending tools with `tool_execution="sequential"` and keep one writer per scope. `InMemorySpendLedger` is the in-process version, for tests and demos. `admit()` takes `ledger=` and a string `spend_scope=` directly.

::: locus.control.spend.SpendLedger
::: locus.control.spend.InMemorySpendLedger
::: locus.control.spend.FileSpendLedger

### Judging the call

The policy sees labels. `ToolCallJudge` reads the call, the tool's name and the arguments the model sent, against an intent written in words and optional per-tool guidance, and returns a typed `allow` / `hold` / `deny` with a one-line reason.

Pass it to `gate_tool(judge=...)` and the verdict reaches `admit()` as `judgement`, where `tighten()` keeps the stricter of the policy's outcome and the judge's: an allow can become a hold or a deny, never the reverse. The verdict is on the audit entry under `judge`; a model error, an off-schema reply or a spent `budget` is a hold.

```python
from locus.control import ControlPolicy, ToolCallJudge, gate_tool

judge = ToolCallJudge(
    small_model,
    intent="Refund a customer's own order; never move money elsewhere.",
    guidance={"issue_refund": "Hold anything over 500 USD."},
    budget=20,
)
refund = gate_tool(issue_refund, policy=ControlPolicy(), trail=trail, judge=judge)
```

Installed as a hook (`hooks=[judge]`) it judges every call and cancels a held
or denied one with a refusal the model can read; with no `model=` it borrows
the agent's. Tools declared `@tool(read_only=True)` are not judged, and
verdicts are cached per (tool, canonical arguments) within a run.

::: locus.control.judge.ToolCallJudge
::: locus.control.judge.JudgeVerdict
::: locus.control.judge.tighten

### Tool-call integrity

By default (`tool_call_integrity=True`) the runtime signs every tool call as
the model's response is parsed and the executor verifies it before it runs;
a call changed in between is refused with `ToolCallIntegrityError`, recorded
on the agent's `AuditTrail` and published as `tool.integrity.refused`. A hook
that rewrites arguments re-signs through `event.resign(by=...)` on its
`BeforeToolCallEvent`; `Agent(tool_call_integrity=False)` turns the check
off. The [control-layer page](../concepts/control-layer.md#what-the-runtime-enforces-tool-call-integrity)
describes what the check guarantees and where it stops.

::: locus.tools.integrity.ToolCallSigner
::: locus.tools.integrity.ToolCallSignature
::: locus.tools.integrity.ToolCallIntegrityError

## Deriving action labels

Turn a tool call into an `Action` using declarative rules, so the labels a
policy matches on are not hand-written per call site.

::: locus.control.action.ActionSpec
::: locus.control.action.resolve_action
::: locus.control.action.default_action
::: locus.control.action.action_from_labels
::: locus.control.action.derive_labels
::: locus.control.action.DerivedLabels
::: locus.control.action.asset_from_args
::: locus.control.action.UNDETERMINED_TAG
::: locus.control.policy.SANDBOXED_TAG

## The record

A hash-chained log of every decision. Each record commits to the previous
hash, so editing any record breaks `verify()`.

!!! warning "Tamper-evident, not tamper-proof"
    By default this is a keyless SHA-256 chain held in memory. It *detects*
    edits when checked against a head hash you retain out-of-band; it does not
    prevent them, and it does not anchor the log. Without that head, anyone
    who can write the log can rebuild an unsigned chain around an edit.
    Persist the JSONL and pin the head hash externally before relying on it as
    compliance evidence, and [sign the trail](#signing-the-trail) to catch a
    rebuild.

### What `verify()` catches, and the one thing it cannot

Called with no arguments, `verify()` catches every edit, every reorder, and
every deletion **from the middle** of the chain — each leaves a record whose
stored hash no longer matches its contents, or whose `prev_hash` no longer
points at the record before it.

It cannot, on its own, catch a **truncation**:

| Attack | `verify()` | `verify(expected_head=…)` |
| --- | --- | --- |
| Edit a record | `False` | `False` |
| Reorder records | `False` | `False` |
| Delete from the middle | `False` | `False` |
| Drop records off the end | `True` | `False` |
| Discard the trail entirely | `True` | `False` |

Dropping the tail leaves a shorter chain that is perfectly valid on its own
terms. That is a property of hash chains in general, not of this
implementation: nothing inside a chain can attest to a link that was never
handed to it. An agent that can reach its own audit trail can therefore erase
the end of it and still pass a bare `verify()`.

Anchoring closes it. Every attack in that table moves the head, so keep
`head` somewhere the agent cannot reach — a WORM bucket, an append-only log,
a co-signer, a transparency log — and pass it back:

```python
trail = AuditTrail()
...
anchor = trail.head                      # to durable, external storage

# later, on the exported chain
restored = AuditTrail.from_records(records)
restored.verify(expected_head=anchor)    # False if anything was removed
```

Added in 2.10.0, alongside a correction: `verify()` previously documented
itself as detecting "no edit, deletion, or reorder", which overstated what a
chain can prove about its own tail.

### Signing the trail

A hash chain proves the records agree with each other, not who wrote them.
Rewrite a record and recompute every hash after it, and a bare `verify()`
passes again. Give the trail a signer and every record's hash is signed with
Ed25519, and the record carries the `key_id` of the key that signed it:

```python
from locus.control import AuditTrail, Ed25519Signer, verify_jsonl

trail = AuditTrail(signer=Ed25519Signer.from_pem(private_pem, key_id="audit-2026-09"))
...
anchor = trail.head                      # to durable, external storage
exported = trail.export_jsonl()

# An auditor with the export and the public key, and no Locus runtime state:
verify_jsonl(exported, keys={"audit-2026-09": public_pem}, expected_head=anchor)
```

With `keys`, an unsigned record, an unknown key id, or a bad signature fails
verification, so a chain rebuilt around an edit fails unless it was signed with
a key the verifier trusts. `trail.verify(keys=...)` runs the same check in
process.

From a terminal, `locus audit verify` runs it over the exported file and exits
0 when the trail verifies, 1 when it does not:

```bash
locus audit verify trail.jsonl --key audit-2026-09=public.pem --head "$ANCHOR"
locus audit verify trail.jsonl --key audit-2026-09=public.pem --format json
```

`--format json` prints the result as data on stdout for both outcomes, for a
job that archives verification results rather than scraping the prose:
`{"file": "trail.jsonl", "verified": true, "records": 12, "keys": 1,
"head_checked": false}`.

Signing does not catch truncation. Records dropped off the end take their
signatures with them, and what is left is still validly signed, so keep passing
an externally held `expected_head`.

To rotate keys, call `trail.use_signer(new_signer)`. Records already written
keep the key they were signed with, so give the verifier both public keys.
Signing needs the `cryptography` package (`pip install "locus-sdk[audit]"`);
an unsigned trail needs nothing and exports exactly as before.

### Keeping the trail out of the agent's reach

`OracleAuditTrail` writes each record as a row of an Oracle AI Database
immutable table — or a blockchain table, whose row hashes the database
re-verifies on `verify()` — and `anchor_trail()` keeps the chain head in an
OCI Object Storage or S3-compatible bucket with a retention rule. Both are
described on the [observability page](../concepts/observability.md#where-the-audit-trail-lives).

```python
from locus.control import OracleAuditTrail, S3AnchorStore, anchor_trail, verify_anchor

trail = OracleAuditTrail(dsn="audit_low", user="AUDIT", password=pw, table_kind="blockchain")
...
store = S3AnchorStore("audit-anchors", endpoint_url="https://objectstorage.example")
anchor_trail(trail, store, require_retention_rule=True)
verify_anchor(trail, store).ok
```

::: locus.control.audit.AuditTrail
::: locus.control.audit.AuditRecord
::: locus.control.audit.AuditSigner
::: locus.control.audit.Ed25519Signer
::: locus.control.audit.verify_jsonl
::: locus.control.governed.AuditHook
::: locus.control.audit_oracle.OracleAuditTrail
::: locus.control.audit_anchor.anchor_trail
::: locus.control.audit_anchor.verify_anchor
::: locus.control.audit_anchor.anchor_periodically
::: locus.control.audit_anchor.latest_anchor
::: locus.control.audit_anchor.Anchor
::: locus.control.audit_anchor.AnchorReport
::: locus.control.audit_anchor.AnchorStore
::: locus.control.audit_anchor.OCIBucketAnchorStore
::: locus.control.audit_anchor.S3AnchorStore

## Governed agents

An `Agent` pre-wired with grounding, guardrails, and an audit trail.

::: locus.control.governed.governed_agent
::: locus.control.governed.GovernedAgent
::: locus.control.governed.GovernanceProfile

## Verification

Evidence quality and adversarial refutation, feeding the
`require_verification_score` and `min_severity` rules on a policy.

::: locus.control.verification.verify
::: locus.control.verification.VerificationResult
::: locus.control.findings.Evidence
::: locus.control.taxonomy.Severity
