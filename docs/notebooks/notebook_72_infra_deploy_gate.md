# Infra deploy gate

An operations agent watches continuous integration (CI). When a build goes
green it proposes the next move: roll the new image onto staging, promote it to
production, or roll production back. Left to act on its own, the agent can take
the checkout API down overnight with nobody watching; requiring a human for
every change stops releases altogether. The middle path is a gate: cheap,
reversible staging changes go through automatically, and any change to the
production environment stops and waits for a human.

The decision of *whether the rollout runs* does not live in the agent's prompt.
It lives in `admit()`, a gate that runs before `kubectl` (the Kubernetes
command-line tool) touches the cluster, and the rule it enforces is written
once, in a `ControlPolicy`: any action carrying the `production` label waits
for a person, however small its blast radius; staging clears on blast radius
alone; and the verification bar is switched off, because a deploy is an
operations action with no security verdict to weigh. The agent can be confused,
manipulated by a prompt injection, or wrong about the blast radius it declares;
a production change still waits for a human, and nothing runs that `admit()`
has not cleared.

`admit()` is the single enforcement point. It asks the policy through
`approve()`, records the decision on the `AuditTrail` whether or not it allows
the action, and only then awaits `perform`; a hold surfaces as an
`AdmissionError` that carries the decision, and a later `admit()` call with
`approved_by=` runs the held action and records the approver on the trail. The
rollout itself is a local stub that appends to an in-memory list instead of
calling `kubectl`, so the script runs offline with no cluster, no credentials
and no network; swap that one function for a real `kubectl rollout` call and
the gate is unchanged.

An optional third part adds a second check that reads the call. The policy
weighs labels, and labels are all it can see: a staging rollout pinned to
`checkout-api:1.8.2` and one to the floating `checkout-api:latest` tag carry
the same labels, so the gate allows both. A `ToolCallJudge` is handed to
`gate_tool(..., judge=judge)` and shown the proposed call, the tool's name and
its arguments, together with the intent written in words and a line of
guidance for this tool; it returns `allow`, `hold` or `deny` with a reason. The
verdict reaches `admit()` as evidence and can only tighten the decision: the
pinned rollout clears, the `latest` rollout the policy had allowed is held, and
a policy hold or deny would stand whatever the judge said. The judge model
follows the series convention in `examples/config.py`: with no provider
configured it is the SDK's scripted test double, so this part runs offline
too; with `LOCUS_MODEL_PROVIDER` set it is the configured model, and its id is
on the audit entries. On an agent it borrows the agent's model unless given
its own.

What this notebook covers:

- Describing each proposal as an `Action` with an `environment`, a `kind`, tags
  and a declared `blast_radius`.
- Writing the deploy rule once, in a `ControlPolicy` with `max_blast_radius=4`,
  `require_human_for={"production"}` and `require_verification_score=0.0`.
- Passing the side effect to `admit()` as a zero-argument async callable that
  runs only on an allow.
- Handling a hold as an `AdmissionError` whose `decision` carries the outcome
  and the reason.
- Reading both decisions back from the hash-chained `AuditTrail` and checking
  that the chain is internally consistent with `trail.verify()`.
- Optionally, gating a `rollout` tool with a `ToolCallJudge` that reads the
  image tag the policy cannot see, and reading its verdict off the same audit
  entry as the admission decision (`judge.model`, `judge.verdict`,
  `judge.cache_hit`).

The drawing below follows both proposals through the gate: `admit()` hands each
`Action` to `approve()` with the policy, the staging deploy clears on blast
radius alone and `perform()` runs the rollout, the production rollback is held
under `require_human_for` even though its blast radius is smaller, and both
decisions are recorded on the hash-chained `AuditTrail` before anything runs.

{{ locus_diagram gate-infra-deploy }}

## Key lines

The side effect is built as a zero-argument async callable that `admit()` runs
only on an allow; here it appends to an in-memory list where the real version
would call `kubectl`:

{{ locus_source examples/notebook_72_infra_deploy_gate.py::make_rollout }}

Each proposed action goes through `admit()` with the module-level `POLICY` and
the trail: an allowed action returns what `perform` produced, a held one raises
`AdmissionError` carrying the decision and its reason, and nothing is applied:

{{ locus_source examples/notebook_72_infra_deploy_gate.py::propose }}

The optional step gates a `rollout` tool with the judge and raises on a
refusal, so a hold reads the same way as in the first two parts; the decision
carries both the policy's own outcome and the judge's escalation:

{{ locus_source examples/notebook_72_infra_deploy_gate.py::propose_judged }}

## Run it

```bash
python examples/notebook_72_infra_deploy_gate.py
```

The gate needs no model: the agent's two proposals are `Action` values built
in `main()` and the rollout is an in-memory stub. With no provider configured
the judge in the optional step runs on `locus.testing.ScriptedModel`, the
SDK's test double, with its two verdicts written down, so the whole example
runs offline with no credentials. Set `LOCUS_MODEL_PROVIDER` (and
`LOCUS_MODEL_ID`) as for any other example and the judge is the configured
model reading the two calls itself:

```bash
LOCUS_MODEL_PROVIDER=oci LOCUS_MODEL_ID=openai.gpt-4.1 python examples/notebook_72_infra_deploy_gate.py
```

The verdict on the unpinned image is then the model's own; the script accepts
a hold or a deny for it, either being a tightening the policy did not make.

## Output

Running it offline — no credentials, bundled mock model — prints a staging deploy allowed and a production rollback held, then the optional judge letting a pinned staging image through and holding an unpinned one the policy had allowed, with both verdicts on the audit entries:

```text
Notebook 72: An ops agent that ships to staging on its own and waits for a human in production
============================================================

--- deploy checkout-api in staging (blast radius 3) ---
  ALLOWED — the agent acted on its own authority.
  Applied: deploy checkout-api -> checkout-api:1.8.2 (staging)

--- rollback checkout-api in production (blast radius 2) ---
  HELD (require_human) — nothing was applied.
  Reason: labels ['production'] require human approval
  Next: route to the on-call human for sign-off.

--- judged: deploy checkout-api -> checkout-api:1.8.2 (staging) ---
  ALLOWED — policy allow, judge allow.
  Applied: deploy checkout-api -> checkout-api:1.8.2 (staging)

--- judged: deploy checkout-api -> checkout-api:latest (staging) ---
  HELD (require_human) — the policy said allow; the judge tightened it. Nothing was applied.
  Reason: tool-call judge escalated allow -> require_human: image tag 'latest' is not pinned; a person should confirm

Audit trail
------------------------------------------------------------
  action-admission: deploy checkout-api -> allow
  action-admission: rollback checkout-api -> require_human
  action-admission: deploy checkout-api -> allow  [judge ScriptedModel: allow, cache_hit=False]
  action-admission: deploy checkout-api -> require_human  [judge ScriptedModel: hold, cache_hit=False]
  chain intact: True

Side effects applied: ['deploy checkout-api -> checkout-api:1.8.2 (staging)', 'deploy checkout-api -> checkout-api:1.8.2 (staging)']

OK — staging shipped, production held, the unpinned image held by the judge; all on the trail.
```
<!-- notebook-output:end -->

## See also

- [Control-layer architecture](../concepts/control-layer.md) — where `admit()`
  sits among the agent loop, approvals, persistence and audit.
- [Admitting an action](../api/control.md#admitting-an-action) — the `admit()`
  and `AdmissionError` reference, including `approved_by`.
- [A second check that reads the call](../concepts/control-layer.md#a-second-check-that-reads-the-call)
  — what the judge adds to the gate, and where it stops.
- [What `verify()` catches](../api/control.md#what-verify-catches-and-the-one-thing-it-cannot)
  — a bare `trail.verify()` catches edits, reorders and deletions from the
  middle of the chain, not a truncation.

## Source

??? note "Full source — `examples/notebook_72_infra_deploy_gate.py`"

    ````python
    --8<-- "examples/notebook_72_infra_deploy_gate.py"
    ````
