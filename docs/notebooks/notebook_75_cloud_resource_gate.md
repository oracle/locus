# Cloud resource gate

A cloud-ops agent is useful and dangerous in equal measure. It can free up a
forgotten development instance in seconds — and it can just as easily
terminate the database the whole product runs on, or grant an outside
principal administrative rights over your account. The model deciding *what*
to do is not the place to enforce *whether* it is allowed to. That belongs in
code that runs before the API call.

This example puts `admit()` in front of every cloud mutation. The agent proposes an `Action`; the gate weighs it against a `ControlPolicy` you wrote; the actual cloud call (here a local stub) fires only if the policy admits it.

Small, contained changes proceed on their own. Anything that touches a production resource, grants Identity and Access Management (IAM) permissions, or touches more than one resource at once is held for a human. Either way the decision lands on a hash-chained `AuditTrail`, so there is no unrecorded path to a side effect.

`admit()` is the single enforcement point. It asks the policy through `approve()`, records the decision on the `AuditTrail` whether or not it allows, and only then awaits `perform`; a hold surfaces as an `AdmissionError` that carries the decision and its reason, and the cloud call never fires.

The script runs three proposals through it:

- Resizing a staging continuous integration (CI) runner clears on blast radius 1 with no restricted label, and the in-memory inventory shows the new instance size.
- Terminating the production primary database is held by the `production` label, and the row is left exactly as it was.
- Granting a deploy service account a new IAM permission is held by the `iam` label, although it is a staging change of blast radius 1.

A model that is jailbroken, prompt-injected or simply mistaken still cannot push the held changes past the gate.

Each side effect is a local stub that edits an in-memory `INVENTORY` or `IAM` dict instead of calling a cloud provider's SDK, so the script runs offline with no cloud account, no credentials and no network; swap those functions for real API calls and the gate is unchanged.

What this notebook covers:

- Describing each proposal as an `Action` with an `environment`, a `kind`, tags
  and a declared `blast_radius`; `Action.labels()` folds the environment, the
  kind and the tags into the one label set the policy matches against.
- Writing the rule once, in a `ControlPolicy` with `max_blast_radius=1`,
  `require_human_for={"production", "iam"}` and
  `require_verification_score=0.0`: one resource at a time may proceed on its
  own, a production resource or an IAM grant always waits for a person, and
  the verification bar is switched off because a cloud mutation is an
  operations action with no security verdict to weigh. `require_human_for`
  matches any label, so a `billing` tag or a specific resource tag works the
  same way.
- Passing each side effect to `admit()` as a zero-argument async callable that
  runs only on an allow; here the callables edit the in-memory `INVENTORY` and
  `IAM` dicts where the real version would call the cloud provider's SDK.
- Handling a hold as an `AdmissionError` whose `decision` carries the outcome
  and the reason, with the resource left as it was.
- Reading all three decisions back from the hash-chained `AuditTrail` and
  checking that the chain is internally consistent with `trail.verify()`.

The drawing below follows the three proposals through the gate: `admit()`
hands each `Action` to `approve()` with the policy, the staging resize clears
and `perform()` edits the inventory, the production terminate and the IAM
grant are held under `require_human_for` and nothing runs, and every decision
is recorded on the `AuditTrail` before `perform()` is awaited or the error is
raised.

{{ locus_diagram gate-cloud-resource }}

## Key lines

Each side effect is built by a small factory that returns a zero-argument
async callable — the shape `admit` expects for `perform` — and that callable
runs only on an allow; here it edits the in-memory inventory where the real
version would call the cloud provider's SDK:

{{ locus_source examples/notebook_75_cloud_resource_gate.py::resize_instance }}

Each proposed `Action` goes through `admit()` with the policy and the trail: an
allowed action returns the receipt `perform` produced, a held one raises
`AdmissionError` carrying the decision and its reason, and the cloud call never
runs:

{{ locus_source examples/notebook_75_cloud_resource_gate.py::attempt }}

## Run it

```bash
python examples/notebook_75_cloud_resource_gate.py
```

No model is involved: the agent's three proposals are `Action` values built in
`main()` and the side effects are in-memory stubs, so the example runs offline
with no provider configuration, no credentials and no network.
`examples/config.py` and `LOCUS_MODEL_PROVIDER` play no part in this notebook.

## Output

Running it offline — no model and no credentials — prints a staging resize allowed, then a production terminate and an IAM grant held:

```text
Notebook 75: gating a cloud-ops agent's actions by blast radius and environment
============================================================

--- agent proposes: resize_instance ---
    asset='i-dev-7a3'  env='staging'  blast_radius=1
  ✅ ADMITTED — the cloud call ran: resized i-dev-7a3 -> t3.medium

--- agent proposes: terminate_instance ---
    asset='i-prod-db1'  env='production'  blast_radius=1
  ⏸  HELD FOR A HUMAN — outcome=require_human
     why: labels ['production'] require human approval
     the cloud call did NOT run; the resource is untouched.

--- agent proposes: grant_iam ---
    asset='svc-deploy'  env='staging'  blast_radius=1
  ⏸  HELD FOR A HUMAN — outcome=require_human
     why: labels ['iam'] require human approval
     the cloud call did NOT run; the resource is untouched.

Cloud account after all three attempts:
------------------------------------------------------------
  i-dev-7a3     env=staging     state=running     size=t3.medium
  i-prod-db1    env=production  state=running     size=r6i.4xlarge
  svc-deploy    permissions=['ecr:Pull', 'ecs:UpdateService']

Audit trail:
------------------------------------------------------------
  #0  resize_instance       -> allow           (i-dev-7a3)
  #1  terminate_instance    -> require_human   (i-prod-db1)
  #2  grant_iam             -> require_human   (svc-deploy)
  chain intact (tamper-evident): True

All assertions passed: low-risk action ran, production and IAM changes held, all audited.
```
<!-- notebook-output:end -->

## See also

- [Control-layer architecture](../concepts/control-layer.md) — where `admit()`
  sits among the agent loop, approvals, persistence and audit.
- [Admitting an action](../api/control.md#admitting-an-action) — the `admit()`
  and `AdmissionError` reference, including `approved_by`.
- [What `verify()` catches, and the one thing it cannot](../api/control.md#what-verify-catches-and-the-one-thing-it-cannot)
  — the limits of a bare `trail.verify()`.

## Source

??? note "Full source — `examples/notebook_75_cloud_resource_gate.py`"

    ````python
    --8<-- "examples/notebook_75_cloud_resource_gate.py"
    ````
