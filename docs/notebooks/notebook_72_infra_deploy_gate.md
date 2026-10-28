# A deploy gate that ships to staging on its own and waits for a human in production

An ops agent watches CI. When a build goes green it proposes the next move:
roll the new image onto staging, or promote / roll back production. Letting it
act freely is how a 3 a.m. agent takes the checkout API down. Blocking everything
for a human is how you stop shipping. The middle path is a gate: cheap,
reversible staging changes go through automatically; any change to the
production environment stops and waits for a named human.

The decision of *whether the rollout runs* does not live in the agent's prompt.
It lives in ``admit``, a gate that runs before ``kubectl`` touches the cluster.
The agent can be confused, jailbroken, or wrong about the blast radius; the
rollout still only happens if the ``ControlPolicy`` admits it.

{{ locus_diagram gate-infra-deploy }}

The rule lives in one ``ControlPolicy``. ``require_human_for={"production"}``
means any action carrying the ``production`` label always stops for a person —
no matter how small the blast radius looks. Staging is left to clear on blast
radius alone, capped by ``max_blast_radius``. ``require_verification_score`` is
set to ``0.0`` because a deploy is an ops action, not a security finding: there
is no verdict to weigh, so the gate reasons purely over environment and scope.

``admit()`` is the single enforcement point. It asks the policy, records the
decision to the ``AuditTrail`` whether or not it allows, and only then awaits
``perform``. There is no path to the side effect that skips the log — the held
production rollback lands on the trail next to the staging deploy that actually
ran, and ``trail.verify()`` confirms the SHA-256 chain is intact.

``perform`` here is a local stub: it appends to an in-memory list instead of
calling ``kubectl``, so the script runs offline with no cluster, no credentials,
no network. Swap that one function for a real ``kubectl rollout`` call and the
gate is unchanged.

Run it (fully offline — no model and no credentials):

    python examples/notebook_72_infra_deploy_gate.py

## Output

Running it offline — no model and no credentials — prints a staging deploy allowed and a production rollback held:

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

Audit trail
------------------------------------------------------------
  action-admission: deploy checkout-api -> allow
  action-admission: rollback checkout-api -> require_human
  chain intact: True

Side effects applied: ['deploy checkout-api -> checkout-api:1.8.2 (staging)']

OK — staging shipped, production held, both on the trail.
```
<!-- notebook-output:end -->

## Source

````python
--8<-- "examples/notebook_72_infra_deploy_gate.py"
````
