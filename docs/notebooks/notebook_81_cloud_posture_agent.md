# Grounded AWS cloud-posture agent

A SOC-analyst-shaped agent that audits an AWS account read-only, then
*grounds* every finding it proposes against the API facts it actually
observed. ``create_soc_analyst`` composes two spec-driven, read-only tools —
``describe_aws`` (discover the shape of AWS from botocore's service models)
and ``use_aws`` (run one read-only operation, return the raw response as
evidence) — behind a ``create_deepagent`` core. The agent proposes findings;
``ground_report`` decides which survive: a proposed finding becomes a typed
``Evidence`` only if its cited evidence clears the GSAR threshold, otherwise it
abstains. The model gathers and proposes; Python decides what ships.

This is the differentiator. A commodity "AWS agent" will confidently narrate
misconfigurations it never actually observed. Here, a finding survives only if
the analyst cited direct API observations for it; inferences alone abstain
(``ground_report``). The control is on the *shape* of the evidence, not its
truth — the agent marks each cited fact as observation or inference (the
``grounded`` flag, which defaults to observation), and the ``ref`` string is
carried for audit, not parsed — so pair this with the raw ``use_aws`` responses
when you need the citation itself checked.

Maps to OWASP ASI: Identity & Privilege Abuse (the root-access-key class of
finding); the read-only-by-construction tooling is the control that keeps the
auditor itself from becoming a liability.

Run it:

    python packages/locus-sdk-security/examples/notebook_81_cloud_posture_agent.py

Part 1 (the grounding decision) runs fully offline and deterministically — no
model, no cloud account. Part 2 builds the live agent; it runs against a real
account only when BOTH a real model provider (``LOCUS_MODEL_PROVIDER=openai`` /
``anthropic``) and AWS credentials (the read-only ``locus-security-audit``
profile, or ``LOCUS_AWS_PROFILE``) are present. With neither, it prints the
bring-your-own-credentials note and exits cleanly.

Prerequisites:

- The DeepAgent notebook — the core this factory wraps.
- For the live Part 2 only: a structured-output-capable provider + an AWS
  identity. The agent is strictly read-only; ``use_aws`` refuses writes.

## Source

````python
--8<-- "examples/notebook_81_cloud_posture_agent.py"
````
