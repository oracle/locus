# Agentic AI-security

This page covers the separate `locus-sdk-security` package (`locus_security`), the SDK's security domain — one of
several action domains (refunds, deployments, records) built on the same
harness. It lets you build agents that test and assess other AI systems.

**In plain terms:** you've deployed an AI — a chatbot, an agent with tools, or a
model behind an API. Can someone jailbreak it, make it leak data, or trick it
into doing something it shouldn't? The module builds an agent that checks,
automatically and continuously, and reports only the flaws it can actually
**prove** — a penetration tester for AI that refuses to cry wolf.

**How it works, in three steps:**

1. **Point** the SDK at a target AI — `Target.endpoint("https://bot.example/chat")`.
2. **Run** a job — `red_team(target)` attacks it; `assure(target)` scores its defenses.
3. **Get evidence** — each result is an `Evidence` finding (the attack worked, here's the proof)
   or an `Abstention` (no proof, so no claim). Never a guess.

More precisely: `locus_security` builds agents whose **subject is another AI
system** —
agents that **red-team** and **assess** other AI (continuous **monitoring** is
on the roadmap), and produce a grounded `Evidence` record or an explicit
`Abstention`, never a hallucinated verdict.

> **A cybersecurity agent** = a *target* (an AI system) × a *job*
> (red-team · assure; monitor is roadmap) × an *output* (grounded `Evidence` |
> `Abstention`) × being *itself trustworthy by construction*.

{{ locus_diagram trust-chain }}

## Why this, and why now

Companies ship LLM agents — with tools, memory, autonomy, and access to data
and actions — into production. That is a new, largely unguarded attack
surface: prompt injection (direct and indirect), tool abuse / excessive
agency, data exfiltration, memory and RAG poisoning, A2A identity confusion,
unknown model and hardware provenance.

Existing tooling either filters I/O without testing it, or tests it and
**hallucinates findings** — the literature is explicit that AI scorers lack
grounding. This module's answer is **abstain-by-construction grounding**
(GSAR): no evidence, no claim.

## The `Target` — what you point at

A `Target` is a uniform handle to the AI system under assessment. One
`.send()` contract over four constructors:

```python
from locus_security import Target

Target.endpoint("https://bot.example/chat")   # a remote LLM / agent endpoint
Target.agent(my_locus_agent)                   # an in-process locus.Agent
Target.a2a(peer_send)                          # an A2A peer's send coroutine
Target.from_callable(fn)                       # any sync/async str -> str
```

Everything below operates on a `Target`, so the same probe runs against a
black-box endpoint, an agent you built, or an offline stub in CI.

## `red_team` — attack, grounded

`red_team` runs the OWASP-ASI probe suite (or the smaller `owasp-llm` one)
against a target and grounds each outcome; MITRE ATLAS appears as tags on the
findings (`Evidence.taxonomy`), not as a suite. A probe whose attack *landed*
yields an `Evidence` finding; an inconclusive one yields an `Abstention`.

```python
import asyncio
from locus_security import Target, red_team, is_finding


async def main():
    report = await red_team(Target.endpoint("https://bot.example/chat"), suite="owasp-asi")
    for r in report:
        if is_finding(r):
            print(r.severity.value, r.title, r.taxonomy)   # e.g. LLM01, ASI02, AML.T0054
        else:
            print("abstained:", r.reason)


asyncio.run(main())
```

The bundled catalogue covers direct and indirect prompt injection, jailbreak,
excessive agency / tool misuse, and sensitive-information disclosure — each
tagged with the taxonomy a SOC (security operations center) and an
AI-assurance reviewer expect.

## `assure` — assess, grounded

Where `red_team` asks *can I break it?*, `assure` asks *how well does it hold
up?* `guardrail_coverage` runs the suite and grounds a posture finding in the
direct observations — INFO when fully hardened, escalating to CRITICAL at 0%
coverage, with the taxonomy listing exactly the gaps.

```python
from locus_security import assure
posture = await assure(target)          # one grounded posture finding per assessment
coverage = posture[0]                   # today assure runs a single assessment: guardrail coverage
# or call guardrail_coverage(target) directly for just that one finding
```

Each item in that list is a *grounded finding*, not a compliance attestation.
`assure` never asserts posture it cannot evidence — it abstains. The SDK deliberately does not
do compliance attestation; its governance acts on the action itself: the
[admission gate](#enforce-it-before-it-acts) — enforced policy, a human on
the actions that warrant one, and a tamper-evident record.

## Secure by default — the floor

An agent that red-teams or assures other AI must itself be trustworthy.
`governed_agent` builds a `locus.Agent` with the security spine on by default —
GSAR grounding, guardrails (PII / injection / a dangerous-tool denylist;
allowlist is opt-in), and a tamper-evident audit trail — and returns it
alongside that trail.

{{ locus_model_tabs }}

```python
from locus.control import governed_agent

secured = governed_agent(model="{{ locus_model }}", tools=[...])
result = secured.run_sync("...")
assert secured.audit_trail.verify()   # the chain is intact (tamper-evident)
```

The `AuditTrail` is an in-memory SHA-256 hash chain: every action commits to
the hash before it, so an edit, deletion, or reorder in the middle breaks
`verify()`. Records cut off the end still leave a valid chain; catching that
needs `verify(expected_head=...)` against a head hash pinned outside the trail.
That makes it tamper-*evident* (it **detects** edits) rather than tamper-proof.
It is unsigned by default, so anyone who can write the log could rebuild the
whole chain around an edit and `verify()` would still pass; only
`verify(expected_head=...)` against a head hash anchored somewhere out of their
reach would catch it. Give it an
`Ed25519Signer` (`AuditTrail(signer=...)`, which needs
`pip install "locus-sdk[audit]"`) and `verify(keys=...)` then fails if any
record is not signed with a key you trust. The signature covers each record's
hash when it is written, so it cannot catch a payload that was wrong before it
was recorded. It exports as JSONL for shipping to your audit store or a SIEM
(a security team's log platform); anyone holding the export can check it with
`verify_jsonl()` and the public keys alone, and write-once retention can
harden it further.

## Enforce it before it acts

Grounding, verification, and the audit trail make an agent *trustworthy*; the
**admission gate** makes that trust *binding*. Run a side-effecting action
through `admit()` (or `ctx.actions.execute()`). The action fires at once
when policy allows it (`approve()` → ALLOW). When policy holds it
(`require_human`), `admit()` raises `AdmissionError` instead of running it; it
runs only if you call `admit()` again naming the person who approved it
(`admit(approved_by=...)`). `ctx.actions.execute()` takes no `approved_by`, so
a hold there is simply refused. Everything else is denied, and no approval
overrides a denial.
Every decision — allowed, held, or denied — is recorded when you pass an
`AuditTrail` to `admit(trail=...)`, and `AdmissionError` is raised whenever the
action does not run. That's the line between an agent that *could* be safe and
a runtime that *enforces* its policy — an action routed through the gate runs
only when policy allows it or a named person approves the hold. See
[SecurityContext](security-context.md#admission-control-the-enforcement-point).

## Regular cyber — classic security operations

This page covers securing *AI*. The same engine works on classic security
operations too — the bundled SOC-analyst factory, the incident-response
playbooks (`phishing_triage`, `ransomware_containment`, `nist_800_61_ir`), and
read-only cloud-posture auditing are a full worked domain, alongside the
general-operations examples elsewhere in the docs. Different target, different
taxonomy — same `Target` + grounded-`Evidence` contract: grounding is exactly
what makes an AI SOC agent's verdicts trustworthy, answering the
false-positive pain teams hit with AI-graded triage.

See also: [Grounded findings](security.md) · [GSAR typed grounding](gsar.md) ·
[Threat scenarios](threat-scenarios.md) · [Cloud-posture agent](cloud-posture.md).
