---
title: Guarantees and boundaries
description: What the SDK's policy gate, grounding, idempotency, and audit trail enforce; what developers configure; and what remains model-dependent.
---

# Why the Oracle Generative AI – Multi-Agent Locus SDK

The SDK is an agentic harness: one `Agent` class, eight
multi-agent shapes, and the runtime around the model (tools, memory, RAG,
observability, guardrails), built first-class for OCI Generative AI and
Oracle Database 26ai and open to every other provider. The [home page](index.md)
covers that ground. This page is
about the part that is easiest to over-claim: what the control layer
guarantees, what you configure, and what stays model-dependent.

Its useful promise is narrow and testable: **a callable routed through
`admit()` is invoked only after the supplied policy allows its declared
`Action`.** The runtime can instead hold the call for a person or deny it, and
it can record that decision.

This is stronger than a prompt instruction, but it is not a claim that every
possible action is automatically safe. Coverage depends on how an application
classifies actions and whether all side-effecting paths use the gate.

## Control at three moments

| Moment | SDK surface | What it does |
|---|---|---|
| Decide what happens next | [Agent loop](concepts/agent-loop.md) | Runs bounded reasoning, tool calls, observations, and termination |
| Assess what may be asserted | [GSAR](concepts/gsar.md) | Scores a judge-produced, typed evidence partition and supports proceed, regenerate, replan, or abstain behavior |
| Decide whether an action executes | [Admission](concepts/control-layer.md) | Evaluates an `Action` against `ControlPolicy` before calling the side effect |

These controls compose, but none silently enables the others. Grounding and
admission are configuration choices, and a high-stakes deployment should make
those choices visible in code and tests.

## The admission guarantee

```python
try:
    result = await admit(
        action,
        perform,
        policy=policy,
        trail=trail,
    )
except AdmissionError as exc:
    handle_refusal(exc.decision)
```

Within that call:

- `perform` runs only for an allowed decision;
- a hold or denial does not invoke `perform`;
- the decision is appended when an `AuditTrail` is supplied.

The guarantee does not extend to another reference that calls the underlying
function directly. `gate_tool()` makes the gated wrapper the tool exposed to
an agent, but application code and review must still prevent bypass paths.

## Runtime, configuration, and judgment

| Capability | Runtime enforces automatically | Developer configures | Remains judgment-dependent |
|---|---|---|---|
| Policy gate | Deterministic evaluation and pre-execution allow/hold/deny | Action labels, rules, gate coverage, approval handling | Whether the classification captures the actual consequence |
| Human approval | Approval state and one-use consumption in the configured store | Durable storage, authenticated approver, resume workflow | Whether the human has enough context to decide |
| GSAR grounding | Score and threshold logic over a typed partition | Evidence corpus, judge, thresholds, recovery/fail behavior | The judge's claim and evidence assessment |
| Typed findings | Carries an explicit score, references, and provenance fields | Criteria for emitting or accepting the finding | Whether source evidence is complete and correctly interpreted |
| Idempotent tools | Reuses a cached result for an identical call in the documented scope | Tool annotation, checkpoints, downstream operation key | Semantic equivalence of differently encoded calls |
| Audit trail | Hash-chains appended records and detects later edits to that chain | Durable export, access controls, external anchoring | Whether all relevant events were sent to the trail |

## Compared with prompt rules and filters

| | Prompt instruction | Input/output filter | The SDK's admission gate |
|---|---|---|---|
| Evaluated where | Model context | Around model text | Around a specific callable |
| Can block the side effect itself | No | Only if application wiring makes it authoritative | Yes, for calls routed through `admit()` |
| Depends on model compliance | Yes | Often, when classification is model-based | Policy evaluation does not; action classification still can |
| Human hold | Application-specific | Framework-specific | Built-in stores and resume flow |
| Decision record | Usually logs | Usually logs | Optional hash-chained `AuditTrail` |

Guardrails remain useful for prompt injection, content policy, and data-loss
prevention. Admission solves a different problem: making an application-level
decision before a named side effect runs.

## Grounding is an assessment, not truth

GSAR makes the evidence assessment inspectable: claims are assigned to
grounded, ungrounded, contradicted, or complementary partitions, then scored.
Threshold logic is deterministic once that partition exists. Producing the
partition is normally an LLM judgment and can be wrong. Evaluate the judge on
your domain, calibrate thresholds, and choose whether low scores cause a
rewrite, replan, abstention, or hard failure.

A typed finding therefore means “this object carries an explicit evidence
assessment,” not “this claim cannot be wrong.”

## Idempotency is scoped

`@tool(idempotent=True)` deduplicates matching calls within the documented
agent/checkpoint scope. It does not by itself close the crash window where an
external service succeeds and the local result has not yet been durably
recorded. For payments, deploys, and messages, also send a stable idempotency
key to the downstream API.

## When the SDK is a good fit

The SDK earns its control layer when an agent can move money, change
infrastructure, alter access, contact a third party, or create another durable
effect. A read-only summarizer may need grounding and evaluation without an
admission gate.

Start with the [first agent](how-to/quickstart.md), then run the
[first controlled action](how-to/first-controlled-action.md). Before
production, read [policy authoring](concepts/policy-authoring.md),
[idempotency](concepts/idempotency.md), and the
[deployment guide](how-to/deploy.md).
