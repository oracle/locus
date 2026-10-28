# SecurityContext

**In plain terms:** an incident responder doesn't think *"I'll query Splunk, then
CrowdStrike, then Okta."* They think *"I'm investigating an incident"* — pull the
logs, check the user, enrich the indicator, look at the host. `SecurityContext` is
that mental model in code: **one handle over the security domains**. Your agent
reasons in *domains* — logs, identity, endpoint, threat-intel, cloud, actions —
and never hard-codes a vendor SDK. Point it at your real stack by injecting a
provider; the investigation code doesn't change. The admission-control
machinery on this page — `admit()`, the policy, the labels — is the product's
general control layer, the same one every domain uses; security is the worked
example.

{{ locus_diagram security-context }}

```python
import asyncio
from locus_security import SecurityContext


async def main():
    ctx = SecurityContext()                              # zero config — runs offline
    await ctx.logs.search("failed login spike", window="6h")
    await ctx.identity.risk("mallory@example.com")
    await ctx.threat_intel.enrich("198.51.100.23")
    await ctx.endpoint.get_host("WS-0142")


asyncio.run(main())
```

## The domains

Each domain is a small `Protocol` — a *port* — that a provider implements. The
handle is the same whether the provider is the offline reference or a live vendor.

| Handle | Domain | Methods |
|---|---|---|
| `ctx.logs` | SIEM / log search | `search(query, window="24h")` |
| `ctx.endpoint` | EDR host forensics + containment | `get_host`, `detections`, `isolate` |
| `ctx.identity` | identity provider (the surface most attacks touch) | `get_user`, `risk`, `signins`, `disable` |
| `ctx.cloud` | cloud control-plane evidence (read-only) | `describe`, `events` |
| `ctx.threat_intel` | IOC reputation / enrichment | `enrich` |
| `ctx.actions` | decide on **and run** a response action | `request_approval(action, finding=…, verdict=…)` · `execute(action, perform, finding=…, verdict=…)` |

Reads are plain domain calls. **Writes** — `endpoint.isolate`, `identity.disable` —
model real-world actions: run them through the [admission gate](#admission-control-the-enforcement-point) so they fire only after the chain clears.

!!! note "Writes are simulated today"
    In the current build the bundled reference adapters **and** the vendor
    templates (Entra/Okta/Auth0 `disable`, CrowdStrike `isolate`) return a
    **simulated offline-sample receipt** — they record the decision but do
    **not** lock an account out or quarantine a host — that includes the
    `disable` in step 4 below. Wire and verify the live vendor write path
    before treating these as enforcement.

## Zero-config by default

Every domain defaults to a bundled **offline reference adapter**, so
`SecurityContext()` works with no credentials, no network, and no vendor account.
The reference identity provider, for instance, returns two deterministic users —
a low-risk `jsmith@example.com` and a high-risk `mallory@example.com` with
impossible-travel sign-ins — so examples, tests, and CI all run the same code path
they would in production, just against benign data.

This matters for adoption: you can write and unit-test an entire investigation
offline, then flip individual domains to live vendors one at a time.

## Going live — inject a provider

Core never imports a vendor. A live provider is a class you write that satisfies
the domain port, and you inject it explicitly:

```python
from locus_security import SecurityContext


class MyIdentity:  # satisfies the identity port
    ...


# Identity goes live; logs, endpoint, threat-intel and cloud stay on the offline reference.
ctx = SecurityContext(identity=MyIdentity())
```

Keep an offline path: resolve credentials from the environment and fall back to a
deterministic sample when none are present, so the same code runs in CI. Cloud
posture is the one live provider in core: `locus_security.aws`
(`pip install locus-sdk[aws]`). [Build an integration](../how-to/build-an-integration.md)
walks through writing one.

## A full investigation — grounded, verified, gated

The facade earns its place in the *whole loop*, not in any one call. An AI agent
that merely reads telemetry is a chatbot; the moment it can **act**, three things
have to be true — the data is real, the claim is verified, and the action is
gated. `SecurityContext` puts the trust spine right in the path:

```python
import asyncio
from locus.control import Action, verify
from locus_security import Evidence, SecurityContext, Severity


async def main():
    ctx = SecurityContext()   # offline reference providers; ControlPolicy() holds production disables for a human

    # 1. Investigate by domain.
    risk = await ctx.identity.risk("mallory@example.com")   # -> {'risk': 'high', 'impossible_travel': True, ...}

    # 2. Form a finding and verify it; a thin claim is refuted, not acted on.
    finding = Evidence(
        title="Account compromise: impossible-travel sign-ins",
        description="High-risk user with impossible travel between two sign-ins.",
        severity=Severity.HIGH,
        asset="mallory@example.com",
        remediation="Disable the account and force a credential reset.",
        evidence_refs=["identity:logs:mallory@example.com"],
        gsar_score=0.86,
    )
    verdict = await verify(finding)
    if not verdict.survives:
        return  # abstain

    # 3. Propose containment through the policy gate.
    decision = ctx.actions.request_approval(
        Action(name="disable_user", asset="mallory@example.com", environment="production"),
        finding=finding,
        verdict=verdict,
    )   # decision.outcome -> "require_human"

    # 4. Act only on approval.
    if decision.allowed:
        await ctx.identity.disable("mallory@example.com")


asyncio.run(main())
```

`"require_human"` is the **held** outcome — every decision comes back allowed,
held, or denied, and a held action waits for a person before anything fires.

> **One investigation, six domains, zero vendor names** in the logic. Swap
> the reference identity provider for your own, or the threat-intel feed for
> another, and steps 1–4 are untouched. That is the platform bet: program against domains —
> security is one of them — and
> the trust spine — [grounding](gsar.md), verification, policy, and a hash-chained
> audit trail — applies no matter whose API is behind the port.

The gate in step 3 is a `ControlPolicy`: `require_human_for={"production"}` by
default, alongside `require_verification_score`, `max_blast_radius`, `deny_for`,
`min_severity`, `require_sandbox_for`, and the spend controls
`require_human_over_usd` / `spend_limit_usd` — the [API reference](../api/control.md)
lists every field. To enforce a custom policy, call `approve(action, policy=…,
finding=…, verdict=…)` directly — `ctx.actions.request_approval` is the
convenience wrapper around it.

`verify()` does not need a finding built by Locus: it accepts a Locus `Evidence`
**or any finding-shaped dict** (it reads `title`, `severity`, `gsar_score`,
`evidence_refs` and `confidence`), so you can verify results that another part of
your pipeline produced.

## Admission control — the enforcement point {#admission-control-the-enforcement-point}

`request_approval()` returns a *decision*; it doesn't run anything. To make the
decision binding, run the action through the **admission gate** — `admit()`, or
`ctx.actions.execute()` on the facade. The side effect fires **only if** the
chain clears (`approve()` → ALLOW); otherwise it raises `AdmissionError`.
`ctx.actions.execute()` takes no trail and records nothing. When you need the
attempt recorded, call `admit(..., trail=...)` directly, which appends the
decision either way, admitted or refused. On the facade:

```python
from locus.control import Action, AdmissionError, verify

verdict = await verify(finding)
try:
    await ctx.actions.execute(
        Action(name="disable_user", asset="mallory@example.com", environment="production"),
        lambda: ctx.identity.disable("mallory@example.com"),   # the side effect
        finding=finding,
        verdict=verdict,
    )
except AdmissionError as exc:
    route_to_human(exc.decision)   # production → require_human, so the disable never fired
```

This is what turns this call path from *advisory* into *enforced*: `execute()`
does not invoke its supplied side effect until the configured checks clear.
Other direct call paths remain the application's responsibility. It's the
admission-controller pattern (think Kubernetes admission webhooks) applied to
agent actions, and it's what makes Locus a *runtime* rather than a library of
trust functions.

### A policy matches labels, not names

Notice what the `Action` above carries: `environment="production"`. That is what
`require_human_for={"production"}` matches. A policy never gates on the *name*
of the action or the tool performing it — it reasons about what the action **is**:
its `environment`, its `kind`, its `blast_radius`, and any `tags`.

This has a consequence worth stating plainly: **an action that declares nothing
cannot be gated by label.** Rather than guess a plausible-looking default, Locus
labels it `environment: "unknown"` — honest, but it matches no rule written for
`production`. Writing `require_human_for={"production"}` and expecting it to
catch an undeclared action is the one mistake this design invites.

So declaring labels is not documentation. It is the input the policy reasons
over, and the thing that lets a tool earn an unattended path:

```python
Action(
    name="refund_customer",
    environment="production",      # which rule set applies
    blast_radius=1,                # how many subjects one call touches
    tags=frozenset({"payment", "irreversible"}),
)
```

A tool that declares no `environment` gets `unknown`, which fails safe rather
than open.

## From facade to agent — `ctx.toolset()`

The domain handles are the *programmatic* facade — what your own code calls. When
you want an autonomous `locus.Agent` to drive the investigation, hand it the
*agent* facade instead:

```python
from locus import Agent

agent = Agent(model="{{ locus_example_model }}", tools=ctx.toolset())
```

`toolset()` returns the agent-ready security tool bundle built from the **bundled
reference adapters** — it does not read the providers you injected into `ctx`, so a
live `identity` or `logs` provider is not reachable from these tools. It is
read-only by default and covers threat intel, SIEM, EDR, scanning and
fingerprinting; pass `ctx.toolset(allow_containment=True)` for `isolate_host`,
`aws=True` for the cloud-posture tools, and `extra=[...]` to merge tools you wrote
against your own live providers. There is no identity tool in the bundle, and
admission (`ctx.actions`) is not a tool — to gate a side-effecting tool you hand
the agent, wrap it with `gate_tool` or call `admit()` in your own code.

## The one-way dependency

Core (`locus-sdk`) defines the ports and ships the offline reference providers;
it **never imports a vendor**. A vendor provider lives in your own package, which
depends on core. You inject providers explicitly, so there is no hidden vendor coupling and the offline path is
always intact.

See also: [Agentic AI-security](agentic-ai-security.md) ·
[Grounded findings](security.md) · [GSAR typed grounding](gsar.md) ·
[Threat scenarios](threat-scenarios.md) · [Cloud-posture agent](cloud-posture.md).
