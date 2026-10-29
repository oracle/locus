# Security

The security domain ships as the separate `locus-sdk-security` package
(`locus_security`). It is not published to PyPI; install it from the
repository at the tag matching your `locus-sdk` version:

```bash
pip install "git+https://github.com/oracle/locus@v5.2.0#subdirectory=packages/locus-sdk-security"
```

The core `locus-sdk` package does not include it. It covers three separable
jobs:

- **Red-teaming** an agent — send adversarial probes at a target and report
  what got through.
- **Grounding a finding** — refuse to assert what the evidence does not
  support, and say so explicitly rather than guessing.
- **Building a SOC agent** — tools that talk to a SIEM, an EDR, a scanner, a
  threat-intel feed, and the OCI control plane, plus playbooks that sequence
  them.

The admission gate also lives under this package for historical reasons. Import
it from [`locus.control`](control.md) instead — that is the domain-neutral
surface and the path that will keep working.

For the concepts, start with [The control layer](../concepts/security-context.md).

## Running a job

The three entry points. Each takes a `Target` and returns a report; none of
them needs an agent instance.

::: locus_security.jobs.red_team
::: locus_security.jobs.assure
::: locus_security.jobs.monitor
::: locus_security.assess.guardrail_coverage

## Naming a target

A `Target` is what to point a job at — an HTTP endpoint, a local callable, or
an agent in this process. The same target works for every job.

::: locus_security.target.Target
::: locus_security.target.Sender

## Probes

A probe is one adversarial attempt with a stated technique and a verdict. The
built-in set maps onto OWASP's ASI and LLM top tens; `suite_probes()` selects
by suite name, which is what `red_team(suite=...)` takes.

::: locus_security.redteam.base.Probe
::: locus_security.redteam.base.ProbeOutcome
::: locus_security.redteam.all_probes
::: locus_security.redteam.suite_probes

### The built-in probes

::: locus_security.redteam.probes.DirectPromptInjection
::: locus_security.redteam.probes.IndirectPromptInjection
::: locus_security.redteam.probes.Jailbreak
::: locus_security.redteam.probes.ExcessiveAgency
::: locus_security.redteam.probes.SensitiveInformationDisclosure
::: locus_security.redteam.probes.UnsandboxedCodeExecution

## Findings and evidence

[`Evidence`](control.md#locus.control.findings.Evidence) is the shape every
finding takes — a claim, the observations behind it, and a confidence. It is
documented on the Control page, which is where a finding first matters.
`Indicator` is the atom a threat-intel lookup returns.

::: locus.control.findings.Indicator
::: locus.control.findings.Confidence

## Grounding — abstention over assertion

The distinctive part. `ground_finding()` returns either an `Evidence` or an
`Abstention`, never a low-confidence guess dressed as a result, and
`is_finding()` is the narrowing check that separates the two. A pipeline that
cannot tell "no evidence" from "no problem" reports clean when it is blind.

::: locus.control.grounded.ground_finding
::: locus.control.grounded.ground_fingerprint
::: locus.control.grounded.is_finding
::: locus.control.grounded.Abstention
::: locus.control.grounded.GroundedFinding

## Verification — trying to refute

[`verify()`](control.md#locus.control.verification.verify) runs skeptics against a
claim rather than a second model that agrees with the first. A skeptic's job is
to refute; what survives is what gets reported. It and
[`VerificationResult`](control.md#locus.control.verification.VerificationResult) are
documented on the Control page — a policy can require a verified finding before
it admits an action, which is where they are load-bearing.

::: locus.control.verification.Refutation
::: locus.control.verification.Skeptic
::: locus.control.verification.EvidenceQualitySkeptic
::: locus.control.verification.AdversarialSkeptic

## Taxonomy

The standard technique vocabularies a finding is tagged with, and the
comparison to use on severity. [`Severity`](control.md#locus.control.taxonomy.Severity)
itself is documented on the Control page, since a policy's `min_severity`
matches on it — but reach for `severity_at_least()` rather than comparing two
directly: it is a string enum, so `>` orders alphabetically and gets the answer
wrong.

::: locus.control.taxonomy.severity_at_least
::: locus.control.taxonomy.SEVERITY_ORDER
::: locus.control.taxonomy.AtlasTechnique
::: locus.control.taxonomy.OwaspASI
::: locus.control.taxonomy.OwaspLLM
::: locus.control.taxonomy.IndicatorType
::: locus.control.taxonomy.TaxonomyTag

## Security context — the ports

`SecurityContext` is the seam between the SDK and your estate. Each port is a
protocol, so the offline reference adapters that ship here and the adapters you
write are interchangeable, and neither is privileged.

::: locus_security.context.SecurityContext
::: locus_security.context.LogSource
::: locus_security.context.EndpointSource
::: locus_security.context.IdentitySource
::: locus_security.context.CloudSource
::: locus_security.context.ThreatIntelSource
::: locus_security.context.ActionsPort

## Writing an adapter

What a vendor adapter has to implement, plus the helpers that keep one small.

::: locus_security.adapter.SecurityAdapter
::: locus_security.adapter.ToolAdapter
::: locus_security.adapter.as_json
::: locus_security.adapter.env
::: locus_security.adapter.indicator_type
::: locus_security.adapter.inference_claim
::: locus_security.adapter.tool_match

## Tools for an agent

Each capability comes in two forms: a plain function you can call, and a
`@tool`-decorated version to hand an agent. `security_toolset()` returns the
whole set at once.

::: locus_security.security_toolset

### Threat intelligence

::: locus_security.intel.enrich_indicator
::: locus_security.intel.enrich_indicator_tool
::: locus_security.intel.classify_indicator
::: locus_security.intel.enrich_to_finding

### SIEM

::: locus_security.siem.query_siem
::: locus_security.siem.siem_query_tool

### Endpoint detection and response

::: locus_security.edr.list_detections
::: locus_security.edr.list_detections_tool
::: locus_security.edr.fetch_host_timeline
::: locus_security.edr.fetch_host_timeline_tool
::: locus_security.edr.isolate_host
::: locus_security.edr.isolate_host_tool

### Scanning

::: locus_security.scanner.scan_endpoint
::: locus_security.scanner.scan_endpoint_tool
::: locus_security.scanner.scan_endpoint_to_finding
::: locus_security.scanner.scan_dependencies
::: locus_security.scanner.scan_dependencies_tool

### OCI cloud posture

The SOC analyst, `SecurityContext().cloud`, the `cloud_posture_audit` playbook
and `security_toolset(oci=True)` all drive the core
[`describe_oci` / `use_oci`](tools.md#oci-tools) tools. `use_oci()` refuses any
operation whose name does not start with a read prefix (`list_`, `get_`,
`summarize_`, …) before any call is made, returning `{"status": "error", ...}`
to the agent; the only override is the explicit `allow_mutations` opt-in, which
nothing in this package sets. This name check is defence-in-depth: run as a
profile whose IAM group grants only `inspect` / `read` verbs
(`SecurityControls(profile=..., auth_type=...)`), because that policy is what
actually enforces read-only access. Writes belong behind `admit()` or a gated
tool of your own.

## Fingerprinting

Identify the model behind an endpoint from response timing alone — no
cooperation from the endpoint required.

::: locus_security.fingerprint.measure_endpoint_timing
::: locus_security.fingerprint.fingerprint_endpoint_tool
::: locus_security.fingerprint.default_classifier
::: locus_security.fingerprint.fingerprint_to_finding
::: locus_security.fingerprint.dispatch_timing_probe_reference
::: locus_security.fingerprint.FEATURE_KEYS
::: locus.control.findings.FingerprintClassifier
::: locus.control.findings.FingerprintFinding
::: locus.control.findings.FingerprintVerdict

## SOC analyst

A prebuilt agent, the report shape it produces, and the grounding pass applied
to that report.

::: locus_security.soc.create_soc_analyst
::: locus_security.soc.ground_report
::: locus_security.soc.submit_posture
::: locus_security.soc.PostureReport
::: locus_security.soc.PostureFinding
::: locus_security.soc.PostureEvidence
::: locus_security.soc.SecurityControls

## Playbooks

Named sequences for the incident types that recur. Each returns a
[`Playbook`](playbooks.md) you can run or edit.

::: locus_security.playbooks.all_playbooks
::: locus_security.playbooks.nist_800_61_ir
::: locus_security.playbooks.phishing_triage
::: locus_security.playbooks.ransomware_containment
::: locus_security.playbooks.cloud_posture_audit
