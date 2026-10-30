# locus-sdk-security

Security-domain tooling for [Locus](https://github.com/oracle/locus)
agents, shipped as a separate, opt-in distribution. The core `locus-sdk`
runtime — the admission gate, policy, audit trail, and the GSAR grounding and
verification layer in `locus.control` — does not include any of it.

It is not published to PyPI. Install it from the repository; its version
tracks `locus-sdk` release for release, so pin the tag you run:

```bash
pip install "git+https://github.com/oracle/locus@v5.3.0#subdirectory=packages/locus-sdk-security"
# + the OCI SDK for the cloud-posture tools (core describe_oci / use_oci):
pip install "locus-sdk[oci]"
```

It is imported as `locus_security`.

## What is in it

Every result goes through the same contract as the rest of Locus: a grounded
`Evidence` tagged against public weakness catalogues (MITRE ATLAS, OWASP LLM /
Agentic Top 10), or an explicit `Abstention` saying why it was withheld.

| Area | API |
|---|---|
| AI red-teaming | `Target`, `red_team`, `assure`, `monitor`, `guardrail_coverage`, the probe library in `locus_security.redteam` |
| Investigation facade | `SecurityContext` — logs, endpoint, identity, cloud, threat-intel sources and an admission-gated actions port |
| SOC triage | `create_soc_analyst`, `ground_report`, `security_toolset`, curated IR playbooks |
| Reference adapters | threat intel, SIEM, EDR, dependency / endpoint scanner, inference fingerprinting; OCI cloud posture on the core `describe_oci` / `use_oci` (read-only) |
| Integration contract | `SecurityAdapter`, `ToolAdapter`, helper toolkit, and the conformance kit in `locus_security.testing` |

The bundled adapters return deterministic, benign offline samples when no
credentials are set, so everything runs standalone. A live vendor adapter is a
class you write in your own package against `SecurityAdapter` and pass in with
`security_toolset(extra=[...])`.

```python
import asyncio

from locus_security import Target, is_finding, red_team


async def main():
    report = await red_team(
        Target.endpoint("https://support-bot.example/chat"), suite="owasp-asi"
    )
    print([f for f in report.findings if is_finding(f)])


asyncio.run(main())
```

The grounding vocabulary (`Evidence`, `Severity`, `ground_finding`, `verify`,
the taxonomy enums) is re-exported from `locus.control`, so domain code needs
one import.

## Examples

[`examples/`](examples/) holds the runnable security notebooks (cloud posture,
IR playbooks, agent red-teaming, a CI security gate, finding verification, SOC
alert triage, model fingerprinting, `SecurityContext` investigations), the
threat-to-defense scenario catalogue, the vendor-integration gists, and the
editable playbook YAML. They run offline by default:

```bash
python examples/notebook_83_agent_red_team.py
python examples/scenarios/run_all.py
```

## Responsible use

Parts of this package act against systems: the red-team probes send adversarial
prompts to a target, the fingerprinting tools time an inference endpoint to
infer what serves it, the scanner connects to hosts, and the EDR adapter can
isolate a host when containment is enabled. Point them only at systems you own
or are explicitly authorised to test, and keep containment and any other
state-changing tool behind the admission gate (`locus.control.admit`) with a
human in the loop. The bundled samples use documentation address ranges
(RFC 5737), `*.example` domains and well-known test artifacts; keep it that
way in anything you publish.

## Moving from `locus.security`

These modules used to ship inside `locus-sdk` as `locus.security`. That path
still works: with this package installed, `from locus.security import red_team`
(and submodule imports such as `locus.security.redteam`) resolve here, and
without it they raise an `ImportError` that says what to install. The
domain-neutral names (`Evidence`, `Severity`, `ground_finding`, `verify`, …)
moved to `locus.control` and still import from `locus.security` with a
`DeprecationWarning`. New code should import from `locus_security` and
`locus.control`.

## License

Copyright (c) 2025, 2026 Oracle and/or its affiliates. Released under the Universal Permissive License v1.0 — see [LICENSE](LICENSE).
