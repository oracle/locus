# tulip-agents-security

Security-domain tooling for [Tulip](https://github.com/tuliplabs-ai/tulip-agents)
agents, shipped as a separate, opt-in distribution. The core `tulip-agents`
runtime — the admission gate, policy, audit trail, and the GSAR grounding and
verification layer in `tulip.control` — does not include any of it.

It is not published to PyPI. Install it from the repository:

```bash
pip install "git+https://github.com/tuliplabs-ai/tulip-agents#subdirectory=packages/tulip-agents-security"
# + boto3 for the AWS posture tools:
pip install "tulip-agents-security[aws] @ git+https://github.com/tuliplabs-ai/tulip-agents#subdirectory=packages/tulip-agents-security"
```

It is imported as `tulip_security`.

## What is in it

Every result goes through the same contract as the rest of Tulip: a grounded
`Evidence` tagged against public weakness catalogues (MITRE ATLAS, OWASP LLM /
Agentic Top 10), or an explicit `Abstention` saying why it was withheld.

| Area | API |
|---|---|
| AI red-teaming | `Target`, `red_team`, `assure`, `monitor`, `guardrail_coverage`, the probe library in `tulip_security.redteam` |
| Investigation facade | `SecurityContext` — logs, endpoint, identity, cloud, threat-intel sources and an admission-gated actions port |
| SOC triage | `create_soc_analyst`, `ground_report`, `security_toolset`, curated IR playbooks |
| Reference adapters | threat intel, SIEM, EDR, dependency / endpoint scanner, inference fingerprinting, AWS posture (read-only) |
| Integration contract | `SecurityAdapter`, `ToolAdapter`, helper toolkit, and the conformance kit in `tulip_security.testing` |

The bundled adapters return deterministic, benign offline samples when no
credentials are set, so everything runs standalone. A live vendor adapter is a
class you write in your own package against `SecurityAdapter` and pass in with
`security_toolset(extra=[...])`.

```python
import asyncio

from tulip_security import Target, is_finding, red_team


async def main():
    report = await red_team(
        Target.endpoint("https://support-bot.example/chat"), suite="owasp-asi"
    )
    print([f for f in report.findings if is_finding(f)])


asyncio.run(main())
```

The grounding vocabulary (`Evidence`, `Severity`, `ground_finding`, `verify`,
the taxonomy enums) is re-exported from `tulip.control`, so domain code needs
one import.

## Examples

[`examples/`](examples/) holds the runnable security notebooks (cloud posture,
IR playbooks, agent red-teaming, a CI security gate, finding verification, SOC
alert triage, model fingerprinting, `SecurityContext` investigations), the
threat-to-defense scenario catalogue, the vendor-integration gists, and the
editable playbook YAML. They run offline by default:

```bash
python examples/notebook_75_agent_red_team.py
python examples/scenarios/run_all.py
```

## Responsible use

Parts of this package act against systems: the red-team probes send adversarial
prompts to a target, the fingerprinting tools time an inference endpoint to
infer what serves it, the scanner connects to hosts, and the EDR adapter can
isolate a host when containment is enabled. Point them only at systems you own
or are explicitly authorised to test, and keep containment and any other
state-changing tool behind the admission gate (`tulip.control.admit`) with a
human in the loop. The bundled samples use documentation address ranges
(RFC 5737), `*.example` domains and well-known test artifacts; keep it that
way in anything you publish.

## Moving from `tulip.security`

These modules used to ship inside `tulip-agents` as `tulip.security`. That path
still works: with this package installed, `from tulip.security import red_team`
(and submodule imports such as `tulip.security.redteam`) resolve here, and
without it they raise an `ImportError` that says what to install. The
domain-neutral names (`Evidence`, `Severity`, `ground_finding`, `verify`, …)
moved to `tulip.control` and still import from `tulip.security` with a
`DeprecationWarning`. New code should import from `tulip_security` and
`tulip.control`.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
