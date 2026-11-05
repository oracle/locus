# Security layer — evidence-grounded findings

The separate `locus-sdk-security` package (`locus_security`) applies the SDK's
runtime to cybersecurity. **A security finding can only be built from
evidence that clears the [GSAR](gsar.md) grounding threshold.** Below the
bar, the agent abstains and records why. An ungrounded finding would be a
false positive, so the API gives you no way to create one: every finding
carries the evidence that earned it.

{{ locus_diagram grounded-findings }}

## The grounding bridge

[`ground_finding()`][gh] takes a candidate finding plus the GSAR
`Partition` of its claims, scores the partition, and returns either a
typed `Evidence` (it ships) or an `Abstention` (it is withheld, with an
audit record of why).

```python
from locus.control import ground_finding, Severity, is_finding
from locus.reasoning.gsar import Claim, EvidenceType, Partition

result = ground_finding(
    title="Expired TLS certificate on 192.0.2.10:443",
    description="The serving endpoint presents an expired certificate.",
    severity=Severity.HIGH,
    asset="192.0.2.10:443",
    remediation="Rotate the certificate; enforce automated renewal.",
    partition=Partition(
        grounded=[
            Claim(
                text="certificate expired 2026-05-30",
                type=EvidenceType.TOOL_MATCH,
                evidence_refs=["tool:tls_scan:not_after=2026-05-30"],
            ),
        ],
    ),
)

if is_finding(result):
    print("SHIPPED", result.title, result.gsar_score)
else:
    print("WITHHELD", result.decision, "—", result.reason)
```

An `Evidence` carries the `gsar_score` that admitted it and the
`evidence_refs` flattened from its claims — it always knows how strongly
it is grounded and what it is grounded in. `Evidence` has no constructor
without a score, so the invariant holds at the type level, not by
convention.

## Threat taxonomy

Findings tag themselves with the published catalogues, so they drop into
a SIEM or a compliance report without translation:

| Enum | Catalogue |
|---|---|
| `AtlasTechnique` | [MITRE ATLAS](https://atlas.mitre.org/) — `AML.Txxxx` |
| `OwaspLLM` | OWASP Top 10 for LLM Applications (2025) — `LLM01`–`LLM10` |
| `OwaspASI` | OWASP Top 10 for Agentic Applications (2026) — `ASI01`–`ASI10` |

```python
from locus.control import AtlasTechnique, OwaspLLM

# tag a finding with the standards it maps to
taxonomy = [OwaspLLM.PROMPT_INJECTION, AtlasTechnique.PROMPT_INJECTION]  # LLM01 / AML.T0051
```

Every ID in the encoded taxonomy enums maps to a runnable defense gist — see the
[threat-scenarios coverage matrix](threat-scenarios.md). (The OWASP LLM/ASI
enums are complete; the `AtlasTechnique` enum encodes a representative subset
of MITRE ATLAS.)

## Inference fingerprinting

`ground_fingerprint()` is an advanced application of the same
admit/abstain contract: identifying the model, inference engine,
and hardware behind an endpoint from **timing side-channels**, with the
timing feature vector as the evidence. A `FingerprintClassifier` protocol
lets a real service plug in behind a deterministic mock; low feature
coverage abstains rather than asserting a fingerprint.

```python
from locus.control import FingerprintVerdict, ground_fingerprint

verdict = classifier(features)              # FingerprintClassifier
finding = ground_fingerprint(
    verdict=verdict,
    asset="192.0.2.20:8000",
    partition=evidence_partition,
)
```

## GPU-level analysis

An agent SDK dispatches GPU probes because some teams need to verify
what actually serves an endpoint — which model, which engine, which
accelerator. For fingerprinting, grounding extends below the
prompt level: it is a measurement problem. The model, engine, and
accelerator serving an endpoint leave a signature in *timing* —
time-to-first-token, inter-token cadence, throughput under contention.

The measurement runs where the hardware is: a probe on target-class
accelerators returns a timing profile, and that profile becomes a grounded
finding. Core ships the `FingerprintClassifier` protocol and
`ground_fingerprint`; `locus_security` adds an offline reference for the
probe dispatch (`dispatch_timing_probe_reference`), and provisioning on a
specific GPU cloud is an integration you write for your cloud. The
grounding contract is the same everywhere, and an under-observed probe
abstains rather than asserting a fingerprint.

The pattern is always the same three steps — **measure → classify →
ground** — with the agent orchestrating tools and `ground_fingerprint`
enforcing the evidence bar:

```python
from locus.control import (
    FingerprintVerdict, ground_fingerprint, is_finding,
    Indicator, IndicatorType, AtlasTechnique,
)
from locus.reasoning.gsar import Claim, EvidenceType, Partition

# 1. MEASURE — a probe runs on a dedicated GPU cluster (RunPod / Lambda /
#    …) against the target endpoint and returns a timing feature
#    vector. Here the values are illustrative.
features = {
    "ttft_ms": 41.2, "inter_token_ms_p50": 7.8,
    "tokens_per_s": 128.0, "contention_slope": 0.34,
}

# 2. CLASSIFY — any object satisfying the FingerprintClassifier protocol.
#    Plug in a real/custom classifier here; the one Locus bundles
#    (`default_classifier`) is a transparent heuristic placeholder. The
#    verdict reports how much of the expected feature space was observed.
verdict = FingerprintVerdict(
    model="open-weights-8b", engine="vllm", hardware="datacenter-gpu",
    confidence=0.93, feature_coverage=1.0,
)

# 3. GROUND — the timing feature vector IS the evidence. Full coverage
#    clears the bar; sparse coverage would abstain.
partition = Partition(grounded=[
    Claim(text="TTFT/cadence profile matches vLLM/8B on datacenter GPU",
          type=EvidenceType.TOOL_MATCH,
          evidence_refs=[f"probe:timing:{k}={v}" for k, v in features.items()]),
])

finding = ground_fingerprint(
    verdict=verdict,
    asset="192.0.2.20:8000",
    partition=partition,
    indicators=[Indicator(type=IndicatorType.ENDPOINT, value="192.0.2.20:8000")],
    taxonomy=[AtlasTechnique.INFERENCE_API_ACCESS],  # AML.T0040
)

if is_finding(finding):
    print("FINGERPRINTED", finding.verdict.model, "→", finding.gsar_score)
else:
    print("ABSTAINED — insufficient feature coverage:", finding.reason)
```

### Measuring the timing

The measurement needs no privileged access and no co-located hardware:
stream a completion from the target and time the token arrivals.
`locus_security` ships this as `measure_endpoint_timing`. With
`OPENAI_API_KEY` set, it streams `samples` completions from the endpoint
(point `TIMING_BASE_URL` at any OpenAI-compatible API) and computes TTFT
p50, mean inter-token latency, its coefficient of variation, and mean
tokens/sec; with no key set it returns the deterministic sample vector
described below. The example keeps an explicit OpenAI model id, the
function's default, because the measurement calls the OpenAI-compatible
streaming API directly rather than an SDK provider class.

```python
from locus_security import measure_endpoint_timing, fingerprint_to_finding

# Live timing measurement against a streaming OpenAI-compatible endpoint.
features = measure_endpoint_timing(model="gpt-4o-mini", samples=5)
# {"ttft_ms_p50": ..., "itl_ms_mean": ..., "itl_cv": ..., "tps_mean": ...}

finding = fingerprint_to_finding(features, asset="api.example/v1")
# Full coverage ships a FingerprintFinding; a thin vector abstains.
```

The measurement feeds the **measure → classify → ground** loop, with
the grounding bar enforced identically — an under-observed endpoint
abstains rather than asserting an identity.

!!! note "Runs in CI without credentials"
    With no key set, `measure_endpoint_timing` returns a deterministic
    sample vector so the notebooks stay
    runnable in CI — never a substitute for the live measurement.

The same shape powers a complete agent workflow in the notebooks
([forensics specialist](../notebooks/notebook_23_specialist_agents.md)),
where a low-coverage probe correctly abstains and a full-coverage probe
ships a finding mapped to compliance controls.

## Where it shows up

- Applied end-to-end: [cloud-posture agent](cloud-posture.md) — grounded,
  read-only OCI auditing built on `ground_finding`.
- Complete defense catalogue: [threat scenarios](threat-scenarios.md) — every
  OWASP LLM / OWASP ASI / MITRE ATLAS ID mapped to a runnable gist.
- Typed grounding notebook: [GSAR typed grounding](../notebooks/notebook_55_gsar_typed_grounding.md)
- Prompt-injection findings: [guardrails](../notebooks/notebook_69_guardrails_security.md)
- Fingerprinting: [forensics specialist](../notebooks/notebook_23_specialist_agents.md)
- RAG poisoning: [intel copilot](../notebooks/notebook_53_rag_agents.md)

[gh]: https://github.com/oracle/locus/blob/main/src/locus/control/grounded.py
