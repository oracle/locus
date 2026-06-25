# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""Notebook 35: structured output — typed security schemas a SOAR can consume.

A security report is only useful if a downstream pipeline can parse it,
so every part extracts typed JSON and prints a
``[model call: X.XXs · prompt→completion tokens]`` banner so you can
see the round-trip happen. The typed contracts come from
``tulip.security`` — the same :class:`Indicator`,
:class:`FingerprintVerdict`, and severity/taxonomy enums a Tulip agent
emits findings with — alongside a couple of notebook-local schemas for
the nested- and tool-selection demos.

- ``extract_json`` and ``parse_structured`` — pull JSON out of a model
  reply and validate it against a Pydantic schema (a typed model that
  the LLM must produce JSON for).
- ``create_schema_prompt`` / ``create_output_instructions`` — emit the
  schema-aware system prompt the model needs to comply.
- ``Agent(output_schema=YourModel)`` — constrained decoding plus a
  prompted-JSON fallback wired into the agent loop, so the parsed
  Pydantic object lands on ``result.parsed``.
- ``StructuredOutputError`` for strict-mode parse failures.
- ``tulip.security.FingerprintVerdict`` — the typed verdict of a timing
  side-channel inference fingerprint (model / engine / hardware), which
  Part 8 has the model produce as constrained JSON.

The inference-fingerprinting surface (Parts 6 and 8) is an AI-security
reconnaissance technique — identifying what serves an endpoint from
timing alone — mapped to MITRE ATLAS ``AML.T0040`` (AI Model Inference
API Access) / ``AML.T0024`` (Exfiltration via AI Inference API).

Run it:
    # The bundled mock model is the default; set TULIP_MODEL_PROVIDER for a live provider.
    TULIP_MODEL_ID=openai.gpt-4.1 python examples/notebook_35_structured_output.py

    # Offline:
    TULIP_MODEL_PROVIDER=mock python examples/notebook_35_structured_output.py

Prerequisites:
- An OpenAI or Anthropic API key (or set ``TULIP_MODEL_PROVIDER`` to
  ``openai`` / ``anthropic`` / ``mock``).
- A model that supports constrained JSON decoding for Part 8 — the
  ``check_structured_output_capable()`` helper exits cleanly under mock
  or Cohere R-series.
"""

import json
import time

from config import get_model
from pydantic import BaseModel, Field

from tulip.agent import Agent
from tulip.core.structured import (
    StructuredOutputError,
    create_output_instructions,
    create_schema_prompt,
    extract_json,
    parse_structured,
)
from tulip.security import FingerprintVerdict, Indicator, IndicatorType, Severity


# ---------------------------------------------------------------------------
# Helpers — every section uses these to fire one model call and print a
# timing/token banner.
# ---------------------------------------------------------------------------


def _banner(result, label: str = "") -> None:
    m = result.metrics
    tag = f" {label}" if label else ""
    print(
        f"  [model call{tag}: {m.duration_ms / 1000.0:.2f}s · "
        f"{m.prompt_tokens}→{m.completion_tokens} tokens]"
    )


def _llm_call(prompt: str, *, system: str = "Reply in one sentence.", max_tokens: int = 100) -> str:
    agent = Agent(model=get_model(max_tokens=max_tokens), system_prompt=system)
    t0 = time.perf_counter()
    res = agent.run_sync(prompt)
    dt = time.perf_counter() - t0
    print(
        f"  [model call: {dt:.2f}s · "
        f"{res.metrics.prompt_tokens}→{res.metrics.completion_tokens} tokens]"
    )
    return res.message.strip()


# ---------------------------------------------------------------------------
# Pydantic schemas — the typed contracts the model must satisfy. The
# Indicator / FingerprintVerdict / Severity contracts come from
# ``tulip.security``; the rest are notebook-local shapes for the nested
# and tool-selection demos.
# ---------------------------------------------------------------------------


class Remediation(BaseModel):
    applied: bool = Field(..., description="Whether the remediation was applied")
    action: str = Field(..., description="What was done")
    risk_reduction: float = Field(default=0.0, description="Estimated risk reduction 0-1")
    tags: list[str] = Field(default_factory=list, description="Related tags (CVE ids, services)")


class NetworkZone(BaseModel):
    segment: str
    site: str
    trust_level: str = "internal"


class Asset(BaseModel):
    hostname: str
    criticality: int
    zone: NetworkZone
    services: list[str] = Field(default_factory=list)


class ToolSelection(BaseModel):
    tool_name: str = Field(..., description="Name of the tool to use")
    arguments: dict = Field(default_factory=dict, description="Tool arguments")
    reasoning: str = Field(..., description="Why this tool was selected")


class ReviewFinding(BaseModel):
    title: str = Field(..., description="One-line finding title")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence 0-1")
    severity: Severity = Field(..., description="info, low, medium, high, or critical")


class FindingList(BaseModel):
    findings: list[ReviewFinding] = Field(..., description="Three findings")


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def main() -> None:
    from config import check_structured_output_capable

    check_structured_output_capable()
    print("=" * 60)
    print("Notebook 35: Structured output — typed security findings")
    print("=" * 60)

    # =========================================================================
    # Part 1: extract_json — pull a JSON object out of a free-form reply
    # =========================================================================
    print("\n=== Part 1: Basic JSON Extraction ===\n")
    raw = _llm_call(
        "Output a single JSON object with value='198.51.100.7' and type='ip' "
        "inside a ```json fenced block. Nothing outside the fence.",
        system="Output only a fenced JSON block.",
        max_tokens=80,
    )
    extracted = extract_json(raw)  # returns the JSON text, pulled out of the fence
    print(f"  extract_json -> {extracted}")
    obj = json.loads(extracted)
    if obj.get("type") in set(IndicatorType):
        print(f"  type '{obj['type']}' is a recognized IndicatorType")

    # =========================================================================
    # Part 2: parse_structured — validate the JSON against a Pydantic schema
    # =========================================================================
    print("\n=== Part 2: Parsing into Pydantic Models ===\n")
    raw = _llm_call(
        "Output a single JSON object {value, type} for the indicator of compromise "
        "phish.example.net, type domain. Inside a ```json block.",
        system="Output only the fenced JSON block. Nothing else.",
        max_tokens=120,
    )
    # tulip.security.Indicator: a typed (IndicatorType, value) observable.
    parsed = parse_structured(raw, Indicator, strict=False)
    print(f"  Success: {parsed.success}  Parsed: {parsed.parsed}")

    # =========================================================================
    # Part 3: Error handling — strict vs non-strict parsing on bad inputs
    # =========================================================================
    print("\n=== Part 3: Error Handling ===\n")
    bad = _llm_call(
        "Reply with the literal string: This is not JSON.",
        system="Reply only with the requested string.",
        max_tokens=40,
    )
    bad_result = parse_structured(bad, Indicator, strict=False)
    print(f"  Invalid JSON - Success: {bad_result.success}  Error: {bad_result.error}")

    missing_type = _llm_call(
        "Output a JSON object with only the field value='aa11bb22cc33', "
        "NO type field. Inside ```json.",
        system="Output only the fenced JSON block.",
        max_tokens=80,
    )
    missing_result = parse_structured(missing_type, Indicator, strict=False)
    print(f"  Missing-field - Success: {missing_result.success}  Error: {missing_result.error}")
    try:
        parse_structured("invalid", Indicator, strict=True)
    except StructuredOutputError as e:
        print(f"  Strict mode raised {type(e).__name__}")

    # =========================================================================
    # Part 4: Schema prompts — tell the model what JSON shape to produce
    # =========================================================================
    print("\n=== Part 4: Creating Schema Prompts ===\n")
    schema_prompt = create_schema_prompt(Remediation)
    print(f"  schema_prompt (head): {schema_prompt[:160]}...")
    instructions = create_output_instructions(Remediation)
    raw = _llm_call(
        "Following these instructions, return a JSON for a completed patch "
        "of CVE-2024-99999 on service `orders-api`:\n" + instructions,
        system="Output only a fenced JSON block matching the schema.",
        max_tokens=200,
    )
    out = parse_structured(raw, Remediation, strict=False)
    if out.success:
        print(
            f"  Parsed: applied={out.parsed.applied} action='{out.parsed.action}' "
            f"tags={out.parsed.tags}"
        )
    else:
        print(f"  Parse error: {out.error}")

    # =========================================================================
    # Part 5: Nested schemas — Asset contains NetworkZone contains primitives
    # =========================================================================
    print("\n=== Part 5: Complex Nested Structures ===\n")
    nested = _llm_call(
        "Output a JSON for an asset web-01, criticality 4, zone "
        "(segment 'dmz-1', site 'tor-dc-2', trust_level 'dmz'), "
        "services [https, ssh]. Inside ```json.",
        system="Output only the fenced JSON block.",
        max_tokens=240,
    )
    asset_res = parse_structured(nested, Asset, strict=False)
    if asset_res.success:
        a = asset_res.parsed
        print(f"  Asset: {a.hostname} (criticality {a.criticality}, {a.zone.segment})")
        print(f"  Services: {', '.join(a.services)}")
    else:
        print(f"  Parse error: {asset_res.error}")

    # =========================================================================
    # Part 6: A real tulip.security type — FingerprintVerdict from a timing
    #         side-channel probe. The model maps observed timing features
    #         (TTFT, tokens/sec, inter-token jitter) to a (model, engine,
    #         hardware) verdict. AML.T0040 (AI Model Inference API Access).
    #         Low feature_coverage is the signal a downstream grounding step
    #         uses to abstain — see notebook 37.
    # =========================================================================
    print("\n=== Part 6: FingerprintVerdict (tulip.security) ===\n")
    fp_instructions = create_output_instructions(FingerprintVerdict)
    raw = _llm_call(
        "A timing side-channel probe of an inference endpoint observed TTFT "
        "cadence and tokens/sec consistent with an open-weights 8B model served "
        "by vLLM on a datacenter A100-class GPU; 5 of the 6 expected timing "
        "features were captured. Return the verdict JSON.\n" + fp_instructions,
        system="Output only a fenced JSON block matching the schema.",
        max_tokens=240,
    )
    verdict_res = parse_structured(raw, FingerprintVerdict, strict=False)
    if verdict_res.success:
        v = verdict_res.parsed
        print(f"  FingerprintVerdict: {v.model} on {v.engine} / {v.hardware}")
        print(f"  Classifier confidence: {v.confidence:.0%}")
        print(f"  Feature coverage: {v.feature_coverage:.0%}")
    else:
        print(f"  Parse error: {verdict_res.error}")

    # =========================================================================
    # Part 7: System-prompt pattern — embed the schema in the system message
    # =========================================================================
    print("\n=== Part 7: Agent ToolSelection prompt ===\n")
    sys_prompt = (
        "You are a SOC assistant with access to security tools.\n\n"
        + create_output_instructions(ToolSelection)
        + "\nThink before selecting."
    )
    pick = _llm_call(
        "We need to check the reputation of IP 192.0.2.66. Pick the right "
        "tool and reply with the JSON.",
        system=sys_prompt,
        max_tokens=200,
    )
    pick_res = parse_structured(pick, ToolSelection, strict=False)
    if pick_res.success:
        ts = pick_res.parsed
        print(f"  tool={ts.tool_name}  args={ts.arguments}")
        print(f"  reasoning={ts.reasoning}")
    else:
        print(f"  Parse error: {pick_res.error}")

    # =========================================================================
    # Part 8: Agent(output_schema=…) — the typed object lands on result.parsed
    # =========================================================================
    print("\n=== Part 8: Agent(output_schema=FindingList) ===\n")
    live_agent = Agent(
        model=get_model(max_tokens=300),
        output_schema=FindingList,
        system_prompt=(
            "You are an application-security reviewer. Report exactly three "
            "findings for the code under review as a structured list."
        ),
    )
    t0 = time.perf_counter()
    live = live_agent.run_sync(
        "Review: a login endpoint compares passwords with ==, logs the raw "
        "password on failure, and has no rate limiting. Top three findings."
    )
    dt = time.perf_counter() - t0
    print(
        f"  [model call: {dt:.2f}s · "
        f"{live.metrics.prompt_tokens}→{live.metrics.completion_tokens} tokens]"
    )
    report: FindingList | None = live.parsed
    if not isinstance(report, FindingList):
        raise TypeError(
            "Review agent returned no parsed FindingList. The configured model "
            "could not honor the JSON schema. Use a stronger model "
            "(e.g. openai.gpt-4o, openai.gpt-5, anthropic.claude-3-5-sonnet) "
            f"for notebook 35 (Part 8). Raw output: {live.message!r}"
        )
    for f in report.findings:
        print(f"  {f.title[:40]:<40}  confidence={f.confidence:.2f}  severity={f.severity.value}")

    print("\n" + "=" * 60)
    print("Done. Next: notebook 36 — reasoning patterns.")
    print("=" * 60)


if __name__ == "__main__":
    main()
