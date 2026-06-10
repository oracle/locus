# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""Notebook 36: structured output — get typed JSON back from an LLM.

Every part calls the configured model and prints a
``[model call: X.XXs · prompt→completion tokens]`` banner so you can
see the round-trip happen.

- ``extract_json`` and ``parse_structured`` — pull JSON out of a model
  reply and validate it against a Pydantic schema (a typed model that
  the LLM must produce JSON for).
- ``create_schema_prompt`` / ``create_output_instructions`` — emit the
  schema-aware system prompt the model needs to comply.
- ``Agent(output_schema=YourModel)`` — constrained decoding plus a
  prompted-JSON fallback wired into the agent loop, so the parsed
  Pydantic object lands on ``result.parsed``.
- ``StructuredOutputError`` for strict-mode parse failures.

Run it:
    # The bundled mock model is the default; set TULIP_MODEL_PROVIDER for a live provider.
    TULIP_MODEL_ID=openai.gpt-4.1 python examples/notebook_41_structured_output.py

    # Offline:
    TULIP_MODEL_PROVIDER=mock python examples/notebook_41_structured_output.py

Prerequisites:
- An OpenAI or Anthropic API key (or set ``TULIP_MODEL_PROVIDER`` to
  ``openai`` / ``anthropic`` / ``mock``).
- A model that supports constrained JSON decoding for Part 8 — the
  ``check_structured_output_capable()`` helper exits cleanly under mock
  or Cohere R-series.
"""

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
# Pydantic schemas — each one is the typed contract the model must satisfy.
# ---------------------------------------------------------------------------


class Person(BaseModel):
    name: str
    age: int
    email: str | None = None


class TaskResult(BaseModel):
    success: bool = Field(..., description="Whether the task succeeded")
    message: str = Field(..., description="Result message")
    score: float = Field(default=0.0, description="Confidence score 0-1")
    tags: list[str] = Field(default_factory=list, description="Related tags")


class Address(BaseModel):
    street: str
    city: str
    country: str = "USA"


class Company(BaseModel):
    name: str
    founded: int
    address: Address
    employees: list[str] = Field(default_factory=list)


class AnalysisResult(BaseModel):
    summary: str = Field(..., description="Brief summary of findings")
    root_cause: str | None = Field(None, description="Root cause if identified")
    confidence: float = Field(..., description="Confidence level 0-1")
    recommendations: list[str] = Field(default_factory=list)
    requires_action: bool = Field(default=False)


class ToolSelection(BaseModel):
    tool_name: str = Field(..., description="Name of the tool to use")
    arguments: dict = Field(default_factory=dict, description="Tool arguments")
    reasoning: str = Field(..., description="Why this tool was selected")


class Vendor(BaseModel):
    name: str = Field(..., description="Vendor brand name")
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence 0-1")
    region: str = Field(..., description="Primary geographic region")


class VendorList(BaseModel):
    vendors: list[Vendor] = Field(..., description="Three picks")


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def main() -> None:
    from config import check_structured_output_capable

    check_structured_output_capable()
    print("=" * 60)
    print("Notebook 36: Structured output — typed JSON from an LLM")
    print("=" * 60)

    # =========================================================================
    # Part 1: extract_json — pull a JSON object out of a free-form reply
    # =========================================================================
    print("\n=== Part 1: Basic JSON Extraction ===\n")
    raw = _llm_call(
        "Output a single JSON object with name=Alice and age=30 inside a "
        "```json fenced block. Nothing outside the fence.",
        system="Output only a fenced JSON block.",
        max_tokens=80,
    )
    extracted = extract_json(raw)
    print(f"  extract_json -> {extracted}")

    # =========================================================================
    # Part 2: parse_structured — validate the JSON against a Pydantic schema
    # =========================================================================
    print("\n=== Part 2: Parsing into Pydantic Models ===\n")
    raw = _llm_call(
        "Output a single JSON object {name, age, email} for the person "
        "Diana, 28, diana@example.com. Inside a ```json block.",
        system="Output only the fenced JSON block. Nothing else.",
        max_tokens=120,
    )
    parsed = parse_structured(raw, Person, strict=False)
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
    bad_result = parse_structured(bad, Person, strict=False)
    print(f"  Invalid JSON - Success: {bad_result.success}  Error: {bad_result.error}")

    missing_age = _llm_call(
        "Output a JSON object with only the field name=Frank, NO age field. Inside ```json.",
        system="Output only the fenced JSON block.",
        max_tokens=80,
    )
    missing_result = parse_structured(missing_age, Person, strict=False)
    print(f"  Missing-field - Success: {missing_result.success}  Error: {missing_result.error}")
    try:
        parse_structured("invalid", Person, strict=True)
    except StructuredOutputError as e:
        print(f"  Strict mode raised {type(e).__name__}")

    # =========================================================================
    # Part 4: Schema prompts — tell the model what JSON shape to produce
    # =========================================================================
    print("\n=== Part 4: Creating Schema Prompts ===\n")
    schema_prompt = create_schema_prompt(TaskResult)
    print(f"  schema_prompt (head): {schema_prompt[:160]}...")
    instructions = create_output_instructions(TaskResult)
    raw = _llm_call(
        "Following these instructions, return a JSON for a successful "
        "deploy of service `orders-api`:\n" + instructions,
        system="Output only a fenced JSON block matching the schema.",
        max_tokens=200,
    )
    out = parse_structured(raw, TaskResult, strict=False)
    if out.success:
        print(
            f"  Parsed: success={out.parsed.success} message='{out.parsed.message}' "
            f"tags={out.parsed.tags}"
        )
    else:
        print(f"  Parse error: {out.error}")

    # =========================================================================
    # Part 5: Nested schemas — Company contains Address contains primitives
    # =========================================================================
    print("\n=== Part 5: Complex Nested Structures ===\n")
    nested = _llm_call(
        "Output a JSON for a company TechCorp, founded 2020, address "
        "(street '123 Main St', city 'San Francisco', country 'USA'), "
        "employees [Alice, Bob, Charlie]. Inside ```json.",
        system="Output only the fenced JSON block.",
        max_tokens=240,
    )
    company_res = parse_structured(nested, Company, strict=False)
    if company_res.success:
        c = company_res.parsed
        print(f"  Company: {c.name} (founded {c.founded}, {c.address.city})")
        print(f"  Employees: {', '.join(c.employees)}")
    else:
        print(f"  Parse error: {company_res.error}")

    # =========================================================================
    # Part 6: Production shape — incident triage as a structured AnalysisResult
    # =========================================================================
    print("\n=== Part 6: Real-world AnalysisResult ===\n")
    raw = _llm_call(
        "Diagnose an incident: 'connection pool saturated, P99=2500ms'. "
        "Return an AnalysisResult JSON inside ```json with fields summary, "
        "root_cause, confidence, recommendations, requires_action.",
        system="Output only the fenced JSON block.",
        max_tokens=300,
    )
    analysis_res = parse_structured(raw, AnalysisResult, strict=False)
    if analysis_res.success:
        a = analysis_res.parsed
        print(f"  Summary: {a.summary}")
        print(f"  Root cause: {a.root_cause}")
        print(f"  Confidence: {a.confidence:.0%}")
        print(f"  Requires action: {a.requires_action}")
        for rec in a.recommendations:
            print(f"    - {rec}")
    else:
        print(f"  Parse error: {analysis_res.error}")

    # =========================================================================
    # Part 7: System-prompt pattern — embed the schema in the system message
    # =========================================================================
    print("\n=== Part 7: Agent ToolSelection prompt ===\n")
    sys_prompt = (
        "You are an AI assistant with access to tools.\n\n"
        + create_output_instructions(ToolSelection)
        + "\nThink before selecting."
    )
    pick = _llm_call(
        "We need to look up a customer email. Pick the right tool and reply with the JSON.",
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
    print("\n=== Part 8: Agent(output_schema=VendorList) ===\n")
    live_agent = Agent(
        model=get_model(max_tokens=300),
        output_schema=VendorList,
        system_prompt=(
            "You are a cloud-procurement analyst. Recommend exactly three "
            "cloud vendors as a structured list."
        ),
    )
    t0 = time.perf_counter()
    live = live_agent.run_sync("Top three cloud vendors for a $2M enterprise compute spend.")
    dt = time.perf_counter() - t0
    print(
        f"  [model call: {dt:.2f}s · "
        f"{live.metrics.prompt_tokens}→{live.metrics.completion_tokens} tokens]"
    )
    picks: VendorList | None = live.parsed
    if not isinstance(picks, VendorList):
        raise TypeError(
            "Vendor agent returned no parsed VendorList. The configured model "
            "could not honor the JSON schema. Use a stronger model "
            "(e.g. openai.gpt-4o, openai.gpt-5, anthropic.claude-3-5-sonnet) "
            f"for notebook 35 (Part 8). Raw output: {live.message!r}"
        )
    for v in picks.vendors:
        print(f"  {v.name:<14}  score={v.score:.2f}  region={v.region}")

    print("\n" + "=" * 60)
    print("Done. Next: notebook 36 — reasoning patterns.")
    print("=" * 60)


if __name__ == "__main__":
    main()
