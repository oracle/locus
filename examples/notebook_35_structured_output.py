# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""Notebook 35: structured output — typed support schemas a help desk can consume.

A support agent's reply is only useful to the rest of the help desk if a
downstream workflow can parse it — route the ticket, update the CRM, fire
a satisfaction survey. So every part extracts typed JSON and prints a
``[model call: X.XXs · prompt→completion tokens]`` banner so you can see
the round-trip happen. The typed contracts are ordinary Pydantic models —
the same ``ContactPoint``, ``Resolution``, and priority enums a Tulip
support agent emits ticket updates with.

- ``extract_json`` and ``parse_structured`` — pull JSON out of a model
  reply and validate it against a Pydantic schema (a typed model that
  the LLM must produce JSON for).
- ``create_schema_prompt`` / ``create_output_instructions`` — emit the
  schema-aware system prompt the model needs to comply.
- ``Agent(output_schema=YourModel)`` — constrained decoding plus a
  prompted-JSON fallback wired into the agent loop, so the parsed
  Pydantic object lands on ``result.parsed``.
- ``StructuredOutputError`` for strict-mode parse failures.
- ``ConversationVerdict`` — a notebook-local schema the model produces as
  constrained JSON in Part 6, classifying a chat transcript into
  (intent, sentiment) with a confidence and an ``evidence_coverage``
  signal a downstream grounding step uses to abstain.

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

import asyncio
import json
import time
from enum import StrEnum

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


async def _llm_call(
    prompt: str, *, system: str = "Reply in one sentence.", max_tokens: int = 100
) -> str:
    agent = Agent(model=get_model(max_tokens=max_tokens), system_prompt=system)
    t0 = time.perf_counter()
    res = await agent.arun(prompt)
    dt = time.perf_counter() - t0
    print(
        f"  [model call: {dt:.2f}s · "
        f"{res.metrics.prompt_tokens}→{res.metrics.completion_tokens} tokens]"
    )
    return res.message.strip()


# ---------------------------------------------------------------------------
# Pydantic schemas — the typed contracts the model must satisfy. These are
# the everyday shapes a support agent emits: a typed contact point, a ticket
# resolution, a customer record, a tool choice, and a triaged issue list.
# ---------------------------------------------------------------------------


class Priority(StrEnum):
    low = "low"
    normal = "normal"
    high = "high"
    urgent = "urgent"


class ChannelType(StrEnum):
    email = "email"
    phone = "phone"
    chat = "chat"
    order_id = "order_id"
    account_id = "account_id"


class ContactPoint(BaseModel):
    value: str = Field(..., description="The contact identifier or reference")
    type: ChannelType = Field(..., description="email, phone, chat, order_id, or account_id")


class Resolution(BaseModel):
    resolved: bool = Field(..., description="Whether the ticket was resolved")
    action: str = Field(..., description="What the agent did")
    satisfaction_delta: float = Field(default=0.0, description="Estimated CSAT change 0-1")
    tags: list[str] = Field(default_factory=list, description="Related tags (product, topic)")


class AccountTier(BaseModel):
    plan: str
    region: str
    standing: str = "good"


class Customer(BaseModel):
    name: str
    lifetime_value: int
    tier: AccountTier
    products: list[str] = Field(default_factory=list)


class ToolSelection(BaseModel):
    tool_name: str = Field(..., description="Name of the tool to use")
    arguments: dict = Field(default_factory=dict, description="Tool arguments")
    reasoning: str = Field(..., description="Why this tool was selected")


class TicketIssue(BaseModel):
    title: str = Field(..., description="One-line issue title")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence 0-1")
    priority: Priority = Field(..., description="low, normal, high, or urgent")


class IssueList(BaseModel):
    issues: list[TicketIssue] = Field(..., description="Three issues")


class ConversationVerdict(BaseModel):
    """A typed read of a chat transcript: what the customer wants, how they
    feel, and how well the evidence supported the call."""

    intent: str = Field(..., description="The customer's primary intent")
    sentiment: str = Field(..., description="positive, neutral, or negative")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classifier confidence 0-1")
    evidence_coverage: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Fraction of expected signals captured 0-1"
    )


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


async def main() -> None:
    from config import check_structured_output_capable

    check_structured_output_capable()
    print("=" * 60)
    print("Notebook 35: Structured output — typed support tickets")
    print("=" * 60)

    # =========================================================================
    # Part 1: extract_json — pull a JSON object out of a free-form reply
    # =========================================================================
    print("\n=== Part 1: Basic JSON Extraction ===\n")
    raw = await _llm_call(
        "Output a single JSON object with value='order-100847' and type='order_id' "
        "inside a ```json fenced block. Nothing outside the fence.",
        system="Output only a fenced JSON block.",
        max_tokens=80,
    )
    extracted = extract_json(raw)  # returns the JSON text, pulled out of the fence
    print(f"  extract_json -> {extracted}")
    obj = json.loads(extracted)
    if obj.get("type") in set(ChannelType):
        print(f"  type '{obj['type']}' is a recognized ChannelType")

    # =========================================================================
    # Part 2: parse_structured — validate the JSON against a Pydantic schema
    # =========================================================================
    print("\n=== Part 2: Parsing into Pydantic Models ===\n")
    raw = await _llm_call(
        "Output a single JSON object {value, type} for the customer contact point "
        "jane@example.com, type email. Inside a ```json block.",
        system="Output only the fenced JSON block. Nothing else.",
        max_tokens=120,
    )
    # ContactPoint: a typed (ChannelType, value) reference the CRM can route on.
    parsed = parse_structured(raw, ContactPoint, strict=False)
    print(f"  Success: {parsed.success}  Parsed: {parsed.parsed}")

    # =========================================================================
    # Part 3: Error handling — strict vs non-strict parsing on bad inputs
    # =========================================================================
    print("\n=== Part 3: Error Handling ===\n")
    bad = await _llm_call(
        "Reply with the literal string: This is not JSON.",
        system="Reply only with the requested string.",
        max_tokens=40,
    )
    bad_result = parse_structured(bad, ContactPoint, strict=False)
    print(f"  Invalid JSON - Success: {bad_result.success}  Error: {bad_result.error}")

    missing_type = await _llm_call(
        "Output a JSON object with only the field value='+1-555-0142', "
        "NO type field. Inside ```json.",
        system="Output only the fenced JSON block.",
        max_tokens=80,
    )
    missing_result = parse_structured(missing_type, ContactPoint, strict=False)
    print(f"  Missing-field - Success: {missing_result.success}  Error: {missing_result.error}")
    try:
        parse_structured("invalid", ContactPoint, strict=True)
    except StructuredOutputError as e:
        print(f"  Strict mode raised {type(e).__name__}")

    # =========================================================================
    # Part 4: Schema prompts — tell the model what JSON shape to produce
    # =========================================================================
    print("\n=== Part 4: Creating Schema Prompts ===\n")
    schema_prompt = create_schema_prompt(Resolution)
    print(f"  schema_prompt (head): {schema_prompt[:160]}...")
    instructions = create_output_instructions(Resolution)
    raw = await _llm_call(
        "Following these instructions, return a JSON for a resolved ticket where "
        "the agent issued a refund for a damaged `wireless-headset` order:\n" + instructions,
        system="Output only a fenced JSON block matching the schema.",
        max_tokens=200,
    )
    out = parse_structured(raw, Resolution, strict=False)
    if out.success:
        print(
            f"  Parsed: resolved={out.parsed.resolved} action='{out.parsed.action}' "
            f"tags={out.parsed.tags}"
        )
    else:
        print(f"  Parse error: {out.error}")

    # =========================================================================
    # Part 5: Nested schemas — Customer contains AccountTier contains primitives
    # =========================================================================
    print("\n=== Part 5: Complex Nested Structures ===\n")
    nested = await _llm_call(
        "Output a JSON for a customer Acme Co, lifetime_value 4800, tier "
        "(plan 'enterprise', region 'us-east', standing 'good'), "
        "products [seats, analytics]. Inside ```json.",
        system="Output only the fenced JSON block.",
        max_tokens=240,
    )
    cust_res = parse_structured(nested, Customer, strict=False)
    if cust_res.success:
        c = cust_res.parsed
        print(f"  Customer: {c.name} (LTV {c.lifetime_value}, {c.tier.plan})")
        print(f"  Products: {', '.join(c.products)}")
    else:
        print(f"  Parse error: {cust_res.error}")

    # =========================================================================
    # Part 6: A typed verdict schema — ConversationVerdict from a chat
    #         transcript. The model maps the conversation to an (intent,
    #         sentiment) read with a confidence. Low evidence_coverage is the
    #         signal a downstream grounding step uses to abstain rather than
    #         auto-route on a guess — see notebook 37.
    # =========================================================================
    print("\n=== Part 6: ConversationVerdict ===\n")
    cv_instructions = create_output_instructions(ConversationVerdict)
    raw = await _llm_call(
        "A live-chat transcript shows a customer whose subscription renewed at a "
        "higher price than expected; they are frustrated and asking to cancel "
        "unless it is fixed. 5 of the 6 expected signals were present in the "
        "transcript. Return the verdict JSON.\n" + cv_instructions,
        system="Output only a fenced JSON block matching the schema.",
        max_tokens=240,
    )
    verdict_res = parse_structured(raw, ConversationVerdict, strict=False)
    if verdict_res.success:
        v = verdict_res.parsed
        print(f"  ConversationVerdict: intent={v.intent} / sentiment={v.sentiment}")
        print(f"  Classifier confidence: {v.confidence:.0%}")
        print(f"  Evidence coverage: {v.evidence_coverage:.0%}")
    else:
        print(f"  Parse error: {verdict_res.error}")

    # =========================================================================
    # Part 7: System-prompt pattern — embed the schema in the system message
    # =========================================================================
    print("\n=== Part 7: Agent ToolSelection prompt ===\n")
    sys_prompt = (
        "You are a customer-support assistant with access to help-desk tools.\n\n"
        + create_output_instructions(ToolSelection)
        + "\nThink before selecting."
    )
    pick = await _llm_call(
        "We need to check the shipping status of order 100847. Pick the right "
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
    print("\n=== Part 8: Agent(output_schema=IssueList) ===\n")
    live_agent = Agent(
        model=get_model(max_tokens=300),
        output_schema=IssueList,
        system_prompt=(
            "You are a support-ticket triager. Report exactly three "
            "distinct issues for the ticket under review as a structured list."
        ),
    )
    t0 = time.perf_counter()
    live = await live_agent.arun(
        "Triage: a customer reports the mobile app crashes on checkout, they were "
        "double-charged for one order, and the help center login link is broken. "
        "Top three issues."
    )
    dt = time.perf_counter() - t0
    print(
        f"  [model call: {dt:.2f}s · "
        f"{live.metrics.prompt_tokens}→{live.metrics.completion_tokens} tokens]"
    )
    report: IssueList | None = live.parsed
    if not isinstance(report, IssueList):
        raise TypeError(
            "Triage agent returned no parsed IssueList. The configured model "
            "could not honor the JSON schema. Use a stronger model "
            "(e.g. openai.gpt-4o, openai.gpt-5, anthropic.claude-3-5-sonnet) "
            f"for notebook 35 (Part 8). Raw output: {live.message!r}"
        )
    for f in report.issues:
        print(f"  {f.title[:40]:<40}  confidence={f.confidence:.2f}  priority={f.priority.value}")

    print("\n" + "=" * 60)
    print("Done. Next: notebook 36 — reasoning patterns.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
