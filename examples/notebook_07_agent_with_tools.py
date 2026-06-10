# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""
Notebook 07: IOC enrichment with tools.

A model without tools can only guess about indicators from what's
already in its context — and guessed verdicts are how false positives
are born. Tools let the agent reach out — look up a hash, pull WHOIS
data — and bring real evidence back into the conversation. Tulip runs
this as a small ReAct loop: the model decides whether to call a tool,
Tulip runs the tool, the result is fed back into the next model call.

Key ideas:
- ``@tool`` turns a plain Python function into something the model can
  call. The docstring is the description the model sees.
- Pass tools to ``Agent(tools=[...])`` and the agent picks when to use
  them.
- Each tool call shows up as a ``ToolStartEvent`` / ``ToolCompleteEvent``
  pair in the event stream — an auditable record of every lookup.
- Tools can take typed arguments (including optional ones) and return
  anything JSON-serialisable — strings, dicts, lists.

The indicators here are benign by design: clearly fake hashes, RFC 5737
documentation IPs, and ``*.example`` domains. The credential-stuffing
sightings map to ATT&CK T1110.004.

Run it:
    .venv/bin/python examples/notebook_07_agent_with_tools.py

The default provider is the mock model; set TULIP_MODEL_PROVIDER for a live one (e.g.
``openai.gpt-4.1`` or ``meta.llama-3.3-70b-instruct``). Drop in
``TULIP_MODEL_PROVIDER=mock`` for an offline run. Tool-calling also
works against OpenAI, Anthropic.

Prerequisite: notebook 06.
"""

import asyncio
from datetime import datetime

# Import shared config
from config import get_model, print_config

from tulip.agent import Agent
from tulip.tools import tool


# =============================================================================
# Part 1: define an enrichment tool
# =============================================================================

# A tool is a plain Python function decorated with @tool. The docstring
# is what the model reads to decide when to call it. All intel data
# below is invented — clearly fake hashes and RFC 5737 / .example IOCs.


@tool
def lookup_hash(sha256: str) -> str:
    """Look up a file hash in the malware intel database."""
    known = {
        "aa11bb22cc33dd44": "EICAR test file — flagged by 60/70 engines",
        "aa11aa11aa11aa11": "FakeLoader sample (test corpus) — flagged by 41/70 engines",
    }
    return known.get(sha256.lower(), f"Hash {sha256} not present in any intel feed")


@tool
def whois_domain(domain: str) -> str:
    """Look up WHOIS registration data for a domain."""
    records = {
        "evil.example": "registered 12 days ago, registrant redacted, NS ns1.cheaphost.example",
        "phish.example.net": "registered 3 days ago, registrant redacted, free TLS certificate",
    }
    return records.get(domain.lower(), f"{domain}: registered 2014, corporate registrant on file")


def example_simple_tools():
    """Show the tool metadata Tulip generates from a decorated function."""
    print("=== Part 1: Simple Tools ===\n")

    result = lookup_hash("aa11bb22cc33dd44")
    print(f"Direct call: lookup_hash('aa11bb22cc33dd44') = {result}")

    print(f"\nTool name: {lookup_hash.name}")
    print(f"Tool description: {lookup_hash.description}")
    print(f"Tool parameters: {lookup_hash.parameters}")

    import time as _t

    agent = Agent(
        model=get_model(max_tokens=80),
        system_prompt="Reply in one short sentence.",
    )
    t0 = _t.perf_counter()
    desc = agent.run_sync(
        f"In one sentence, when would a SOC agent use a tool called '{lookup_hash.name}' "
        f"that {lookup_hash.description}?"
    )
    dt = _t.perf_counter() - t0
    print(
        f"  [model call: {dt:.2f}s · "
        f"{desc.metrics.prompt_tokens}→{desc.metrics.completion_tokens} tokens]"
    )
    print(f"  AI commentary: {desc.message.strip()}")
    print()


# =============================================================================
# Part 2: hand tools to an enrichment agent
# =============================================================================


def example_agent_with_tools():
    """Wire tools into an Agent and let the model decide when to call them."""
    print("=== Part 2: Agent Using Tools ===\n")

    model = get_model(max_tokens=200)

    agent = Agent(
        model=model,
        tools=[lookup_hash, whois_domain],
        system_prompt="You are an IOC-enrichment assistant. Use the provided tools to look up "
        "indicators before giving a verdict.",
    )

    print(f"Agent has {len(agent.tools)} tools registered")

    result = agent.run_sync("Is the hash aa11bb22cc33dd44 known malware?")
    print("\nQ: Is the hash aa11bb22cc33dd44 known malware?")
    print(f"A: {result.message}")
    print(f"Tool calls made: {result.metrics.tool_calls}")
    print()


# =============================================================================
# Part 3: tools with optional and typed arguments
# =============================================================================


@tool
def get_current_time() -> str:
    """Get the current date and time, for case-log timestamps."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def certificate_age(issued_year: int) -> str:
    """Calculate how old a TLS certificate is given its issuance year."""
    current_year = datetime.now().year
    age = current_year - issued_year
    return f"A certificate issued in {issued_year} is {age} years old."


@tool
def format_case_title(indicator: str, urgent: bool = False) -> str:
    """Create a case title for an indicator under investigation.

    Args:
        indicator: The IOC the case is about
        urgent: Whether to flag the case as urgent (default: False)
    """
    if urgent:
        return f"[URGENT] Investigation: {indicator} — immediate triage required."
    return f"Investigation: {indicator} — routine triage."


def example_complex_tools():
    """Tools with default arguments and varied return types."""
    print("=== Part 3: Complex Tools ===\n")

    model = get_model(max_tokens=200)

    agent = Agent(
        model=model,
        tools=[get_current_time, certificate_age, format_case_title],
        system_prompt="You are a SOC assistant with access to time and case-management tools.",
    )

    prompts = [
        "What time is it right now? I need it for the case log.",
        "How old is a TLS certificate issued in 2019?",
        "Give me an urgent case title for the domain evil.example",
    ]

    for prompt in prompts:
        result = agent.run_sync(prompt)
        print(f"Q: {prompt}")
        print(f"A: {result.message}")
        print()


# =============================================================================
# Part 4: watch enrichment lookups happen in the event stream
# =============================================================================


async def example_tool_events():
    """Stream events to see the model plan, call a tool, and use its evidence."""
    print("=== Part 4: Tool Execution Events ===\n")

    model = get_model(max_tokens=200)

    agent = Agent(
        model=model,
        tools=[lookup_hash, whois_domain],
        system_prompt="Use tools to enrich indicators. Always look up hashes and domains "
        "before answering.",
    )

    print("Q: Enrich hash aa11bb22cc33dd44 and domain evil.example, then give a verdict.\n")
    print("Events:")

    async for event in agent.run(
        "Enrich hash aa11bb22cc33dd44 and domain evil.example, then give a verdict."
    ):
        event_type = event.event_type

        if event_type == "tool_start":
            print(f"  TOOL_START: {event.tool_name}({event.arguments})")
        elif event_type == "tool_complete":
            print(f"  TOOL_COMPLETE: {event.tool_name} -> {event.result}")
        elif event_type == "think":
            if event.tool_calls:
                print(f"  THINK: Planning to call {len(event.tool_calls)} tool(s)")
        elif event_type == "terminate":
            print(f"  TERMINATE: {event.reason}")
            if event.final_message:
                print(f"\nFinal Answer: {event.final_message}")

    print()


# =============================================================================
# Part 5: tools that return structured data
# =============================================================================


@tool
def search_iocs(query: str, max_results: int = 3) -> list[dict]:
    """Search for indicators of compromise in the intel catalogue.

    Args:
        query: Search query (matches indicator or type)
        max_results: Maximum number of results to return
    """
    # In-memory catalogue stands in for a threat-intel platform. The
    # search logic below is the part worth reading. All IOCs are fake.
    iocs = [
        {"id": 1, "indicator": "198.51.100.7", "type": "ip", "verdict": "suspicious"},
        {
            "id": 2,
            "indicator": "evil.example",
            "type": "domain",
            "verdict": "malicious",
        },
        {"id": 3, "indicator": "203.0.113.9", "type": "ip", "verdict": "suspicious"},
        {"id": 4, "indicator": "phish.example.net", "type": "domain", "verdict": "malicious"},
        {"id": 5, "indicator": "aa11bb22cc33dd44", "type": "hash", "verdict": "malicious"},
        {"id": 6, "indicator": "192.0.2.55", "type": "ip", "verdict": "benign"},
        {
            "id": 7,
            "indicator": "corp.example",
            "type": "domain",
            "verdict": "benign",
        },
        {
            "id": 8,
            "indicator": "aa11aa11aa11aa11",
            "type": "hash",
            "verdict": "malicious",
        },
    ]

    # Case-insensitive match on indicator OR type.
    q = query.lower()
    matches = [i for i in iocs if q in i["indicator"].lower() or q in i["type"].lower()]
    return matches[:max_results]


@tool
def get_ioc_details(ioc_id: int) -> dict:
    """Get detailed intel about a specific indicator of compromise."""
    details = {
        1: {
            "id": 1,
            "indicator": "198.51.100.7",
            "verdict": "suspicious",
            "notes": "seen in 2 credential-stuffing campaigns (ATT&CK T1110.004), last 30 days",
        },
        2: {
            "id": 2,
            "indicator": "evil.example",
            "verdict": "malicious",
            "notes": "phishing landing pages, registered 12 days ago",
        },
        3: {
            "id": 3,
            "indicator": "203.0.113.9",
            "verdict": "suspicious",
            "notes": "scanning activity reported by 3 partner orgs",
        },
        4: {
            "id": 4,
            "indicator": "phish.example.net",
            "verdict": "malicious",
            "notes": "credential-harvesting kit",
        },
        5: {"id": 5, "indicator": "aa11bb22cc33dd44", "verdict": "malicious", "notes": "EICAR"},
        6: {
            "id": 6,
            "indicator": "192.0.2.55",
            "verdict": "benign",
            "notes": "documentation range, no sightings",
        },
        7: {
            "id": 7,
            "indicator": "corp.example",
            "verdict": "benign",
            "notes": "company-owned domain, allowlisted",
        },
        8: {
            "id": 8,
            "indicator": "aa11aa11aa11aa11",
            "verdict": "malicious",
            "notes": "test-corpus loader sample",
        },
    }
    return details.get(ioc_id, {"error": f"IOC {ioc_id} not found"})


def example_structured_tools():
    """Tools can return dicts and lists — the model parses them on the next turn."""
    print("=== Part 5: Structured Data Tools ===\n")

    model = get_model(max_tokens=300)

    agent = Agent(
        model=model,
        tools=[search_iocs, get_ioc_details],
        system_prompt="You are a threat-intel assistant. Help analysts look up indicators.",
    )

    result = agent.run_sync("Find domain IOCs, then tell me more about evil.example")
    print("Q: Find domain IOCs, then tell me more about evil.example")
    print(f"A: {result.message}")
    print(f"\nTool calls made: {result.metrics.tool_calls}")
    print()


# =============================================================================
# Main
# =============================================================================


def main():
    """Run all notebook parts."""
    print("=" * 60)
    print("Notebook 07: IOC Enrichment with Tools")
    print("=" * 60)
    print()

    print_config()
    print()

    example_simple_tools()
    example_agent_with_tools()
    example_complex_tools()
    asyncio.run(example_tool_events())
    example_structured_tools()

    print("=" * 60)
    print("Next: Notebook 08 — Investigation Memory")
    print("=" * 60)


if __name__ == "__main__":
    main()
