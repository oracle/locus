# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""
Notebook 08: investigation memory — the Vault, backed by Redis.

Give an analyst copilot a checkpointer and every turn of the
investigation is written to the Vault: the SOC's durable store of
checkpointed case state. Restart the process, attach a fresh agent to
the same connection, hand it the same ``thread_id``, and the case picks
up where it left off — earlier alerts, indicators, and notes intact.
This is the production memory story for Tulip.

Key ideas:
- ``RedisBackend(...)`` opens a connection to Redis and stores agent
  state under keys you scope.
- ``thread_id`` keys investigations — one Redis can host many
  independent cases side by side without cross-contamination.
- ``checkpoint_every_n_iterations=1`` writes after every loop iteration,
  so a crash mid-tool-call still resumes cleanly.
- The saved state is rich: messages, tool history, iteration count,
  Reflexion confidence — all loadable for case review.
- Findings can be typed: ``tulip.security.Indicator`` gives each
  recorded observable a ``type`` + ``value`` instead of a bare string.

Run it:
    export REDIS_URL=redis://localhost:6379/0
    python examples/notebook_08_agent_memory.py

The agent itself goes through whichever provider you configure via
``TULIP_MODEL_PROVIDER`` (``openai`` / ``anthropic``).
Set ``TULIP_MODEL_PROVIDER=mock`` to bypass the model for offline runs.

If REDIS_URL is missing the script prints a skip banner and
exits cleanly — it never falls back to an in-memory checkpointer.
"""

import asyncio
import os
import sys

from config import get_model, print_config

from tulip.agent import Agent
from tulip.memory.backends import RedisBackend
from tulip.security import Indicator, IndicatorType
from tulip.tools import tool


_REQUIRED_ENV = ("REDIS_URL",)


def _missing_env() -> list[str]:
    return [name for name in _REQUIRED_ENV if not os.environ.get(name)]


def _build_checkpointer(table_suffix: str = "default"):
    """Build a Redis-backed checkpointer described by env vars."""
    backend = RedisBackend(
        url=os.environ["REDIS_URL"],
        namespace=f"tulip_notebook_08_{table_suffix}",
    )
    return backend.as_checkpointer()


# =============================================================================
# Part 1: a two-turn investigation that survives in Redis
# =============================================================================


def example_conversation_memory():
    """Same thread_id across two run_sync calls — the copilot recalls turn one."""
    print("=== Part 1: Investigation memory (Redis) ===\n")

    model = get_model(max_tokens=100)
    checkpointer = _build_checkpointer("case")

    agent = Agent(
        model=model,
        system_prompt="You are a SOC analyst copilot. Remember case details the analyst shares.",
        checkpointer=checkpointer,
    )

    thread_id = "case_soc_1042"

    result1 = agent.run_sync(
        "We're investigating alert SOC-1042: a suspicious login from 198.51.100.7.",
        thread_id=thread_id,
    )
    print("Analyst: We're investigating alert SOC-1042: a suspicious login from 198.51.100.7.")
    print(f"Copilot: {result1.message}")

    # Same thread_id — the agent loads prior case state from Redis
    # before the next model call.
    result2 = agent.run_sync("Which IP are we investigating?", thread_id=thread_id)
    print("\nAnalyst: Which IP are we investigating?")
    print(f"Copilot: {result2.message}")
    print()


# =============================================================================
# Part 2: write a checkpoint after every iteration
# =============================================================================


_CASE_NOTES: list[str] = []


@tool
def save_case_note(content: str) -> str:
    """Save an investigation note for later reference."""
    _CASE_NOTES.append(content)
    return f"Case note saved: {content}"


@tool
def get_case_notes() -> str:
    """Get all saved investigation notes."""
    if not _CASE_NOTES:
        return "No case notes saved yet."
    lines = "\n".join(f"- {n}" for n in _CASE_NOTES)
    return f"You have {len(_CASE_NOTES)} case note(s):\n{lines}"


def example_checkpointing_with_tools():
    """checkpoint_every_n_iterations=1 means a mid-loop crash still recovers."""
    print("=== Part 2: Checkpoint after each iteration ===\n")

    model = get_model(max_tokens=150)
    checkpointer = _build_checkpointer("notes")

    agent = Agent(
        model=model,
        tools=[save_case_note, get_case_notes],
        system_prompt="You are an incident note-taking assistant.",
        checkpointer=checkpointer,
        checkpoint_every_n_iterations=1,
    )

    thread_id = "case_notes_session"

    result1 = agent.run_sync(
        "Save a case note: 198.51.100.7 also appears in alert SOC-0991",
        thread_id=thread_id,
    )
    print("Analyst: Save a case note: 198.51.100.7 also appears in alert SOC-0991")
    print(f"Copilot: {result1.message}")
    print(f"Tool calls: {result1.metrics.tool_calls}")

    result2 = agent.run_sync("What case notes do we have so far?", thread_id=thread_id)
    print("\nAnalyst: What case notes do we have so far?")
    print(f"Copilot: {result2.message}")
    print()


# =============================================================================
# Part 3: a fresh agent picks up where the old one stopped
# =============================================================================


def example_persistence_across_processes():
    """Two Agent objects, one Redis. The second loads the first's case state."""
    print("=== Part 3: Cross-process persistence ===\n")

    model = get_model(max_tokens=100)

    agent1 = Agent(
        model=model,
        system_prompt="You are a SOC analyst copilot.",
        checkpointer=_build_checkpointer("persist"),
    )

    thread_id = "persistent_case"

    result1 = agent1.run_sync(
        "Remember: the incident ticket for this investigation is IR-2026-042.",
        thread_id=thread_id,
    )
    print("Analyst: Remember: the incident ticket for this investigation is IR-2026-042.")
    print(f"Copilot: {result1.message}")

    # Pretend the analyst's shift ended and the process restarted. A *new*
    # Agent + checkpointer object connects to the same Redis and reads
    # back the case state.
    agent2 = Agent(
        model=model,
        system_prompt="You are a SOC analyst copilot.",
        checkpointer=_build_checkpointer("persist"),
    )

    result2 = agent2.run_sync("What was the incident ticket number?", thread_id=thread_id)
    print("\n[New process — same Redis]")
    print("Analyst: What was the incident ticket number?")
    print(f"Copilot: {result2.message}")
    print()


# =============================================================================
# Part 4: many investigations sharing one database
# =============================================================================


def example_multiple_threads():
    """Two cases, two thread_ids, one Redis — no cross-talk between them."""
    print("=== Part 4: Multiple investigations ===\n")

    model = get_model(max_tokens=100)
    checkpointer = _build_checkpointer("multi")

    agent = Agent(
        model=model,
        system_prompt="You are a SOC analyst copilot.",
        checkpointer=checkpointer,
    )

    thread_phishing = "case_phishing"
    thread_malware = "case_malware"

    agent.run_sync(
        "This case is about a phishing email from phish.example.net.",
        thread_id=thread_phishing,
    )
    agent.run_sync(
        "This case is about a quarantined binary with hash aa11bb22cc33dd44.",
        thread_id=thread_malware,
    )

    result_phishing = agent.run_sync("What is this case about?", thread_id=thread_phishing)
    print("Thread 'case_phishing': What is this case about?")
    print(f"Copilot: {result_phishing.message}")

    result_malware = agent.run_sync("What is this case about?", thread_id=thread_malware)
    print("\nThread 'case_malware': What is this case about?")
    print(f"Copilot: {result_malware.message}")
    print()


# =============================================================================
# Part 5: load and walk a saved checkpoint
# =============================================================================


async def example_inspect_checkpoint():
    """Load the persisted AgentState directly and look at every field.

    Reflexion's confidence score only moves when tools succeed, so the
    inspector gets a ``record_finding`` tool. After two turns that each
    fire a tool call, ``state.confidence`` should be > 0.
    """
    print("=== Part 5: Inspecting a Redis-backed checkpoint ===\n")

    model = get_model(max_tokens=200)
    checkpointer = _build_checkpointer("inspect")

    _findings: list[str] = []
    _indicators: list[Indicator] = []

    @tool
    def record_finding(finding: str, indicator: str = "", indicator_type: str = "") -> str:
        """Persist an observation from the investigation so the case record keeps it.

        Args:
            finding: Short summary of the observation.
            indicator: An optional IOC value (e.g. an IP or domain) to type.
            indicator_type: One of ip, domain, url, sha256, md5, email, file_path.
        """
        _findings.append(finding)
        # Type the IOC instead of leaving it as a bare string. Unknown
        # types fall back to recording the finding text only.
        if indicator and indicator_type in {t.value for t in IndicatorType}:
            _indicators.append(Indicator(type=IndicatorType(indicator_type), value=indicator))
        return f"Recorded finding #{len(_findings)}: {finding}"

    agent = Agent(
        agent_id="case_inspector",
        model=model,
        system_prompt=(
            "You are a SOC analyst copilot. Whenever the analyst reports an "
            "observation (an IOC, affected account, suspicious event, etc.), "
            "call record_finding exactly once with a short summary of that "
            "observation — and the indicator value and type when one is "
            "present — then reply naturally."
        ),
        tools=[record_finding],
        checkpointer=checkpointer,
        reflexion=True,
    )

    thread_id = "inspect_case"

    agent.run_sync(
        "Observed: 3 failed logins from 198.51.100.7 before a successful one.",
        thread_id=thread_id,
    )
    agent.run_sync(
        "Observed: the targeted account belongs to a finance-team service user.",
        thread_id=thread_id,
    )

    state = await checkpointer.load(thread_id)

    if state:
        print(f"Thread ID: {thread_id}")
        print(f"Agent ID: {state.agent_id}")
        print(f"Iteration: {state.iteration}")
        print(f"Message count: {len(state.messages)}")
        print(f"Tool calls so far: {len(state.tool_history)}")
        print(f"Confidence: {state.confidence:.2f}")
        print(f"Confidence history: {[round(c, 2) for c in state.confidence_history]}")

        print("\nMessages:")
        for i, msg in enumerate(state.messages):
            content = (
                msg.content[:50] + "..." if msg.content and len(msg.content) > 50 else msg.content
            )
            print(f"  {i}. [{msg.role.value}] {content}")

    print(f"\nFindings recorded by record_finding: {_findings}")
    if _indicators:
        print("Typed indicators (tulip.security.Indicator):")
        for ind in _indicators:
            print(f"  - {ind.type.value}: {ind.value}")
    print()


# =============================================================================
# Main
# =============================================================================


def _print_skip_banner(missing: list[str]) -> None:
    print("\n--- Notebook 08: The Vault — Investigation Memory ---")
    print(
        "Required environment variables not set; skipping the live demo "
        "so this file still runs cleanly in CI.\n"
    )
    for name in missing:
        print(f"  - {name}")
    print("\nStart a Redis instance, set REDIS_URL, and re-run.")


def main():
    """Run all notebook parts."""
    print("=" * 60)
    print("Notebook 08: The Vault — Investigation Memory on Redis")
    print("=" * 60)
    print()

    missing = _missing_env()
    if missing:
        _print_skip_banner(missing)
        return

    print_config()
    print()

    example_conversation_memory()
    example_checkpointing_with_tools()
    example_persistence_across_processes()
    example_multiple_threads()
    asyncio.run(example_inspect_checkpoint())

    print("=" * 60)
    print("Next: Notebook 11 — Streaming a Phishing Analysis")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
