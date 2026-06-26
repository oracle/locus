# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""
Notebook 11: streaming a data-subject-request (DSAR) analysis.

Every Tulip agent run is a stream of events: thinking, tool start, tool
complete, terminate. ``agent.run(prompt)`` returns an async iterator
that yields them in order — so a privacy analyst can watch a data-subject
request unfold instead of waiting for an opaque verdict. This notebook
walks through the full event set, then shows useful patterns built on
top — a live console UI, an audit log via event filtering, metric
collection, and a progress bar.

Key ideas:
- ``ThinkEvent``: the model produced a plan (and possibly tool calls).
- ``ToolStartEvent`` / ``ToolCompleteEvent``: a tool ran; the complete
  event carries the result and duration.
- ``TerminateEvent``: the loop ended; carries the final message, stop
  reason, and iteration count.
- You filter the stream by ``isinstance(event, EventType)``.
- ``StructuredStream`` (mentioned at the end) wraps the stream and
  yields partial Pydantic instances as JSON arrives.

The scenario is data-privacy triage (GDPR Art. 15 right of access). All
subjects, records, and contact details are fabricated — ``*.example``
domains and 555-01xx phone numbers.

Run it:
    .venv/bin/python examples/notebook_11_agent_streaming.py

The default provider is the mock model; set TULIP_MODEL_PROVIDER for a live one (e.g.
``openai.gpt-4.1`` or ``meta.llama-3.3-70b-instruct``). For an offline
run set ``TULIP_MODEL_PROVIDER=mock``; OpenAI, Anthropic
work too.

Prerequisite: notebook 07.
"""

import asyncio
import re
from datetime import datetime

# Import shared config
from config import get_model, print_config

from tulip.agent import Agent
from tulip.core.events import (
    TerminateEvent,
    ThinkEvent,
    ToolCompleteEvent,
    ToolStartEvent,
)
from tulip.tools import tool


# =============================================================================
# Part 1: every event type the agent emits
# =============================================================================


# Deterministic local extraction — plain regexes, no network calls.
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"\b\d{3}[-.]\d{3}[-.]\d{4}\b")
_SENSITIVE_CUES = ("ssn", "social security", "date of birth", "passport", "credit card")


def _extract_personal_data(text: str) -> dict:
    """Pull emails, phone numbers, and sensitive-data cues out of raw record text."""
    lowered = text.lower()
    return {
        "emails": _EMAIL_RE.findall(text),
        "phones": _PHONE_RE.findall(text),
        "cues": [cue for cue in _SENSITIVE_CUES if cue in lowered],
    }


@tool
def extract_pii(text: str) -> str:
    """Extract email addresses, phone numbers, and sensitive-data cues from raw record text."""
    found = _extract_personal_data(text)
    if not any(found.values()):
        return "No PII or sensitive-data cues found"
    return f"emails={found['emails']} phones={found['phones']} cues={found['cues']}"


async def example_all_events():
    """Print every event with its distinguishing fields."""
    print("=== Part 1: Understanding Events ===\n")

    model = get_model(max_tokens=200)

    agent = Agent(
        model=model,
        tools=[extract_pii],
        system_prompt="You are a data-privacy analyst. Always run extract_pii on the record.",
    )

    record = (
        "Subject record: contact jane.doe@example.com or 555-0142. SSN and date of birth on file."
    )
    print(f"Running: 'Analyze this record: {record}'\n")
    print("Events received:")

    async for event in agent.run(f"Analyze this record: {record}"):
        print(f"\n  Event Type: {event.event_type}")
        print(f"  Timestamp:  {event.timestamp}")

        if isinstance(event, ThinkEvent):
            print(f"  Iteration:  {event.iteration}")
            print(f"  Tool Calls: {len(event.tool_calls)}")
            if event.reasoning:
                preview = (
                    event.reasoning[:80] + "..." if len(event.reasoning) > 80 else event.reasoning
                )
                print(f"  Reasoning:  {preview}")

        elif isinstance(event, ToolStartEvent):
            print(f"  Tool Name:  {event.tool_name}")
            print(f"  Arguments:  {event.arguments}")

        elif isinstance(event, ToolCompleteEvent):
            print(f"  Tool Name:  {event.tool_name}")
            print(f"  Result:     {event.result}")
            print(f"  Duration:   {event.duration_ms:.1f}ms")

        elif isinstance(event, TerminateEvent):
            print(f"  Reason:     {event.reason}")
            print(f"  Iterations: {event.iterations_used}")
            if event.final_message:
                print(f"  Answer:     {event.final_message}")

    print()


# =============================================================================
# Part 2: a live console UI driven by the event stream
# =============================================================================


async def example_console_ui():
    """Render a tiny progress UI as the agent fetches and checks the record."""
    print("=== Part 2: Console UI ===\n")

    model = get_model(max_tokens=200)

    @tool
    def fetch_record(subject_id: str) -> str:
        """Fetch a data-subject record by subject id."""
        # Sleep so the UI has visible work to show.
        import time

        time.sleep(0.5)
        return f"{subject_id}: jane.doe@example.com, marketing profile, last updated 2026-01-12"

    @tool
    def check_consent(data_category: str) -> str:
        """Check the consent status for a data category."""
        return f"{data_category}: consent withdrawn 9 days ago, retention basis expired"

    agent = Agent(
        model=model,
        tools=[fetch_record, check_consent],
        system_prompt="You are a DSAR triage assistant. Fetch records and check consent.",
    )

    print("Query: Triage data-subject request DSR-1042 and check its marketing consent\n")

    async for event in agent.run("Triage data-subject request DSR-1042 and check its consent"):
        if isinstance(event, ThinkEvent):
            print("[Thinking...]")
            if event.tool_calls:
                for tc in event.tool_calls:
                    print(f"  Planning to call: {tc.name}")

        elif isinstance(event, ToolStartEvent):
            print(f"[Running] {event.tool_name}...", end=" ", flush=True)

        elif isinstance(event, ToolCompleteEvent):
            if event.error:
                print(f"ERROR: {event.error}")
            else:
                print(f"Done ({event.duration_ms:.0f}ms)")

        elif isinstance(event, TerminateEvent):
            print(f"\n[Complete] {event.reason}")
            if event.final_message:
                print(f"\nVerdict: {event.final_message}")

    print()


# =============================================================================
# Part 3: pick out just the events you care about
# =============================================================================


async def example_event_filtering():
    """Build an audit log by keeping only ToolStart/ToolComplete events."""
    print("=== Part 3: Event Filtering ===\n")

    model = get_model(max_tokens=200)

    agent = Agent(
        model=model,
        tools=[extract_pii],
        system_prompt="Use the extract_pii tool on any record text you're given.",
    )

    tool_log = []

    async for event in agent.run(
        "Extract PII from: 'Reach me at john.roe@example.org or 555-0188, passport number on file'"
    ):
        if isinstance(event, ToolStartEvent):
            tool_log.append(
                {
                    "action": "start",
                    "tool": event.tool_name,
                    "args": event.arguments,
                    "time": datetime.now().isoformat(),
                }
            )

        elif isinstance(event, ToolCompleteEvent):
            tool_log.append(
                {
                    "action": "complete",
                    "tool": event.tool_name,
                    "result": event.result,
                    "duration_ms": event.duration_ms,
                }
            )

    print("Tool execution audit log:")
    for entry in tool_log:
        print(f"  {entry}")
    print()


# =============================================================================
# Part 4: roll up metrics from the event stream
# =============================================================================


async def example_collect_metrics():
    """Count events and sum tool durations to get per-run telemetry."""
    print("=== Part 4: Collecting Metrics ===\n")

    model = get_model(max_tokens=200)

    @tool
    def parse_request(data: str) -> str:
        """First processing step — parse the request and locate the subject."""
        return f"Parsed request and located subject for: {data}"

    @tool
    def classify_request(data: str) -> str:
        """Second processing step — classify as access, rectification, or erasure."""
        return f"Classified: {data} is a right-to-erasure request under Art. 17"

    agent = Agent(
        model=model,
        tools=[parse_request, classify_request],
        system_prompt="Process requests through parse_request then classify_request.",
    )

    metrics = {
        "think_events": 0,
        "tool_starts": 0,
        "tool_completes": 0,
        "total_tool_time_ms": 0,
        "iterations": 0,
    }

    start_time = datetime.now()

    async for event in agent.run("Run the data-subject request 'DSR-1043' through both steps"):
        if isinstance(event, ThinkEvent):
            metrics["think_events"] += 1
        elif isinstance(event, ToolStartEvent):
            metrics["tool_starts"] += 1
        elif isinstance(event, ToolCompleteEvent):
            metrics["tool_completes"] += 1
            metrics["total_tool_time_ms"] += event.duration_ms or 0
        elif isinstance(event, TerminateEvent):
            metrics["iterations"] = event.iterations_used

    elapsed = (datetime.now() - start_time).total_seconds() * 1000

    print("Execution Metrics:")
    print(f"  Think events:    {metrics['think_events']}")
    print(f"  Tool starts:     {metrics['tool_starts']}")
    print(f"  Tool completes:  {metrics['tool_completes']}")
    print(f"  Tool time:       {metrics['total_tool_time_ms']:.1f}ms")
    print(f"  Iterations:      {metrics['iterations']}")
    print(f"  Total time:      {elapsed:.1f}ms")
    print()


# =============================================================================
# Part 5: a progress bar from ToolCompleteEvent
# =============================================================================


async def example_progress_tracking():
    """Draw a textual progress bar as each processing step completes."""
    print("=== Part 5: Progress Tracking ===\n")

    model = get_model(max_tokens=300)

    # Deterministic tool bodies: fetch_records returns a real summary of
    # an in-memory subject store, scan_records hashes its input. No canned
    # strings, no network — all subject data below is invented.
    sources = {
        "pending": [
            {"id": "DSR-1042", "subject": "jane.doe@example.com", "type": "access"},
            {"id": "DSR-1043", "subject": "john.roe@example.org", "type": "erasure"},
            {"id": "DSR-1044", "subject": "sam.lee@example.net", "type": "rectification"},
        ],
        "escalated": [
            {"id": "DSR-0991", "subject": "alex.kim@example.com", "type": "erasure"},
            {"id": "DSR-0987", "subject": "pat.ray@example.org", "type": "access"},
        ],
    }

    @tool
    def fetch_records(source: str) -> str:
        """Fetch requests from a named queue (pending, escalated, ...)."""
        import json
        import time

        time.sleep(0.3)
        rows = sources.get(source.lower())
        if rows is None:
            return f"unknown queue {source!r}; try one of: {sorted(sources)}"
        return f"{len(rows)} request(s) from {source}: {json.dumps(rows)}"

    @tool
    def scan_records(data: str) -> str:
        """Scan fetched records — counts bytes and emits a checksum for the case file."""
        import hashlib
        import time

        time.sleep(0.2)
        digest = hashlib.sha256(data.encode()).hexdigest()[:12]
        return f"scanned {len(data)} bytes, sha256:{digest}"

    @tool
    def file_report(results: str) -> str:
        """File the triage report in the case system."""
        import time

        time.sleep(0.1)
        return "Triage report filed successfully"

    agent = Agent(
        model=model,
        tools=[fetch_records, scan_records, file_report],
        system_prompt="Fetch requests from 'pending', scan them, and file a report.",
    )

    steps_done = 0
    total_steps = 3  # we expect three tool calls: fetch, scan, file

    print("Processing pending queue...")

    async for event in agent.run("Fetch, scan, and report on the pending queue"):
        if isinstance(event, ToolCompleteEvent):
            steps_done += 1
            progress = (steps_done / total_steps) * 100
            bar = "#" * int(progress / 10) + "-" * (10 - int(progress / 10))
            print(f"  [{bar}] {progress:.0f}% - {event.tool_name} complete")

        elif isinstance(event, TerminateEvent):
            print(f"\nDone! Final message: {event.final_message}")

    print()


# =============================================================================
# Main
# =============================================================================


def main():
    """Run all notebook parts."""
    print("=" * 60)
    print("Notebook 11: Streaming a Data-Subject-Request Analysis")
    print("=" * 60)
    print()

    print_config()
    print()

    asyncio.run(example_all_events())
    asyncio.run(example_console_ui())
    asyncio.run(example_event_filtering())
    asyncio.run(example_collect_metrics())
    asyncio.run(example_progress_tracking())

    # =========================================================================
    # See also: structured streaming
    # =========================================================================
    print("=== See also: StructuredStream ===\n")
    print(
        "Want incremental Pydantic instances instead of raw chunks?\n"
        "Wrap any agent stream with StructuredStream:\n"
    )
    print(
        """    from tulip.streaming import StructuredStream

    stream = StructuredStream(agent.run("Top 3 PII items in this record."), schema=PiiList)
    async for partial in stream:
        ui.render(partial)               # may have 0, 1, 2, then 3 items
    final: PiiList | None = stream.final
"""
    )
    print(
        "Tulip buffers each ModelChunkEvent, closes any unbalanced JSON\n"
        "braces / brackets / strings, runs the result through\n"
        "schema.model_validate, and yields the parsed instance if it\n"
        "validates.\n"
    )

    print("=" * 60)
    print("Next: Notebook 12 — Audit Hooks")
    print("=" * 60)


if __name__ == "__main__":
    main()
