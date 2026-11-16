#!/usr/bin/env python3
# ruff: noqa: T201
"""Capture what a notebook actually prints, and keep the page honest about it.

72 of the 75 notebook pages were prose plus a source include, and the three
that looked like exceptions were showing ASCII architecture diagrams rather
than output. So a site visitor never saw what running an example looks like —
which is most of the persuasive value for the examples whose entire point is
the printed decision trail. `notebook_86` ends on ``chain intact
(tamper-evident): True`` and none of that reached the page.

Output on a page rots the moment the example changes, and rotted output is
worse than none: a reader who runs it and gets something else has no way to
tell which of the two is wrong. So this does both jobs from one place ::

    python scripts/notebook_output.py --update   # re-capture into the pages
    python scripts/notebook_output.py --check    # CI: still matches?

``--check`` runs in the docs build job, which is the one that has the SDK
installed. It pins the same checkout the pages render from, so the output
shown is the output of the source shown.

Only deterministic, offline notebooks belong here. Every entry in
:data:`PAGES` was verified to produce byte-identical output across runs before
being added; one that starts varying will fail ``--check`` loudly rather than
drifting quietly, which is the intended outcome.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs" / "notebooks"

#: Notebook stem -> why its output earns space on the page.
#:
#: Deliberately not "all of them". These are the examples whose printed trail
#: *is* the point: a gate holding, a decision recorded, a probe refused.
PAGES: dict[str, str] = {
    "notebook_03_agent_with_tools": "the tools answering direct calls, the single lookup the mock model makes in each agent run (`Tool calls made: 1`; one `tool_start` / `tool_complete` pair in Part 4) followed by a fixed verdict keyed on the prompt's wording rather than derived from the lookup, and the blockers the structured readiness report lists for the failing service",
    "notebook_08_termination": "the `(stop, reason)` each predicate and composition returns (`max_iterations`, `text_mention:TICKET_RESOLVED`, `max_iterations AND token_budget`, `confidence_met`), so the OR and AND semantics are visible before the agent runs, then the agent's normalized stop reason, the resolution read back from `state.metadata['resolution']` and the two replies the callable system prompt produced",
    "notebook_10_hooks_advanced": "the blocked tool result RELEASE GUARD feeds back to the model, the read-only error that names the writable event fields, and the gate/ledger call sequence reversing on the way out",
    "notebook_15_human_in_the_loop": "the pause-and-resume trail of every gate: where each graph paused, the question it surfaced, the answer given and the resulting state, including the blast-radius routing that auto-approves 2 pods and escalates 20 and 200 to the site reliability engineering (SRE) on-call and the release manager",
    "notebook_27_multiagent_human_in_loop": "the pause-and-resume trail of all three patterns: the node each graph pauses at, the approver's question, the resume value and the final outcome, with Pattern B's escalation routed to the human's answer and Pattern C's snapshot key count",
    "notebook_29_cognitive_router": "a fallback notice for each request (the mock extractor produces no `GoalFrame`, so the hand-built frame is used), the protocol the router selected and the runtime shape it compiled to for the five requests that run, the `PolicyGate` verdict that holds the sixth, HIGH-risk node drain for approval, and a closing protocol histogram",
    "notebook_46_mcp_integration": "the MCP schema derived from a Locus tool and its round trip back with parameters preserved, the `tools/list` catalogue and a `tools/call` result (Part 4 ran because the optional `fastmcp` package was installed for this capture; without it the script prints an install note there instead), then the GSAR ship/abstain trail (SHIP at S=1.0000 on full refund signals, ABSTAIN/REPLAN at S=0.0000 on sparse ones)",
    "notebook_47_plugins": "the plugin's audit trail of model and tool hooks in firing order, the event types a callback handler receives, and the `cancelled` stop reason with the count of tool calls that completed before `Agent.cancel()` took effect",
    "notebook_53_rag_agents": "one skip notice for the four retrieval sections, which need OPENAI_API_KEY, then the answer-grounding gate's decision trail: S=1.0 and proceed for the remediation that cites retrieved runbook chunks, with its evidence references, and S=0.0 and replan for the unsupported cross-incident claim",
    "notebook_55_gsar_typed_grounding": "the admit/withhold trail: a grounded TLS finding shipping with its score arithmetic, an ungrounded and a contradicted candidate withheld with their reasons, the Appendix-B evidence-weight table, the same finding under three threshold profiles, and the Algorithm-1 loop abstaining through its replan budget to a degraded result",
    "notebook_57_multimodal_providers": "the four `isinstance` checks against the provider Protocols, the auto-registered tool names (`web_fetch`, then `web_search`) and the first lines of the example.com page, which Part 3 fetches live, so even this keyless run needs outbound network access",
    "notebook_60_observability_basics": "the bus singleton still `None` after a reply made outside `run_context`, the canonical event trail a tool-calling ticket reply leaves on the bus (`agent.think`, `agent.tool.started`, `agent.tool.completed`, `agent.think`, `agent.terminate`), the two events a late subscriber replays from history after the run has finished, then the span tree `TelemetryHook` records for the same reply under the GenAI semantic conventions (`invoke_agent support-desk` over two `locus.iteration` spans holding the `chat` and `execute_tool lookup_order` spans, token counts on each) with the conventions check passing",
    "notebook_62_eventbus_subscribers": "the event kinds the global telemetry forwarder and the per-deployment subscriber each received across two concurrent rollouts, then the `bus.stats()` snapshot (`active_runs`, `global_subscribers`, `history_runs`, `dropped_events_total`, `queue_depths`)",
    "notebook_63_event_catalogue": "the canonical `event_type` catalogue grouped by prefix (every `EV_*` constant read from the installed `locus.observability.emit` module when the script runs) and then the `composition.*` events the two-stage dispute-triage pipeline publishes, with no model text in the trail",
    "notebook_71_payment_refund_gate": "a small refund paid, a large one held",
    "notebook_72_infra_deploy_gate": "a staging deploy allowed and a production rollback held, then the optional judge letting a pinned staging image through and holding an unpinned one the policy had allowed, with both verdicts on the audit entries",
    "notebook_73_support_account_gate": "a small credit applied, a plan upgrade held, and the trail that records both; then the same upgrade held by a gated tool with the run parked, the hold read back from the thread's checkpoint, approved from another process with `locus approvals`, and performed once by a fresh agent resuming the thread",
    "notebook_74_data_deletion_gate": "the export allowed on its own, the erasure held with nothing deleted until the Data Protection Officer signs off and it is re-admitted, and the three-record trail that verifies intact",
    "notebook_75_cloud_resource_gate": "a staging resize allowed, then a production terminate and an IAM grant held",
    "notebook_80_vendor_integrations": "the offline sample each integration returns: the risk verdict for three identifiers, the two catalog records that hold the subject's data with their sensitivity, and the four PII categories a scan of `crm.customers` finds, then the STEWARD triage result",
}

_MARKER = "## Output"

#: The block this script owns, so ``--update`` replaces rather than accretes.
_BLOCK = re.compile(
    r"\n## Output\n.*?<!-- notebook-output:end -->\n",
    re.DOTALL,
)


def sdk_dir() -> Path:
    """Where the SDK is checked out, matching the docs build's own lookup."""
    for candidate in (
        os.environ.get("LOCUS_SDK_DIR"),
        ROOT / ".sdk",
        ROOT.parent / "locus-sdk",
    ):
        if candidate and Path(candidate).is_dir():
            # Resolved, because the subprocess below runs with ``cwd`` set to
            # this directory: a relative path would then be interpreted
            # relative to itself. CI passes ``LOCUS_SDK_DIR=./.sdk``, which
            # turned into ``.sdk/.sdk/examples/...`` and failed only there.
            return Path(candidate).resolve()
    raise SystemExit("SDK checkout not found — set LOCUS_SDK_DIR")


def notebook_script(stem: str, sdk: Path) -> Path:
    """The notebook's source under the checkout's examples directory."""
    return sdk / "examples" / f"{stem}.py"


def run_notebook(stem: str, sdk: Path) -> str:
    """Run one notebook offline and return exactly what it printed."""
    script = notebook_script(stem, sdk)
    if not script.is_file():
        raise SystemExit(f"{script} not found — is the SDK checkout current?")
    completed = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=script.parent.parent,
        # No provider set: the notebooks fall back to the bundled mock, which
        # is what makes this reproducible and keyless.
        env={
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", ""),
            "PYTHONPATH": str(sdk / "src"),
        },
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(f"{stem} exited {completed.returncode}:\n{completed.stderr[-2000:]}")
    # Trailing whitespace is dropped line by line: a page that carried it would
    # fail the repository's whitespace hook, and it carries no information.
    return "\n".join(line.rstrip() for line in completed.stdout.splitlines()).rstrip("\n")


def uses_model(stem: str, sdk: Path) -> bool:
    """Whether the notebook builds a model at all.

    Most gate notebooks call ``admit()`` directly and never touch a model, so a
    lead-in claiming a "bundled mock model" would describe something the reader
    cannot find in the source. Read it from the source rather than asserting it.
    """
    script = notebook_script(stem, sdk)
    if not script.is_file():
        return False
    source = script.read_text()
    return "get_model" in source or "ScriptedModel" in source


def render(stem: str, why: str, output: str, *, with_model: bool = True) -> str:
    """The page section, with an end marker so ``--update`` can find it again."""
    how = "no credentials, bundled mock model" if with_model else "no model and no credentials"
    return (
        f"\n{_MARKER}\n\n"
        f"Running it offline — {how} — prints "
        f"{why}:\n\n"
        f"```text\n{output}\n```\n"
        f"<!-- notebook-output:end -->\n"
    )


def page_for(stem: str) -> Path:
    path = DOCS / f"{stem}.md"
    if not path.is_file():
        raise SystemExit(f"no page for {stem} at {path}")
    return path


def apply(stem: str, why: str, output: str, *, write: bool, with_model: bool = True) -> bool:
    """Update or verify one page. Returns True when it already matched."""
    path = page_for(stem)
    text = path.read_text()
    section = render(stem, why, output, with_model=with_model)

    if _BLOCK.search(text):
        updated = _BLOCK.sub(lambda _: section, text, count=1)
    else:
        # New section goes before "## Source": read what it does, then read it.
        anchor = "\n## Source\n"
        if anchor not in text:
            raise SystemExit(f"{path.name} has no '## Source' section to anchor against")
        updated = text.replace(anchor, section + anchor, 1)

    if updated == text:
        return True
    if write:
        path.write_text(updated)
    return False


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="fail if a page is stale")
    group.add_argument("--update", action="store_true", help="re-capture into the pages")
    args = parser.parse_args(argv[1:])

    sdk = sdk_dir()
    stale = []
    for stem, why in PAGES.items():
        output = run_notebook(stem, sdk)
        if not apply(stem, why, output, write=args.update, with_model=uses_model(stem, sdk)):
            stale.append(stem)

    if args.update:
        print(f"updated {len(stale)} page(s); {len(PAGES) - len(stale)} already current")
        return 0

    print(f"checked {len(PAGES)} notebook page(s)")
    if not stale:
        print("output on every page matches a real run")
        return 0
    print("\nthese pages show output the notebook no longer produces:\n")
    for stem in stale:
        print(f"  docs/notebooks/{stem}.md")
    print("\nRefresh them with: python scripts/notebook_output.py --update")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
