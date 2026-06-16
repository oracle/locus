# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Live integration test for #280 — empty-content safety net.

Reproduces the field bug shape against OCI Gemini 2.5 Flash:

  - Long system prompt (5K-token CONTEXT BLOCK padding)
  - One trivial tool the model rarely needs
  - User question that the model can answer briefly

Before the fix in src/locus/agent/runtime_loop.py the safety-net branch:
the model would return content=None or content='' on iteration 1
(Gemini 2.5 Flash's thinking-mode behavior under long-context loads),
the runtime would emit ``TerminateEvent(final_message=None)``, and
AgentResult.text would land as ''.

After the fix: the runtime detects the empty content, injects a
"[Final answer requested]" system message, forces one no-tool
completion, and that response lands as result.text. The operator
sees a real answer end-to-end.

Gated by ``RUN_LIVE_OCI=1`` so CI doesn't burn tokens.

Cost guard: 4-6 iterations max, 2K-token outputs. ~1-2 cents per run.
"""

from __future__ import annotations

import os

import pytest

from locus import create_deepagent
from locus.tools.decorator import tool


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_OCI") != "1",
    reason="Live OCI tests gated by RUN_LIVE_OCI=1 to avoid CI token spend",
)


@tool
def list_db_tables() -> str:
    """Return a hard-coded table list to give the model something it
    can choose to call (or not). The bug we're testing is the path
    where the model writes content without tool calls — we want a
    tool here so the agent loop has the option."""
    return "Tables available:\n- iron_metabolism_v1 (45 docs)\n- sre_knowledge_v2 (1,240 docs)\n"


def _build_chat_model():
    """Build the OCI Gemini 2.5 Flash model exactly the way the
    gists do (matches Federico's API_FREE_TIER setup per his
    CLAUDE.md)."""
    from locus.models import get_model

    profile = os.environ.get("OCI_PROFILE", "API_FREE_TIER")
    region = os.environ.get("OCI_REGION", "us-chicago-1")
    compartment = os.environ.get("OCI_COMPARTMENT")
    if not compartment:
        compartment = os.environ.get(
            "OCI_TENANCY",
            "ocid1.tenancy.oc1..aaaaaaaaqlhpnytg33ztkwrdpq62p5yxx5gn5ltmkah23m7qebwjzc7x3lcq",
        )
    return get_model(
        "oci:google.gemini-2.5-flash",
        profile=profile,
        region=region,
        compartment_id=compartment,
    )


# ---------------------------------------------------------------------------
# The bug-shape reproduction
# ---------------------------------------------------------------------------


def test_long_prompt_gemini_safety_net_produces_real_output() -> None:
    """Long-context + Gemini 2.5 Flash. Without the safety net this
    returns empty text. With the fix, ``result.text`` must contain
    the model's actual answer (or at least the fallback summary)."""
    model = _build_chat_model()
    agent = create_deepagent(
        model=model,
        tools=[list_db_tables],
        # ~5K tokens of padding — the field shape that surfaces the
        # bug. Tuned high enough to trigger Gemini's thinking-mode
        # branch reliably on a free-tier endpoint.
        system_prompt=(
            "You are a deep research agent. Use list_db_tables if "
            "you need it; otherwise answer directly. " + ("CONTEXT BLOCK. " * 700)
        ),
        reflexion=False,
        grounding=False,
        max_iterations=4,
        max_output_tokens=2048,
        # NOTE: no output_schema (Gemini rejects additionalProperties —
        # tracked in #281; out of scope here)
        # NOTE: no total_token_budget (would attach TokenLimit and
        # confound the test with #278's bug)
    )
    result = agent.run_sync("What tables are in the database?")
    text = getattr(result, "text", "") or ""
    last_msg = getattr(result, "last_assistant_message", "") or ""
    metrics = getattr(result, "metrics", None)

    # The fix's contract: result.text MUST contain real content,
    # whether from the model's direct response on iteration 1 OR
    # from the safety-net's forced summary call. Both paths are OK;
    # what we're testing is that the FINAL output is non-empty.
    assert text or last_msg, (
        "After the safety-net fix, a long-prompt Gemini run must "
        "produce non-empty result.text (or at minimum land the "
        "answer on last_assistant_message). Empty output across "
        "both indicates the safety net didn't fire OR the fallback "
        "summary also failed. "
        f"text_len={len(text)}, last_msg_len={len(last_msg)}, "
        f"metrics={metrics!r}"
    )

    # Sanity check: completion_tokens > 0 means the model actually
    # generated something. If we have completion_tokens but empty
    # text, the bug is still firing somewhere.
    if metrics is not None and getattr(metrics, "completion_tokens", 0) > 0:
        # We got real tokens; the safety net should have caught any
        # empty-content iteration and surfaced something.
        assert text, (
            f"Model generated {metrics.completion_tokens} completion "
            "tokens but result.text is empty — the safety net failed "
            "to flush them. This is the exact #280 bug shape."
        )

    if hasattr(agent, "close"):
        agent.close()


def test_long_prompt_gemini_with_tool_use_still_produces_output() -> None:
    """Same long-prompt shape, but the user's question explicitly
    asks the model to use the tool. After the tool runs, the model
    might (or might not) emit a final summary; the safety net
    catches either path."""
    model = _build_chat_model()
    agent = create_deepagent(
        model=model,
        tools=[list_db_tables],
        system_prompt=(
            "You are a deep research agent. ALWAYS call list_db_tables "
            "first, then summarize what's available. " + ("CONTEXT BLOCK. " * 700)
        ),
        reflexion=False,
        grounding=False,
        max_iterations=6,
        max_output_tokens=2048,
    )
    result = agent.run_sync("List the tables and summarize.")
    text = getattr(result, "text", "") or ""
    tool_execs = list(getattr(result, "tool_executions", []) or [])
    metrics = getattr(result, "metrics", None)

    # Tool should have fired (the prompt requests it explicitly).
    # If it didn't, that's a different failure than #280 — flag but
    # don't fail (model might decline the tool on this exact prompt).
    if tool_execs:
        tool_names = [getattr(e, "tool_name", "") for e in tool_execs]
        assert "list_db_tables" in tool_names

    # Result text must not be empty — this is the fix's contract.
    assert text, (
        "Tool use OR direct answer — either way, result.text must "
        "contain the model's final response. The safety net catches "
        "the case where the model finishes tool use with an empty "
        "summary iteration. "
        f"tools={len(tool_execs)}, metrics={metrics!r}"
    )

    if hasattr(agent, "close"):
        agent.close()
