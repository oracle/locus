# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""Notebook 01: pick the right Oracle Cloud Infrastructure (OCI) Generative AI client for your model.

Locus ships three model classes for OCI GenAI. They are not
interchangeable — the right one depends on which model family you call
and whether you want OCI to hold the conversation history. This file
builds an agent against each, runs the same prompt, and prints a
decision table.

Key concepts:

- ``OCIChatCompletionsModel`` is the default. It targets the OpenAI-compatible
  endpoint at ``/openai/v1/chat/completions`` and covers OpenAI, Meta,
  Mistral, xAI, Gemini, and non-R-series Cohere in one class.
- ``OCIModel`` targets OCI's proprietary chat endpoint
  (``/20231130/actions/chat``) via the official OCI Python SDK. It is
  required for Cohere R-series (``cohere.command-r*``), which the
  OpenAI-compatible endpoint rejects.
- ``OCIResponsesModel`` targets the Responses endpoint
  (``/openai/v1/responses``). Use it for Responses-only models such as
  ``openai.gpt-5.5-pro`` or to push conversation state to the server.
- The runtime reads ``model.server_stateful``: when it is ``True``,
  Locus sends only the latest turn and threads OCI's continuation id
  via ``AgentState.provider_state``.
- ``ConversationManager`` is the one Locus primitive that bypasses on
  the Responses path (no client-side history to trim). Memory, hooks,
  tool calling, streaming, structured output, and checkpointing all
  apply identically across the three classes.

Run it::

    export OCI_PROFILE=<your-profile>
    export OCI_REGION=us-chicago-1                   # or your region
    export OCI_COMPARTMENT=ocid1.compartment.oc1..…  # GenAI access
    python examples/notebook_01_oci_transports.py
    python examples/notebook_01_oci_transports.py --transport v1

Difficulty: Beginner
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from typing import Any


def _env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        sys.stderr.write(
            f"skipped: missing env var {name} — see the prerequisites in the notebook docstring\n"
        )
        # 77 = skipped: scripts/run_notebooks.py reports SKIP, not FAIL.
        sys.exit(77)
    return val


# =============================================================================
# OCIResponsesModel — Responses endpoint, server-side history
# =============================================================================


def example_responses_transport() -> None:
    """Reach Responses-only models and let OCI hold the thread."""
    from locus.agent import Agent
    from locus.models.providers.oci import OCIResponsesModel

    print("=== OCIResponsesModel (server-stateful) ===\n")

    model = OCIResponsesModel(
        # gpt-5.5-pro is reachable only through the Responses endpoint on OCI.
        # Regular gpt-5.5 works on both endpoints.
        model="openai.gpt-5.5-pro",
        profile=_env("OCI_PROFILE"),
        region=os.environ.get("OCI_REGION", "us-chicago-1"),
        compartment_id=os.environ.get("OCI_COMPARTMENT"),
    )

    agent = Agent(model=model, system_prompt="Answer in one short sentence.")
    result = agent.run_sync("What is the largest mammal on Earth?")
    print(f"  reply:           {result.message.strip()}")
    print(f"  iterations:      {result.metrics.iterations}")
    # After turn 1, AgentState carries the continuation id, so a follow-up
    # agent.run(...) on the same thread_id resumes the OCI-side thread.
    print(f"  server_stateful: {type(model).server_stateful}")
    print()


# =============================================================================
# OCIModel — OCI SDK against the proprietary chat endpoint (Cohere R-series)
# =============================================================================


def example_sdk_transport() -> None:
    """Required for Cohere R-series. OpenAI-compatible endpoint rejects them."""
    from locus.agent import Agent
    from locus.models.providers.oci import OCIModel

    print("=== OCIModel (OCI SDK, Cohere R-series) ===\n")

    model = OCIModel(
        model_id="cohere.command-r-plus-08-2024",
        profile_name=_env("OCI_PROFILE"),
        # api_key, security_token, instance_principal, resource_principal all supported
        auth_type="api_key",
        compartment_id=os.environ.get("OCI_COMPARTMENT"),
        service_endpoint=(
            f"https://inference.generativeai.{os.environ.get('OCI_REGION', 'us-chicago-1')}"
            ".oci.oraclecloud.com"
        ),
    )

    agent = Agent(model=model, system_prompt="Answer in one short sentence.")
    result = agent.run_sync("What is the largest mammal on Earth?")
    print(f"  reply:           {result.message.strip()}")
    print(f"  iterations:      {result.metrics.iterations}")
    print(f"  server_stateful: {getattr(type(model), 'server_stateful', False)}")
    print()


# =============================================================================
# OCIChatCompletionsModel — OpenAI-compatible endpoint, default for everything else
# =============================================================================


def example_v1_transport() -> None:
    """The default. One class for OpenAI, Meta, Mistral, xAI, Gemini, and non-R Cohere."""
    from locus.agent import Agent
    from locus.models.providers.oci import OCIChatCompletionsModel

    print("=== OCIChatCompletionsModel (OpenAI-compatible endpoint, default) ===\n")

    model = OCIChatCompletionsModel(
        model="meta.llama-3.3-70b-instruct",
        profile=_env("OCI_PROFILE"),
        region=os.environ.get("OCI_REGION", "us-chicago-1"),
        compartment_id=os.environ.get("OCI_COMPARTMENT"),
    )

    agent = Agent(model=model, system_prompt="Answer in one short sentence.")
    result = agent.run_sync("What is the largest mammal on Earth?")
    print(f"  reply:           {result.message.strip()}")
    print(f"  iterations:      {result.metrics.iterations}")
    print(f"  server_stateful: {getattr(type(model), 'server_stateful', False)}")
    print()


# =============================================================================
# Side-by-side comparison table
# =============================================================================


def print_comparison() -> None:
    """A quick decision table the reader can refer back to."""
    print("=== Decision table ===\n")
    rows: list[tuple[str, str, str, str]] = [
        (
            "Endpoint",
            "/openai/v1/responses",
            "/20231130/actions/chat",
            "/openai/v1/chat/completions",
        ),
        ("Class", "OCIResponsesModel", "OCIModel", "OCIChatCompletionsModel"),
        ("Stateful?", "server-side", "stateless", "stateless"),
        (
            "Model families",
            "openai, xai, gemini (Responses-only)",
            "cohere.command-r*",
            "all (default)",
        ),
        ("Project OCID", "optional", "no", "no"),
        ("ConversationManager", "skipped", "applies", "applies"),
        ("Memory / hooks / GSAR", "all apply", "all apply", "all apply"),
        ("Streaming", "yes (SSE)", "yes", "yes (SSE)"),
        ("Structured output", "yes", "limited", "yes"),
    ]
    width = max(len(r[0]) for r in rows) + 2
    cols = [width, 26, 24, 30]
    header = (
        "Field",
        "Responses (server-stateful)",
        "OCIModel (SDK)",
        "OCIChatCompletionsModel (default)",
    )
    print(" ".join(f"{c:<{w}}" for c, w in zip(header, cols, strict=False)))
    print(" ".join("-" * w for w in cols))
    for r in rows:
        print(" ".join(f"{c:<{w}}" for c, w in zip(r, cols, strict=False)))
    print()


# =============================================================================
# Main
# =============================================================================


async def _amain(which: str) -> None:
    if which in ("all", "responses"):
        try:
            example_responses_transport()
        except Exception as e:  # noqa: BLE001 — surface clearly to the reader
            print(f"  [responses transport raised: {type(e).__name__}: {e}]\n")
    if which in ("all", "sdk"):
        try:
            example_sdk_transport()
        except Exception as e:  # noqa: BLE001
            print(f"  [sdk transport raised: {type(e).__name__}: {e}]\n")
    if which in ("all", "v1"):
        try:
            example_v1_transport()
        except Exception as e:  # noqa: BLE001
            print(f"  [v1 transport raised: {type(e).__name__}: {e}]\n")
    print_comparison()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transport",
        choices=("all", "responses", "sdk", "v1"),
        default="all",
        help="which transport to demo (default: all)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Notebook 01 — OCI Generative AI: pick the right client class")
    print("=" * 70)
    print()
    print(f"OCI_PROFILE     = {os.environ.get('OCI_PROFILE', '(unset)')}")
    print(f"OCI_REGION      = {os.environ.get('OCI_REGION', 'us-chicago-1 (default)')}")
    print(f"OCI_COMPARTMENT = {os.environ.get('OCI_COMPARTMENT', '(unset)')[:60]}...")
    print()

    asyncio.run(_amain(args.transport))

    print("=" * 70)
    print("Next: notebook 02 — OCIChatCompletionsModel deep dive (the default path)")
    print("=" * 70)


if __name__ == "__main__":
    main()
