# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""Notebook 04: target an Oracle Cloud Infrastructure (OCI) Dedicated AI Cluster endpoint.

A Dedicated AI Cluster (DAC) is OCI GenAI's provisioned-capacity
serving mode: instead of pay-per-token inference against a shared model
id, you address a dedicated endpoint by its OCID
(``ocid1.generativeaiendpoint.oc1.<region>....``) and OCI routes
inference to your cluster. This notebook wires Locus to one.

Key concepts:

- Pass the endpoint OCID as the model id and Locus routes it through
  ``OCIModel`` (the OCI SDK client). The OpenAI-compatible client
  cannot speak ``DedicatedServingMode``, which is required for DACs.
- Tool schemas pass through unchanged. Whether the model on the cluster
  emits structured ``tool_calls`` (OpenAI / Llama / Cohere) or text
  ``<tool_call>`` blocks (default Qwen3) depends on the deployment.
- Both ``complete()`` and SSE ``stream()`` work end-to-end against a
  DAC — the call sites look identical to a public model.

Part-by-part:

- Part 1 — how Locus picks the OCI SDK client for a DAC OCID.
- Part 2 — build an ``Agent`` against a DAC endpoint.
- Part 3 — single ``complete()`` round-trip.
- Part 4 — SSE ``stream()`` with deltas printed inline.
- Part 5 — tool-using ``Agent`` on top of the DAC.

Run it::

    pip install -e ".[oci]"
    export OCI_DAC_ENDPOINT_OCID=ocid1.generativeaiendpoint.oc1.uk-london-1....
    export OCI_DAC_COMPARTMENT_ID=ocid1.compartment.oc1....
    export OCI_DAC_REGION=uk-london-1
    export OCI_PROFILE=MY_DAC_PROFILE
    python examples/notebook_04_oci_dac.py

Without the DAC env vars Parts 2-5 print the wiring snippet and skip,
so the file still runs cleanly in CI.

Difficulty: Intermediate
"""

from __future__ import annotations

import asyncio
import os


# =============================================================================
# Part 1: routing — why a DAC OCID goes through OCIModel
# =============================================================================


def example_routing() -> None:
    """How Locus picks the OCI SDK client for a DAC OCID."""
    print("=== Part 1: routing ===\n")

    print("locus.models.registry inspects the model id:")
    print()
    print("  1. ocid1.generativeaiendpoint.<region>....   -> OCIModel (OCI SDK)")
    print("  2. cohere.command-r-*                         -> OCIModel (OCI SDK)")
    print("  3. everything else                            -> OCIChatCompletionsModel")
    print()
    print("Both of these end up on OCIModel:")
    print()
    print('  Agent(model="oci:cohere.command-r-plus-08-2024")  # rule 2')
    print('  Agent(model="oci:ocid1.generativeaiendpoint....")  # rule 1')
    print()
    print("A DAC needs OCIModel because DedicatedServingMode (endpoint_id=...)")
    print("is part of the OCI proprietary chat shape, not the OpenAI-compatible")
    print("/openai/v1/chat/completions endpoint.")


# =============================================================================
# Part 2: build an Agent against a DAC endpoint
# =============================================================================


def _dac_env_ready() -> bool:
    return bool(
        os.environ.get("OCI_DAC_ENDPOINT_OCID") and os.environ.get("OCI_DAC_COMPARTMENT_ID")
    )


def example_configure_agent() -> None:
    """Build an Agent pointed at a DAC. Same shape as any other model."""
    print("\n=== Part 2: configure Agent against a DAC ===\n")

    if not _dac_env_ready():
        print("OCI_DAC_ENDPOINT_OCID / OCI_DAC_COMPARTMENT_ID not set.")
        print()
        print("Wiring (with the env vars set):")
        print("""
  from locus.agent import Agent
  from locus.models import get_model

  # Pre-build the model: DAC needs provider-specific kwargs that
  # AgentConfig doesn't accept on its keyword path.
  region = os.environ["OCI_DAC_REGION"]
  model = get_model(
      f"oci:{os.environ['OCI_DAC_ENDPOINT_OCID']}",
      compartment_id=os.environ["OCI_DAC_COMPARTMENT_ID"],
      profile_name=os.environ.get("OCI_PROFILE", "DEFAULT"),
      service_endpoint=(
          f"https://inference.generativeai.{region}.oci.oraclecloud.com"
      ),
  )
  agent = Agent(
      model=model,
      system_prompt="You are a concise assistant.",
  )
""")
        return

    from locus.agent import Agent
    from locus.models import get_model

    region = os.environ.get("OCI_DAC_REGION", "us-chicago-1")
    model = get_model(
        f"oci:{os.environ['OCI_DAC_ENDPOINT_OCID']}",
        compartment_id=os.environ["OCI_DAC_COMPARTMENT_ID"],
        profile_name=os.environ.get("OCI_PROFILE", "DEFAULT"),
        service_endpoint=f"https://inference.generativeai.{region}.oci.oraclecloud.com",
    )
    agent = Agent(
        model=model,
        system_prompt="You are a concise assistant. Reply briefly.",
        max_iterations=2,
    )
    print(f"Agent configured against DAC endpoint in {region}.")
    print(f"  underlying model class:  {type(agent._model).__name__}")
    print("  serving mode:             DedicatedServingMode")


# =============================================================================
# Part 3: complete() — single round-trip against the DAC
# =============================================================================


async def example_complete() -> None:
    """One chat against the DAC and print what comes back."""
    print("\n=== Part 3: complete() against the DAC ===\n")

    if not _dac_env_ready():
        print("Skipping — env vars not set.")
        return

    from locus.core.messages import Message
    from locus.models.providers.oci import OCIAuthType, OCIModel

    region = os.environ.get("OCI_DAC_REGION", "us-chicago-1")
    model = OCIModel(
        model_id=os.environ["OCI_DAC_ENDPOINT_OCID"],
        compartment_id=os.environ["OCI_DAC_COMPARTMENT_ID"],
        profile_name=os.environ.get("OCI_PROFILE", "DEFAULT"),
        auth_type=OCIAuthType.API_KEY,
        service_endpoint=f"https://inference.generativeai.{region}.oci.oraclecloud.com",
        max_tokens=128,
    )

    response = await model.complete(
        messages=[
            Message.user("In one sentence, what model are you?"),
        ],
        tools=None,
    )
    content = (response.message.content or "").strip()
    print(f"Reply:        {content!r}")
    print(f"usage:        {response.usage}")
    print(f"stop_reason:  {response.stop_reason}")


# =============================================================================
# Part 4: stream() — SSE deltas from the DAC
# =============================================================================


async def example_stream() -> None:
    """Stream from the DAC and print each delta as it arrives."""
    print("\n=== Part 4: stream() against the DAC ===\n")

    if not _dac_env_ready():
        print("Skipping — env vars not set.")
        return

    from locus.core.messages import Message
    from locus.models.providers.oci import OCIAuthType, OCIModel

    region = os.environ.get("OCI_DAC_REGION", "us-chicago-1")
    model = OCIModel(
        model_id=os.environ["OCI_DAC_ENDPOINT_OCID"],
        compartment_id=os.environ["OCI_DAC_COMPARTMENT_ID"],
        profile_name=os.environ.get("OCI_PROFILE", "DEFAULT"),
        auth_type=OCIAuthType.API_KEY,
        service_endpoint=f"https://inference.generativeai.{region}.oci.oraclecloud.com",
        max_tokens=64,
    )

    print("Streaming reply (chunks shown inline):")
    print("  ", end="", flush=True)
    async for event in model.stream(
        messages=[Message.user("Count from 1 to 5, separated by commas.")],
        tools=None,
    ):
        if event.content:
            print(event.content, end="", flush=True)
        if event.done:
            break
    print()


# =============================================================================
# Part 5: tool-using Agent on top of the DAC
# =============================================================================


async def example_agent_with_tool() -> None:
    """Wire a tool-using Agent on top of the DAC.

    Locus sends the same OpenAI-style tool definitions to the DAC that
    it sends to on-demand models. Whether the model on the cluster
    emits structured tool calls or text-format ones depends on the
    model and the deployment:

    - **OpenAI / Llama / Cohere on OCI** emit structured ``tool_calls``.
      Locus extracts them automatically.
    - **Qwen on a DAC** by default emits ``<tool_call>...</tool_call>``
      text blocks. Locus's parser doesn't pick those up, so
      ``result.metrics.tool_calls`` will be 0 even though the model
      "called" the tool in its content. Two fixes:
        (a) Enable OpenAI-compatible tool-call output on the deployment
            (Qwen3 supports ``--enable-auto-tool-choice``).
        (b) Add a post-processor that converts the text blocks into
            Locus ``ToolCall`` objects.

    This part shows the wiring. The output depends on the model.
    """
    print("\n=== Part 5: Agent + @tool against the DAC ===\n")

    if not _dac_env_ready():
        print("Skipping — env vars not set.")
        return

    from locus.agent import Agent
    from locus.models import get_model
    from locus.tools.decorator import tool

    @tool(name="add_two_numbers")
    def add_two_numbers(a: int, b: int) -> int:
        """Return the sum of two integers."""
        return a + b

    region = os.environ.get("OCI_DAC_REGION", "us-chicago-1")
    model = get_model(
        f"oci:{os.environ['OCI_DAC_ENDPOINT_OCID']}",
        compartment_id=os.environ["OCI_DAC_COMPARTMENT_ID"],
        profile_name=os.environ.get("OCI_PROFILE", "DEFAULT"),
        service_endpoint=f"https://inference.generativeai.{region}.oci.oraclecloud.com",
    )
    agent = Agent(
        model=model,
        tools=[add_two_numbers],
        system_prompt="You can call add_two_numbers when asked to add. Reply briefly.",
        max_iterations=4,
    )

    result = await asyncio.to_thread(
        agent.run_sync,
        "Use the add_two_numbers tool to add 7 and 35, then state the result in one sentence.",
    )
    print(f"final message:        {result.message.strip()[:200]}")
    print(f"iterations:           {result.metrics.iterations}")
    print(f"locus tool calls:     {result.metrics.tool_calls}")
    if "<tool_call>" in (result.message or ""):
        print()
        print("Note: the model emitted a <tool_call> text block instead of a")
        print("structured tool_call. See the docstring for how to handle this")
        print("(deployment flag or post-processing parser).")


# =============================================================================
# Main
# =============================================================================


async def _async_main() -> None:
    example_routing()
    example_configure_agent()
    await example_complete()
    await example_stream()
    await example_agent_with_tool()


if __name__ == "__main__":
    asyncio.run(_async_main())
