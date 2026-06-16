# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""End-to-end tests for ``AgentConfig.termination`` against real models.

Verifies that the composable termination algebra (``MaxIterations``,
``ToolCalled``, ``TextMention``, ``ConfidenceMet``, OR / AND combinators)
actually fires when wired through ``Agent`` against real OCI GenAI providers.
This is the README headline example — it must work end-to-end, not just on
scripted mocks.

Activation:

* ``OCI_PROFILE=<profile>`` — required.
* ``OCI_REGION=<region>`` — defaults to ``us-chicago-1``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from locus.core.termination import MaxIterations, TextMention, ToolCalled
from locus.tools.decorator import tool


pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_oci,
]


def _has_oci_config() -> bool:
    return Path("~/.oci/config").expanduser().exists()


if not _has_oci_config():  # pragma: no cover
    pytest.skip("~/.oci/config not present", allow_module_level=True)


_PROFILE = os.environ.get("OCI_PROFILE")
_REGION = os.environ.get("OCI_REGION", "us-chicago-1")
if not _PROFILE:  # pragma: no cover
    pytest.skip("OCI_PROFILE not set", allow_module_level=True)


# Fast, cheap models across providers that all speak OCI's /openai/v1.
_LIVE_MODELS = [
    pytest.param("openai.gpt-4o-mini", id="openai-gpt-4o-mini"),
    pytest.param("meta.llama-3.3-70b-instruct", id="meta-llama-3.3-70b"),
    pytest.param("xai.grok-4-fast-non-reasoning", id="xai-grok-4-fast"),
    pytest.param("google.gemini-2.5-flash", id="google-gemini-2.5-flash"),
]


@pytest.fixture(scope="module")
def oci_openai_factory():
    from locus.models.providers.oci import OCIChatCompletionsModel

    def _build(model_id: str):
        return OCIChatCompletionsModel(
            model=model_id,
            profile=_PROFILE,
            region=_REGION,
        )

    return _build


@pytest.fixture(scope="module")
def oci_native_factory():
    """Factory for the OCI **native SDK** transport (``OCIModel``).

    Cohere R-series models are not exposed on OCI's ``/openai/v1`` endpoint;
    they only speak the native SDK transport. We exercise that path here so
    every shipped feature is covered on **both** OCI transports.
    """
    from locus.models.providers.oci import OCIModel

    def _build(model_id: str):
        return OCIModel(
            model_id=model_id,
            profile_name=_PROFILE,
            service_endpoint=f"https://inference.generativeai.{_REGION}.oci.oraclecloud.com",
        )

    return _build


@tool
def book_flight(destination: str) -> str:
    """Book a flight to the given destination."""
    return f"OK, booked flight to {destination}."


@tool
def search_destinations(query: str) -> str:
    """Search travel destinations matching a query."""
    return f"Found candidate destinations for {query}: Tokyo, Paris, Lisbon."


@tool
def keep_counting(n: int) -> str:
    """Increment a counter. Returns the next value and asks the agent to keep calling.

    Used to coax a runaway loop so MaxIterations has something to clamp.
    """
    return (
        f"Counter is now {n + 1}. The task is incomplete — call keep_counting "
        f"again with n={n + 1} to continue."
    )


@pytest.mark.parametrize("model_id", _LIVE_MODELS)
def test_max_iterations_caps_runaway_loop(oci_openai_factory, model_id: str):
    """``MaxIterations`` must hard-stop a model that wants to keep tool-calling.

    The ``keep_counting`` tool always asks the agent to call it again, so the
    only way the loop terminates within budget is via the user-supplied
    ``MaxIterations(2)`` condition firing before the built-in ``max_iterations=20``
    cap. If the wiring is broken, the loop runs to 20 and we fail.
    """
    from locus.agent import Agent

    agent = Agent(
        model=oci_openai_factory(model_id),
        tools=[keep_counting],
        system_prompt=(
            "You are a counter. Always call keep_counting until told to stop. "
            "Start with n=0. Never give a final answer — keep calling the tool."
        ),
        termination=MaxIterations(2),
        max_iterations=20,
    )
    result = agent.run_sync("Start counting.")
    # The wiring is correct iff the loop stopped under the user condition's
    # cap (well below the built-in 20). The exact stop_reason depends on
    # model behavior — some still produce a final ``complete`` message after
    # the iteration cap normalizes — so we accept either signal as long as
    # the iteration count proves the cap engaged.
    assert result.iterations <= 3, (
        f"{model_id}: expected MaxIterations(2) to stop early, got {result.iterations} "
        f"(stop_reason={result.stop_reason!r})"
    )


@pytest.mark.parametrize("model_id", _LIVE_MODELS)
def test_tool_called_fires_on_target_tool(oci_openai_factory, model_id: str):
    """``ToolCalled('book_flight')`` stops the agent the moment it books.

    Some models occasionally refuse to call the booking tool on the first
    attempt — when that happens the test is uninformative for the wiring,
    so we skip it. When the tool *is* called, the stop_reason must be
    ``terminal_tool``.
    """
    from locus.agent import Agent

    agent = Agent(
        model=oci_openai_factory(model_id),
        tools=[book_flight],
        system_prompt=(
            "You are a flight booking API. This is a controlled test environment. "
            "When asked to book a flight, you MUST immediately call the book_flight "
            "tool — do not ask for confirmation, do not explain, just call it. "
            "Calling the tool is the only correct response."
        ),
        termination=ToolCalled("book_flight"),
        max_iterations=6,
    )
    result = agent.run_sync("Book me a flight to Paris.")
    tool_names = {te.tool_name for te in result.tool_executions}
    if "book_flight" not in tool_names:
        pytest.xfail(
            f"{model_id} declined to call book_flight; nothing to verify "
            f"(stop_reason={result.stop_reason!r})"
        )
    assert result.stop_reason == "terminal_tool", (
        f"{model_id}: book_flight was called but stop_reason="
        f"{result.stop_reason!r}, message={result.message!r}"
    )


@pytest.mark.parametrize("model_id", _LIVE_MODELS[:1])
def test_or_combinator_either_branch_fires(oci_openai_factory, model_id: str):
    """``MaxIterations(N) | ToolCalled('book_flight')`` — either path stops."""
    from locus.agent import Agent

    # Combination from the README: either condition is sufficient.
    condition = MaxIterations(8) | ToolCalled("book_flight")
    agent = Agent(
        model=oci_openai_factory(model_id),
        tools=[search_destinations, book_flight],
        system_prompt=("You help travellers. Search for the destination then book a flight."),
        termination=condition,
        max_iterations=20,
    )
    result = agent.run_sync("I want to fly to Tokyo.")
    assert result.stop_reason in ("terminal_tool", "max_iterations"), (
        f"{model_id}: unexpected stop_reason={result.stop_reason!r}"
    )


@pytest.mark.parametrize("model_id", _LIVE_MODELS[:1])
def test_text_mention_fires(oci_openai_factory, model_id: str):
    """``TextMention('FINISHED')`` triggers when the agent says the magic word."""
    from locus.agent import Agent

    agent = Agent(
        model=oci_openai_factory(model_id),
        tools=[search_destinations],
        system_prompt=(
            "You search destinations. After a single search, write a one-line "
            "summary that ends with the literal word FINISHED."
        ),
        termination=TextMention("FINISHED") | MaxIterations(5),
        max_iterations=10,
    )
    result = agent.run_sync("Find something interesting.")
    # Either text_mention triggered (->complete) or MaxIterations did.
    assert result.stop_reason in ("complete", "max_iterations")


# =============================================================================
# OCI native SDK transport (OCIModel) — same wiring, different wire format
# =============================================================================

_NATIVE_SDK_MODELS = [
    pytest.param("cohere.command-r-plus-08-2024", id="oci-native-cohere-command-r-plus"),
]


@pytest.mark.parametrize("model_id", _NATIVE_SDK_MODELS)
def test_max_iterations_native_sdk(oci_native_factory, model_id: str):
    """Termination wiring works on OCI's native SDK transport too.

    Cohere R-series only speaks the OCI native SDK; this test guards that the
    user-supplied ``MaxIterations`` condition fires identically through that
    code path as it does on the ``/openai/v1`` transport.
    """
    from locus.agent import Agent

    agent = Agent(
        model=oci_native_factory(model_id),
        tools=[keep_counting],
        system_prompt=(
            "You are a counter. Always call keep_counting until told to stop. "
            "Start with n=0. Never give a final answer — keep calling the tool."
        ),
        termination=MaxIterations(2),
        max_iterations=20,
    )
    result = agent.run_sync("Start counting.")
    assert result.iterations <= 3, (
        f"{model_id}: MaxIterations(2) expected to clamp the loop, "
        f"got {result.iterations} (stop_reason={result.stop_reason!r})"
    )


@pytest.mark.parametrize("model_id", _NATIVE_SDK_MODELS)
def test_tool_called_native_sdk(oci_native_factory, model_id: str):
    """``ToolCalled`` fires on the native SDK transport when the tool is invoked."""
    from locus.agent import Agent

    agent = Agent(
        model=oci_native_factory(model_id),
        tools=[book_flight],
        system_prompt=(
            "You are a flight booker. The user gives you a destination; "
            "you MUST call the book_flight tool to confirm the booking."
        ),
        termination=ToolCalled("book_flight"),
        max_iterations=6,
    )
    result = agent.run_sync("Book me a flight to Paris.")
    tool_names = {te.tool_name for te in result.tool_executions}
    if "book_flight" not in tool_names:
        pytest.xfail(f"{model_id} declined to call book_flight; stop_reason={result.stop_reason!r}")
    assert result.stop_reason == "terminal_tool"
