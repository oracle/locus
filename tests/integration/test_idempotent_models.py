# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""End-to-end tests for ``@tool(idempotent=True)`` against real models.

Verifies that the idempotent dedup wired into ``Agent.run()`` actually
short-circuits a second call with the same arguments — across providers.
The README hero example ("idempotent writes fire once") would otherwise
be a silent no-op for any user not using the lower-level ExecuteNode.

Activation:

* ``OCI_PROFILE=<profile>`` — required.
* ``OCI_REGION=<region>`` — defaults to ``us-chicago-1``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from locus.core.termination import MaxIterations
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
    """Factory for the OCI native SDK transport (``OCIModel``)."""
    from locus.models.providers.oci import OCIModel

    def _build(model_id: str):
        return OCIModel(
            model_id=model_id,
            profile_name=_PROFILE,
            service_endpoint=f"https://inference.generativeai.{_REGION}.oci.oraclecloud.com",
        )

    return _build


# Module-level counter so the closure captures it cleanly across calls.
class _Counter:
    n = 0


@tool(idempotent=True)
def submit_invoice(invoice_id: str, amount_usd: float) -> str:
    """Submit an invoice for payment. SAFE TO RETRY (idempotent)."""
    _Counter.n += 1
    return f"Submitted invoice {invoice_id} for ${amount_usd:.2f}. (call #{_Counter.n})"


@pytest.mark.parametrize("model_id", _LIVE_MODELS)
def test_idempotent_tool_fires_once_when_model_retries(oci_openai_factory, model_id: str):
    """When the model tries to submit the same invoice twice in one run, the
    underlying body must execute exactly once. The second call short-circuits
    via the idempotent cache.

    We coax the model into two attempts by instructing it explicitly. If the
    model only calls once on its own, the test still passes (one body run is
    fine, and there's nothing to dedup) — but the assertion will skip.
    """
    from locus.agent import Agent

    _Counter.n = 0

    agent = Agent(
        model=oci_openai_factory(model_id),
        tools=[submit_invoice],
        system_prompt=(
            "You are a finance assistant. To confirm the user's invoice you MUST "
            "call submit_invoice EXACTLY THREE TIMES with the SAME parameters "
            "(invoice_id='INV-42', amount_usd=100.0). Three calls are required "
            "by audit policy. After the third call, briefly report success."
        ),
        # Don't terminate on first ToolCalled — let the loop run several iterations
        # so the model gets a chance to retry.
        termination=MaxIterations(6),
        max_iterations=10,
    )
    result = agent.run_sync(
        "Process INV-42 for $100.00. Remember: three submit_invoice calls, same args."
    )

    invoice_calls = [te for te in result.tool_executions if te.tool_name == "submit_invoice"]
    if len(invoice_calls) < 2:
        pytest.xfail(
            f"{model_id} only invoked submit_invoice {len(invoice_calls)}x; no duplicate to dedup"
        )

    # At least one of the executions after the first must be a cache hit.
    cache_hits = [te for te in invoice_calls if te.idempotent_cache_hit]
    assert cache_hits, (
        f"{model_id}: {len(invoice_calls)} submit_invoice executions but no cache "
        f"hit recorded. arguments seen: {[te.arguments for te in invoice_calls]}"
    )
    # Body must have run strictly fewer times than total invocations.
    assert _Counter.n < len(invoice_calls), (
        f"{model_id}: body ran {_Counter.n} times for {len(invoice_calls)} "
        "invocations — dedup did not fire."
    )


# =============================================================================
# OCI native SDK transport (OCIModel) — same wiring, different wire format
# =============================================================================

_NATIVE_SDK_MODELS = [
    pytest.param("cohere.command-r-plus-08-2024", id="oci-native-cohere-command-r-plus"),
]


@pytest.mark.parametrize("model_id", _NATIVE_SDK_MODELS)
def test_idempotent_dedup_native_sdk(oci_native_factory, model_id: str):
    """Idempotent dedup must fire on the OCI native SDK transport too.

    Same scripted scenario as the v1 transport test, just routed through
    ``OCIModel`` (Cohere R+ via the OCI GenAI SDK) instead of
    ``OCIChatCompletionsModel``. Cache hits + body invocation count are asserted
    identically.
    """
    from locus.agent import Agent

    _Counter.n = 0

    agent = Agent(
        model=oci_native_factory(model_id),
        tools=[submit_invoice],
        system_prompt=(
            "You are a finance assistant. To confirm the user's invoice you MUST "
            "call submit_invoice EXACTLY THREE TIMES with the SAME parameters "
            "(invoice_id='INV-42', amount_usd=100.0). Three calls are required "
            "by audit policy. After the third call, briefly report success."
        ),
        termination=MaxIterations(6),
        max_iterations=10,
    )
    result = agent.run_sync(
        "Process INV-42 for $100.00. Remember: three submit_invoice calls, same args."
    )

    invoice_calls = [te for te in result.tool_executions if te.tool_name == "submit_invoice"]
    if len(invoice_calls) < 2:
        pytest.xfail(
            f"{model_id} only invoked submit_invoice {len(invoice_calls)}x; no duplicate to dedup"
        )

    cache_hits = [te for te in invoice_calls if te.idempotent_cache_hit]
    assert cache_hits, (
        f"{model_id}: {len(invoice_calls)} submit_invoice executions but no cache "
        f"hit recorded. arguments seen: {[te.arguments for te in invoice_calls]}"
    )
    assert _Counter.n < len(invoice_calls), (
        f"{model_id}: body ran {_Counter.n} times for {len(invoice_calls)} "
        "invocations — dedup did not fire."
    )
