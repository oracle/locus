# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Live: ``Agent(auxiliary_model=...)`` routes side calls correctly.

Drives a real two-iteration agent against OCI v1 with a small primary model
and an even smaller auxiliary model, then asserts the auxiliary model
received at least one call (the max-iterations summary). The auxiliary
acts as the cheap-tier helper so the primary's budget stays for the
actual task.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from locus.tools.decorator import tool


pytestmark = [pytest.mark.integration]


_PROFILE = os.environ.get("OCI_PROFILE")
_REGION = os.environ.get("OCI_REGION", "us-chicago-1")


def _has_oci_config() -> bool:
    return Path("~/.oci/config").expanduser().exists()


if not (_has_oci_config() and _PROFILE):  # pragma: no cover
    pytest.skip("OCI_PROFILE not set", allow_module_level=True)


@tool
def keep_searching(topic: str) -> str:
    """Search and ask the agent to keep searching.

    Returned text instructs the agent to call again, so the loop runs to
    max_iterations and triggers the summary path we want to observe.
    """
    return f"found partial results for {topic}; call keep_searching again to refine"


class _CountingModel:
    """Forwards complete()/stream() to a real model and counts calls.

    Pydantic models reject ``model.complete = ...`` patching, so we wrap.
    """

    def __init__(self, inner):
        self._inner = inner
        self.calls = 0

    async def complete(self, *args, **kwargs):
        self.calls += 1
        return await self._inner.complete(*args, **kwargs)

    async def stream(self, *args, **kwargs):
        return self._inner.stream(*args, **kwargs)


def test_auxiliary_model_handles_summary_call_live():
    """Two OCI models — the auxiliary wrapper should receive the summary call."""
    from locus.agent import Agent
    from locus.models.providers.oci import OCIChatCompletionsModel

    primary = OCIChatCompletionsModel(model="openai.gpt-4o-mini", profile=_PROFILE, region=_REGION)
    aux_inner = OCIChatCompletionsModel(
        model="openai.gpt-4o-mini", profile=_PROFILE, region=_REGION
    )
    auxiliary = _CountingModel(aux_inner)

    agent = Agent(
        model=primary,
        auxiliary_model=auxiliary,
        tools=[keep_searching],
        system_prompt=(
            "You are a researcher. Always call keep_searching for any topic. "
            "Never give a final answer until told."
        ),
        max_iterations=2,
    )
    result = agent.run_sync("Research HNSW indexes.")

    # Loop ran to max_iterations -> the agent should have routed the
    # summary call through the auxiliary model wrapper.
    assert auxiliary.calls >= 1, (
        f"auxiliary_model never received a call; result.stop_reason="
        f"{result.stop_reason!r}, message={result.message!r}"
    )
    assert result.stop_reason == "max_iterations"
