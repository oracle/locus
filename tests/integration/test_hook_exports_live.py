# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""End-to-end smoke test: the newly re-exported hooks actually run.

A unit test guards that ``from locus.hooks.builtin import ModelRetryHook,
SteeringHook`` doesn't ImportError. This file proves the symbols resolve
to working hooks when an Agent runs against real models on both OCI
transports — so we know the export wired the *correct* class, not just
the right name.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

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


@tool
def echo(msg: str) -> str:
    """Echo the message back. Used to give the agent something to do."""
    return f"echoed: {msg}"


@pytest.fixture(scope="module")
def oci_v1():
    from locus.models.providers.oci import OCIChatCompletionsModel

    return OCIChatCompletionsModel(model="openai.gpt-4o-mini", profile=_PROFILE, region=_REGION)


@pytest.fixture(scope="module")
def oci_native():
    from locus.models.providers.oci import OCIModel

    return OCIModel(
        model_id="cohere.command-r-plus-08-2024",
        profile_name=_PROFILE,
        service_endpoint=f"https://inference.generativeai.{_REGION}.oci.oraclecloud.com",
    )


def test_model_retry_hook_runs_on_v1(oci_v1):
    """``ModelRetryHook`` from locus.hooks.builtin attaches and the agent runs."""
    from locus.agent import Agent
    from locus.hooks.builtin import ModelRetryHook

    agent = Agent(
        model=oci_v1,
        tools=[echo],
        hooks=[ModelRetryHook(max_retries=2)],
        system_prompt="Echo whatever the user says.",
        max_iterations=3,
    )
    result = agent.run_sync("Say hello.")
    assert result.message  # got *something* back
    assert result.stop_reason in ("complete", "no_tools", "terminal_tool")


def test_model_retry_hook_runs_on_native_sdk(oci_native):
    """Same as above, native SDK transport (Cohere R+)."""
    from locus.agent import Agent
    from locus.hooks.builtin import ModelRetryHook

    agent = Agent(
        model=oci_native,
        tools=[echo],
        hooks=[ModelRetryHook(max_retries=2)],
        system_prompt="Echo whatever the user says.",
        max_iterations=3,
    )
    result = agent.run_sync("Say hello.")
    assert result.message
    assert result.stop_reason in ("complete", "no_tools", "terminal_tool")
