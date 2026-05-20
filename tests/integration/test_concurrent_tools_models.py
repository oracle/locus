# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Per-transport wall-time guard for ``tool_execution="concurrent"`` (#210).

The unit test ``tests/unit/test_agent_concurrent_tools.py`` pins the contract
deterministically with a scripted model. The single-transport twin in
``tests/integration/test_comprehensive_agent.py`` confirms it under one real
model.

This file completes the matrix: parallelism is pinned per-transport so a
regression in either ``OCIChatCompletionsModel`` (OpenAI-compat wire) or ``OCIModel``
(OCI native SDK wire) is caught. Mirrors the model list used by
``test_idempotent_models.py`` so dedup and parallelism share a transport
coverage baseline.

Activation:

* ``OCI_PROFILE=<profile>`` — required.
* ``OCI_REGION=<region>`` — defaults to ``us-chicago-1``.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any

import pytest

from locus.core.events import ThinkEvent, ToolCompleteEvent, ToolStartEvent
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


_PROFILE_OPT = os.environ.get("OCI_PROFILE")
_REGION = os.environ.get("OCI_REGION", "us-chicago-1")
if not _PROFILE_OPT:  # pragma: no cover
    pytest.skip("OCI_PROFILE not set", allow_module_level=True)
# Above check narrows ``_PROFILE_OPT`` to ``str`` — re-bind so the factories
# downstream can pass it to OCI constructors typed as ``str`` (not ``str | None``).
_PROFILE: str = _PROFILE_OPT


# Models known to fan out multiple tool_calls per response on OCI's
# OpenAI-compat wire. Same matrix as ``test_idempotent_models.py`` so a
# regression on either parallelism or dedup is caught against the same
# baseline.
_OPENAI_COMPAT_MODELS = [
    pytest.param("openai.gpt-4o-mini", id="openai-gpt-4o-mini"),
    pytest.param("meta.llama-3.3-70b-instruct", id="meta-llama-3.3-70b"),
    # Frontier non-xAI vendor to keep the matrix off shared xAI team
    # RPM limits that were tripping ``xai.grok-4-fast-non-reasoning``
    # under matrix sweeps. OpenAI gpt-4.1 (full, not mini) is a
    # different frontier class than the other slot's gpt-4o-mini and
    # exercises the same OpenAI-compat ``tool_calls`` codepath.
    pytest.param("openai.gpt-4.1", id="openai-gpt-4.1"),
    pytest.param("google.gemini-2.5-flash", id="google-gemini-2.5-flash"),
]

_NATIVE_SDK_MODELS = [
    pytest.param("cohere.command-r-plus-08-2024", id="oci-native-cohere-command-r-plus"),
]


@pytest.fixture(scope="module")
def oci_openai_factory() -> Any:
    from locus.models.providers.oci import OCIChatCompletionsModel

    def _build(model_id: str) -> Any:
        return OCIChatCompletionsModel(
            model=model_id,
            profile=_PROFILE,
            region=_REGION,
        )

    return _build


@pytest.fixture(scope="module")
def oci_native_factory() -> Any:
    from locus.models.providers.oci import OCIModel

    def _build(model_id: str) -> Any:
        return OCIModel(
            model_id=model_id,
            profile_name=_PROFILE,
            service_endpoint=f"https://inference.generativeai.{_REGION}.oci.oraclecloud.com",
        )

    return _build


# Tool sleep is large enough to clear network jitter but small enough not
# to slow the suite. Three parallel sleeps land at ~SLEEP_PER_TOOL; three
# serial ones at ~3*SLEEP_PER_TOOL. The /2 ceiling in the assertion is
# generous for slow CI.
_SLEEP_PER_TOOL = 0.6
_N_TARGET = 3


@tool(name="lookup_async")
async def lookup_async(topic: str) -> str:
    """Async lookup that sleeps for ``_SLEEP_PER_TOOL`` (simulates remote I/O)."""
    await asyncio.sleep(_SLEEP_PER_TOOL)
    return f"info about {topic}"


async def _measure_fanout_gap(agent: Any, prompt: str) -> tuple[int, float]:
    """Drive ``agent.run`` to completion, then return (n_calls, wall_seconds)
    for the first iteration that fanned out >= 2 ``ToolStartEvent``s.

    Returns (0, 0.0) if the model never fanned out — caller can xfail.
    """
    events: list[tuple[float, Any]] = []
    async for ev in agent.run(prompt):
        events.append((time.perf_counter(), ev))

    per_iter_starts: list[list[tuple[float, ToolStartEvent]]] = []
    per_iter_completes: list[list[tuple[float, ToolCompleteEvent]]] = []
    current_starts: list[tuple[float, ToolStartEvent]] = []
    current_completes: list[tuple[float, ToolCompleteEvent]] = []
    for ts, ev in events:
        if isinstance(ev, ThinkEvent):
            if current_starts:
                per_iter_starts.append(current_starts)
                per_iter_completes.append(current_completes)
            current_starts = []
            current_completes = []
        elif isinstance(ev, ToolStartEvent):
            current_starts.append((ts, ev))
        elif isinstance(ev, ToolCompleteEvent):
            current_completes.append((ts, ev))
    if current_starts:
        per_iter_starts.append(current_starts)
        per_iter_completes.append(current_completes)

    fanout_idx = next(
        (i for i, sts in enumerate(per_iter_starts) if len(sts) >= 2),
        None,
    )
    if fanout_idx is None:
        return 0, 0.0

    starts = per_iter_starts[fanout_idx]
    completes = per_iter_completes[fanout_idx]
    n = len(starts)
    assert len(completes) == n
    return n, completes[-1][0] - starts[0][0]


def _system_prompt() -> str:
    return (
        "You are a research assistant. You MUST call lookup_async "
        f"{_N_TARGET} times in your VERY FIRST response, all in a "
        "single message — one call per topic the user asks about. "
        "Do NOT call them one-at-a-time across iterations."
    )


def _user_prompt() -> str:
    return (
        "Look up these three topics: python, quantum, oracle. "
        "Call the tool three times in one response."
    )


def _assert_concurrent_wall_time(n: int, gap: float, model_id: str) -> None:
    sequential_floor = n * _SLEEP_PER_TOOL
    assert gap < sequential_floor / 2, (
        f"{model_id}: {n} concurrent tool calls took {gap:.2f}s "
        f"(sequential floor {sequential_floor:.2f}s) — "
        f"runtime loop may be serializing the executor again (#210)"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("model_id", _OPENAI_COMPAT_MODELS)
async def test_parallel_tool_calls_openai_compat_transport(
    oci_openai_factory: Any, model_id: str
) -> None:
    """OCI OpenAI-compat wire: parallel tool calls in one response must run
    concurrently, not serially.

    Skips cleanly if the live model declines to emit >=2 tool_calls in any
    single iteration (small models sometimes won't).
    """
    from locus.agent import Agent

    agent = Agent(
        model=oci_openai_factory(model_id),
        tools=[lookup_async],
        system_prompt=_system_prompt(),
        max_iterations=3,
        tool_execution="concurrent",
        max_concurrency=10,
        termination=MaxIterations(3),
    )
    try:
        n, gap = await _measure_fanout_gap(agent, _user_prompt())
    except Exception as e:  # noqa: BLE001
        # OCI's Meta Llama wire currently rejects multi-tool-call responses
        # with HTTP 400 "Parallel tool call is not supported yet." That's a
        # server-side limitation, not a runtime-loop regression — record it
        # as an xfail so the matrix stays informative without going red.
        msg = str(e)
        if "Parallel tool call is not supported" in msg:
            pytest.xfail(f"{model_id}: OCI server rejects parallel tool calls ({msg})")
        raise
    if n == 0:
        pytest.xfail(
            f"{model_id} did not fan out >=2 tool_calls in any iteration; nothing to measure"
        )
    _assert_concurrent_wall_time(n, gap, model_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("model_id", _NATIVE_SDK_MODELS)
async def test_parallel_tool_calls_native_sdk_transport(
    oci_native_factory: Any, model_id: str
) -> None:
    """OCI native SDK wire (Cohere R+): same parallelism guarantee.

    The native transport normalises Cohere's text-style tool calls into
    ``response.message.tool_calls`` before the runtime loop sees them, so
    the same #210 fix applies. This test verifies it actually does.
    """
    from locus.agent import Agent

    agent = Agent(
        model=oci_native_factory(model_id),
        tools=[lookup_async],
        system_prompt=_system_prompt(),
        max_iterations=3,
        tool_execution="concurrent",
        max_concurrency=10,
        termination=MaxIterations(3),
    )
    n, gap = await _measure_fanout_gap(agent, _user_prompt())
    if n == 0:
        pytest.xfail(
            f"{model_id} did not fan out >=2 tool_calls in any iteration; nothing to measure"
        )
    _assert_concurrent_wall_time(n, gap, model_id)
