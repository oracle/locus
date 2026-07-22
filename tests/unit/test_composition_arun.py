# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Composition pipelines drive agents through the native async ``arun``.

Pipelines are async, so they must not depend on threads to run their agents —
``Agent.run_sync`` spins up a worker thread, which is unavailable under WASM /
Pyodide (the browser workbench). These tests pin that pipelines prefer the
thread-free ``arun`` when the agent exposes it, and still fall back to
``run_sync`` for older agent-likes that don't.
"""

from __future__ import annotations

import types
from typing import Any

import pytest

from tulip.agent.composition import LoopAgent, ParallelPipeline, SequentialPipeline


class _ArunAgent:
    """Agent stand-in exposing both ``arun`` and ``run_sync``; records which
    path a pipeline actually used."""

    def __init__(self, reply: str = "ok") -> None:
        self.reply = reply
        self.arun_calls = 0
        self.run_sync_calls = 0

    async def arun(self, prompt: str) -> Any:  # noqa: ARG002 — match interface
        self.arun_calls += 1
        return types.SimpleNamespace(message=self.reply, success=True)

    def run_sync(self, prompt: str) -> Any:  # noqa: ARG002 — match interface
        self.run_sync_calls += 1
        return types.SimpleNamespace(message=self.reply, success=True)


class _RunSyncOnlyAgent:
    """Legacy agent-like with only ``run_sync`` (no ``arun``)."""

    def __init__(self, reply: str = "legacy") -> None:
        self.reply = reply
        self.run_sync_calls = 0

    def run_sync(self, prompt: str) -> Any:  # noqa: ARG002 — match interface
        self.run_sync_calls += 1
        return types.SimpleNamespace(message=self.reply, success=True)


class TestPipelinesPreferArun:
    """When an agent has ``arun``, no pipeline touches ``run_sync`` (no threads)."""

    @pytest.mark.asyncio
    async def test_sequential_uses_arun(self) -> None:
        a, b = _ArunAgent("first"), _ArunAgent("second")
        result = await SequentialPipeline(agents=[a, b]).run("go")
        assert result.success
        assert a.arun_calls == 1
        assert b.arun_calls == 1
        assert a.run_sync_calls == 0
        assert b.run_sync_calls == 0

    @pytest.mark.asyncio
    async def test_parallel_uses_arun(self) -> None:
        agents = [_ArunAgent(f"a{i}") for i in range(3)]
        result = await ParallelPipeline(agents=agents).run("go")
        assert result.success
        assert all(ag.arun_calls == 1 for ag in agents)
        assert all(ag.run_sync_calls == 0 for ag in agents)

    @pytest.mark.asyncio
    async def test_loop_uses_arun(self) -> None:
        agent = _ArunAgent("iter")
        await LoopAgent(agent=agent, condition=lambda _out: True, max_loops=3).run("go")
        assert agent.arun_calls == 1  # condition met on the first pass
        assert agent.run_sync_calls == 0


class TestRunSyncFallback:
    """Agent-likes without ``arun`` still work via the run_sync fallback."""

    @pytest.mark.asyncio
    async def test_sequential_falls_back(self) -> None:
        agent = _RunSyncOnlyAgent("legacy")
        result = await SequentialPipeline(agents=[agent]).run("go")
        assert result.success
        assert result.final_output == "legacy"
        assert agent.run_sync_calls == 1
