# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Regression guards for defects found in the 2026-08 SDK audit.

Each test here corresponds to a shipped bug where the documentation and the
code disagreed, or where a method failed silently. They are grouped in one
file because they share a cause rather than a module: nothing executed the
claim, so nothing caught the drift.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from tulip.multiagent.graph import StateGraph


REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------
# #86 — MCPClient was advertised in the README capability matrix while absent
# from ``tulip.integrations.__all__``, so the documented import raised.
# --------------------------------------------------------------------------


def test_mcp_client_is_importable_from_the_package_root() -> None:
    from tulip.integrations import MCPClient

    assert MCPClient is not None


def test_mcp_client_is_public_api() -> None:
    """DEPRECATION.md defines public API as membership in ``__all__``."""
    from tulip import integrations

    assert "MCPClient" in integrations.__all__


# --------------------------------------------------------------------------
# #94 — starlette was a *core* runtime dependency that nothing under src/
# imported. It is a test-collection dependency and belongs in ``dev``.
# --------------------------------------------------------------------------


def _pyproject() -> dict[str, Any]:
    with (REPO_ROOT / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)


def test_starlette_is_not_a_core_dependency() -> None:
    core = _pyproject()["project"]["dependencies"]
    assert not [d for d in core if d.startswith("starlette")], (
        "starlette is imported nowhere under src/; keep it in the dev extra "
        "so `pip install tulip-agents` does not pull it."
    )


def test_core_dependencies_stay_small() -> None:
    """Leanness is a claim the project makes; make it a checked one."""
    core = _pyproject()["project"]["dependencies"]
    assert len(core) <= 4, f"core deps grew to {len(core)}: {core}"


def test_starlette_pin_survives_in_dev() -> None:
    """The <1.4 cap is load-bearing for test collection — do not lose it."""
    dev = _pyproject()["project"]["optional-dependencies"]["dev"]
    assert any(d.startswith("starlette") for d in dev)


# --------------------------------------------------------------------------
# #93 — StateGraph.aget_state() returned None unconditionally. Ported
# LangGraph code reading a paused run got a silent empty answer.
# --------------------------------------------------------------------------


class _Saved:
    def __init__(self, state: dict[str, Any] | None) -> None:
        self.metadata = {"graph_state": state} if state is not None else {}


class _Checkpointer:
    """Minimal stand-in exposing only what ``aget_state`` uses."""

    def __init__(self, saved: _Saved | None) -> None:
        self._saved = saved
        self.loaded_with: list[str] = []

    async def load(self, thread_id: str) -> _Saved | None:
        self.loaded_with.append(thread_id)
        return self._saved


class _State(BaseModel):
    step: int = 0


def _graph(checkpointer: Any = None, thread_id: str | None = None) -> StateGraph:
    graph: StateGraph = StateGraph(state_schema=_State)
    graph.config.checkpointer = checkpointer
    graph.config.thread_id = thread_id
    return graph


@pytest.mark.asyncio
async def test_aget_state_returns_the_saved_graph_state() -> None:
    cp = _Checkpointer(_Saved({"step": 3}))
    graph = _graph(cp)
    assert await graph.aget_state({"configurable": {"thread_id": "t-1"}}) == {"step": 3}
    assert cp.loaded_with == ["t-1"]


@pytest.mark.asyncio
async def test_aget_state_accepts_a_bare_thread_id_mapping() -> None:
    cp = _Checkpointer(_Saved({"step": 1}))
    assert await _graph(cp).aget_state({"thread_id": "t-2"}) == {"step": 1}
    assert cp.loaded_with == ["t-2"]


@pytest.mark.asyncio
async def test_aget_state_accepts_a_plain_string() -> None:
    cp = _Checkpointer(_Saved({"step": 2}))
    assert await _graph(cp).aget_state("t-3") == {"step": 2}


@pytest.mark.asyncio
async def test_aget_state_falls_back_to_the_graph_thread() -> None:
    cp = _Checkpointer(_Saved({"step": 4}))
    assert await _graph(cp, thread_id="t-4").aget_state() == {"step": 4}
    assert cp.loaded_with == ["t-4"]


@pytest.mark.asyncio
async def test_aget_state_returns_none_when_the_thread_has_no_checkpoint() -> None:
    assert await _graph(_Checkpointer(None)).aget_state("unseen") is None


@pytest.mark.asyncio
async def test_aget_state_returns_none_when_the_record_has_no_graph_state() -> None:
    assert await _graph(_Checkpointer(_Saved(None))).aget_state("t") is None


@pytest.mark.asyncio
async def test_aget_state_without_a_checkpointer_raises() -> None:
    with pytest.raises(ValueError, match="checkpointer"):
        await _graph().aget_state("t")


@pytest.mark.asyncio
async def test_aget_state_without_a_thread_id_raises() -> None:
    with pytest.raises(ValueError, match="thread id"):
        await _graph(_Checkpointer(None)).aget_state()


@pytest.mark.asyncio
async def test_aget_state_never_silently_returns_none_on_misuse() -> None:
    """The original bug in one assertion.

    Both misuse paths must raise. A ``None`` here is indistinguishable from
    "no checkpoint yet", which is exactly what made the stub dangerous.
    """
    for call in (_graph().aget_state("t"), _graph(_Checkpointer(None)).aget_state()):
        with pytest.raises(ValueError):
            await call


# --------------------------------------------------------------------------
# #95 — the bundled MockModel returned prose with no tool_calls, so every
# tool-centric example printed "Tool calls made: 0". It is the shared fixture
# for all 74 numbered examples, so it is worth guarding here.
# --------------------------------------------------------------------------


def _mock_model() -> Any:
    """Import ``examples/config.py``, which is outside the package."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_examples_config", REPO_ROOT / "examples" / "config.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MockModel()


_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Look up the weather.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


@pytest.mark.asyncio
async def test_mock_model_emits_a_tool_call_when_tools_are_bound() -> None:
    from tulip.core.messages import Message

    resp = await _mock_model().complete(
        [Message.user("What is the weather in Paris?")], tools=[_WEATHER_TOOL]
    )
    assert resp.tool_calls, "the mock must call a bound tool, or examples show nothing"
    assert resp.tool_calls[0].name == "get_weather"


@pytest.mark.asyncio
async def test_mock_model_synthesises_schema_valid_arguments() -> None:
    from tulip.core.messages import Message

    resp = await _mock_model().complete(
        [Message.user("What is the weather in Paris?")], tools=[_WEATHER_TOOL]
    )
    args = resp.tool_calls[0].arguments
    assert set(args) == {"city"}  # required only; defaults keep their meaning
    assert args["city"] == "Paris"  # read from the prompt, so traces stay legible


@pytest.mark.asyncio
async def test_mock_model_answers_in_prose_once_a_tool_has_replied() -> None:
    """Without this the mock would loop until the iteration cap."""
    from tulip.core.messages import Message, ToolResult

    messages = [
        Message.user("What is the weather in Paris?"),
        Message.tool(ToolResult(tool_call_id="1", name="get_weather", content="18C")),
    ]
    resp = await _mock_model().complete(messages, tools=[_WEATHER_TOOL])
    assert not resp.tool_calls
    assert resp.content


@pytest.mark.asyncio
async def test_mock_model_returns_prose_when_no_tools_are_bound() -> None:
    from tulip.core.messages import Message

    resp = await _mock_model().complete([Message.user("Hello there")], tools=None)
    assert not resp.tool_calls
    assert resp.content
