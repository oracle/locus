# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Tests for the AgentConfig.tool_result_store wiring.

Covers the agent.py integration that replaces the lossy
head-truncation path with a checkpointer-backed offload when the
config slot is set, and confirms the legacy truncation path still
works when the slot is left ``None``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from locus.agent.agent import Agent
from locus.agent.config import AgentConfig
from locus.core.messages import Message, Role, ToolCall
from locus.models import ModelResponse
from locus.tools.decorator import tool
from locus.tools.result_storage import (
    REFERENCE_MARKER,
    ToolResultStore,
    extract_reference_key,
)


class _StubModel:
    name = "stub"

    def __init__(self, *, scripted: list[ModelResponse]) -> None:
        self._scripted = list(scripted)

    async def complete(
        self, messages: list[Message], tools: Any = None, **kwargs: Any
    ) -> ModelResponse:
        return self._scripted.pop(0)

    async def stream(self, *a: Any, **kw: Any) -> AsyncIterator[Any]:
        raise NotImplementedError
        yield  # pragma: no cover


@tool
def big_tool() -> str:
    """Returns a large blob that should trip the size cap."""
    return "PAYLOAD-CONTENT " * 5_000  # ~80 kB


def _build_agent(*, store: ToolResultStore | None) -> Agent:
    primary = _StubModel(
        scripted=[
            ModelResponse(
                message=Message.assistant(
                    tool_calls=[ToolCall(id="t1", name="big_tool", arguments={})]
                )
            ),
            ModelResponse(message=Message.assistant("done")),
        ]
    )
    return Agent(
        config=AgentConfig(
            model=primary,
            tools=[big_tool],
            max_iterations=3,
            max_tool_result_length=2_000,
            tool_result_store=store,
        )
    )


# ---------------------------------------------------------------------------
# Legacy path: no store → head truncation as before.
# ---------------------------------------------------------------------------


class TestLegacyTruncation:
    def test_no_store_truncates_head(self) -> None:
        agent = _build_agent(store=None)
        result = agent.run_sync("Use the big tool.")

        tool_msgs = [m for m in result.state.messages if m.role == Role.TOOL]
        assert tool_msgs
        content = tool_msgs[0].content or ""
        assert "[OUTPUT TRUNCATED" in content
        assert REFERENCE_MARKER not in content
        # Truncated to ~2k chars + the suffix marker.
        assert len(content) < 2_500


# ---------------------------------------------------------------------------
# New path: store wired → offload + reference key inlined.
# ---------------------------------------------------------------------------


class TestStoreOffload:
    def test_store_offloads_full_payload(self) -> None:
        backing: dict[str, str] = {}
        store = ToolResultStore(
            save=lambda k, v: backing.__setitem__(k, v),
            load=backing.get,
            threshold_chars=2_000,
            preview_chars=500,
        )

        agent = _build_agent(store=store)
        result = agent.run_sync("Use the big tool.")

        tool_msgs = [m for m in result.state.messages if m.role == Role.TOOL]
        assert tool_msgs
        content = tool_msgs[0].content or ""
        # New path: the inline content carries the marker, NOT the
        # legacy truncation suffix.
        assert REFERENCE_MARKER in content
        assert "[OUTPUT TRUNCATED" not in content

        # Recoverable through the store, with the original payload intact.
        key = extract_reference_key(content)
        assert key is not None
        full = store.load(key)
        assert full is not None
        assert "PAYLOAD-CONTENT " in full
        assert len(full) > 70_000


# ---------------------------------------------------------------------------
# Under-threshold path passes through unchanged regardless of store.
# ---------------------------------------------------------------------------


@tool
def small_tool() -> str:
    return "tiny output"


class TestSmallToolUnchanged:
    @pytest.mark.parametrize("with_store", [True, False])
    def test_small_output_not_offloaded(self, with_store: bool) -> None:
        backing: dict[str, str] = {}
        store: ToolResultStore | None = (
            ToolResultStore(
                save=lambda k, v: backing.__setitem__(k, v),
                load=backing.get,
                threshold_chars=2_000,
                preview_chars=500,
            )
            if with_store
            else None
        )
        primary = _StubModel(
            scripted=[
                ModelResponse(
                    message=Message.assistant(
                        tool_calls=[ToolCall(id="t1", name="small_tool", arguments={})]
                    )
                ),
                ModelResponse(message=Message.assistant("done")),
            ]
        )
        agent = Agent(
            config=AgentConfig(
                model=primary,
                tools=[small_tool],
                max_iterations=3,
                max_tool_result_length=2_000,
                tool_result_store=store,
            )
        )
        result = agent.run_sync("Use the small tool.")
        tool_msgs = [m for m in result.state.messages if m.role == Role.TOOL]
        assert tool_msgs[0].content == "tiny output"
        assert backing == {}
