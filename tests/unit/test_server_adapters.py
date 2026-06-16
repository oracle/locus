# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for ``locus.server.adapters.GraphRunnable`` (closes #213).

The adapter lets ``AgentServer`` / ``A2AServer`` publish a graph by
making it look like an Agent. The tests use a scripted graph stand-in
so we exercise the translation logic — ``StreamEvent`` → ``ThinkEvent``
per node, plus a terminal ``TerminateEvent`` with the user-visible
reply pulled from the final state — without spinning up a real model.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from locus.core.events import TerminateEvent, ThinkEvent
from locus.multiagent.graph import StreamEvent, StreamMode
from locus.server import GraphRunnable


class _ScriptedGraph:
    """Yields a fixed list of StreamEvents from .stream() — no compile step,
    no real nodes. Captures the inputs the adapter passes so tests can
    assert on the input-key plumbing."""

    name = "scripted"

    def __init__(self, events: list[StreamEvent]) -> None:
        self._events = list(events)
        self.last_inputs: dict[str, Any] | None = None

    async def stream(
        self,
        inputs: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        self.last_inputs = inputs
        for ev in self._events:
            yield ev


def _node_event(node: str, data: Any) -> StreamEvent:
    return StreamEvent(mode=StreamMode.NODES, node_id=node, data=data)


def _values_event(data: Any) -> StreamEvent:
    return StreamEvent(mode=StreamMode.VALUES, node_id=None, data=data)


@pytest.mark.asyncio
async def test_run_emits_think_events_per_node_then_terminate() -> None:
    """Each per-node StreamEvent becomes a ThinkEvent; the final-state event
    becomes a TerminateEvent whose ``final_message`` is ``final_state[output_key]``.
    """
    graph = _ScriptedGraph(
        [
            _node_event("planner", {"plan": "step 1, step 2"}),
            _node_event("executor", {"intermediate": "done step 1"}),
            _values_event({"answer": "42"}),
        ]
    )
    runnable = GraphRunnable(graph, input_key="prompt", output_key="answer")

    events = [ev async for ev in runnable.run("what is the meaning of life?")]

    think_events = [ev for ev in events if isinstance(ev, ThinkEvent)]
    terminate_events = [ev for ev in events if isinstance(ev, TerminateEvent)]

    assert len(think_events) == 3
    assert "node=planner" in (think_events[0].reasoning or "")
    assert "node=executor" in (think_events[1].reasoning or "")
    assert len(terminate_events) == 1
    assert terminate_events[0].reason == "complete"
    assert terminate_events[0].final_message == "42"
    assert terminate_events[0].iterations_used == 3
    # Input-key plumbing.
    assert graph.last_inputs == {"prompt": "what is the meaning of life?"}


@pytest.mark.asyncio
async def test_custom_input_key_is_respected() -> None:
    """``input_key`` overrides the default ``"prompt"`` so callers can match
    whichever state key the graph actually reads."""
    graph = _ScriptedGraph(
        [
            _values_event({"answer": "ok"}),
        ]
    )
    runnable = GraphRunnable(graph, input_key="user_query", output_key="answer")

    [_ async for _ in runnable.run("hi")]

    assert graph.last_inputs == {"user_query": "hi"}


@pytest.mark.asyncio
async def test_output_key_unset_falls_back_to_state_str() -> None:
    """Without ``output_key`` the adapter stringifies the whole final state.

    Less useful in production but a reasonable default for diagnostic dumps.
    """
    graph = _ScriptedGraph(
        [
            _values_event({"answer": "ok", "trace": ["a", "b"]}),
        ]
    )
    runnable = GraphRunnable(graph)

    events = [ev async for ev in runnable.run("hi")]
    terminate = next(ev for ev in events if isinstance(ev, TerminateEvent))
    assert "answer" in (terminate.final_message or "")
    assert "trace" in (terminate.final_message or "")


@pytest.mark.asyncio
async def test_non_dict_final_state_is_stringified() -> None:
    """A graph whose final-state event isn't a dict (uncommon — strict
    StateGraph state dicts) gets stringified verbatim."""
    graph = _ScriptedGraph(
        [
            _values_event("final reply as a bare string"),
        ]
    )
    runnable = GraphRunnable(graph, output_key="answer")

    events = [ev async for ev in runnable.run("hi")]
    terminate = next(ev for ev in events if isinstance(ev, TerminateEvent))
    assert terminate.final_message == "final reply as a bare string"


@pytest.mark.asyncio
async def test_empty_graph_still_emits_terminate() -> None:
    """A graph that yields zero events still terminates cleanly with an
    empty reply — protects against a graph that fast-paths to no-op."""
    graph = _ScriptedGraph([])
    runnable = GraphRunnable(graph, output_key="answer")

    events = [ev async for ev in runnable.run("hi")]
    assert len(events) == 1
    assert isinstance(events[0], TerminateEvent)
    assert events[0].final_message == ""
    assert events[0].iterations_used == 0


@pytest.mark.asyncio
async def test_thread_id_and_metadata_accepted_but_ignored() -> None:
    """The adapter takes ``thread_id`` / ``metadata`` to honour the Agent
    contract so servers passing them through don't blow up — currently
    they're a no-op (graphs use their own checkpointer layer for state)."""
    graph = _ScriptedGraph(
        [
            _values_event({"answer": "ok"}),
        ]
    )
    runnable = GraphRunnable(graph, output_key="answer")

    # Should not raise.
    [ev async for ev in runnable.run("hi", thread_id="t-1", metadata={"k": "v"})]


@pytest.mark.asyncio
async def test_satisfies_agent_contract_for_a2a_server() -> None:
    """``A2AServer`` and ``AgentServer`` duck-type their ``agent`` argument
    to ``run(prompt, thread_id=..., metadata=...) -> AsyncIterator[LocusEvent]``.
    Verify ``GraphRunnable`` actually matches that signature so wiring it
    in doesn't surface at runtime."""
    import inspect

    runnable = GraphRunnable(_ScriptedGraph([]))
    sig = inspect.signature(runnable.run)
    assert {"prompt", "thread_id", "metadata"} <= set(sig.parameters)
