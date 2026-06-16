# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Runnable adapters — make non-``Agent`` shapes look like an Agent to
``AgentServer`` and ``A2AServer``.

Locus's HTTP/SSE server and A2A server both consume the Agent contract::

    async for event in obj.run(prompt, thread_id=..., metadata=...) -> AsyncIterator[LocusEvent]

A ``StateGraph`` (or any object with a ``stream(inputs) -> AsyncIterator``)
does not natively satisfy that contract — its inputs are dicts, its
events are :class:`locus.multiagent.graph.StreamEvent` instances, and
its final state is a dict.

:class:`GraphRunnable` is a thin shim that bridges those two surfaces.
Wrap a compiled graph and use it everywhere an ``Agent`` is expected:

.. code-block:: python

    from locus.multiagent.graph import StateGraph
    from locus.server import AgentServer, GraphRunnable
    from locus.a2a import A2AServer, AgentSkill

    graph = StateGraph(...).compile()
    runnable = GraphRunnable(
        graph,
        input_key="prompt",  # str prompt → {"prompt": "..."}
        output_key="answer",  # final_state["answer"] → user-visible reply
    )

    # As an HTTP/SSE server:
    AgentServer(agent=runnable).run(port=8000)

    # As a spec-compliant A2A endpoint:
    A2AServer(
        agent=runnable,
        api_key="...",
        name="my-graph",
        skills=[AgentSkill(id="planner", name="Planner", description="...")],
    ).run(port=7421)

Closes #213. The adapter is intentionally minimal — it preserves the
caller's graph as-is and only translates the streaming surface.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from locus.core.events import LocusEvent, TerminateEvent, ThinkEvent


if TYPE_CHECKING:
    from locus.multiagent.graph import StateGraph


# Sentinel meaning "use the whole final_state as the user-visible message".
_USE_FULL_STATE = object()


class GraphRunnable:
    """Wrap a graph so :class:`AgentServer` / :class:`A2AServer` can publish it.

    The adapter speaks the Agent contract — ``async def run(prompt, *,
    thread_id=None, metadata=None) -> AsyncIterator[LocusEvent]`` — by
    driving ``graph.stream({input_key: prompt})`` underneath. Each per-node
    ``StreamEvent`` becomes a :class:`ThinkEvent` so SSE / A2A clients see
    activity in real time; the terminal final-state event becomes a
    :class:`TerminateEvent` with ``final_message`` set to
    ``final_state[output_key]`` (or the full state stringified when
    ``output_key`` is unset).

    Args:
        graph: A compiled :class:`StateGraph` (or any object with an
            async ``stream(inputs) -> AsyncIterator``). Duck-typed so
            future runnable shapes (compiled pipelines, custom DAGs)
            slot in unchanged.
        input_key: Key under which the user prompt is placed in the
            graph's initial state. Defaults to ``"prompt"`` because
            both ``KEY_PROMPT`` (deep research) and most StateGraph
            examples use that name.
        output_key: Key in the graph's final state that holds the
            user-visible reply. ``None`` (default) means stringify the
            whole final-state dict — usable for diagnostics but most
            callers will set this explicitly.
        name: Optional display name (echoed into ``ThinkEvent``
            reasoning lines so SSE consumers can tell graphs apart from
            agents).

    Notes:
        * ``thread_id`` and ``metadata`` from the server are accepted but
          not propagated yet — graphs don't have a native notion of a
          server-managed thread. Per-graph checkpointers handle the same
          concern at a different layer.
        * The adapter does NOT touch the agent loop; agents inside the
          graph keep emitting their own ``LocusEvent`` stream onto the
          locus event bus. The shim only translates the *graph*'s outer
          stream.
    """

    def __init__(
        self,
        graph: StateGraph | Any,
        *,
        input_key: str = "prompt",
        output_key: str | None = None,
        name: str | None = None,
    ) -> None:
        self.graph = graph
        self.input_key = input_key
        self.output_key = output_key
        self.name = name or getattr(graph, "name", None) or type(graph).__name__

    async def run(
        self,
        prompt: str,
        *,
        thread_id: str | None = None,  # noqa: ARG002 — accepted for the Agent contract
        metadata: dict[str, Any] | None = None,  # noqa: ARG002 — same
    ) -> AsyncIterator[LocusEvent]:
        """Drive ``self.graph.stream(...)`` and yield LocusEvent shapes.

        Yields:
            One :class:`ThinkEvent` per intermediate ``StreamEvent`` (so
            consumers see node-by-node progress on the SSE wire), then a
            terminal :class:`TerminateEvent` with the user-visible reply
            extracted from the graph's final state.
        """
        inputs = {self.input_key: prompt}

        iteration = 0
        last_data: Any = None
        async for ev in self.graph.stream(inputs):
            # ``StreamEvent`` is the graph's per-node envelope. We treat
            # the *last* event as the terminal final-state and only emit
            # a TerminateEvent for it; everything in between becomes a
            # ThinkEvent so the SSE stream stays alive and informative.
            iteration += 1
            last_data = ev.data
            node_id = getattr(ev, "node_id", None)
            mode = getattr(ev, "mode", None)
            label = f"[{self.name}]"
            if node_id:
                label += f" node={node_id}"
            if mode is not None:
                label += f" mode={mode}"
            yield ThinkEvent(
                iteration=iteration,
                reasoning=f"{label} produced data of type {type(ev.data).__name__}",
                tool_calls=[],
            )

        final_message = self._extract_reply(last_data)
        yield TerminateEvent(
            reason="complete",
            iterations_used=iteration,
            final_confidence=1.0,
            total_tool_calls=0,
            final_message=final_message,
        )

    def _extract_reply(self, last_data: Any) -> str:
        """Pull the user-visible reply out of the graph's final state."""
        if last_data is None:
            return ""
        if not isinstance(last_data, dict):
            return str(last_data)
        if self.output_key is not None:
            value = last_data.get(self.output_key)
            if value is None:
                return ""
            return value if isinstance(value, str) else str(value)
        # Fallback: stringify the whole dict. Useful for diagnostics
        # but most real callers should pass ``output_key``.
        return str(last_data)
