# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Running an eval suite against a :class:`~tulip.multiagent.graph.StateGraph`.

``docs/concepts/multi-agent/production.md`` claimed *"Run regression suites
against any agent or graph"* and showed ``EvalRunner(agent=graph).run(cases)``.
That could not work. :class:`~tulip.evaluation.EvalRunner` calls
``run_sync(prompt)`` and then reads ``.message``, ``.iterations`` and
``.tool_executions``; ``StateGraph.run_sync`` takes a **dict**, not a prompt,
and returns a ``GraphResult`` with none of those three. Every case errored.

The claim was worth honouring. A graph is where the interesting regressions
live — a routing change sends work down the wrong branch, a node stops
contributing to the final state — and those are exactly what a regression
suite is for.

:func:`as_eval_target` is the adapter, and it has to make two translations
that a graph cannot make for itself:

**A prompt is not a state.** ``EvalCase.prompt`` is a string; a graph starts
from a dict whose shape is the graph's own business. ``input_key`` says which
field the prompt becomes, and it has no universally right default, so
``"prompt"`` is a guess that a caller is expected to correct. Getting it wrong
is loud — the graph sees an empty input — rather than quietly evaluating
nothing.

**Nodes are the graph's tools.** A graph has no tool executions; it has an
execution order of node ids. Mapping those onto ``tool_executions`` is what
makes ``expected_tools`` and ``expected_tool_sequence`` work against a graph,
and asserting *which nodes ran, in what order* is the single most useful thing
a graph regression suite can do. The docstrings say "node" where the eval
vocabulary says "tool" so nobody has to guess which is meant.
"""

from __future__ import annotations

import json
from typing import Any

from tulip.agent.result import ToolExecution


__all__ = ["GraphEvalTarget", "as_eval_target"]


def _as_text(value: Any) -> str:
    """Whatever the graph ended with, as something a rubric can read."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str, sort_keys=True)
    except (TypeError, ValueError):  # pragma: no cover - json is very forgiving
        return str(value)


class GraphEvalResult:
    """The three fields ``EvalRunner`` reads, sourced from a ``GraphResult``.

    Deliberately not an ``AgentResult``: most of that class describes a ReAct
    loop a graph never ran, and filling it with defaults would invent a
    stop_reason, a token count and a message history that nothing measured.
    The real ``GraphResult`` stays reachable on :attr:`graph_result` so a
    caller who needs the node-level detail is not cut off from it.
    """

    def __init__(
        self, graph_result: Any, *, output_key: str | None, output_node: str | None
    ) -> None:
        self.graph_result = graph_result
        state = getattr(graph_result, "final_state", None) or {}
        outputs = getattr(graph_result, "final_outputs", None) or {}

        if output_node is not None:
            self.message = _as_text(outputs.get(output_node))
        elif output_key is not None:
            self.message = _as_text(state.get(output_key))
        else:
            # No key given: the whole accumulated state, minus the ``_node_*``
            # entries the graph keeps for its own bookkeeping. Those duplicate
            # every node's return value, so leaving them in would hand a rubric
            # each answer twice and a substring check a guaranteed match.
            self.message = _as_text({k: v for k, v in state.items() if not k.startswith("_node_")})

        self.iterations = int(getattr(graph_result, "iterations", 0) or 0)

        # Nodes, in the order they ran, presented as the eval vocabulary's
        # "tool executions" so expected_tools / expected_tool_sequence work.
        self.tool_executions = tuple(
            ToolExecution(
                tool_name=node_id,
                tool_call_id=f"node-{index}",
                # A node takes the whole state, not named arguments, so there
                # is nothing honest to put here.
                arguments={},
                result=None,
            )
            for index, node_id in enumerate(getattr(graph_result, "execution_order", ()) or ())
        )

        self.error = None if getattr(graph_result, "success", True) else "graph execution failed"


class GraphEvalTarget:
    """A ``StateGraph`` wearing the shape ``EvalRunner`` expects.

    Args:
        graph: The graph under test.
        input_key: Which field of the graph's initial state the case prompt
            becomes. There is no universally right default — a graph's input
            schema is its own — so ``"prompt"`` is a starting guess to correct
            rather than a convention to rely on.
        output_key: Which field of the graph's **final state** holds the
            answer to grade. That is where a graph's result accumulates —
            ``final_outputs`` is keyed by node id, which is a different
            question and has its own argument below.
        output_node: Grade the output of one named **node** instead. Use this
            when the answer never lands in shared state, or when only the last
            node's own return value matters.
        initial_state: Fields merged under the prompt on every case — a graph
            that needs a config blob or a session id to start at all should not
            have to encode it in every ``EvalCase``.
    """

    def __init__(
        self,
        graph: Any,
        *,
        input_key: str = "prompt",
        output_key: str | None = None,
        output_node: str | None = None,
        initial_state: dict[str, Any] | None = None,
    ) -> None:
        if output_key is not None and output_node is not None:
            raise ValueError(
                "pass output_key (a field of the final state) or output_node "
                "(one node's own output), not both — they name different things"
            )
        self.graph = graph
        self.input_key = input_key
        self.output_key = output_key
        self.output_node = output_node
        self.initial_state = dict(initial_state or {})

    def _inputs(self, prompt: str) -> dict[str, Any]:
        return {**self.initial_state, self.input_key: prompt}

    async def arun(self, prompt: str, **_: Any) -> GraphEvalResult:
        result = await self.graph.execute(self._inputs(prompt))
        return GraphEvalResult(result, output_key=self.output_key, output_node=self.output_node)

    def run_sync(self, prompt: str, **_: Any) -> GraphEvalResult:
        import asyncio  # noqa: PLC0415

        return asyncio.run(self.arun(prompt))


def as_eval_target(
    graph: Any,
    *,
    input_key: str = "prompt",
    output_key: str | None = None,
    output_node: str | None = None,
    initial_state: dict[str, Any] | None = None,
) -> GraphEvalTarget:
    """Wrap a graph so :class:`~tulip.evaluation.EvalRunner` can run it.

        runner = EvalRunner(agent=as_eval_target(graph, input_key="question"))
        report = runner.run([
            EvalCase(
                name="routes_billing_to_the_billing_node",
                prompt="I was charged twice",
                expected_tool_sequence=["classify", "billing"],
            )
        ])

    ``expected_tools`` and ``expected_tool_sequence`` match on **node ids**;
    everything else — ``expected_output_contains``, budgets, a rubric graded by
    :class:`~tulip.evaluation.LLMJudge` — behaves exactly as it does for an
    agent.
    """
    return GraphEvalTarget(
        graph,
        input_key=input_key,
        output_key=output_key,
        output_node=output_node,
        initial_state=initial_state,
    )
