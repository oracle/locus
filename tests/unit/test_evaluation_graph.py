# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Running an eval suite against a graph, which the docs promised and could not do.

``production.md`` showed ``EvalRunner(agent=graph).run(cases)`` under *"Run
regression suites against any agent or graph"*. Every case errored:
``EvalRunner`` calls ``run_sync(prompt)`` and reads ``.message``,
``.iterations`` and ``.tool_executions``; ``StateGraph.run_sync`` takes a dict
and returns a ``GraphResult`` with none of those.

The graph built here is the one worth testing — a classifier routing to one of
two branches — because the regression a graph suite exists to catch is exactly
"a change sent this down the wrong branch".
"""

from __future__ import annotations

from typing import Any

import pytest

from tulip.evaluation import EvalCase, EvalRunner, as_eval_target
from tulip.evaluation.graph import GraphEvalTarget
from tulip.multiagent.graph import StateGraph


def _routing_graph() -> StateGraph:
    def classify(state: dict[str, Any]) -> dict[str, Any]:
        return {"topic": "billing" if "charged" in state.get("prompt", "") else "general"}

    def billing(state: dict[str, Any]) -> dict[str, Any]:
        return {"answer": "Refund issued for the duplicate charge."}

    def general(state: dict[str, Any]) -> dict[str, Any]:
        return {"answer": "Passing you to a human."}

    graph = StateGraph()
    for name, fn in (("classify", classify), ("billing", billing), ("general", general)):
        graph.add_node(name, fn)
    graph.set_entry_point("classify")
    graph.add_conditional_edges(
        "classify", lambda s: s.get("topic"), {"billing": "billing", "general": "general"}
    )
    graph.set_finish_point("billing")
    graph.set_finish_point("general")
    return graph


def _run(cases: list[EvalCase], **kwargs: Any) -> Any:
    target = as_eval_target(_routing_graph(), **kwargs)
    return EvalRunner(agent=target).run(cases)


# --------------------------------------------------------------------------
# The claim itself
# --------------------------------------------------------------------------


def test_a_suite_runs_against_a_graph_at_all() -> None:
    report = _run([EvalCase(name="smoke", prompt="hello")])

    assert report.results[0].error is None, report.results[0].error
    assert report.total_cases == 1


def test_routing_is_asserted_by_node_order() -> None:
    """The regression a graph suite is for: work went down the wrong branch."""
    report = _run(
        [
            EvalCase(
                name="billing",
                prompt="I was charged twice",
                expected_tool_sequence=["classify", "billing"],
            ),
            EvalCase(
                name="general",
                prompt="what are your hours",
                expected_tool_sequence=["classify", "general"],
            ),
        ],
        output_key="answer",
    )

    assert report.passed == 2, report.summary()


def test_a_wrong_route_fails_the_case() -> None:
    """A suite that cannot fail is not a suite."""
    report = _run(
        [
            EvalCase(
                name="misrouted",
                prompt="what are your hours",
                expected_tool_sequence=["classify", "billing"],
            )
        ],
        output_key="answer",
    )

    assert not report.results[0].passed
    assert report.results[0].checks["tool_sequence"] is False


def test_node_membership_works_too() -> None:
    report = _run(
        [EvalCase(name="reached", prompt="I was charged twice", expected_tools=["billing"])],
        output_key="answer",
    )

    assert report.results[0].checks["tool_called:billing"] is True


# --------------------------------------------------------------------------
# Addressing the answer — final state vs one node's output
# --------------------------------------------------------------------------


def test_output_key_reads_the_final_state() -> None:
    report = _run(
        [
            EvalCase(
                name="graded",
                prompt="I was charged twice",
                expected_output_contains=["Refund"],
            )
        ],
        output_key="answer",
    )

    assert report.results[0].passed
    assert "Refund" in report.results[0].output


def test_output_node_reads_one_nodes_own_output() -> None:
    """A different question from ``output_key``, and it has its own argument."""
    report = _run(
        [
            EvalCase(
                name="graded",
                prompt="I was charged twice",
                expected_output_contains=["Refund"],
            )
        ],
        output_node="billing",
    )

    assert report.results[0].passed


def test_naming_both_is_rejected_rather_than_silently_preferring_one() -> None:
    with pytest.raises(ValueError, match="not both"):
        as_eval_target(_routing_graph(), output_key="answer", output_node="billing")


def test_with_no_key_the_bookkeeping_fields_are_left_out() -> None:
    """``_node_*`` duplicates every node's return value into the state.

    Leaving them in hands a rubric each answer twice, and gives a substring
    check a second place to match — so a ``expected_output_contains`` could
    pass on the bookkeeping copy after the real field stopped being written.
    """
    report = _run([EvalCase(name="raw", prompt="I was charged twice")])

    output = report.results[0].output
    assert "_node_" not in output
    assert "Refund" in output


# --------------------------------------------------------------------------
# Feeding the graph
# --------------------------------------------------------------------------


def test_the_prompt_lands_where_the_graph_expects_it() -> None:
    """A graph's input schema is its own; ``input_key`` is how it is told."""

    def echo(state: dict[str, Any]) -> dict[str, Any]:
        return {"answer": f"asked: {state.get('question', '')}"}

    graph = StateGraph()
    graph.add_node("echo", echo)
    graph.set_entry_point("echo")
    graph.set_finish_point("echo")

    report = EvalRunner(agent=as_eval_target(graph, input_key="question", output_key="answer")).run(
        [EvalCase(name="echo", prompt="ping", expected_output_contains=["asked: ping"])]
    )

    assert report.results[0].passed, report.results[0].output


def test_initial_state_is_merged_under_every_case() -> None:
    """A graph needing a session id to start should not need it in each case."""

    def echo(state: dict[str, Any]) -> dict[str, Any]:
        return {"answer": f"{state.get('tenant')}:{state.get('prompt')}"}

    graph = StateGraph()
    graph.add_node("echo", echo)
    graph.set_entry_point("echo")
    graph.set_finish_point("echo")

    report = EvalRunner(
        agent=as_eval_target(graph, output_key="answer", initial_state={"tenant": "acme"})
    ).run([EvalCase(name="scoped", prompt="hi", expected_output_contains=["acme:hi"])])

    assert report.results[0].passed, report.results[0].output


def test_the_prompt_wins_over_a_clashing_initial_state_field() -> None:
    """Otherwise every case would silently evaluate the same fixed input."""
    target = GraphEvalTarget(
        _routing_graph(), input_key="prompt", initial_state={"prompt": "leftover"}
    )

    assert target._inputs("real question")["prompt"] == "real question"


# --------------------------------------------------------------------------
# The async path, and what the adapter deliberately does not pretend to be
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_async_runner_grades_a_graph_the_same_way() -> None:
    report = await EvalRunner(
        agent=as_eval_target(_routing_graph(), output_key="answer"), concurrency=1
    ).arun(
        [
            EvalCase(
                name="billing",
                prompt="I was charged twice",
                expected_tool_sequence=["classify", "billing"],
            )
        ]
    )

    assert report.passed == 1, report.summary()


@pytest.mark.asyncio
async def test_the_real_graph_result_stays_reachable() -> None:
    """Node-level detail is not thrown away to fit the agent-shaped hole."""
    result = await as_eval_target(_routing_graph()).arun("I was charged twice")

    assert result.graph_result.execution_order == ["classify", "billing"]
    assert result.graph_result.success is True
