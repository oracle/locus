# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for ``multiagent.visualize`` (Mermaid + ASCII diagrams).

The visualisation helpers don't share a Pydantic schema with
:class:`StateGraph` — they duck-type against ``graph.nodes`` /
``graph.edges`` / ``graph.conditional_edges``. The tests below build
minimal stub graphs that satisfy that contract directly so we don't
drag the full StateGraph machinery into the diagram tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from locus.multiagent.visualize import draw_ascii, draw_mermaid


# ---------------------------------------------------------------------------
# Tiny stand-in graph types — duck-type a StateGraph for the visualise calls.
# ---------------------------------------------------------------------------


@dataclass
class _Node:
    name: str


@dataclass
class _Edge:
    source_id: str
    target_id: str


@dataclass
class _CondEdge:
    source_id: str
    targets: dict[str, str]
    default_target: str | None = None


@dataclass
class _Graph:
    nodes: dict[str, _Node]
    edges: list[_Edge]
    conditional_edges: list[_CondEdge] = field(default_factory=list)


def _linear_graph() -> _Graph:
    """``__START__ → process → __END__`` — the canonical example."""
    return _Graph(
        nodes={
            "__START__": _Node(name="__START__"),
            "process": _Node(name="process"),
            "__END__": _Node(name="__END__"),
        },
        edges=[
            _Edge(source_id="__START__", target_id="process"),
            _Edge(source_id="process", target_id="__END__"),
        ],
    )


def _branching_graph() -> _Graph:
    """A graph with a conditional edge to exercise the dotted-arrow branch."""
    return _Graph(
        nodes={
            "__START__": _Node(name="__START__"),
            "router": _Node(name="router"),
            "branch_a": _Node(name="branch_a"),
            "branch_b": _Node(name="branch_b"),
            "__END__": _Node(name="__END__"),
        },
        edges=[_Edge(source_id="__START__", target_id="router")],
        conditional_edges=[
            _CondEdge(
                source_id="router",
                targets={"a": "branch_a", "b": "branch_b"},
                default_target="__END__",
            )
        ],
    )


# ---------------------------------------------------------------------------
# Mermaid
# ---------------------------------------------------------------------------


class TestDrawMermaid:
    def test_emits_graph_directive(self) -> None:
        out = draw_mermaid(_linear_graph())
        assert out.startswith("graph TD")

    def test_respects_direction_arg(self) -> None:
        out = draw_mermaid(_linear_graph(), direction="LR")
        assert out.startswith("graph LR")

    def test_renders_start_and_end_with_round_node_shape(self) -> None:
        out = draw_mermaid(_linear_graph())
        assert "__START__([Start])" in out
        assert "__END__([End])" in out

    def test_renders_intermediate_node_with_square_shape(self) -> None:
        out = draw_mermaid(_linear_graph())
        assert "process[process]" in out

    def test_renders_solid_arrow_per_edge(self) -> None:
        out = draw_mermaid(_linear_graph())
        assert "__START__ --> process" in out
        assert "process --> __END__" in out

    def test_renders_conditional_edges_as_dotted_arrows(self) -> None:
        out = draw_mermaid(_branching_graph())
        assert "router -.->|a| branch_a" in out
        assert "router -.->|b| branch_b" in out

    def test_renders_default_target_with_default_label(self) -> None:
        out = draw_mermaid(_branching_graph())
        assert "router -.->|default| __END__" in out

    def test_node_id_safe_substitution(self) -> None:
        graph = _Graph(
            nodes={"my-node": _Node(name="my node")},
            edges=[],
        )
        out = draw_mermaid(graph)
        # Hyphens and spaces in IDs are normalised so Mermaid doesn't barf.
        assert "my_node[my node]" in out

    def test_node_without_name_attr_falls_back_to_id(self) -> None:
        # Plain object with no ``name`` attribute — the helper falls back
        # to the node_id string for the label.
        @dataclass
        class _NamelessNode:
            pass

        graph = _Graph(
            nodes={"plain": _NamelessNode()},  # type: ignore[arg-type]
            edges=[],
        )
        out = draw_mermaid(graph)
        assert "plain[plain]" in out

    def test_conditional_edge_without_default_target(self) -> None:
        # Cover the branch where ``default_target`` is None — only the
        # explicit conditional targets are emitted.
        graph = _Graph(
            nodes={
                "router": _Node("router"),
                "a": _Node("a"),
                "b": _Node("b"),
            },
            edges=[],
            conditional_edges=[_CondEdge(source_id="router", targets={"a": "a", "b": "b"})],
        )
        out = draw_mermaid(graph)
        assert "router -.->|a| a" in out
        assert "router -.->|b| b" in out
        assert "default" not in out

    def test_handles_graph_without_conditional_edges_attr(self) -> None:
        # ``getattr(graph, "conditional_edges", [])`` is the relaxed
        # contract — duck-typed graphs that don't expose the attribute
        # at all must still render.
        @dataclass
        class _MinimalGraph:
            nodes: dict[str, _Node]
            edges: list[_Edge]

        graph = _MinimalGraph(
            nodes={"__START__": _Node("__START__"), "__END__": _Node("__END__")},
            edges=[_Edge("__START__", "__END__")],
        )
        out = draw_mermaid(graph)  # type: ignore[arg-type]
        assert "__START__ --> __END__" in out


# ---------------------------------------------------------------------------
# ASCII
# ---------------------------------------------------------------------------


class TestDrawAscii:
    def test_linear_graph_renders_single_arrow_per_edge(self) -> None:
        out = draw_ascii(_linear_graph())
        assert "[Start] --> [process]" in out
        assert "[process] --> [End]" in out

    def test_branching_graph_renders_set_of_targets(self) -> None:
        out = draw_ascii(_branching_graph())
        # Multi-target nodes use ``{a, b, ...}`` syntax.
        assert "[router] --> {" in out
        assert "branch_a" in out
        assert "branch_b" in out

    def test_does_not_revisit_nodes(self) -> None:
        # Diamond: __START__ -> a -> b, __START__ -> b — ``b`` should only
        # render its outgoing edges once.
        graph = _Graph(
            nodes={
                "__START__": _Node("__START__"),
                "a": _Node("a"),
                "b": _Node("b"),
                "__END__": _Node("__END__"),
            },
            edges=[
                _Edge("__START__", "a"),
                _Edge("a", "b"),
                _Edge("__START__", "b"),
                _Edge("b", "__END__"),
            ],
        )
        out = draw_ascii(graph)
        # ``[b] --> [End]`` must appear exactly once even though ``b``
        # is reachable via two paths.
        assert out.count("[b] --> [End]") == 1

    def test_conditional_edges_contribute_targets(self) -> None:
        out = draw_ascii(_branching_graph())
        # The router has 2 conditional + 1 default → 3 targets total.
        # The router appears exactly once on the LHS (``[router] -->``).
        assert out.count("[router] -->") == 1
        assert "branch_a" in out
        assert "branch_b" in out
        assert "End" in out

    def test_node_without_name_attr_falls_back_to_id(self) -> None:
        @dataclass
        class _NamelessNode:
            pass

        graph = _Graph(
            nodes={
                "__START__": _Node("__START__"),
                "plain": _NamelessNode(),  # type: ignore[dict-item]
            },
            edges=[_Edge("__START__", "plain")],
        )
        out = draw_ascii(graph)
        assert "[Start] --> [plain]" in out

    def test_ascii_conditional_edge_without_default_target(self) -> None:
        # Cover the ``if cond_edge.default_target`` branch in
        # ``draw_ascii`` when no default is set.
        graph = _Graph(
            nodes={
                "__START__": _Node("__START__"),
                "router": _Node("router"),
                "a": _Node("a"),
            },
            edges=[_Edge("__START__", "router")],
            conditional_edges=[_CondEdge(source_id="router", targets={"a": "a"})],
        )
        out = draw_ascii(graph)
        # Single target for ``router`` because no default was added.
        assert "[router] --> [a]" in out
