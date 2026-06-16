# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Graph visualization — Mermaid and ASCII diagrams.

Generate visual representations of graph workflows for documentation,
debugging, and stakeholder communication.

Example:
    from locus.multiagent.visualize import draw_mermaid, draw_ascii

    graph = StateGraph()
    graph.add_node("process", process_fn)
    graph.add_edge(START, "process")
    graph.add_edge("process", END)

    print(draw_mermaid(graph))
    # graph TD
    #     __START__([Start]) --> process[process]
    #     process --> __END__([End])

    print(draw_ascii(graph))
    # [Start] → [process] → [End]
"""

from __future__ import annotations

from typing import Any


def draw_mermaid(graph: Any, direction: str = "TD") -> str:
    """Generate a Mermaid diagram from a graph.

    Args:
        graph: A StateGraph or Graph instance.
        direction: Flow direction — TD (top-down), LR (left-right).

    Returns:
        Mermaid diagram as a string.

    Example output:
        graph TD
            __START__([Start]) --> process[process]
            process --> __END__([End])
    """
    lines = [f"graph {direction}"]

    # Node shapes
    for node_id, node in graph.nodes.items():
        if node_id == "__START__":
            lines.append("    __START__([Start])")
        elif node_id == "__END__":
            lines.append("    __END__([End])")
        else:
            label = node.name if hasattr(node, "name") else node_id
            lines.append(f"    {_safe_id(node_id)}[{label}]")

    lines.append("")

    # Edges
    for edge in graph.edges:
        src = _safe_id(edge.source_id)
        tgt = _safe_id(edge.target_id)
        lines.append(f"    {src} --> {tgt}")

    # Conditional edges
    for cond_edge in getattr(graph, "conditional_edges", []):
        src = _safe_id(cond_edge.source_id)
        for target_name, target_id in cond_edge.targets.items():
            tgt = _safe_id(target_id)
            lines.append(f"    {src} -.->|{target_name}| {tgt}")
        if cond_edge.default_target:
            tgt = _safe_id(cond_edge.default_target)
            lines.append(f"    {src} -.->|default| {tgt}")

    return "\n".join(lines)


def draw_ascii(graph: Any) -> str:
    """Generate a simple ASCII diagram from a graph.

    Args:
        graph: A StateGraph or Graph instance.

    Returns:
        ASCII diagram as a string.
    """
    lines: list[str] = []

    # Build adjacency
    adj: dict[str, list[str]] = {}
    for edge in graph.edges:
        adj.setdefault(edge.source_id, []).append(edge.target_id)

    for cond_edge in getattr(graph, "conditional_edges", []):
        targets = list(cond_edge.targets.values())
        if cond_edge.default_target:
            targets.append(cond_edge.default_target)
        adj.setdefault(cond_edge.source_id, []).extend(targets)

    # Walk from START
    visited: set[str] = set()
    queue = ["__START__"]
    while queue:
        node_id = queue.pop(0)
        if node_id in visited:
            continue
        visited.add(node_id)

        label = _display_name(node_id, graph)
        targets = adj.get(node_id, [])

        if targets:
            target_labels = [_display_name(t, graph) for t in targets]
            if len(target_labels) == 1:
                lines.append(f"[{label}] --> [{target_labels[0]}]")
            else:
                lines.append(f"[{label}] --> {{{', '.join(target_labels)}}}")
            queue.extend(t for t in targets if t not in visited)

    return "\n".join(lines)


def _safe_id(node_id: str) -> str:
    """Make a node ID safe for Mermaid syntax."""
    return node_id.replace("-", "_").replace(" ", "_")


def _display_name(node_id: str, graph: Any) -> str:
    """Get display name for a node."""
    if node_id == "__START__":
        return "Start"
    if node_id == "__END__":
        return "End"
    node = graph.nodes.get(node_id)
    if node and hasattr(node, "name"):
        return str(node.name)
    return node_id
