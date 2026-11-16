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

A thread's checkpoint history, with a fork drawn as a branch of the thread
it came from, is :func:`draw_mermaid_history`; it takes the checkpointer and
the thread ids to draw.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from locus.memory.time_travel import CheckpointInfo

#: Characters a Mermaid node id may not contain, replaced by ``_``.
_UNSAFE = re.compile(r"[^0-9A-Za-z_]")


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


async def draw_mermaid_history(
    checkpointer: Any, *thread_ids: str, limit: int = 50, direction: str = "LR"
) -> str:
    """Mermaid flowchart of one or more threads' checkpoints; a fork is a branch.

    One subgraph per thread, its checkpoints oldest to newest with an arrow
    from each to the one that descends from it (``parent_checkpoint_id``
    from ``checkpointer.list``). A thread forked from another is joined to
    the checkpoint it was forked at by a dashed ``fork`` arrow when that
    thread is drawn too, so the two threads read as one history that
    branches. A run started ``from_checkpoint=`` shows the same way inside
    its own thread: its checkpoints hang off the one it started from.

    Args:
        checkpointer: Any :class:`~locus.memory.BaseCheckpointer`.
        *thread_ids: The threads to draw; give both a thread and its fork
            to see the branch.
        limit: Checkpoints drawn per thread, newest first.
        direction: Flow direction — LR (left-right) or TD (top-down).

    Example output::

        graph LR
            subgraph thread_main["main"]
                direction LR
                cp_main__c1["c1<br/>iteration 1"] --> cp_main__c2["c2<br/>iteration 2"]
            end
            subgraph thread_main_alt["main-alt"]
                direction LR
                cp_main_alt__f1["f1<br/>iteration 1<br/>fork"]
            end
            cp_main__c1 -.->|fork| cp_main_alt__f1
    """
    lines = [f"graph {direction}"]
    nodes: dict[tuple[str, str], str] = {}
    histories: dict[str, list[CheckpointInfo]] = {}
    for thread_id in thread_ids:
        infos = await checkpointer.list(thread_id, limit=limit)
        histories[thread_id] = infos
        lines.append(f'    subgraph {_history_id("thread_" + thread_id)}["{_quote(thread_id)}"]')
        lines.append(f"        direction {direction}")
        for info in reversed(infos):
            node = _history_id(f"cp_{thread_id}__{info.checkpoint_id}")
            nodes[(thread_id, info.checkpoint_id)] = node
            lines.append(f'        {node}["{_history_label(info)}"]')
        lines.append("    end")

    forks: list[str] = []
    for thread_id, infos in histories.items():
        for info in infos:
            if info.parent_checkpoint_id is None:
                continue
            parent_thread = info.forked_from or thread_id
            parent = nodes.get((parent_thread, info.parent_checkpoint_id))
            if parent is None:
                # The parent is outside the drawn window, or its thread was not given.
                continue
            child = nodes[(thread_id, info.checkpoint_id)]
            if parent_thread == thread_id:
                lines.append(f"    {parent} --> {child}")
            else:
                forks.append(f"    {parent} -.->|fork| {child}")
    lines.extend(forks)
    return "\n".join(lines)


def _history_id(raw: str) -> str:
    """A Mermaid node id from a thread or checkpoint id (which may hold anything)."""
    return _UNSAFE.sub("_", raw)


def _quote(text: str) -> str:
    """``text`` safe inside a double-quoted Mermaid label."""
    return text.replace('"', "#quot;")


def _history_label(info: CheckpointInfo) -> str:
    """What a checkpoint's node shows: a short id, the iteration, and how it was written."""
    checkpoint_id = info.checkpoint_id
    parts = [_quote(checkpoint_id if len(checkpoint_id) <= 12 else checkpoint_id[:12] + "…")]
    if info.iteration is not None:
        parts.append(f"iteration {info.iteration}")
    if info.source:
        parts.append(_quote(info.source))
    return "<br/>".join(parts)


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
