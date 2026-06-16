# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Causal inference chains for root cause analysis.

This module provides tools for building and analyzing causal relationships
between events, enabling agents to distinguish root causes from symptoms
and detect causal conflicts.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class NodeType(StrEnum):
    """Types of nodes in a causal graph."""

    ROOT_CAUSE = "root_cause"
    INTERMEDIATE = "intermediate"
    SYMPTOM = "symptom"
    UNKNOWN = "unknown"


class RelationshipType(StrEnum):
    """Types of causal relationships."""

    CAUSES = "causes"
    CORRELATES_WITH = "correlates_with"
    PRECEDES = "precedes"
    INHIBITS = "inhibits"


class CausalNode(BaseModel):
    """A node in the causal graph representing an event or condition.

    Attributes:
        id: Unique identifier for this node.
        label: Human-readable label describing the event/condition.
        node_type: Classification of this node in the causal chain.
        evidence: Evidence supporting this node's existence.
        confidence: Confidence in this node's classification (0.0 to 1.0).
        metadata: Additional metadata about this node.
    """

    id: str = Field(default_factory=lambda: f"node_{uuid4().hex[:8]}")
    label: str = Field(..., description="Human-readable description")
    node_type: NodeType = Field(
        default=NodeType.UNKNOWN,
        description="Classification of this node",
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="Evidence supporting this node",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence in classification",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional node metadata",
    )

    model_config = {"frozen": True}

    def with_type(self, node_type: NodeType) -> CausalNode:
        """Return a new node with updated type."""
        return self.model_copy(update={"node_type": node_type})

    def with_evidence(self, evidence: str) -> CausalNode:
        """Return a new node with additional evidence."""
        return self.model_copy(update={"evidence": [*self.evidence, evidence]})

    def with_confidence(self, confidence: float) -> CausalNode:
        """Return a new node with updated confidence."""
        return self.model_copy(update={"confidence": max(0.0, min(1.0, confidence))})


class CausalEdge(BaseModel):
    """An edge representing a causal relationship between nodes.

    Attributes:
        source_id: ID of the source node (cause).
        target_id: ID of the target node (effect).
        relationship: Type of causal relationship.
        confidence: Confidence in this relationship (0.0 to 1.0).
        evidence: Evidence supporting this relationship.
        reasoning: Explanation of the causal link.
    """

    source_id: str = Field(..., description="ID of the source node (cause)")
    target_id: str = Field(..., description="ID of the target node (effect)")
    relationship: RelationshipType = Field(
        default=RelationshipType.CAUSES,
        description="Type of causal relationship",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence in this relationship",
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="Evidence supporting this relationship",
    )
    reasoning: str | None = Field(
        default=None,
        description="Explanation of the causal link",
    )

    model_config = {"frozen": True}

    @property
    def is_causal(self) -> bool:
        """Whether this edge represents a causal (not correlative) relationship."""
        return self.relationship in (RelationshipType.CAUSES, RelationshipType.INHIBITS)


class CausalConflict(BaseModel):
    """Represents a conflict in the causal graph.

    Conflicts occur when edges create logical inconsistencies,
    such as cycles or contradictory relationships.

    Attributes:
        conflict_type: Type of conflict detected.
        involved_nodes: Node IDs involved in the conflict.
        involved_edges: Edges involved in the conflict.
        description: Human-readable description of the conflict.
        resolution_hint: Suggested resolution approach.
    """

    conflict_type: str = Field(..., description="Type of conflict")
    involved_nodes: list[str] = Field(
        default_factory=list,
        description="Node IDs involved in the conflict",
    )
    involved_edges: list[tuple[str, str]] = Field(
        default_factory=list,
        description="Edge pairs (source_id, target_id) involved",
    )
    description: str = Field(..., description="Description of the conflict")
    resolution_hint: str | None = Field(
        default=None,
        description="Suggested resolution",
    )

    model_config = {"frozen": True}


class CausalChain:
    """Builder for causal inference chains.

    CausalChain allows agents to construct and analyze causal graphs,
    identifying root causes, symptoms, and potential conflicts.

    Attributes:
        nodes: Dictionary of nodes by ID.
        edges: List of causal edges.
    """

    def __init__(self) -> None:
        """Initialize an empty causal chain."""
        self._nodes: dict[str, CausalNode] = {}
        self._edges: list[CausalEdge] = []
        self._adjacency: dict[str, list[str]] = {}  # source -> [targets]
        self._reverse_adjacency: dict[str, list[str]] = {}  # target -> [sources]

    @property
    def nodes(self) -> dict[str, CausalNode]:
        """Get all nodes in the graph."""
        return dict(self._nodes)

    @property
    def edges(self) -> list[CausalEdge]:
        """Get all edges in the graph."""
        return list(self._edges)

    def add_node(self, node: CausalNode) -> CausalNode:
        """Add a node to the causal graph.

        Args:
            node: The node to add.

        Returns:
            The added node.

        Raises:
            ValueError: If a node with this ID already exists.
        """
        if node.id in self._nodes:
            msg = f"Node with ID '{node.id}' already exists"
            raise ValueError(msg)

        self._nodes[node.id] = node
        self._adjacency[node.id] = []
        self._reverse_adjacency[node.id] = []
        return node

    def create_node(
        self,
        label: str,
        node_type: NodeType = NodeType.UNKNOWN,
        evidence: list[str] | None = None,
        confidence: float = 0.5,
        **metadata: Any,
    ) -> CausalNode:
        """Create and add a new node to the graph.

        Args:
            label: Human-readable description.
            node_type: Classification of this node.
            evidence: Supporting evidence.
            confidence: Confidence in classification.
            **metadata: Additional metadata.

        Returns:
            The created and added node.
        """
        node = CausalNode(
            label=label,
            node_type=node_type,
            evidence=evidence or [],
            confidence=confidence,
            metadata=metadata,
        )
        return self.add_node(node)

    def add_edge(self, edge: CausalEdge) -> CausalEdge:
        """Add an edge to the causal graph.

        Args:
            edge: The edge to add.

        Returns:
            The added edge.

        Raises:
            ValueError: If source or target node doesn't exist.
        """
        if edge.source_id not in self._nodes:
            msg = f"Source node '{edge.source_id}' not found"
            raise ValueError(msg)
        if edge.target_id not in self._nodes:
            msg = f"Target node '{edge.target_id}' not found"
            raise ValueError(msg)

        self._edges.append(edge)
        self._adjacency[edge.source_id].append(edge.target_id)
        self._reverse_adjacency[edge.target_id].append(edge.source_id)
        return edge

    def link(
        self,
        source_id: str,
        target_id: str,
        relationship: RelationshipType = RelationshipType.CAUSES,
        confidence: float = 0.5,
        evidence: list[str] | None = None,
        reasoning: str | None = None,
    ) -> CausalEdge:
        """Create and add an edge between existing nodes.

        Args:
            source_id: ID of the source node.
            target_id: ID of the target node.
            relationship: Type of relationship.
            confidence: Confidence in the relationship.
            evidence: Supporting evidence.
            reasoning: Explanation of the link.

        Returns:
            The created edge.
        """
        edge = CausalEdge(
            source_id=source_id,
            target_id=target_id,
            relationship=relationship,
            confidence=confidence,
            evidence=evidence or [],
            reasoning=reasoning,
        )
        return self.add_edge(edge)

    def get_node(self, node_id: str) -> CausalNode | None:
        """Get a node by ID.

        Args:
            node_id: The node ID to look up.

        Returns:
            The node or None if not found.
        """
        return self._nodes.get(node_id)

    def get_edges_from(self, node_id: str) -> list[CausalEdge]:
        """Get all edges originating from a node.

        Args:
            node_id: The source node ID.

        Returns:
            List of edges from this node.
        """
        target_ids = self._adjacency.get(node_id, [])
        return [e for e in self._edges if e.source_id == node_id and e.target_id in target_ids]

    def get_edges_to(self, node_id: str) -> list[CausalEdge]:
        """Get all edges pointing to a node.

        Args:
            node_id: The target node ID.

        Returns:
            List of edges to this node.
        """
        source_ids = self._reverse_adjacency.get(node_id, [])
        return [e for e in self._edges if e.target_id == node_id and e.source_id in source_ids]

    def identify_root_causes(self) -> list[CausalNode]:
        """Identify nodes that are root causes.

        Root causes are nodes with outgoing causal edges but no incoming
        causal edges, or nodes explicitly marked as root_cause.

        Returns:
            List of root cause nodes.
        """
        root_causes: list[CausalNode] = []

        for node_id, node in self._nodes.items():
            # Check explicit marking
            if node.node_type == NodeType.ROOT_CAUSE:
                root_causes.append(node)
                continue

            # Check graph structure
            incoming = self._reverse_adjacency.get(node_id, [])
            outgoing = self._adjacency.get(node_id, [])

            # Has outgoing but no incoming = likely root cause
            if outgoing and not incoming:
                root_causes.append(node)

        return root_causes

    def identify_symptoms(self) -> list[CausalNode]:
        """Identify nodes that are symptoms.

        Symptoms are nodes with incoming causal edges but no outgoing
        causal edges, or nodes explicitly marked as symptom.

        Returns:
            List of symptom nodes.
        """
        symptoms: list[CausalNode] = []

        for node_id, node in self._nodes.items():
            # Check explicit marking
            if node.node_type == NodeType.SYMPTOM:
                symptoms.append(node)
                continue

            # Check graph structure
            incoming = self._reverse_adjacency.get(node_id, [])
            outgoing = self._adjacency.get(node_id, [])

            # Has incoming but no outgoing = likely symptom
            if incoming and not outgoing:
                symptoms.append(node)

        return symptoms

    def get_causal_path(
        self,
        source_id: str,
        target_id: str,
    ) -> list[CausalNode] | None:
        """Find a causal path between two nodes.

        Uses BFS to find the shortest path through causal edges.

        Args:
            source_id: Starting node ID.
            target_id: Ending node ID.

        Returns:
            List of nodes in the path, or None if no path exists.
        """
        if source_id not in self._nodes or target_id not in self._nodes:
            return None

        if source_id == target_id:
            return [self._nodes[source_id]]

        # BFS
        visited: set[str] = set()
        queue: list[list[str]] = [[source_id]]

        while queue:
            path = queue.pop(0)
            current = path[-1]

            if current in visited:
                continue
            visited.add(current)

            for neighbor in self._adjacency.get(current, []):
                new_path = [*path, neighbor]
                if neighbor == target_id:
                    return [self._nodes[n] for n in new_path]
                queue.append(new_path)

        return None

    def detect_conflicts(self) -> list[CausalConflict]:
        """Detect conflicts in the causal graph.

        Checks for:
        - Cycles (A causes B causes A)
        - Bidirectional causation (A causes B and B causes A)
        - Contradictory relationships (A causes B and A inhibits B)

        Returns:
            List of detected conflicts.
        """
        conflicts: list[CausalConflict] = []

        # Check for cycles
        cycle_conflicts = self._detect_cycles()
        conflicts.extend(cycle_conflicts)

        # Check for bidirectional causation
        bidirectional_conflicts = self._detect_bidirectional()
        conflicts.extend(bidirectional_conflicts)

        # Check for contradictory relationships
        contradictory_conflicts = self._detect_contradictory()
        conflicts.extend(contradictory_conflicts)

        return conflicts

    def _detect_cycles(self) -> list[CausalConflict]:
        """Detect cycles in the causal graph using DFS."""
        conflicts: list[CausalConflict] = []

        # Track visited and recursion stack for each DFS
        visited: set[str] = set()
        rec_stack: set[str] = set()
        path: list[str] = []

        def dfs(node_id: str) -> list[str] | None:
            visited.add(node_id)
            rec_stack.add(node_id)
            path.append(node_id)

            for neighbor in self._adjacency.get(node_id, []):
                if neighbor not in visited:
                    result = dfs(neighbor)
                    if result:
                        return result
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    return path[cycle_start:]

            path.pop()
            rec_stack.remove(node_id)
            return None

        for node_id in self._nodes:
            if node_id not in visited:
                cycle = dfs(node_id)
                if cycle:
                    conflicts.append(
                        CausalConflict(
                            conflict_type="cycle",
                            involved_nodes=cycle,
                            involved_edges=[
                                (cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))
                            ],
                            description=f"Causal cycle detected: {' -> '.join(cycle)} -> {cycle[0]}",
                            resolution_hint="Review the causal chain and break the cycle by removing or revising one edge",
                        )
                    )

        return conflicts

    def _detect_bidirectional(self) -> list[CausalConflict]:
        """Detect bidirectional causal relationships."""
        conflicts: list[CausalConflict] = []
        checked: set[tuple[str, str]] = set()

        for edge in self._edges:
            if not edge.is_causal:
                continue

            ordered = sorted([edge.source_id, edge.target_id])
            pair: tuple[str, str] = (ordered[0], ordered[1])
            if pair in checked:
                continue
            checked.add(pair)

            # Check for reverse edge
            reverse_edges = [
                e
                for e in self._edges
                if e.source_id == edge.target_id and e.target_id == edge.source_id and e.is_causal
            ]

            if reverse_edges:
                conflicts.append(
                    CausalConflict(
                        conflict_type="bidirectional_causation",
                        involved_nodes=[edge.source_id, edge.target_id],
                        involved_edges=[
                            (edge.source_id, edge.target_id),
                            (edge.target_id, edge.source_id),
                        ],
                        description=(
                            f"Bidirectional causation between "
                            f"'{self._nodes[edge.source_id].label}' and "
                            f"'{self._nodes[edge.target_id].label}'"
                        ),
                        resolution_hint="Determine the primary causal direction or model as correlation",
                    )
                )

        return conflicts

    def _detect_contradictory(self) -> list[CausalConflict]:
        """Detect contradictory relationships between the same nodes."""
        conflicts: list[CausalConflict] = []
        edge_map: dict[tuple[str, str], list[CausalEdge]] = {}

        for edge in self._edges:
            key = (edge.source_id, edge.target_id)
            if key not in edge_map:
                edge_map[key] = []
            edge_map[key].append(edge)

        for (source_id, target_id), edges in edge_map.items():
            if len(edges) < 2:
                continue

            # Check for contradictory relationships
            has_causes = any(e.relationship == RelationshipType.CAUSES for e in edges)
            has_inhibits = any(e.relationship == RelationshipType.INHIBITS for e in edges)

            if has_causes and has_inhibits:
                conflicts.append(
                    CausalConflict(
                        conflict_type="contradictory_relationship",
                        involved_nodes=[source_id, target_id],
                        involved_edges=[(source_id, target_id)],
                        description=(
                            f"Contradictory relationships: "
                            f"'{self._nodes[source_id].label}' both causes and inhibits "
                            f"'{self._nodes[target_id].label}'"
                        ),
                        resolution_hint="Resolve by determining the dominant effect or adding context",
                    )
                )

        return conflicts

    def classify_nodes(self) -> dict[str, NodeType]:
        """Automatically classify all nodes based on graph structure.

        Returns:
            Dictionary mapping node IDs to their inferred types.
        """
        classifications: dict[str, NodeType] = {}

        for node_id in self._nodes:
            incoming = self._reverse_adjacency.get(node_id, [])
            outgoing = self._adjacency.get(node_id, [])

            # Preserve explicit classifications
            if self._nodes[node_id].node_type != NodeType.UNKNOWN:
                classifications[node_id] = self._nodes[node_id].node_type
            elif outgoing and not incoming:
                classifications[node_id] = NodeType.ROOT_CAUSE
            elif incoming and not outgoing:
                classifications[node_id] = NodeType.SYMPTOM
            elif incoming and outgoing:
                classifications[node_id] = NodeType.INTERMEDIATE
            else:
                classifications[node_id] = NodeType.UNKNOWN

        return classifications

    def update_node_types(self) -> None:
        """Update node types based on graph structure (in place)."""
        classifications = self.classify_nodes()

        for node_id, node_type in classifications.items():
            if self._nodes[node_id].node_type == NodeType.UNKNOWN:
                self._nodes[node_id] = self._nodes[node_id].with_type(node_type)

    def get_chain_summary(self) -> dict[str, Any]:
        """Get a summary of the causal chain.

        Returns:
            Dictionary with chain statistics and structure.
        """
        classifications = self.classify_nodes()

        return {
            "total_nodes": len(self._nodes),
            "total_edges": len(self._edges),
            "root_causes": [
                self._nodes[n].label for n, t in classifications.items() if t == NodeType.ROOT_CAUSE
            ],
            "symptoms": [
                self._nodes[n].label for n, t in classifications.items() if t == NodeType.SYMPTOM
            ],
            "intermediates": [
                self._nodes[n].label
                for n, t in classifications.items()
                if t == NodeType.INTERMEDIATE
            ],
            "conflicts": len(self.detect_conflicts()),
            "avg_confidence": (
                sum(n.confidence for n in self._nodes.values()) / len(self._nodes)
                if self._nodes
                else 0.0
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize the causal chain to a dictionary.

        Returns:
            Dictionary representation of the chain.
        """
        return {
            "nodes": [n.model_dump() for n in self._nodes.values()],
            "edges": [e.model_dump() for e in self._edges],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CausalChain:
        """Deserialize a causal chain from a dictionary.

        Args:
            data: Dictionary with nodes and edges.

        Returns:
            CausalChain instance.
        """
        chain = cls()

        for node_data in data.get("nodes", []):
            node = CausalNode.model_validate(node_data)
            chain.add_node(node)

        for edge_data in data.get("edges", []):
            edge = CausalEdge.model_validate(edge_data)
            chain.add_edge(edge)

        return chain


def build_causal_chain(
    events: list[dict[str, Any]],
    auto_classify: bool = True,
) -> CausalChain:
    """Convenience function to build a causal chain from event data.

    Args:
        events: List of event dictionaries with label and optional causes.
        auto_classify: Whether to auto-classify node types.

    Returns:
        Built CausalChain.

    Example:
        events = [
            {"label": "Database connection failed"},
            {"label": "Query timeout", "causes": ["Database connection failed"]},
            {"label": "User sees error page", "causes": ["Query timeout"]},
        ]
        chain = build_causal_chain(events)
    """
    chain = CausalChain()
    label_to_id: dict[str, str] = {}

    # First pass: create all nodes
    for event in events:
        label = event["label"]
        node = chain.create_node(
            label=label,
            node_type=NodeType(event.get("type", "unknown")),
            evidence=event.get("evidence", []),
            confidence=event.get("confidence", 0.5),
        )
        label_to_id[label] = node.id

    # Second pass: create edges
    for event in events:
        label = event["label"]
        causes = event.get("causes", [])
        target_id = label_to_id[label]

        for cause_label in causes:
            if cause_label in label_to_id:
                source_id = label_to_id[cause_label]
                chain.link(
                    source_id=source_id,
                    target_id=target_id,
                    relationship=RelationshipType.CAUSES,
                )

    if auto_classify:
        chain.update_node_types()

    return chain
