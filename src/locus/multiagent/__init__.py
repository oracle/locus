# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Multi-agent orchestration for Locus.

This module provides multiple patterns for coordinating agents:

1. **Graph**: DAG-based workflows with dependency resolution
2. **Orchestrator**: Central coordinator that routes to specialists
3. **Specialist**: Domain-specific agents with focused capabilities
4. **Swarm**: Self-organizing agents with shared context
5. **Handoff**: Agent-to-agent context transfer
"""

from locus.multiagent.graph import (
    END,
    # Special nodes
    START,
    CachePolicy,
    ConditionalEdge,
    # Core graph types
    Edge,
    Graph,
    GraphConfig,
    GraphResult,
    Node,
    NodeResult,
    NodeStatus,
    RetryPolicy,
    StateGraph,
    StreamEvent,
    StreamMode,
    # Convenience functions
    create_graph,
    # Streaming
    emit_custom,
    node,
)
from locus.multiagent.handoff import (
    Handoff,
    HandoffAgent,
    HandoffContext,
    HandoffEvent,
    HandoffReason,
    HandoffResult,
    create_handoff_agent,
    create_handoff_manager,
)
from locus.multiagent.orchestrator import (
    Orchestrator,
    OrchestratorResult,
    RoutingDecision,
    create_orchestrator,
)
from locus.multiagent.specialist import (
    Playbook,
    PlaybookStep,
    Specialist,
    SpecialistResult,
    create_code_analyst,
    create_log_analyst,
    create_metrics_analyst,
    create_trace_analyst,
)
from locus.multiagent.swarm import (
    SharedContext,
    Swarm,
    SwarmAgent,
    SwarmResult,
    SwarmTask,
    TaskStatus,
    create_swarm,
    create_swarm_agent,
)


__all__ = [
    # Graph - Core types
    "Edge",
    "ConditionalEdge",
    "Graph",
    "StateGraph",
    "GraphConfig",
    "GraphResult",
    "Node",
    "NodeResult",
    "NodeStatus",
    "StreamMode",
    "StreamEvent",
    # Graph - Special nodes
    "START",
    "END",
    # Graph - Node policies
    "CachePolicy",
    "RetryPolicy",
    # Graph - Convenience
    "create_graph",
    "node",
    # Graph - Streaming
    "emit_custom",
    # Handoff
    "Handoff",
    "HandoffAgent",
    "HandoffContext",
    "HandoffEvent",
    "HandoffReason",
    "HandoffResult",
    "create_handoff_agent",
    "create_handoff_manager",
    # Orchestrator
    "Orchestrator",
    "OrchestratorResult",
    "RoutingDecision",
    "create_orchestrator",
    # Specialist
    "Playbook",
    "PlaybookStep",
    "Specialist",
    "SpecialistResult",
    "create_code_analyst",
    "create_log_analyst",
    "create_metrics_analyst",
    "create_trace_analyst",
    # Swarm
    "SharedContext",
    "Swarm",
    "SwarmAgent",
    "SwarmResult",
    "SwarmTask",
    "TaskStatus",
    "create_swarm",
    "create_swarm_agent",
]
