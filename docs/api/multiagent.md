# Multi-agent

## Composition

The agent-level pipelines. Same `AgentResult` shape at the boundary as
a plain `Agent.run_sync(...)` call — these are drop-in for any place
that takes an agent.

::: tulip.agent.composition.SequentialPipeline
::: tulip.agent.composition.ParallelPipeline
::: tulip.agent.composition.LoopAgent

## Orchestrator + Specialists

A central `Orchestrator` routes each turn to one of N domain-focused
`Specialist`s. `RoutingDecision` is the typed decision record; the
built-in specialists below are factory-built for common roles.

::: tulip.multiagent.orchestrator.Orchestrator
::: tulip.multiagent.orchestrator.OrchestratorResult
::: tulip.multiagent.orchestrator.RoutingDecision
::: tulip.multiagent.orchestrator.create_orchestrator
::: tulip.multiagent.specialist.Specialist
::: tulip.multiagent.specialist.SpecialistResult
::: tulip.multiagent.specialist.Playbook
::: tulip.multiagent.specialist.PlaybookStep

### Built-in specialists

::: tulip.multiagent.specialist.create_code_analyst
::: tulip.multiagent.specialist.create_log_analyst
::: tulip.multiagent.specialist.create_metrics_analyst
::: tulip.multiagent.specialist.create_trace_analyst

## Swarm

Self-organizing agents that pick up tasks from a shared queue and
share progress via `SharedContext`.

::: tulip.multiagent.swarm.Swarm
::: tulip.multiagent.swarm.SharedContext
::: tulip.multiagent.swarm.SwarmAgent
::: tulip.multiagent.swarm.SwarmTask
::: tulip.multiagent.swarm.SwarmResult
::: tulip.multiagent.swarm.TaskStatus
::: tulip.multiagent.swarm.create_swarm
::: tulip.multiagent.swarm.create_swarm_agent

## Handoff

Agent-to-agent context transfer. `HandoffReason` is the typed reason
the source agent emitted; `HandoffContext` is the carried state.

::: tulip.multiagent.handoff.Handoff
::: tulip.multiagent.handoff.HandoffAgent
::: tulip.multiagent.handoff.HandoffContext
::: tulip.multiagent.handoff.HandoffReason
::: tulip.multiagent.handoff.HandoffEvent
::: tulip.multiagent.handoff.HandoffResult
::: tulip.multiagent.handoff.create_handoff_agent
::: tulip.multiagent.handoff.create_handoff_manager

## StateGraph

DAG-based workflow with explicit nodes, edges, reducers, and a typed
state. The most expressive composition primitive — used by
`create_research_workflow` (see [DeepAgent](deepagent.md)) and the
router's compiled `Runnable`s.

::: tulip.multiagent.graph.StateGraph
::: tulip.multiagent.graph.GraphConfig
::: tulip.multiagent.graph.Graph
::: tulip.multiagent.graph.GraphResult
::: tulip.multiagent.graph.Node
::: tulip.multiagent.graph.NodeResult
::: tulip.multiagent.graph.NodeStatus
::: tulip.multiagent.graph.Edge
::: tulip.multiagent.graph.ConditionalEdge
::: tulip.multiagent.graph.CachePolicy
::: tulip.multiagent.graph.RetryPolicy
::: tulip.multiagent.graph.StreamMode
::: tulip.multiagent.graph.StreamEvent

### Special nodes

::: tulip.multiagent.graph.START
::: tulip.multiagent.graph.END

### Convenience builders

::: tulip.multiagent.graph.create_graph
::: tulip.multiagent.graph.node
::: tulip.multiagent.graph.emit_custom

## Functional API

::: tulip.multiagent.functional.task
::: tulip.multiagent.functional.entrypoint

## A2A protocol

::: tulip.a2a.protocol.A2AServer
::: tulip.a2a.protocol.A2AClient
