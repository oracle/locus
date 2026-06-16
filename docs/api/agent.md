# Agent

## `Agent` class

::: locus.agent.agent.Agent
    options:
      show_root_heading: true
      members_order: source

## AgentConfig

::: locus.agent.config.AgentConfig

### Reasoning configs

`AgentConfig.reflexion`, `AgentConfig.grounding`, and `AgentConfig.gsar` accept
typed configuration objects defined alongside the config itself. The boolean
shorthand (`reflexion=True`) coerces into a default instance — the explicit
form is what to reach for when you want to tune thresholds, swap models, or
override defaults.

::: locus.agent.config.ReflexionConfig
::: locus.agent.config.GroundingConfig
::: locus.agent.config.GSARConfig

## AgentResult

::: locus.agent.result.AgentResult

### Result sub-types

`AgentResult.metrics` is an `ExecutionMetrics`, `AgentResult.stop_reason` is a
`StopReason` literal, and the streaming entry point yields `StreamingResult`
between events.

::: locus.agent.result.ExecutionMetrics
::: locus.agent.result.StopReason
::: locus.agent.result.StreamingResult

## AgentState

::: locus.core.state.AgentState

### State sub-types

`AgentState.tool_executions` is a tuple of `ToolExecution`,
`AgentState.reasoning_steps` is a tuple of `ReasoningStep`. Both are surfaced
on `AgentResult` via the matching properties.

::: locus.core.state.ToolExecution
::: locus.core.state.ReasoningStep

## Composition

The composition helpers chain or fan-out multiple agents while keeping the
same `AgentResult` shape at the boundary. See the
[Multi-agent composition](multiagent.md) page for the full pipeline classes;
the functional builders below are the ergonomic entry points.

::: locus.agent.composition.PipelineResult
::: locus.agent.composition.sequential
::: locus.agent.composition.parallel
::: locus.agent.composition.loop
