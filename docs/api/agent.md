# Agent

## `Agent` class

::: tulip.agent.agent.Agent
    options:
      show_root_heading: true
      members_order: source

## AgentConfig

::: tulip.agent.config.AgentConfig

### Reasoning configs

`AgentConfig.reflexion`, `AgentConfig.grounding`, and `AgentConfig.gsar` accept
typed configuration objects defined alongside the config itself. The boolean
shorthand (`reflexion=True`) coerces into a default instance — the explicit
form is what to reach for when you want to tune thresholds, swap models, or
override defaults.

::: tulip.agent.config.ReflexionConfig
::: tulip.agent.config.GroundingConfig
::: tulip.agent.config.GSARConfig

## AgentResult

::: tulip.agent.result.AgentResult

### Result sub-types

`AgentResult.metrics` is an `ExecutionMetrics`, `AgentResult.stop_reason` is a
`StopReason` literal, and the streaming entry point yields `StreamingResult`
between events.

::: tulip.agent.result.ExecutionMetrics
::: tulip.agent.result.StopReason
::: tulip.agent.result.StreamingResult

## AgentState

::: tulip.core.state.AgentState

### State sub-types

`AgentState.tool_executions` is a tuple of `ToolExecution`,
`AgentState.reasoning_steps` is a tuple of `ReasoningStep`. Both are surfaced
on `AgentResult` via the matching properties.

::: tulip.core.state.ToolExecution
::: tulip.core.state.ReasoningStep

## Composition

The composition helpers chain or fan-out multiple agents while keeping the
same `AgentResult` shape at the boundary. See the
[Multi-agent composition](multiagent.md) page for the full pipeline classes;
the functional builders below are the ergonomic entry points.

::: tulip.agent.composition.PipelineResult
::: tulip.agent.composition.sequential
::: tulip.agent.composition.parallel
::: tulip.agent.composition.loop
