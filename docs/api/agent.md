# Agent

The high-level entry point — configure an `Agent`, run it, and read
back a typed `AgentResult`.

For the concepts, start with [Agent](../concepts/agent.md) and
[the agent loop](../concepts/agent-loop.md).

## `Agent` class

```text
Agent(
    model: str | ModelProtocol | None = None,
    tools: list[Tool] | None = None,
    system_prompt: str | None = None,
    reflexion: ReflexionConfig | bool | None = None,
    grounding: GroundingConfig | bool | None = None,
    max_iterations: int = 20,
    conversation_manager: ConversationManager | None = None,
    checkpointer: Checkpointer | None = None,
    hooks: list[HookProvider] | None = None,
    config: AgentConfig | None = None,
    **kwargs,
)
```

::: locus.agent.agent.Agent
    options:
      show_root_heading: true
      show_signature: false
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

`AgentResult.metrics` is an `ExecutionMetrics` and `AgentResult.stop_reason`
is a `StopReason` literal. `StreamingResult` is a partial-result model
exported from `locus.agent`; the streaming entry point (`agent.run()`) yields
`LocusEvent`s, not `StreamingResult`.

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
