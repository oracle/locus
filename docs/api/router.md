# Cognitive Router

The cognitive router compiles natural-language tasks onto proven
orchestration shapes. The LLM fills one typed `GoalFrame`; everything
after that — protocol selection, policy gating, compilation — is
rule-based.

## Dispatch

::: locus.router.runtime.Router
::: locus.router.runtime.FrameExtractionError
::: locus.router.runnable.RunnableResult

## Goal Frame

The typed contract the LLM fills. Picker / compiler read this structure
deterministically; the LLM never names a protocol directly.

::: locus.router.goal_frame.GoalFrame
::: locus.router.goal_frame.TaskType
::: locus.router.goal_frame.Risk
::: locus.router.goal_frame.Complexity

## Protocol registry

A protocol is a named, parameterized orchestration shape (e.g.
"specialist_fanout", "deep_research", "human_review"). Built-ins are
in `builtin_protocols()`; users register custom shapes via
`ProtocolRegistry.register()`.

::: locus.router.protocol.Protocol
::: locus.router.protocol.ProtocolRegistry
::: locus.router.protocol.builtin_protocols
::: locus.router.protocol.BuilderContext
::: locus.router.protocol.NoMatchingProtocolError

## Picker

The picker decides which protocol best matches a `GoalFrame`.
`LLMProtocolPicker` is the default; `PickedProtocol` is the decision
record returned to the compiler.

::: locus.router.picker.LLMProtocolPicker
::: locus.router.picker.PickedProtocol
::: locus.router.picker.PickerError

## Policy gate

Pre-execution policy checks (cost, risk, capability requirements,
optional human approval). `PolicyVerdict` either allows compilation
to proceed or rejects with a structured reason.

::: locus.router.policy.PolicyGate
::: locus.router.policy.PolicyVerdict
::: locus.router.policy.PolicyDeniedError

## Compiler

Turns a `PickedProtocol` + `GoalFrame` into an executable `Runnable`.
Approval callbacks (HITL gates) plug in here.

::: locus.router.compiler.CognitiveCompiler
::: locus.router.compiler.ApprovalCallback

## Runnables

The compiled execution shape. `Runnable` is the base Protocol — every
concrete runnable below produces a `RunnableResult` when invoked.

::: locus.router.runnable.Runnable
::: locus.router.runnable.AgentRunnable
::: locus.router.runnable.PipelineRunnable
::: locus.router.runnable.OrchestratorRunnable
::: locus.router.runnable.DebateRunnable
::: locus.router.runnable.A2ARunnable

## Capabilities

The skill / tool index the compiler binds protocol slots to. Custom
capabilities register themselves via `CapabilityIndex.annotate(...)`
or by exposing a `SkillIndex`.

::: locus.router.capability.Capability
::: locus.router.capability.CapabilityIndex
::: locus.router.capability.HUMAN_SENTINEL
::: locus.router.skill_index.SkillIndex
