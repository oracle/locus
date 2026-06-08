# Cognitive Router

The cognitive router compiles natural-language tasks onto proven
orchestration shapes. The LLM fills one typed `GoalFrame`; everything
after that — protocol selection, policy gating, compilation — is
rule-based.

## Dispatch

::: tulip.router.runtime.Router
::: tulip.router.runtime.FrameExtractionError
::: tulip.router.runnable.RunnableResult

## Goal Frame

The typed contract the LLM fills. Picker / compiler read this structure
deterministically; the LLM never names a protocol directly.

::: tulip.router.goal_frame.GoalFrame
::: tulip.router.goal_frame.TaskType
::: tulip.router.goal_frame.Risk
::: tulip.router.goal_frame.Complexity

## Protocol registry

A protocol is a named, parameterized orchestration shape (e.g.
"specialist_fanout", "deep_research", "human_review"). Built-ins are
in `builtin_protocols()`; users register custom shapes via
`ProtocolRegistry.register()`.

::: tulip.router.protocol.Protocol
::: tulip.router.protocol.ProtocolRegistry
::: tulip.router.protocol.builtin_protocols
::: tulip.router.protocol.BuilderContext
::: tulip.router.protocol.NoMatchingProtocolError

## Picker

The picker decides which protocol best matches a `GoalFrame`.
`LLMProtocolPicker` is the default; `PickedProtocol` is the decision
record returned to the compiler.

::: tulip.router.picker.LLMProtocolPicker
::: tulip.router.picker.PickedProtocol
::: tulip.router.picker.PickerError

## Policy gate

Pre-execution policy checks (cost, risk, capability requirements,
optional human approval). `PolicyVerdict` either allows compilation
to proceed or rejects with a structured reason.

::: tulip.router.policy.PolicyGate
::: tulip.router.policy.PolicyVerdict
::: tulip.router.policy.PolicyDeniedError

## Compiler

Turns a `PickedProtocol` + `GoalFrame` into an executable `Runnable`.
Approval callbacks (HITL gates) plug in here.

::: tulip.router.compiler.CognitiveCompiler
::: tulip.router.compiler.ApprovalCallback

## Runnables

The compiled execution shape. `Runnable` is the base Protocol — every
concrete runnable below produces a `RunnableResult` when invoked.

::: tulip.router.runnable.Runnable
::: tulip.router.runnable.AgentRunnable
::: tulip.router.runnable.PipelineRunnable
::: tulip.router.runnable.OrchestratorRunnable
::: tulip.router.runnable.DebateRunnable
::: tulip.router.runnable.A2ARunnable

## Capabilities

The skill / tool index the compiler binds protocol slots to. Custom
capabilities register themselves via `CapabilityIndex.annotate(...)`
or by exposing a `SkillIndex`.

::: tulip.router.capability.Capability
::: tulip.router.capability.CapabilityIndex
::: tulip.router.capability.HUMAN_SENTINEL
::: tulip.router.skill_index.SkillIndex
