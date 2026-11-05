---
title: API reference
description: Signatures and docstrings for the locus package, from agents, tools and models to memory, RAG, multi-agent workflows, serving and the control layer, each linked to the guide that explains it.
---

# API reference

Signatures and docstrings for the `locus` package, rendered from
`locus-sdk` {{ locus_sdk_version }}, the SDK version this site is built
against. Each page covers one area of the framework. For what a piece is for
and how it fits with the rest, start with the guide linked beside it, then
come back here for the exact signature.

The everyday names import from the package root. Everything else imports from
the module its page names:

```python
from locus import Agent, AgentConfig, StateGraph, tool
from locus.core.termination import MaxIterations, ToolCalled
```

## Agents and runs

| Reference | What it documents | Guide |
|---|---|---|
| [Agent](agent.md) | `Agent`, `AgentConfig`, `AgentResult` and `AgentState`: configure an agent, run it, and read back a typed result | [Agent](../concepts/agent.md), [The agent loop](../concepts/agent-loop.md) |
| [Termination](termination.md) | Stop conditions such as `MaxIterations`, `TimeLimit` and `ToolCalled`, composed into one rule and set on `AgentConfig.termination` | [Termination](../concepts/termination.md) |
| [Hooks](hooks.md) | `HookProvider` and `HookRegistry`, which keeps providers sorted by `priority` and dispatches events to them (an `Agent` runs the hooks you pass it in list order); the hook event types; and the built-in logging, telemetry, retry, guardrail and steering hooks | [Hooks](../concepts/hooks.md) |
| [Events](events.md) | The typed, frozen `LocusEvent` models that `async for event in agent.run(...)` yields | [Events](../concepts/events.md) |
| [Streaming](streaming.md) | Handlers that send the event stream to a terminal, a Server-Sent Events response or a buffer, and structured-output streaming | [Streaming](../concepts/streaming.md), [Graph streaming](../concepts/graph-streaming.md) |
| [Core primitives](core.md) | The shared types underneath: settings, protocols, messages, the error hierarchy, and the `Command` and `interrupt` helpers for graph control flow | [State](../concepts/state.md), [Errors](../concepts/errors.md) |

## Tools and capabilities

| Reference | What it documents | Guide |
|---|---|---|
| [Tools](tools.md) | The `@tool` decorator, which builds a tool's schema from the function's type hints and docstring; `ToolContext`, the registry and the executors | [Tools](../concepts/tools.md) |
| [Capability providers](providers.md) | Web search, web fetch, image generation and speech providers. Set one on `AgentConfig` and the agent registers a matching tool | [Multi-modal providers](../concepts/multi-modal-providers.md), [Tools](../concepts/tools.md) |
| [Integrations](integrations.md) | MCP in both directions: serve an agent to any MCP client, or turn MCP tools into native tools. Cloud posture runs on the core `use_oci`, which refuses any OCI operation whose name does not start with a read prefix such as `list_`, `get_` or `summarize_` | [MCP](../concepts/mcp.md) |
| [Skills](skills.md) | `Skill` and `SkillsPlugin`: `SKILL.md` instruction bundles the agent loads on demand, attached with `AgentConfig.skills` | [Skills](../concepts/skills.md) |
| [Playbooks](playbooks.md) | Declared step sequences with expected tools. Set `AgentConfig.playbook` and a `PlaybookEnforcerHook` checks each tool call against the current step | [Playbooks](../concepts/playbooks.md) |

## Models and reasoning

| Reference | What it documents | Guide |
|---|---|---|
| [Models](models.md) | `get_model` and the provider registry, where the part of a model string before the colon picks the provider; the `ModelProtocol` contract; and the native provider classes | [Model providers](../concepts/models.md) |
| [Reasoning](reasoning.md) | `Reflector`, which evaluates the agent's progress after each iteration; `GroundingEvaluator`, which checks claims against the evidence the tools gathered; `CausalChain`; and GSAR's thresholds and results | [Reasoning](../concepts/reasoning.md), [GSAR](../concepts/gsar.md) |

## Memory and retrieval

| Reference | What it documents | Guide |
|---|---|---|
| [Memory](memory.md) | Conversation managers that keep, window or summarize the history within a run; the cross-thread `BaseStore`; and the long-term memory manager, which decides what to keep from a finished session | [Conversation management](../concepts/conversation-management.md), [Long-term memory](../concepts/memory-manager.md) |
| [Checkpointers](checkpointers.md) | `BaseCheckpointer` and its storage backends. Prior turns are reloaded only when the agent has a checkpointer and you pass a `thread_id` | [Checkpointers](../concepts/checkpointers.md) |
| [RAG](rag.md) | `RAGRetriever` over pluggable embedders and vector stores, rerankers, multimodal processing, and helpers that expose retrieval as a tool | [RAG](../concepts/rag.md) |

## Multi-agent

| Reference | What it documents | Guide |
|---|---|---|
| [Multi-agent](multiagent.md) | Pipelines (`SequentialPipeline`, `ParallelPipeline`, `LoopAgent`), an orchestrator with specialists, swarms, handoffs, and the `StateGraph` workflow | [Multi-agent workflows](../concepts/multi-agent.md) |
| [DeepAgent](deepagent.md) | `create_deepagent`, a research-shaped `Agent` with optional filesystem, todo and subagent tools; and `create_research_workflow`, a `StateGraph` that scores its summary's grounding and rewrites or replans, up to set limits, when the score is below threshold | [DeepAgent](../concepts/deepagent.md) |
| [A2A](a2a.md) | `A2AServer` and `A2AClient`: one agent calls another over HTTP with the Agent-to-Agent protocol. The server accepts both the v1 wire format and the earlier one | [A2A — Agent-to-Agent](../concepts/multi-agent/a2a.md) |

## Serving, observing and testing

| Reference | What it documents | Guide |
|---|---|---|
| [Agent Server](server.md) | `AgentServer`, which wraps an agent as a FastAPI app with `/invoke`, `/stream`, `/threads/{thread_id}` and `/health` routes; and `GraphRunnable`, for serving a compiled `StateGraph` | [Agent Server](../concepts/server.md) |
| [Observability](observability.md) | The in-process `EventBus`, which fans events out to subscribers by run id; the `emit()` helpers; run context; and `EventBusHook`, which republishes an agent's hook events onto the bus | [Observability](../concepts/observability.md), [SSE event catalogue](../concepts/sse-events.md) |
| [Evaluation](evaluation.md) | `EvalCase`, `EvalRunner` and `EvalReport` for test suites; `LLMJudge` for answers with no single right string; `check_trajectory` for the order tools were called in; and graph evaluation targets | [Evaluation](../concepts/evaluation.md), [Testing agents](../concepts/testing.md) |

## Control

| Reference | What it documents | Guide |
|---|---|---|
| [Control](control.md) | `admit()`, which checks an action against a `ControlPolicy` you write, before the action runs: allow, hold for a person, or deny (API values `allow`, `require_human`, `deny`). `gate_tool` puts the same check in front of a tool, approval stores keep held calls until someone decides, and `AuditTrail` records each decision when you pass one. Only the calls you route through the gate are checked | [Control-layer architecture](../concepts/control-layer.md), [Writing a policy that holds](../concepts/policy-authoring.md) |

## Security domain

| Reference | What it documents | Guide |
|---|---|---|
| [Security](security.md) | One domain built on the framework: red-team probes and jobs, grounded findings that come back as evidence or an `Abstention`, verification skeptics, and MITRE ATLAS and OWASP taxonomy tags. Import the admission gate from `locus.control`, not from here | [Agentic AI security](../concepts/agentic-ai-security.md), [Security layer](../concepts/security.md) |

## Deprecated

[ReAct loop](loop.md) documents `locus.loop`, a second ReAct implementation
that `Agent` has never used. It is deprecated: it still imports, each name
you import from it emits `LocusDeprecationWarning`, and it will be removed
in a later major release, per the
[deprecation policy](https://github.com/oracle/locus/blob/main/DEPRECATION.md).
Use [`Agent`](agent.md) instead.
