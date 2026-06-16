# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""
Locus - A zero-LangChain agentic SDK.

Built-in Reflexion, Grounding Evaluation, and production-grade orchestration.
100% Pydantic. No magic.

Usage:
    from locus import Agent, tool

    @tool
    def search(query: str) -> str:
        '''Search the knowledge base.'''
        return "results..."

    agent = Agent(
        model="openai:gpt-4o",  # or oci:cohere.command-r-plus
        tools=[search],
        system_prompt="You are a helpful assistant.",
    )

    async for event in agent.run("Find information about X"):
        print(event)
"""

from locus.core.config import LocusSettings
from locus.core.errors import LocusError
from locus.core.events import (
    GroundingEvent,
    LocusEvent,
    ReflectEvent,
    TerminateEvent,
    ThinkEvent,
    ToolCompleteEvent,
    ToolStartEvent,
)
from locus.core.messages import Message, Role, ToolCall
from locus.core.state import AgentState
from locus.tools.context import ToolContext
from locus.tools.decorator import tool


# Lazy import mapping for optional dependencies
_LAZY_IMPORTS = {
    "Agent": ("locus.agent.agent", "Agent"),
    "AgentConfig": ("locus.agent.config", "AgentConfig"),
    "AgentResult": ("locus.agent.result", "AgentResult"),
    # Composition primitives — graph-free orchestration shapes.
    "SequentialPipeline": ("locus.agent.composition", "SequentialPipeline"),
    "ParallelPipeline": ("locus.agent.composition", "ParallelPipeline"),
    "LoopAgent": ("locus.agent.composition", "LoopAgent"),
    # Multi-agent primitives — graph + handoff + orchestrator/specialist.
    "StateGraph": ("locus.multiagent.graph", "StateGraph"),
    "GraphConfig": ("locus.multiagent.graph", "GraphConfig"),
    "START": ("locus.multiagent.graph", "START"),
    "END": ("locus.multiagent.graph", "END"),
    "Send": ("locus.core.send", "Send"),
    "Handoff": ("locus.multiagent.handoff", "Handoff"),
    "HandoffContext": ("locus.multiagent.handoff", "HandoffContext"),
    "HandoffReason": ("locus.multiagent.handoff", "HandoffReason"),
    "create_handoff_agent": ("locus.multiagent.handoff", "create_handoff_agent"),
    "create_handoff_manager": ("locus.multiagent.handoff", "create_handoff_manager"),
    "Orchestrator": ("locus.multiagent.orchestrator", "Orchestrator"),
    "RoutingDecision": ("locus.multiagent.orchestrator", "RoutingDecision"),
    "create_orchestrator": ("locus.multiagent.orchestrator", "create_orchestrator"),
    "Specialist": ("locus.multiagent.specialist", "Specialist"),
    # Public name "Reflexion" maps to the actual class "Reflector". The
    # mismatch was an import error on locus 0.1.0 — keep the alias so existing
    # docs / code that does ``from locus import Reflexion`` keeps working.
    "Reflexion": ("locus.reasoning.reflexion", "Reflector"),
    "Reflector": ("locus.reasoning.reflexion", "Reflector"),
    "GroundingEvaluator": ("locus.reasoning.grounding", "GroundingEvaluator"),
    "CausalChain": ("locus.reasoning.causal", "CausalChain"),
    "HookProvider": ("locus.hooks.provider", "HookProvider"),
    "HookRegistry": ("locus.hooks.registry", "HookRegistry"),
    # RAG
    "RAGRetriever": ("locus.rag.retriever", "RAGRetriever"),
    "OCIEmbeddings": ("locus.rag.embeddings.oci", "OCIEmbeddings"),
    "OracleVectorStore": ("locus.rag.stores.oracle", "OracleVectorStore"),
    # PRISM router — bounded-graph generation atop locus primitives.
    "Router": ("locus.router.runtime", "Router"),
    "GoalFrame": ("locus.router.goal_frame", "GoalFrame"),
    "TaskType": ("locus.router.goal_frame", "TaskType"),
    "Risk": ("locus.router.goal_frame", "Risk"),
    "Complexity": ("locus.router.goal_frame", "Complexity"),
    "Capability": ("locus.router.capability", "Capability"),
    "CapabilityIndex": ("locus.router.capability", "CapabilityIndex"),
    "Protocol": ("locus.router.protocol", "Protocol"),
    "ProtocolRegistry": ("locus.router.protocol", "ProtocolRegistry"),
    "PolicyGate": ("locus.router.policy", "PolicyGate"),
    "PolicyVerdict": ("locus.router.policy", "PolicyVerdict"),
    "CognitiveCompiler": ("locus.router.compiler", "CognitiveCompiler"),
    "RunnableResult": ("locus.router.runnable", "RunnableResult"),
    "SkillIndex": ("locus.router.skill_index", "SkillIndex"),
    "builtin_protocols": ("locus.router.protocol", "builtin_protocols"),
    # Deep research — research-shaped Agent factory + provider protocol.
    # Submodule is ``locus.deepagent`` (Pythonic path-name convention).
    # Factory is ``create_deepagent`` (matches ``create_orchestrator``,
    # ``create_handoff_agent`` — the existing locus naming for "build me
    # a configured X").
    "create_deepagent": ("locus.deepagent", "create_deepagent"),
    "create_research_workflow": ("locus.deepagent.workflow", "create_research_workflow"),
    "make_execute_node": ("locus.deepagent.workflow", "make_execute_node"),
    "make_causal_inference_node": ("locus.deepagent.workflow", "make_causal_inference_node"),
    "make_summarize_node": ("locus.deepagent.workflow", "make_summarize_node"),
    "make_grounding_eval_node": ("locus.deepagent.workflow", "make_grounding_eval_node"),
    "make_regenerate_summary_node": ("locus.deepagent.workflow", "make_regenerate_summary_node"),
    "make_replan_node": ("locus.deepagent.workflow", "make_replan_node"),
    "route_after_grounding": ("locus.deepagent.workflow", "route_after_grounding"),
    # Research workflow state keys
    "KEY_PROMPT": ("locus.deepagent.workflow", "KEY_PROMPT"),
    "KEY_EXECUTE_PROMPT": ("locus.deepagent.workflow", "KEY_EXECUTE_PROMPT"),
    "KEY_EVIDENCE": ("locus.deepagent.workflow", "KEY_EVIDENCE"),
    "KEY_GROUNDING_FACTS": ("locus.deepagent.workflow", "KEY_GROUNDING_FACTS"),
    "KEY_CAUSAL_CHAIN": ("locus.deepagent.workflow", "KEY_CAUSAL_CHAIN"),
    "KEY_CAUSAL_HYPOTHESIS": ("locus.deepagent.workflow", "KEY_CAUSAL_HYPOTHESIS"),
    "KEY_CAUSAL_CONFIDENCE": ("locus.deepagent.workflow", "KEY_CAUSAL_CONFIDENCE"),
    "KEY_SUMMARY": ("locus.deepagent.workflow", "KEY_SUMMARY"),
    "KEY_STRUCTURED_OUTPUT": ("locus.deepagent.workflow", "KEY_STRUCTURED_OUTPUT"),
    "KEY_GROUNDING_SCORE": ("locus.deepagent.workflow", "KEY_GROUNDING_SCORE"),
    "KEY_UNGROUNDED_CLAIMS": ("locus.deepagent.workflow", "KEY_UNGROUNDED_CLAIMS"),
    "KEY_REPLAN_COUNT": ("locus.deepagent.workflow", "KEY_REPLAN_COUNT"),
    "KEY_REGENERATION_COUNT": ("locus.deepagent.workflow", "KEY_REGENERATION_COUNT"),
    "KEY_STOP_REASON": ("locus.deepagent.workflow", "KEY_STOP_REASON"),
    "KnowledgeProvider": ("locus.deepagent", "KnowledgeProvider"),
    "KnowledgeRow": ("locus.deepagent", "KnowledgeRow"),
    "ItemRef": ("locus.deepagent", "ItemRef"),
    "Grounding": ("locus.deepagent", "Grounding"),
}


def __getattr__(name: str) -> object:
    """Lazy import for Agent and model classes."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        import importlib

        module = importlib.import_module(module_path)
        return getattr(module, attr_name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__version__ = "0.1.0"
__all__ = [
    "Agent",
    "AgentConfig",
    "AgentResult",
    "AgentState",
    "CausalChain",
    "END",
    "GraphConfig",
    "GroundingEvaluator",
    "GroundingEvent",
    "Handoff",
    "HandoffContext",
    "HandoffReason",
    "HookProvider",
    "HookRegistry",
    "LocusError",
    "LocusEvent",
    "LocusSettings",
    "LoopAgent",
    "Message",
    "Orchestrator",
    "ParallelPipeline",
    "ReflectEvent",
    "Reflector",
    "Reflexion",
    "Role",
    "RoutingDecision",
    "START",
    "Send",
    "SequentialPipeline",
    "Specialist",
    "StateGraph",
    "TerminateEvent",
    "ThinkEvent",
    "ToolCall",
    "ToolCompleteEvent",
    "ToolContext",
    "ToolStartEvent",
    "__version__",
    "create_handoff_agent",
    "create_handoff_manager",
    "create_orchestrator",
    "tool",
    # RAG (lazy)
    "RAGRetriever",
    "OCIEmbeddings",
    "OracleVectorStore",
    # PRISM router (lazy)
    "Capability",
    "CapabilityIndex",
    "CognitiveCompiler",
    "Complexity",
    "GoalFrame",
    "PolicyGate",
    "PolicyVerdict",
    "Protocol",
    "ProtocolRegistry",
    "Risk",
    "Router",
    "RunnableResult",
    "SkillIndex",
    "TaskType",
    "builtin_protocols",
    # Deep research — agent factory (lazy)
    "create_deepagent",
    "KnowledgeProvider",
    "KnowledgeRow",
    "ItemRef",
    "Grounding",
    # Deep research — research workflow primitives (lazy)
    "create_research_workflow",
    "make_execute_node",
    "make_causal_inference_node",
    "make_summarize_node",
    "make_grounding_eval_node",
    "make_regenerate_summary_node",
    "make_replan_node",
    "route_after_grounding",
    # Research workflow state keys (lazy)
    "KEY_PROMPT",
    "KEY_EXECUTE_PROMPT",
    "KEY_EVIDENCE",
    "KEY_GROUNDING_FACTS",
    "KEY_CAUSAL_CHAIN",
    "KEY_CAUSAL_HYPOTHESIS",
    "KEY_CAUSAL_CONFIDENCE",
    "KEY_SUMMARY",
    "KEY_STRUCTURED_OUTPUT",
    "KEY_GROUNDING_SCORE",
    "KEY_UNGROUNDED_CLAIMS",
    "KEY_REPLAN_COUNT",
    "KEY_REGENERATION_COUNT",
    "KEY_STOP_REASON",
]
