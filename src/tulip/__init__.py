# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""
Tulip - A zero-LangChain agentic SDK.

Built-in Reflexion, Grounding Evaluation, and production-grade orchestration.
100% Pydantic. No magic.

Usage:
    from tulip import Agent, tool

    @tool
    def search(query: str) -> str:
        '''Search the knowledge base.'''
        return "results..."

    agent = Agent(
        model="openai:gpt-4o",  # or anthropic:claude-sonnet-4-6
        tools=[search],
        system_prompt="You are a helpful assistant.",
    )

    async for event in agent.run("Find information about X"):
        print(event)
"""

from tulip.core.config import TulipSettings
from tulip.core.errors import TulipError
from tulip.core.events import (
    GroundingEvent,
    ReflectEvent,
    TerminateEvent,
    ThinkEvent,
    ToolCompleteEvent,
    ToolStartEvent,
    TulipEvent,
)
from tulip.core.messages import Message, Role, ToolCall
from tulip.core.state import AgentState
from tulip.tools.context import ToolContext
from tulip.tools.decorator import tool


# Lazy import mapping for optional dependencies
_LAZY_IMPORTS = {
    "Agent": ("tulip.agent.agent", "Agent"),
    "AgentConfig": ("tulip.agent.config", "AgentConfig"),
    "AgentResult": ("tulip.agent.result", "AgentResult"),
    # Composition primitives — graph-free orchestration shapes.
    "SequentialPipeline": ("tulip.agent.composition", "SequentialPipeline"),
    "ParallelPipeline": ("tulip.agent.composition", "ParallelPipeline"),
    "LoopAgent": ("tulip.agent.composition", "LoopAgent"),
    # Multi-agent primitives — graph + handoff + orchestrator/specialist.
    "StateGraph": ("tulip.multiagent.graph", "StateGraph"),
    "GraphConfig": ("tulip.multiagent.graph", "GraphConfig"),
    "START": ("tulip.multiagent.graph", "START"),
    "END": ("tulip.multiagent.graph", "END"),
    "Send": ("tulip.core.send", "Send"),
    "Handoff": ("tulip.multiagent.handoff", "Handoff"),
    "HandoffContext": ("tulip.multiagent.handoff", "HandoffContext"),
    "HandoffReason": ("tulip.multiagent.handoff", "HandoffReason"),
    "create_handoff_agent": ("tulip.multiagent.handoff", "create_handoff_agent"),
    "create_handoff_manager": ("tulip.multiagent.handoff", "create_handoff_manager"),
    "Orchestrator": ("tulip.multiagent.orchestrator", "Orchestrator"),
    "RoutingDecision": ("tulip.multiagent.orchestrator", "RoutingDecision"),
    "create_orchestrator": ("tulip.multiagent.orchestrator", "create_orchestrator"),
    "Specialist": ("tulip.multiagent.specialist", "Specialist"),
    # Public name "Reflexion" maps to the actual class "Reflector". The
    # mismatch was an import error on tulip 0.1.0 — keep the alias so existing
    # docs / code that does ``from tulip import Reflexion`` keeps working.
    "Reflexion": ("tulip.reasoning.reflexion", "Reflector"),
    "Reflector": ("tulip.reasoning.reflexion", "Reflector"),
    "GroundingEvaluator": ("tulip.reasoning.grounding", "GroundingEvaluator"),
    "CausalChain": ("tulip.reasoning.causal", "CausalChain"),
    "HookProvider": ("tulip.hooks.provider", "HookProvider"),
    "HookRegistry": ("tulip.hooks.registry", "HookRegistry"),
    # RAG
    "RAGRetriever": ("tulip.rag.retriever", "RAGRetriever"),
    "OpenAIEmbeddings": ("tulip.rag.embeddings.openai", "OpenAIEmbeddings"),
    # PRISM router — bounded-graph generation atop tulip primitives.
    "Router": ("tulip.router.runtime", "Router"),
    "GoalFrame": ("tulip.router.goal_frame", "GoalFrame"),
    "TaskType": ("tulip.router.goal_frame", "TaskType"),
    "Risk": ("tulip.router.goal_frame", "Risk"),
    "Complexity": ("tulip.router.goal_frame", "Complexity"),
    "Capability": ("tulip.router.capability", "Capability"),
    "CapabilityIndex": ("tulip.router.capability", "CapabilityIndex"),
    "Protocol": ("tulip.router.protocol", "Protocol"),
    "ProtocolRegistry": ("tulip.router.protocol", "ProtocolRegistry"),
    "PolicyGate": ("tulip.router.policy", "PolicyGate"),
    "PolicyVerdict": ("tulip.router.policy", "PolicyVerdict"),
    "CognitiveCompiler": ("tulip.router.compiler", "CognitiveCompiler"),
    "RunnableResult": ("tulip.router.runnable", "RunnableResult"),
    "SkillIndex": ("tulip.router.skill_index", "SkillIndex"),
    "builtin_protocols": ("tulip.router.protocol", "builtin_protocols"),
    # Deep research — research-shaped Agent factory + provider protocol.
    # Submodule is ``tulip.deepagent`` (Pythonic path-name convention).
    # Factory is ``create_deepagent`` (matches ``create_orchestrator``,
    # ``create_handoff_agent`` — the existing tulip naming for "build me
    # a configured X").
    "create_deepagent": ("tulip.deepagent", "create_deepagent"),
    "create_research_workflow": ("tulip.deepagent.workflow", "create_research_workflow"),
    "make_execute_node": ("tulip.deepagent.workflow", "make_execute_node"),
    "make_causal_inference_node": ("tulip.deepagent.workflow", "make_causal_inference_node"),
    "make_summarize_node": ("tulip.deepagent.workflow", "make_summarize_node"),
    "make_grounding_eval_node": ("tulip.deepagent.workflow", "make_grounding_eval_node"),
    "make_regenerate_summary_node": ("tulip.deepagent.workflow", "make_regenerate_summary_node"),
    "make_replan_node": ("tulip.deepagent.workflow", "make_replan_node"),
    "route_after_grounding": ("tulip.deepagent.workflow", "route_after_grounding"),
    # Research workflow state keys
    "KEY_PROMPT": ("tulip.deepagent.workflow", "KEY_PROMPT"),
    "KEY_EXECUTE_PROMPT": ("tulip.deepagent.workflow", "KEY_EXECUTE_PROMPT"),
    "KEY_EVIDENCE": ("tulip.deepagent.workflow", "KEY_EVIDENCE"),
    "KEY_GROUNDING_FACTS": ("tulip.deepagent.workflow", "KEY_GROUNDING_FACTS"),
    "KEY_CAUSAL_CHAIN": ("tulip.deepagent.workflow", "KEY_CAUSAL_CHAIN"),
    "KEY_CAUSAL_HYPOTHESIS": ("tulip.deepagent.workflow", "KEY_CAUSAL_HYPOTHESIS"),
    "KEY_CAUSAL_CONFIDENCE": ("tulip.deepagent.workflow", "KEY_CAUSAL_CONFIDENCE"),
    "KEY_SUMMARY": ("tulip.deepagent.workflow", "KEY_SUMMARY"),
    "KEY_STRUCTURED_OUTPUT": ("tulip.deepagent.workflow", "KEY_STRUCTURED_OUTPUT"),
    "KEY_GROUNDING_SCORE": ("tulip.deepagent.workflow", "KEY_GROUNDING_SCORE"),
    "KEY_UNGROUNDED_CLAIMS": ("tulip.deepagent.workflow", "KEY_UNGROUNDED_CLAIMS"),
    "KEY_REPLAN_COUNT": ("tulip.deepagent.workflow", "KEY_REPLAN_COUNT"),
    "KEY_REGENERATION_COUNT": ("tulip.deepagent.workflow", "KEY_REGENERATION_COUNT"),
    "KEY_STOP_REASON": ("tulip.deepagent.workflow", "KEY_STOP_REASON"),
    "KnowledgeProvider": ("tulip.deepagent", "KnowledgeProvider"),
    "KnowledgeRow": ("tulip.deepagent", "KnowledgeRow"),
    "ItemRef": ("tulip.deepagent", "ItemRef"),
    "Grounding": ("tulip.deepagent", "Grounding"),
    # Security — evidence-grounded findings (the cybersecurity layer).
    "Evidence": ("tulip.security", "Evidence"),
    "Indicator": ("tulip.security", "Indicator"),
    "Severity": ("tulip.security", "Severity"),
    "IndicatorType": ("tulip.security", "IndicatorType"),
    "FingerprintFinding": ("tulip.security", "FingerprintFinding"),
    "FingerprintVerdict": ("tulip.security", "FingerprintVerdict"),
    "FingerprintClassifier": ("tulip.security", "FingerprintClassifier"),
    "Abstention": ("tulip.security", "Abstention"),
    "GroundedFinding": ("tulip.security", "GroundedFinding"),
    "ground_finding": ("tulip.security", "ground_finding"),
    "ground_fingerprint": ("tulip.security", "ground_fingerprint"),
    "is_finding": ("tulip.security", "is_finding"),
    "AtlasTechnique": ("tulip.security", "AtlasTechnique"),
    "OwaspLLM": ("tulip.security", "OwaspLLM"),
    "OwaspASI": ("tulip.security", "OwaspASI"),
    "severity_at_least": ("tulip.security", "severity_at_least"),
}


def __getattr__(name: str) -> object:
    """Lazy import for Agent and model classes."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        import importlib

        module = importlib.import_module(module_path)
        return getattr(module, attr_name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__version__ = "2.1.0"
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
    "TulipError",
    "TulipEvent",
    "TulipSettings",
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
    "OpenAIEmbeddings",
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
    # Security — evidence-grounded findings (lazy)
    "Abstention",
    "AtlasTechnique",
    "Evidence",
    "FingerprintClassifier",
    "FingerprintFinding",
    "FingerprintVerdict",
    "GroundedFinding",
    "Indicator",
    "IndicatorType",
    "OwaspASI",
    "OwaspLLM",
    "Severity",
    "ground_finding",
    "ground_fingerprint",
    "is_finding",
    "severity_at_least",
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
