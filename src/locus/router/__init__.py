# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Bounded graph generation via typed goal frames.

A meta-orchestration layer that compiles natural-language requests onto
existing locus primitives (``Agent``, ``Pipeline``, ``Orchestrator``,
``StateGraph``, ``Handoff``, ``A2AClient``). The LLM only fills a typed
:class:`GoalFrame`; protocol selection, capability binding, policy
checks and graph compilation are all deterministic.

Quick start::

    from locus import Agent, tool
    from locus.router import (
        Router,
        CognitiveCompiler,
        ProtocolRegistry,
        PolicyGate,
        CapabilityIndex,
        GoalFrame,
        builtin_protocols,
    )
    from locus.tools.registry import create_registry


    @tool
    def search(q: str) -> str: ...


    tools = create_registry(search)
    capabilities = CapabilityIndex(tools)
    capabilities.annotate(
        "kb_search",
        tool_name="search",
        description="Knowledge base search.",
        domain="research",
    )

    protocols = ProtocolRegistry()
    protocols.register_many(builtin_protocols())

    extractor = Agent(model="...", output_schema=GoalFrame)
    compiler = CognitiveCompiler(
        protocols=protocols,
        capabilities=capabilities,
        policy=PolicyGate(),
        model="...",
    )
    router = Router(extractor=extractor, compiler=compiler)

    result = await router.dispatch("What does X mean?")
    print(result.text, result.protocol_id)
"""

from __future__ import annotations

from locus.router.capability import HUMAN_SENTINEL, Capability, CapabilityIndex
from locus.router.compiler import ApprovalCallback, CognitiveCompiler
from locus.router.goal_frame import Complexity, GoalFrame, Risk, TaskType
from locus.router.picker import LLMProtocolPicker, PickedProtocol, PickerError
from locus.router.policy import PolicyDeniedError, PolicyGate, PolicyVerdict
from locus.router.protocol import (
    BuilderContext,
    NoMatchingProtocolError,
    Protocol,
    ProtocolRegistry,
    builtin_protocols,
)
from locus.router.runnable import (
    A2ARunnable,
    AgentRunnable,
    DebateRunnable,
    OrchestratorRunnable,
    PipelineRunnable,
    Runnable,
    RunnableResult,
)
from locus.router.runtime import FrameExtractionError, Router
from locus.router.skill_index import SkillIndex


__all__ = [
    "A2ARunnable",
    "AgentRunnable",
    "ApprovalCallback",
    "BuilderContext",
    "Capability",
    "CapabilityIndex",
    "DebateRunnable",
    "CognitiveCompiler",
    "Complexity",
    "FrameExtractionError",
    "GoalFrame",
    "HUMAN_SENTINEL",
    "LLMProtocolPicker",
    "NoMatchingProtocolError",
    "OrchestratorRunnable",
    "PickedProtocol",
    "PickerError",
    "PipelineRunnable",
    "PolicyDeniedError",
    "PolicyGate",
    "PolicyVerdict",
    "Protocol",
    "ProtocolRegistry",
    "Risk",
    "Router",
    "Runnable",
    "RunnableResult",
    "SkillIndex",
    "TaskType",
    "builtin_protocols",
]
