# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Bounded graph generation via typed goal frames.

A meta-orchestration layer that compiles natural-language requests onto
existing locus primitives (``Agent``, ``Pipeline``, ``Orchestrator``,
``StateGraph``, ``Handoff``, ``A2AClient``). The LLM only fills a typed
:class:`GoalFrame`; protocol selection, capability binding, policy
checks and graph compilation are all deterministic.

What that buys, and what it does not. The model never names a shape, so
the router cannot emit a topology outside the registered protocols, and a
prompt that tells it to invent one has nothing to grab. Every frame passes
the :class:`PolicyGate` before anything runs, and a frame too risky for
the shapes that handle it is held for approval. The same frame always
compiles to the same shape, and :meth:`Router.explain` shows why without
a model call. Routing accuracy, measured in
``tests/integration/test_router_benchmark.py``, is comparable to letting
the model pick the protocol directly: better on some models, worse on
others, with the misses visible per prompt. The guarantees above are the
reason to use it; accuracy is not the claim.

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

The extractor is one :class:`RequestClassifier`; ``Router(classifier=...)``
takes any other ``async classify(text) -> GoalFrame``, and
``CognitiveCompiler(protocol_picker=...)`` takes any :class:`ProtocolPicker`.
"""

from __future__ import annotations

from locus.router.capability import HUMAN_SENTINEL, Capability, CapabilityIndex
from locus.router.classifier import AgentFrameExtractor, RequestClassifier
from locus.router.compiler import ApprovalCallback, CognitiveCompiler
from locus.router.goal_frame import Complexity, GoalFrame, Risk, TaskType
from locus.router.picker import (
    LLMProtocolPicker,
    PickedProtocol,
    PickerError,
    ProtocolPicker,
)
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
    ProtocolExecutionError,
    Runnable,
    RunnableResult,
    SynthesizedPipelineRunnable,
)
from locus.router.runtime import FrameExtractionError, Router
from locus.router.skill_index import SkillIndex


__all__ = [
    "A2ARunnable",
    "AgentFrameExtractor",
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
    "SynthesizedPipelineRunnable",
    "ProtocolExecutionError",
    "PolicyDeniedError",
    "PolicyGate",
    "PolicyVerdict",
    "Protocol",
    "ProtocolPicker",
    "ProtocolRegistry",
    "RequestClassifier",
    "Risk",
    "Router",
    "Runnable",
    "RunnableResult",
    "SkillIndex",
    "TaskType",
    "builtin_protocols",
]
