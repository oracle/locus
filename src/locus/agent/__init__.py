# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Agent implementation for Locus."""

from locus.agent.agent import Agent
from locus.agent.completion import CompletionCheck, Continuation, chain_verifiers
from locus.agent.composition import (
    LoopAgent,
    ParallelPipeline,
    PipelineResult,
    SequentialPipeline,
    loop,
    parallel,
    sequential,
)
from locus.agent.config import (
    AgentConfig,
    CompactionConfig,
    GroundingConfig,
    ModelRetryConfig,
    ObservationPackConfig,
    ReflexionConfig,
)
from locus.agent.result import AgentResult, ExecutionMetrics, StopReason, StreamingResult
from locus.agent.run_context import answer_tokens_held
from locus.agent.specs import AgentSpec, load_agent_specs
from locus.agent.subagent import Subagent, SubagentResult, run_subagent
from locus.agent.tasks import TaskRegistry, task_tool


__all__ = [
    "Agent",
    "AgentConfig",
    "AgentResult",
    "AgentSpec",
    "ExecutionMetrics",
    "CompactionConfig",
    "CompletionCheck",
    "Continuation",
    "GroundingConfig",
    "ModelRetryConfig",
    "ObservationPackConfig",
    "LoopAgent",
    "ParallelPipeline",
    "PipelineResult",
    "ReflexionConfig",
    "SequentialPipeline",
    "StopReason",
    "StreamingResult",
    "Subagent",
    "SubagentResult",
    "TaskRegistry",
    "answer_tokens_held",
    "chain_verifiers",
    "load_agent_specs",
    "loop",
    "parallel",
    "run_subagent",
    "sequential",
    "task_tool",
]
