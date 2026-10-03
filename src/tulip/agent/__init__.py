# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Agent implementation for Tulip."""

from tulip.agent.agent import Agent
from tulip.agent.completion import CompletionCheck, Continuation, chain_verifiers
from tulip.agent.composition import (
    LoopAgent,
    ParallelPipeline,
    PipelineResult,
    SequentialPipeline,
    loop,
    parallel,
    sequential,
)
from tulip.agent.config import (
    AgentConfig,
    CompactionConfig,
    GroundingConfig,
    ModelRetryConfig,
    ObservationPackConfig,
    ReflexionConfig,
)
from tulip.agent.result import AgentResult, ExecutionMetrics, StopReason, StreamingResult
from tulip.agent.specs import AgentSpec, load_agent_specs
from tulip.agent.subagent import Subagent, SubagentResult, run_subagent
from tulip.agent.tasks import TaskRegistry, task_tool


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
    "chain_verifiers",
    "load_agent_specs",
    "loop",
    "parallel",
    "run_subagent",
    "sequential",
    "task_tool",
]
