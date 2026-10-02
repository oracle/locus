# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Agent implementation for Tulip."""

from tulip.agent.agent import Agent
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
    ReflexionConfig,
)
from tulip.agent.result import AgentResult, ExecutionMetrics, StopReason, StreamingResult
from tulip.agent.subagent import SubagentResult, run_subagent


__all__ = [
    "Agent",
    "AgentConfig",
    "AgentResult",
    "ExecutionMetrics",
    "CompactionConfig",
    "GroundingConfig",
    "ModelRetryConfig",
    "LoopAgent",
    "ParallelPipeline",
    "PipelineResult",
    "ReflexionConfig",
    "SequentialPipeline",
    "StopReason",
    "StreamingResult",
    "SubagentResult",
    "loop",
    "parallel",
    "run_subagent",
    "sequential",
]
