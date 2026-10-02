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
from tulip.agent.config import AgentConfig, GroundingConfig, ModelRetryConfig, ReflexionConfig
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
    "GroundingConfig",
    "ModelRetryConfig",
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
    "load_agent_specs",
    "loop",
    "parallel",
    "run_subagent",
    "sequential",
    "task_tool",
]
