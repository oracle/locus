# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Agent implementation for Locus."""

from locus.agent.agent import Agent
from locus.agent.composition import (
    LoopAgent,
    ParallelPipeline,
    PipelineResult,
    SequentialPipeline,
    loop,
    parallel,
    sequential,
)
from locus.agent.config import AgentConfig, GroundingConfig, ReflexionConfig
from locus.agent.result import AgentResult, ExecutionMetrics, StopReason, StreamingResult


__all__ = [
    "Agent",
    "AgentConfig",
    "AgentResult",
    "ExecutionMetrics",
    "GroundingConfig",
    "LoopAgent",
    "ParallelPipeline",
    "PipelineResult",
    "ReflexionConfig",
    "SequentialPipeline",
    "StopReason",
    "StreamingResult",
    "loop",
    "parallel",
    "sequential",
]
