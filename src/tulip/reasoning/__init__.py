# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Reasoning patterns for Tulip (Reflexion, Grounding, Causal).

This module provides three key reasoning capabilities:

1. **Reflexion**: Self-evaluation and progress tracking with confidence adjustment.
2. **Grounding**: Verification that claims are supported by evidence.
3. **Causal**: Building and analyzing causal inference chains.

Example usage:

    from tulip.reasoning import Reflector, GroundingEvaluator, CausalChain

    # Reflexion
    reflector = Reflector()
    result = reflector.reflect(agent_state)
    if result.assessment == "loop_detected":
        print(result.guidance)

    # Grounding
    evaluator = GroundingEvaluator()
    grounding = evaluator.evaluate(claims, evidence)
    if grounding.requires_replan:
        print(evaluator.get_replan_guidance(grounding))

    # Causal
    chain = CausalChain()
    node1 = chain.create_node("Database failure", node_type="root_cause")
    node2 = chain.create_node("Service unavailable")
    chain.link(node1.id, node2.id)
    root_causes = chain.identify_root_causes()
"""

from tulip.reasoning.causal import (
    CausalChain,
    CausalConflict,
    CausalEdge,
    CausalNode,
    NodeType,
    RelationshipType,
    build_causal_chain,
)
from tulip.reasoning.grounding import (
    ClaimEvaluation,
    GroundingEvaluator,
    GroundingResult,
    evaluate_grounding,
)
from tulip.reasoning.reflexion import (
    AssessmentCategory,
    ReflectionResult,
    Reflector,
    evaluate_progress,
)


__all__ = [
    # Reflexion
    "AssessmentCategory",
    "ReflectionResult",
    "Reflector",
    "evaluate_progress",
    # Grounding
    "ClaimEvaluation",
    "GroundingEvaluator",
    "GroundingResult",
    "evaluate_grounding",
    # Causal
    "CausalChain",
    "CausalConflict",
    "CausalEdge",
    "CausalNode",
    "NodeType",
    "RelationshipType",
    "build_causal_chain",
]
