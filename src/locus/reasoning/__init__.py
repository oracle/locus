# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Reasoning patterns for Locus (Reflexion, Grounding, Causal).

This module provides three key reasoning capabilities:

1. **Reflexion**: Self-evaluation and progress tracking with confidence adjustment.
2. **Grounding**: Verification that claims are supported by evidence.
3. **Causal**: Building and analyzing causal inference chains.

Example usage:

    from locus.reasoning import Reflector, GroundingEvaluator, CausalChain

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

from locus.reasoning.causal import (
    CausalChain,
    CausalConflict,
    CausalEdge,
    CausalNode,
    NodeType,
    RelationshipType,
    build_causal_chain,
)
from locus.reasoning.grounding import (
    ClaimEvaluation,
    GroundingEvaluator,
    GroundingResult,
    evaluate_grounding,
)
from locus.reasoning.reflexion import (
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
