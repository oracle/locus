# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Comprehensive tests for reasoning modules: Reflexion, Grounding, and Causal."""

import pytest
from pydantic import ValidationError

from locus.core.state import AgentState, ToolExecution
from locus.reasoning import (
    AssessmentCategory,
    CausalChain,
    CausalConflict,
    CausalEdge,
    CausalNode,
    ClaimEvaluation,
    GroundingEvaluator,
    GroundingResult,
    NodeType,
    ReflectionResult,
    Reflector,
    RelationshipType,
    build_causal_chain,
    evaluate_grounding,
    evaluate_progress,
)


# =============================================================================
# Reflexion Tests
# =============================================================================


class TestReflectionResult:
    """Tests for ReflectionResult model."""

    def test_create_default(self):
        """Create with default values."""
        result = ReflectionResult()

        assert result.confidence_delta == 0.0
        assert result.assessment == AssessmentCategory.ON_TRACK
        assert result.guidance is None
        assert result.loop_pattern is None
        assert result.findings_summary is None

    def test_create_with_values(self):
        """Create with custom values."""
        result = ReflectionResult(
            confidence_delta=0.15,
            assessment=AssessmentCategory.NEW_FINDINGS,
            guidance="Continue investigating",
            findings_summary="Found relevant data",
        )

        assert result.confidence_delta == 0.15
        assert result.assessment == AssessmentCategory.NEW_FINDINGS
        assert result.guidance == "Continue investigating"
        assert result.findings_summary == "Found relevant data"

    def test_confidence_delta_bounds(self):
        """Confidence delta is bounded."""
        with pytest.raises(ValidationError, match="less than or equal to 1"):
            ReflectionResult(confidence_delta=1.5)

        with pytest.raises(ValidationError, match="greater than or equal to -1"):
            ReflectionResult(confidence_delta=-1.5)

    def test_frozen(self):
        """Result is immutable."""
        result = ReflectionResult()

        with pytest.raises(ValidationError):
            result.confidence_delta = 0.5  # type: ignore[misc]


class TestReflector:
    """Tests for Reflector class."""

    def test_create_with_defaults(self):
        """Create reflector with defaults."""
        reflector = Reflector()

        assert reflector.loop_threshold == 3
        assert reflector.success_weight == 0.15
        assert reflector.error_penalty == 0.2
        assert reflector.diminishing_returns is True

    def test_create_with_custom_values(self):
        """Create reflector with custom configuration."""
        reflector = Reflector(
            loop_threshold=5,
            success_weight=0.2,
            error_penalty=0.3,
            diminishing_returns=False,
        )

        assert reflector.loop_threshold == 5
        assert reflector.success_weight == 0.2
        assert reflector.error_penalty == 0.3
        assert reflector.diminishing_returns is False

    def test_reflect_empty_state(self):
        """Reflect on empty state.

        With ``completion_bonus`` (default ``0.05``), an iteration that
        produced an assistant turn but no tool calls gets a small positive
        delta — otherwise tool-less chat agents would never raise their
        confidence above ``0.0``.
        """
        reflector = Reflector()
        state = AgentState()

        result = reflector.reflect(state)

        assert result.assessment == AssessmentCategory.ON_TRACK
        assert result.confidence_delta == pytest.approx(0.05)

    def test_reflect_empty_state_with_completion_bonus_disabled(self):
        """``completion_bonus=0.0`` opts out of the no-tool-activity bump.

        Restores the legacy behaviour where a zero-success / zero-error
        iteration produced no confidence change.
        """
        reflector = Reflector(completion_bonus=0.0)
        state = AgentState()

        result = reflector.reflect(state)

        assert result.assessment == AssessmentCategory.ON_TRACK
        assert result.confidence_delta == pytest.approx(0.0)

    def test_completion_bonus_only_applies_with_no_tool_activity(self):
        """The bonus is *replaced* by success/error scoring whenever
        any tool actually fired — it doesn't stack.
        """
        reflector = Reflector(completion_bonus=0.5, success_weight=0.1)

        executions = [
            ToolExecution(
                tool_name="search",
                tool_call_id="call_1",
                arguments={"q": "test"},
                result="ok",
            ),
        ]

        result = reflector.reflect(AgentState(), executions)

        # success_weight=0.1, no error penalty, diminishing returns kick in
        # but the value is the success-driven delta, NOT the larger 0.5 bonus.
        assert result.confidence_delta < 0.5

    def test_reflect_with_successful_executions(self):
        """Reflect with successful tool executions."""
        reflector = Reflector()

        executions = [
            ToolExecution(
                tool_name="search",
                tool_call_id="call_1",
                arguments={"q": "test"},
                result="Found 5 results with relevant information about the query",
            ),
            ToolExecution(
                tool_name="read",
                tool_call_id="call_2",
                arguments={"file": "data.txt"},
                result="File contents: important data here with more than 100 characters to meet the threshold for findings detection.",
            ),
        ]

        result = reflector.reflect(AgentState(), executions)

        assert result.confidence_delta > 0.0
        assert result.assessment in (
            AssessmentCategory.ON_TRACK,
            AssessmentCategory.NEW_FINDINGS,
        )

    def test_reflect_with_failed_executions(self):
        """Reflect with failed tool executions."""
        reflector = Reflector()

        executions = [
            ToolExecution(
                tool_name="search",
                tool_call_id="call_1",
                arguments={"q": "test"},
                error="Connection timeout",
            ),
            ToolExecution(
                tool_name="read",
                tool_call_id="call_2",
                arguments={"file": "data.txt"},
                error="File not found",
            ),
        ]

        result = reflector.reflect(AgentState(), executions)

        assert result.confidence_delta < 0.0
        assert result.assessment == AssessmentCategory.STUCK
        assert result.guidance is not None

    def test_detect_single_tool_loop(self):
        """Detect repeated single tool calls across iterations."""
        from locus.core.messages import ToolCall
        from locus.core.state import ReasoningStep

        reflector = Reflector(loop_threshold=3)

        state = AgentState()
        for i in range(3):
            step = ReasoningStep(
                iteration=i + 1,
                thought=f"Search {i}",
                tool_calls=[ToolCall(name="search", arguments={"q": "same query"})],
            )
            state = state.with_reasoning_step(step)
            state = state.with_tool_execution(
                ToolExecution(
                    tool_name="search", tool_call_id=f"call_{i}", arguments={"q": "same query"}
                )
            )
            state = state.next_iteration()

        result = reflector.reflect(state)

        assert result.assessment == AssessmentCategory.LOOP_DETECTED
        assert result.loop_pattern is not None
        assert "search" in result.loop_pattern
        assert result.confidence_delta < 0.0

    def test_detect_alternating_loop(self):
        """Detect alternating tool pattern across iterations."""
        from locus.core.messages import ToolCall
        from locus.core.state import ReasoningStep

        reflector = Reflector(loop_threshold=4)

        state = AgentState()
        tools = ["search", "read", "search", "read"]
        for i, tool_name in enumerate(tools):
            step = ReasoningStep(
                iteration=i + 1,
                thought=f"Step {i}",
                tool_calls=[ToolCall(name=tool_name, arguments={})],
            )
            state = state.with_reasoning_step(step)
            state = state.with_tool_execution(
                ToolExecution(tool_name=tool_name, tool_call_id=f"call_{i}", arguments={})
            )
            state = state.next_iteration()

        result = reflector.reflect(state)

        assert result.assessment == AssessmentCategory.LOOP_DETECTED
        assert result.loop_pattern is not None

    def test_no_loop_with_varied_tools(self):
        """No loop detected with varied tool usage."""
        reflector = Reflector(loop_threshold=3)

        state = AgentState()
        for tool_name in ["search", "read", "calculate"]:
            execution = ToolExecution(
                tool_name=tool_name,
                tool_call_id="call",
                arguments={},
                result="success",
            )
            state = state.with_tool_execution(execution)

        result = reflector.reflect(state)

        assert result.assessment != AssessmentCategory.LOOP_DETECTED

    def test_diminishing_returns_applied(self):
        """Verify diminishing returns affect confidence delta."""
        reflector = Reflector(diminishing_returns=True, success_weight=0.2)

        # Low confidence - should get nearly full delta
        low_conf_delta = reflector._calculate_confidence_delta(2, 0, 0.1)

        # High confidence - should get reduced delta
        high_conf_delta = reflector._calculate_confidence_delta(2, 0, 0.9)

        assert low_conf_delta > high_conf_delta

    def test_no_diminishing_returns(self):
        """Verify no diminishing returns when disabled."""
        reflector = Reflector(diminishing_returns=False, success_weight=0.2)

        low_conf_delta = reflector._calculate_confidence_delta(2, 0, 0.1)
        high_conf_delta = reflector._calculate_confidence_delta(2, 0, 0.9)

        assert low_conf_delta == high_conf_delta

    def test_adjust_state_confidence(self):
        """Adjust state confidence through reflector."""
        reflector = Reflector(diminishing_returns=True)
        state = AgentState().with_confidence(0.5)

        result = ReflectionResult(confidence_delta=0.2)
        new_state = reflector.adjust_state_confidence(state, result)

        # With diminishing returns: 0.5 + 0.2 * (1 - 0.5) = 0.6
        assert new_state.confidence == pytest.approx(0.6)

    def test_create_guidance_message(self):
        """Create guidance message from reflection."""
        reflector = Reflector()

        result = ReflectionResult(
            assessment=AssessmentCategory.STUCK,
            guidance="Try a different approach",
            confidence_delta=-0.1,
        )

        message = reflector.create_guidance_message(result)

        assert message is not None
        assert "stuck" in message.lower()
        assert "different approach" in message

    def test_create_guidance_message_with_loop(self):
        """Create guidance message for loop detection."""
        reflector = Reflector()

        result = ReflectionResult(
            assessment=AssessmentCategory.LOOP_DETECTED,
            guidance="Break the loop",
            loop_pattern="Tool 'search' called 3 times",
        )

        message = reflector.create_guidance_message(result)

        assert message is not None
        assert "loop" in message.lower()
        assert "search" in message

    def test_no_guidance_message_when_on_track(self):
        """No guidance message when assessment is on_track without guidance."""
        reflector = Reflector()

        result = ReflectionResult(
            assessment=AssessmentCategory.ON_TRACK,
            guidance=None,
        )

        message = reflector.create_guidance_message(result)

        assert message is None


class TestEvaluateProgress:
    """Tests for the evaluate_progress convenience function."""

    def test_evaluate_progress_basic(self):
        """Basic progress evaluation."""
        state = AgentState()

        result = evaluate_progress(state)

        assert isinstance(result, ReflectionResult)

    def test_evaluate_progress_with_custom_params(self):
        """Progress evaluation with custom parameters."""
        state = AgentState()

        result = evaluate_progress(
            state,
            loop_threshold=5,
            success_weight=0.25,
        )

        assert isinstance(result, ReflectionResult)


# =============================================================================
# Grounding Tests
# =============================================================================


class TestClaimEvaluation:
    """Tests for ClaimEvaluation model."""

    def test_create_evaluation(self):
        """Create a claim evaluation."""
        evaluation = ClaimEvaluation(
            claim="The server is down",
            score=0.8,
            supporting_evidence=["Error log shows connection refused"],
            reasoning="Direct evidence of server issue",
        )

        assert evaluation.claim == "The server is down"
        assert evaluation.score == 0.8
        assert len(evaluation.supporting_evidence) == 1
        assert evaluation.is_grounded is True

    def test_is_grounded_threshold(self):
        """Test grounding threshold."""
        grounded = ClaimEvaluation(claim="test", score=0.5)
        ungrounded = ClaimEvaluation(claim="test", score=0.49)

        assert grounded.is_grounded is True
        assert ungrounded.is_grounded is False

    def test_score_bounds(self):
        """Score is bounded 0-1."""
        with pytest.raises(ValidationError, match="less than or equal to 1"):
            ClaimEvaluation(claim="test", score=1.5)

        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            ClaimEvaluation(claim="test", score=-0.1)


class TestGroundingResult:
    """Tests for GroundingResult model."""

    def test_create_result(self):
        """Create a grounding result."""
        claims = [
            ClaimEvaluation(claim="A", score=0.9),
            ClaimEvaluation(claim="B", score=0.3),
        ]

        result = GroundingResult(
            score=0.6,
            claims=claims,
            ungrounded_claims=["B"],
            requires_replan=False,
        )

        assert result.score == 0.6
        assert len(result.claims) == 2
        assert result.ungrounded_claims == ["B"]

    def test_grounded_claims_property(self):
        """Get grounded claims from result."""
        claims = [
            ClaimEvaluation(claim="A", score=0.9),
            ClaimEvaluation(claim="B", score=0.3),
            ClaimEvaluation(claim="C", score=0.7),
        ]

        result = GroundingResult(score=0.6, claims=claims)

        grounded = result.grounded_claims
        assert len(grounded) == 2
        assert all(c.claim in ("A", "C") for c in grounded)

    def test_grounding_ratio(self):
        """Calculate grounding ratio."""
        claims = [
            ClaimEvaluation(claim="A", score=0.9),
            ClaimEvaluation(claim="B", score=0.3),
            ClaimEvaluation(claim="C", score=0.7),
            ClaimEvaluation(claim="D", score=0.2),
        ]

        result = GroundingResult(score=0.5, claims=claims)

        assert result.grounding_ratio == 0.5  # 2/4 grounded

    def test_grounding_ratio_empty(self):
        """Grounding ratio with no claims."""
        result = GroundingResult(score=1.0, claims=[])

        assert result.grounding_ratio == 1.0


class TestGroundingEvaluator:
    """Tests for GroundingEvaluator class."""

    def test_create_with_defaults(self):
        """Create evaluator with defaults."""
        evaluator = GroundingEvaluator()

        assert evaluator.replan_threshold == 0.65
        assert evaluator.claim_threshold == 0.5

    def test_evaluate_no_claims(self):
        """Evaluate with no claims."""
        evaluator = GroundingEvaluator()

        result = evaluator.evaluate([], ["some evidence"])

        assert result.score == 1.0
        assert result.requires_replan is False

    def test_evaluate_exact_match(self):
        """Evaluate claims with exact evidence match."""
        evaluator = GroundingEvaluator()

        claims = ["The server returned error 500"]
        evidence = ["The server returned error 500"]

        result = evaluator.evaluate(claims, evidence)

        assert result.score == 1.0
        assert len(result.ungrounded_claims) == 0

    def test_evaluate_substring_match(self):
        """Evaluate claims found as substring in evidence."""
        evaluator = GroundingEvaluator()

        claims = ["error 500"]
        evidence = ["The server returned error 500 at 10:00 AM"]

        result = evaluator.evaluate(claims, evidence)

        assert result.score >= 0.8
        assert len(result.ungrounded_claims) == 0

    def test_evaluate_keyword_match(self):
        """Evaluate claims with keyword overlap."""
        evaluator = GroundingEvaluator()

        claims = ["Database connection failed"]
        evidence = ["The database reported a connection failure"]

        result = evaluator.evaluate(claims, evidence)

        # Should have some grounding due to keyword overlap
        assert result.score > 0.0

    def test_evaluate_ungrounded(self):
        """Evaluate completely ungrounded claims."""
        evaluator = GroundingEvaluator()

        claims = ["The moon is made of cheese"]
        evidence = ["Server health check passed", "Database is online"]

        result = evaluator.evaluate(claims, evidence)

        assert result.score == 0.0
        assert "The moon is made of cheese" in result.ungrounded_claims

    def test_requires_replan_when_below_threshold(self):
        """Trigger replan when score below threshold."""
        evaluator = GroundingEvaluator(replan_threshold=0.7)

        claims = ["Completely unverified claim"]
        evidence = ["Unrelated evidence about something else"]

        result = evaluator.evaluate(claims, evidence)

        assert result.requires_replan is True

    def test_should_replan_method(self):
        """Test should_replan method."""
        evaluator = GroundingEvaluator()

        replan_result = GroundingResult(score=0.5, requires_replan=True)
        no_replan_result = GroundingResult(score=0.8, requires_replan=False)

        assert evaluator.should_replan(replan_result) is True
        assert evaluator.should_replan(no_replan_result) is False

    def test_get_replan_guidance(self):
        """Get guidance for replanning."""
        evaluator = GroundingEvaluator(replan_threshold=0.65)

        result = GroundingResult(
            score=0.4,
            ungrounded_claims=["Claim A", "Claim B"],
            requires_replan=True,
        )

        guidance = evaluator.get_replan_guidance(result)

        assert "Claim A" in guidance
        assert "Claim B" in guidance
        assert "evidence" in guidance.lower()

    def test_get_replan_guidance_no_claims(self):
        """Guidance when all claims grounded."""
        evaluator = GroundingEvaluator()

        result = GroundingResult(
            score=0.9,
            ungrounded_claims=[],
            requires_replan=False,
        )

        guidance = evaluator.get_replan_guidance(result)

        assert "grounded" in guidance.lower()

    def test_require_evidence_setting(self):
        """Test require_evidence parameter."""
        strict = GroundingEvaluator(require_evidence=True)
        lenient = GroundingEvaluator(require_evidence=False)

        claims = ["Some claim"]
        evidence: list[str] = []  # No evidence

        strict_result = strict.evaluate(claims, evidence)
        lenient_result = lenient.evaluate(claims, evidence)

        assert strict_result.score < lenient_result.score


class TestEvaluateGrounding:
    """Tests for the evaluate_grounding convenience function."""

    def test_evaluate_grounding_basic(self):
        """Basic grounding evaluation."""
        result = evaluate_grounding(
            claims=["Test claim"],
            evidence=["Test claim is verified"],
        )

        assert isinstance(result, GroundingResult)

    def test_evaluate_grounding_with_threshold(self):
        """Grounding evaluation with custom threshold."""
        result = evaluate_grounding(
            claims=["Unverified"],
            evidence=["Something else"],
            threshold=0.9,
        )

        assert result.requires_replan is True


# =============================================================================
# Causal Tests
# =============================================================================


class TestCausalNode:
    """Tests for CausalNode model."""

    def test_create_node(self):
        """Create a causal node."""
        node = CausalNode(
            label="Database failure",
            node_type=NodeType.ROOT_CAUSE,
            evidence=["Error log entry"],
            confidence=0.9,
        )

        assert node.label == "Database failure"
        assert node.node_type == NodeType.ROOT_CAUSE
        assert len(node.evidence) == 1
        assert node.confidence == 0.9

    def test_node_auto_id(self):
        """Node generates ID automatically."""
        node = CausalNode(label="Test")

        assert node.id.startswith("node_")
        assert len(node.id) > 5

    def test_with_type(self):
        """Update node type immutably."""
        node = CausalNode(label="Test", node_type=NodeType.UNKNOWN)

        updated = node.with_type(NodeType.SYMPTOM)

        assert node.node_type == NodeType.UNKNOWN  # Original unchanged
        assert updated.node_type == NodeType.SYMPTOM

    def test_with_evidence(self):
        """Add evidence immutably."""
        node = CausalNode(label="Test", evidence=["First"])

        updated = node.with_evidence("Second")

        assert len(node.evidence) == 1  # Original unchanged
        assert len(updated.evidence) == 2
        assert "Second" in updated.evidence

    def test_with_confidence(self):
        """Update confidence immutably."""
        node = CausalNode(label="Test", confidence=0.5)

        updated = node.with_confidence(0.8)

        assert node.confidence == 0.5  # Original unchanged
        assert updated.confidence == 0.8

    def test_confidence_clamped(self):
        """Confidence is clamped to valid range."""
        node = CausalNode(label="Test")

        assert node.with_confidence(1.5).confidence == 1.0
        assert node.with_confidence(-0.5).confidence == 0.0


class TestCausalEdge:
    """Tests for CausalEdge model."""

    def test_create_edge(self):
        """Create a causal edge."""
        edge = CausalEdge(
            source_id="node_1",
            target_id="node_2",
            relationship=RelationshipType.CAUSES,
            confidence=0.8,
            reasoning="Direct observation",
        )

        assert edge.source_id == "node_1"
        assert edge.target_id == "node_2"
        assert edge.relationship == RelationshipType.CAUSES
        assert edge.is_causal is True

    def test_is_causal_property(self):
        """Test is_causal for different relationship types."""
        causes = CausalEdge(
            source_id="a",
            target_id="b",
            relationship=RelationshipType.CAUSES,
        )
        inhibits = CausalEdge(
            source_id="a",
            target_id="b",
            relationship=RelationshipType.INHIBITS,
        )
        correlates = CausalEdge(
            source_id="a",
            target_id="b",
            relationship=RelationshipType.CORRELATES_WITH,
        )
        precedes = CausalEdge(
            source_id="a",
            target_id="b",
            relationship=RelationshipType.PRECEDES,
        )

        assert causes.is_causal is True
        assert inhibits.is_causal is True
        assert correlates.is_causal is False
        assert precedes.is_causal is False


class TestCausalChain:
    """Tests for CausalChain class."""

    def test_create_empty_chain(self):
        """Create empty causal chain."""
        chain = CausalChain()

        assert len(chain.nodes) == 0
        assert len(chain.edges) == 0

    def test_add_node(self):
        """Add a node to the chain."""
        chain = CausalChain()
        node = CausalNode(label="Test")

        added = chain.add_node(node)

        assert added.id in chain.nodes
        assert chain.nodes[added.id].label == "Test"

    def test_add_duplicate_node_fails(self):
        """Cannot add node with duplicate ID."""
        chain = CausalChain()
        node = CausalNode(id="same_id", label="First")
        chain.add_node(node)

        duplicate = CausalNode(id="same_id", label="Second")

        with pytest.raises(ValueError, match="already exists"):
            chain.add_node(duplicate)

    def test_create_node(self):
        """Create and add node in one step."""
        chain = CausalChain()

        node = chain.create_node(
            label="Database failure",
            node_type=NodeType.ROOT_CAUSE,
            evidence=["Error log"],
        )

        assert node.id in chain.nodes
        assert node.node_type == NodeType.ROOT_CAUSE

    def test_add_edge(self):
        """Add an edge between nodes."""
        chain = CausalChain()
        node1 = chain.create_node("Cause")
        node2 = chain.create_node("Effect")

        edge = CausalEdge(
            source_id=node1.id,
            target_id=node2.id,
            relationship=RelationshipType.CAUSES,
        )
        chain.add_edge(edge)

        assert len(chain.edges) == 1

    def test_add_edge_missing_source_fails(self):
        """Cannot add edge with missing source node."""
        chain = CausalChain()
        node = chain.create_node("Effect")

        edge = CausalEdge(
            source_id="nonexistent",
            target_id=node.id,
        )

        with pytest.raises(ValueError, match="Source node"):
            chain.add_edge(edge)

    def test_add_edge_missing_target_fails(self):
        """Cannot add edge with missing target node."""
        chain = CausalChain()
        node = chain.create_node("Cause")

        edge = CausalEdge(
            source_id=node.id,
            target_id="nonexistent",
        )

        with pytest.raises(ValueError, match="Target node"):
            chain.add_edge(edge)

    def test_link_nodes(self):
        """Link nodes using convenience method."""
        chain = CausalChain()
        node1 = chain.create_node("Cause")
        node2 = chain.create_node("Effect")

        edge = chain.link(
            source_id=node1.id,
            target_id=node2.id,
            relationship=RelationshipType.CAUSES,
            confidence=0.9,
        )

        assert edge.source_id == node1.id
        assert edge.target_id == node2.id
        assert len(chain.edges) == 1

    def test_get_node(self):
        """Get node by ID."""
        chain = CausalChain()
        node = chain.create_node("Test")

        found = chain.get_node(node.id)
        not_found = chain.get_node("nonexistent")

        assert found is not None
        assert found.label == "Test"
        assert not_found is None

    def test_get_edges_from(self):
        """Get edges originating from a node."""
        chain = CausalChain()
        node1 = chain.create_node("Root")
        node2 = chain.create_node("Child1")
        node3 = chain.create_node("Child2")

        chain.link(node1.id, node2.id)
        chain.link(node1.id, node3.id)

        edges = chain.get_edges_from(node1.id)

        assert len(edges) == 2

    def test_get_edges_to(self):
        """Get edges pointing to a node."""
        chain = CausalChain()
        node1 = chain.create_node("Parent1")
        node2 = chain.create_node("Parent2")
        node3 = chain.create_node("Child")

        chain.link(node1.id, node3.id)
        chain.link(node2.id, node3.id)

        edges = chain.get_edges_to(node3.id)

        assert len(edges) == 2

    def test_identify_root_causes(self):
        """Identify root cause nodes."""
        chain = CausalChain()
        root = chain.create_node("Database failure", NodeType.ROOT_CAUSE)
        intermediate = chain.create_node("Query timeout")
        symptom = chain.create_node("Error page")

        chain.link(root.id, intermediate.id)
        chain.link(intermediate.id, symptom.id)

        root_causes = chain.identify_root_causes()

        assert len(root_causes) == 1
        assert root_causes[0].label == "Database failure"

    def test_identify_symptoms(self):
        """Identify symptom nodes."""
        chain = CausalChain()
        root = chain.create_node("Database failure")
        intermediate = chain.create_node("Query timeout")
        symptom = chain.create_node("Error page", NodeType.SYMPTOM)

        chain.link(root.id, intermediate.id)
        chain.link(intermediate.id, symptom.id)

        symptoms = chain.identify_symptoms()

        assert len(symptoms) == 1
        assert symptoms[0].label == "Error page"

    def test_get_causal_path(self):
        """Find path between nodes."""
        chain = CausalChain()
        node1 = chain.create_node("A")
        node2 = chain.create_node("B")
        node3 = chain.create_node("C")

        chain.link(node1.id, node2.id)
        chain.link(node2.id, node3.id)

        path = chain.get_causal_path(node1.id, node3.id)

        assert path is not None
        assert len(path) == 3
        assert path[0].label == "A"
        assert path[-1].label == "C"

    def test_get_causal_path_not_found(self):
        """No path between disconnected nodes."""
        chain = CausalChain()
        node1 = chain.create_node("A")
        node2 = chain.create_node("B")
        # No edge between them

        path = chain.get_causal_path(node1.id, node2.id)

        assert path is None

    def test_get_causal_path_same_node(self):
        """Path from node to itself."""
        chain = CausalChain()
        node = chain.create_node("A")

        path = chain.get_causal_path(node.id, node.id)

        assert path is not None
        assert len(path) == 1

    def test_detect_cycle_conflict(self):
        """Detect cycle in causal graph."""
        chain = CausalChain()
        node1 = chain.create_node("A")
        node2 = chain.create_node("B")
        node3 = chain.create_node("C")

        chain.link(node1.id, node2.id)
        chain.link(node2.id, node3.id)
        chain.link(node3.id, node1.id)  # Creates cycle

        conflicts = chain.detect_conflicts()

        cycle_conflicts = [c for c in conflicts if c.conflict_type == "cycle"]
        assert len(cycle_conflicts) >= 1

    def test_detect_bidirectional_conflict(self):
        """Detect bidirectional causation."""
        chain = CausalChain()
        node1 = chain.create_node("A")
        node2 = chain.create_node("B")

        chain.link(node1.id, node2.id, RelationshipType.CAUSES)
        chain.link(node2.id, node1.id, RelationshipType.CAUSES)

        conflicts = chain.detect_conflicts()

        bidirectional = [c for c in conflicts if c.conflict_type == "bidirectional_causation"]
        assert len(bidirectional) >= 1

    def test_detect_contradictory_conflict(self):
        """Detect contradictory relationships."""
        chain = CausalChain()
        node1 = chain.create_node("A")
        node2 = chain.create_node("B")

        chain.link(node1.id, node2.id, RelationshipType.CAUSES)
        chain.link(node1.id, node2.id, RelationshipType.INHIBITS)

        conflicts = chain.detect_conflicts()

        contradictory = [c for c in conflicts if c.conflict_type == "contradictory_relationship"]
        assert len(contradictory) >= 1

    def test_no_conflicts_in_clean_graph(self):
        """No conflicts in well-formed graph."""
        chain = CausalChain()
        node1 = chain.create_node("Root")
        node2 = chain.create_node("Middle")
        node3 = chain.create_node("Leaf")

        chain.link(node1.id, node2.id)
        chain.link(node2.id, node3.id)

        conflicts = chain.detect_conflicts()

        assert len(conflicts) == 0

    def test_classify_nodes(self):
        """Classify all nodes based on graph structure."""
        chain = CausalChain()
        node1 = chain.create_node("Root")
        node2 = chain.create_node("Middle")
        node3 = chain.create_node("Leaf")

        chain.link(node1.id, node2.id)
        chain.link(node2.id, node3.id)

        classifications = chain.classify_nodes()

        assert classifications[node1.id] == NodeType.ROOT_CAUSE
        assert classifications[node2.id] == NodeType.INTERMEDIATE
        assert classifications[node3.id] == NodeType.SYMPTOM

    def test_update_node_types(self):
        """Update node types in place."""
        chain = CausalChain()
        node1 = chain.create_node("Root")
        node2 = chain.create_node("Leaf")

        chain.link(node1.id, node2.id)
        chain.update_node_types()

        assert chain.get_node(node1.id).node_type == NodeType.ROOT_CAUSE  # type: ignore[union-attr]
        assert chain.get_node(node2.id).node_type == NodeType.SYMPTOM  # type: ignore[union-attr]

    def test_get_chain_summary(self):
        """Get summary of causal chain."""
        chain = CausalChain()
        node1 = chain.create_node("Database failure")
        node2 = chain.create_node("Query timeout")
        node3 = chain.create_node("Error page")

        chain.link(node1.id, node2.id)
        chain.link(node2.id, node3.id)

        summary = chain.get_chain_summary()

        assert summary["total_nodes"] == 3
        assert summary["total_edges"] == 2
        assert summary["conflicts"] == 0
        assert "Database failure" in summary["root_causes"]
        assert "Error page" in summary["symptoms"]

    def test_serialization_roundtrip(self):
        """Serialize and deserialize causal chain."""
        chain = CausalChain()
        node1 = chain.create_node("A", NodeType.ROOT_CAUSE)
        node2 = chain.create_node("B", NodeType.SYMPTOM)
        chain.link(node1.id, node2.id)

        data = chain.to_dict()
        restored = CausalChain.from_dict(data)

        assert len(restored.nodes) == 2
        assert len(restored.edges) == 1
        assert restored.get_node(node1.id).label == "A"  # type: ignore[union-attr]


class TestBuildCausalChain:
    """Tests for the build_causal_chain convenience function."""

    def test_build_basic_chain(self):
        """Build chain from event list."""
        events = [
            {"label": "Database failure"},
            {"label": "Query timeout", "causes": ["Database failure"]},
            {"label": "Error page", "causes": ["Query timeout"]},
        ]

        chain = build_causal_chain(events)

        assert len(chain.nodes) == 3
        assert len(chain.edges) == 2

    def test_build_chain_with_types(self):
        """Build chain with explicit node types."""
        events = [
            {"label": "Root", "type": "root_cause"},
            {"label": "Leaf", "type": "symptom", "causes": ["Root"]},
        ]

        chain = build_causal_chain(events, auto_classify=False)

        root = next(n for n in chain.nodes.values() if n.label == "Root")
        assert root.node_type == NodeType.ROOT_CAUSE

    def test_build_chain_auto_classify(self):
        """Build chain with auto classification."""
        events = [
            {"label": "Root"},
            {"label": "Leaf", "causes": ["Root"]},
        ]

        chain = build_causal_chain(events, auto_classify=True)

        root = next(n for n in chain.nodes.values() if n.label == "Root")
        leaf = next(n for n in chain.nodes.values() if n.label == "Leaf")

        assert root.node_type == NodeType.ROOT_CAUSE
        assert leaf.node_type == NodeType.SYMPTOM

    def test_build_chain_with_evidence(self):
        """Build chain with evidence."""
        events = [
            {
                "label": "Failure",
                "evidence": ["Log entry 1", "Log entry 2"],
                "confidence": 0.9,
            },
        ]

        chain = build_causal_chain(events)

        node = next(iter(chain.nodes.values()))
        assert len(node.evidence) == 2
        assert node.confidence == 0.9


class TestCausalConflict:
    """Tests for CausalConflict model."""

    def test_create_conflict(self):
        """Create a causal conflict."""
        conflict = CausalConflict(
            conflict_type="cycle",
            involved_nodes=["a", "b", "c"],
            involved_edges=[("a", "b"), ("b", "c"), ("c", "a")],
            description="Cycle detected: a -> b -> c -> a",
            resolution_hint="Break one edge",
        )

        assert conflict.conflict_type == "cycle"
        assert len(conflict.involved_nodes) == 3
        assert len(conflict.involved_edges) == 3
