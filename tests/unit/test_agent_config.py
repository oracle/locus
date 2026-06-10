# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for agent configuration."""

import pytest
from pydantic import ValidationError

from tulip.agent.config import (
    AgentConfig,
    GroundingConfig,
    ReflexionConfig,
)


class TestReflexionConfig:
    """Tests for ReflexionConfig."""

    def test_default_config(self):
        """Test creating config with defaults."""
        config = ReflexionConfig()
        assert config.enabled is True
        assert config.confidence_threshold == 0.85
        assert config.diminishing_returns is True
        assert config.evaluate_every_n_iterations == 1
        assert config.include_guidance is True
        assert config.model is None

    def test_custom_config(self):
        """Test creating config with custom values."""
        config = ReflexionConfig(
            enabled=False,
            confidence_threshold=0.9,
            diminishing_returns=False,
            evaluate_every_n_iterations=2,
            include_guidance=False,
            model="openai:gpt-4",
        )
        assert config.enabled is False
        assert config.confidence_threshold == 0.9
        assert config.model == "openai:gpt-4"

    def test_confidence_threshold_validation_min(self):
        """Test confidence threshold minimum validation."""
        with pytest.raises(ValidationError):
            ReflexionConfig(confidence_threshold=-0.1)

    def test_confidence_threshold_validation_max(self):
        """Test confidence threshold maximum validation."""
        with pytest.raises(ValidationError):
            ReflexionConfig(confidence_threshold=1.5)

    def test_evaluate_every_n_iterations_min(self):
        """Test evaluate_every_n_iterations minimum validation."""
        with pytest.raises(ValidationError):
            ReflexionConfig(evaluate_every_n_iterations=0)

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            ReflexionConfig(unknown_field="value")


class TestGroundingConfig:
    """Tests for GroundingConfig."""

    def test_default_config(self):
        """Test creating config with defaults."""
        config = GroundingConfig()
        assert config.enabled is True
        assert config.threshold == 0.65
        assert config.max_replans == 2
        assert config.check_before_final is True
        assert config.model is None

    def test_custom_config(self):
        """Test creating config with custom values."""
        config = GroundingConfig(
            enabled=False,
            threshold=0.8,
            max_replans=5,
            check_before_final=False,
            model="openai:gpt-4",
        )
        assert config.enabled is False
        assert config.threshold == 0.8
        assert config.max_replans == 5

    def test_threshold_validation_min(self):
        """Test threshold minimum validation."""
        with pytest.raises(ValidationError):
            GroundingConfig(threshold=-0.1)

    def test_threshold_validation_max(self):
        """Test threshold maximum validation."""
        with pytest.raises(ValidationError):
            GroundingConfig(threshold=1.5)

    def test_max_replans_min(self):
        """Test max_replans minimum validation."""
        with pytest.raises(ValidationError):
            GroundingConfig(max_replans=-1)


class TestAgentConfig:
    """Tests for AgentConfig."""

    def test_minimal_config(self):
        """Test creating config with minimal fields."""
        config = AgentConfig(model="openai:gpt-4o")
        assert config.model == "openai:gpt-4o"
        assert config.tools == []
        assert config.max_iterations == 20

    def test_full_config(self):
        """Test creating config with all fields."""
        config = AgentConfig(
            model="openai:gpt-4o",
            tools=[],
            system_prompt="You are helpful.",
            max_iterations=10,
            reflexion=ReflexionConfig(),
            grounding=GroundingConfig(),
            terminal_tools={"done"},
            tool_loop_threshold=5,
            tool_execution="sequential",
            max_concurrency=5,
            checkpoint_every_n_iterations=2,
            agent_id="test-agent",
            temperature=0.5,
            max_tokens=2048,
            metadata={"key": "value"},
        )
        assert config.system_prompt == "You are helpful."
        assert config.max_iterations == 10
        assert config.tool_execution == "sequential"
        assert config.agent_id == "test-agent"

    def test_model_validation_requires_colon(self):
        """Test that model string must contain colon."""
        with pytest.raises(ValidationError, match="provider:model"):
            AgentConfig(model="gpt-4o")

    def test_model_validation_valid_string(self):
        """Test that valid model string passes."""
        config = AgentConfig(model="openai:gpt-4o")
        assert config.model == "openai:gpt-4o"

    def test_model_validation_allows_objects(self):
        """Test that model objects are allowed."""

        class FakeModel:
            pass

        config = AgentConfig(model=FakeModel())
        assert isinstance(config.model, FakeModel)

    def test_tools_validation_none(self):
        """Test that None tools becomes empty list."""
        config = AgentConfig(model="openai:gpt-4o", tools=None)
        assert config.tools == []

    def test_tools_validation_single_tool(self):
        """Test that single tool is wrapped in list."""

        class FakeTool:
            pass

        tool = FakeTool()
        config = AgentConfig(model="openai:gpt-4o", tools=tool)
        assert config.tools == [tool]

    def test_max_iterations_min(self):
        """Test max_iterations minimum validation."""
        with pytest.raises(ValidationError):
            AgentConfig(model="openai:gpt-4o", max_iterations=0)

    def test_max_iterations_max(self):
        """Test max_iterations maximum validation (cap is 500)."""
        with pytest.raises(ValidationError):
            AgentConfig(model="openai:gpt-4o", max_iterations=501)

    def test_tool_loop_threshold_min(self):
        """Test tool_loop_threshold minimum validation."""
        with pytest.raises(ValidationError):
            AgentConfig(model="openai:gpt-4o", tool_loop_threshold=1)

    def test_temperature_min(self):
        """Test temperature minimum validation."""
        with pytest.raises(ValidationError):
            AgentConfig(model="openai:gpt-4o", temperature=-0.1)

    def test_temperature_max(self):
        """Test temperature maximum validation."""
        with pytest.raises(ValidationError):
            AgentConfig(model="openai:gpt-4o", temperature=2.5)

    def test_max_tokens_min(self):
        """Test max_tokens minimum validation."""
        with pytest.raises(ValidationError):
            AgentConfig(model="openai:gpt-4o", max_tokens=0)

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            AgentConfig(model="openai:gpt-4o", unknown_field="value")

    def test_with_reflexion(self):
        """Test with_reflexion method."""
        config = AgentConfig(model="openai:gpt-4o")
        new_config = config.with_reflexion(confidence_threshold=0.9)

        assert config.reflexion is None  # Original unchanged
        assert new_config.reflexion is not None
        assert new_config.reflexion.confidence_threshold == 0.9

    def test_with_grounding(self):
        """Test with_grounding method."""
        config = AgentConfig(model="openai:gpt-4o")
        new_config = config.with_grounding(threshold=0.8)

        assert config.grounding is None  # Original unchanged
        assert new_config.grounding is not None
        assert new_config.grounding.threshold == 0.8

    def test_with_hooks(self):
        """Test with_hooks method."""
        config = AgentConfig(model="openai:gpt-4o", hooks=["hook1"])
        new_config = config.with_hooks("hook2", "hook3")

        assert config.hooks == ["hook1"]  # Original unchanged
        assert new_config.hooks == ["hook1", "hook2", "hook3"]

    def test_default_terminal_tools(self):
        """Test default terminal tools."""
        config = AgentConfig(model="openai:gpt-4o")
        assert "submit" in config.terminal_tools
        assert "done" in config.terminal_tools
        assert "finish" in config.terminal_tools
        assert "complete" in config.terminal_tools

    def test_default_system_prompt(self):
        """Test default system prompt."""
        config = AgentConfig(model="openai:gpt-4o")
        assert "helpful" in config.system_prompt.lower()

    def test_name_accepted_and_stored(self):
        """``AgentConfig`` accepts a display name without erroring on
        ``extra='forbid'`` — users putting ``name=`` on the agent
        reasonably expect to pass it on the config too."""
        config = AgentConfig(model="openai:gpt-4o", name="planner")
        assert config.name == "planner"

    def test_name_defaults_to_none(self):
        config = AgentConfig(model="openai:gpt-4o")
        assert config.name is None


class TestReasoningShorthand:
    """Boolean shorthand for reflexion + grounding configs.

    The docs advertise ``Agent(reflexion=True)`` / ``grounding=True``
    as one-line activations. The before-validators on AgentConfig
    coerce the bool into the corresponding default config object.
    """

    def test_reflexion_true_materializes_default_config(self):
        config = AgentConfig(model="openai:gpt-4o", reflexion=True)
        assert isinstance(config.reflexion, ReflexionConfig)
        assert config.reflexion.enabled is True
        assert config.reflexion.confidence_threshold == 0.85

    def test_reflexion_false_disables(self):
        config = AgentConfig(model="openai:gpt-4o", reflexion=False)
        assert config.reflexion is None

    def test_reflexion_explicit_config_passes_through(self):
        cfg = ReflexionConfig(confidence_threshold=0.9)
        config = AgentConfig(model="openai:gpt-4o", reflexion=cfg)
        assert config.reflexion is cfg
        assert config.reflexion.confidence_threshold == 0.9

    def test_reflexion_none_stays_none(self):
        config = AgentConfig(model="openai:gpt-4o", reflexion=None)
        assert config.reflexion is None

    def test_grounding_true_materializes_default_config(self):
        config = AgentConfig(model="openai:gpt-4o", grounding=True)
        assert isinstance(config.grounding, GroundingConfig)
        assert config.grounding.enabled is True
        assert config.grounding.threshold == 0.65

    def test_grounding_false_disables(self):
        config = AgentConfig(model="openai:gpt-4o", grounding=False)
        assert config.grounding is None

    def test_grounding_explicit_config_passes_through(self):
        cfg = GroundingConfig(threshold=0.75, max_replans=5)
        config = AgentConfig(model="openai:gpt-4o", grounding=cfg)
        assert config.grounding is cfg
        assert config.grounding.threshold == 0.75
        assert config.grounding.max_replans == 5

    def test_both_true_together(self):
        config = AgentConfig(model="openai:gpt-4o", reflexion=True, grounding=True)
        assert isinstance(config.reflexion, ReflexionConfig)
        assert isinstance(config.grounding, GroundingConfig)
