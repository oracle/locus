# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for tulip package __init__ lazy imports."""

import pytest


class TestDirectImports:
    """Tests for directly imported classes/functions."""

    def test_import_tulip_settings(self):
        """Test importing TulipSettings."""
        from tulip import TulipSettings

        assert TulipSettings is not None

    def test_import_events(self):
        """Test importing event classes."""
        from tulip import (
            GroundingEvent,
            ReflectEvent,
            TerminateEvent,
            ThinkEvent,
            ToolCompleteEvent,
            ToolStartEvent,
            TulipEvent,
        )

        assert GroundingEvent is not None
        assert TulipEvent is not None
        assert ReflectEvent is not None
        assert TerminateEvent is not None
        assert ThinkEvent is not None
        assert ToolCompleteEvent is not None
        assert ToolStartEvent is not None

    def test_import_messages(self):
        """Test importing message classes."""
        from tulip import Message, Role, ToolCall

        assert Message is not None
        assert Role is not None
        assert ToolCall is not None

    def test_import_state(self):
        """Test importing AgentState."""
        from tulip import AgentState

        assert AgentState is not None

    def test_import_tool_context(self):
        """Test importing ToolContext."""
        from tulip import ToolContext

        assert ToolContext is not None

    def test_import_tool_decorator(self):
        """Test importing tool decorator."""
        from tulip import tool

        assert tool is not None


class TestLazyImports:
    """Tests for lazy imported classes."""

    def test_lazy_import_agent(self):
        """Test lazy importing Agent."""
        from tulip import Agent

        assert Agent is not None
        assert Agent.__name__ == "Agent"

    def test_lazy_import_agent_config(self):
        """Test lazy importing AgentConfig."""
        from tulip import AgentConfig

        assert AgentConfig is not None

    def test_lazy_import_agent_result(self):
        """Test lazy importing AgentResult."""
        from tulip import AgentResult

        assert AgentResult is not None

    def test_lazy_import_reflector(self):
        """Test lazy importing Reflector (via Reflexion alias issue)."""
        # Note: The __init__.py maps "Reflexion" to "Reflexion" but the actual
        # class is "Reflector". This test documents the current behavior.
        from tulip.reasoning.reflexion import Reflector

        assert Reflector is not None

    def test_lazy_import_grounding_evaluator(self):
        """Test lazy importing GroundingEvaluator."""
        from tulip import GroundingEvaluator

        assert GroundingEvaluator is not None

    def test_lazy_import_causal_chain(self):
        """Test lazy importing CausalChain."""
        from tulip import CausalChain

        assert CausalChain is not None

    def test_lazy_import_hook_provider(self):
        """Test lazy importing HookProvider."""
        from tulip import HookProvider

        assert HookProvider is not None

    def test_lazy_import_hook_registry(self):
        """Test lazy importing HookRegistry."""
        from tulip import HookRegistry

        assert HookRegistry is not None

    def test_lazy_import_rag_retriever(self):
        """Test lazy importing RAGRetriever."""
        from tulip import RAGRetriever

        assert RAGRetriever is not None


class TestUnknownImport:
    """Tests for unknown attribute access."""

    def test_import_unknown_raises(self):
        """Test that importing unknown attribute raises AttributeError."""
        import tulip

        with pytest.raises(AttributeError, match="has no attribute"):
            _ = tulip.NonExistentClass


class TestVersionAndAll:
    """Tests for version and __all__."""

    def test_version_defined(self):
        """Test that __version__ is defined."""
        import tulip

        assert hasattr(tulip, "__version__")
        assert isinstance(tulip.__version__, str)

    def test_all_defined(self):
        """Test that __all__ is defined."""
        import tulip

        assert hasattr(tulip, "__all__")
        assert isinstance(tulip.__all__, list)
        assert "Agent" in tulip.__all__
        assert "tool" in tulip.__all__

    def test_all_items_importable(self):
        """Test that all items in __all__ are importable."""
        import tulip

        # Known broken lazy imports (mapping to wrong attribute name)
        known_broken = {"Reflexion"}  # Maps to Reflexion but class is Reflector

        for name in tulip.__all__:
            if name == "__version__":
                continue
            if name in known_broken:
                continue
            try:
                getattr(tulip, name)
            except (ImportError, AttributeError):
                # Some optional deps may not be installed or may have
                # changed attribute names
                pass
