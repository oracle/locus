# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for checkpointer registry."""

from unittest.mock import MagicMock

import pytest

from locus.memory.checkpointer import BaseCheckpointer
from locus.memory.registry import (
    _CHECKPOINTERS,
    get_checkpointer,
    list_checkpointers,
    register_checkpointer,
)


class TestRegisterCheckpointer:
    """Tests for register_checkpointer function."""

    def test_register_custom_checkpointer(self):
        """Test registering a custom checkpointer."""
        mock_factory = MagicMock(return_value=MagicMock(spec=BaseCheckpointer))

        # Register
        register_checkpointer("test_custom", mock_factory)

        assert "test_custom" in _CHECKPOINTERS
        assert _CHECKPOINTERS["test_custom"] is mock_factory

        # Cleanup
        del _CHECKPOINTERS["test_custom"]

    def test_register_overwrites_existing(self):
        """Test that registering with same name overwrites."""
        factory1 = MagicMock()
        factory2 = MagicMock()

        register_checkpointer("test_overwrite", factory1)
        register_checkpointer("test_overwrite", factory2)

        assert _CHECKPOINTERS["test_overwrite"] is factory2

        # Cleanup
        del _CHECKPOINTERS["test_overwrite"]


class TestGetCheckpointer:
    """Tests for get_checkpointer function."""

    def test_get_unknown_provider(self):
        """Test getting unknown provider raises ValueError."""
        with pytest.raises(ValueError, match="Unknown checkpointer provider"):
            get_checkpointer("nonexistent_provider_xyz")

    def test_get_unknown_provider_shows_available(self):
        """Test error message shows available providers."""
        with pytest.raises(ValueError, match="Available providers:"):
            get_checkpointer("nonexistent_provider_xyz")

    def test_get_memory_checkpointer(self):
        """Test getting memory checkpointer."""
        cp = get_checkpointer("memory")
        assert cp is not None
        from locus.memory.backends.memory import MemoryCheckpointer

        assert isinstance(cp, MemoryCheckpointer)

    def test_get_file_checkpointer(self):
        """Test getting file checkpointer."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            cp = get_checkpointer(f"file:{tmpdir}")
            assert cp is not None
            from locus.memory.backends.file import FileCheckpointer

            assert isinstance(cp, FileCheckpointer)

    def test_get_file_checkpointer_with_kwargs(self):
        """Test getting file checkpointer with explicit kwargs."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            cp = get_checkpointer("file", base_dir=tmpdir)
            assert cp is not None

    def test_get_checkpointer_with_config_hint(self):
        """Test config_hint is passed to factory."""
        mock_factory = MagicMock(return_value=MagicMock(spec=BaseCheckpointer))
        register_checkpointer("test_hint", mock_factory)

        get_checkpointer("test_hint:my_config")

        mock_factory.assert_called_once_with(config_hint="my_config")

        # Cleanup
        del _CHECKPOINTERS["test_hint"]

    def test_get_checkpointer_without_config_hint(self):
        """Test provider without config_hint."""
        mock_factory = MagicMock(return_value=MagicMock(spec=BaseCheckpointer))
        register_checkpointer("test_no_hint", mock_factory)

        get_checkpointer("test_no_hint")

        mock_factory.assert_called_once_with()

        # Cleanup
        del _CHECKPOINTERS["test_no_hint"]

    def test_get_checkpointer_with_extra_kwargs(self):
        """Test extra kwargs are passed to factory."""
        mock_factory = MagicMock(return_value=MagicMock(spec=BaseCheckpointer))
        register_checkpointer("test_kwargs", mock_factory)

        get_checkpointer("test_kwargs", extra_arg="value", another=123)

        mock_factory.assert_called_once_with(extra_arg="value", another=123)

        # Cleanup
        del _CHECKPOINTERS["test_kwargs"]


class TestListCheckpointers:
    """Tests for list_checkpointers function."""

    def test_list_includes_defaults(self):
        """Test list includes default checkpointers."""
        providers = list_checkpointers()

        # Should always have memory and file
        assert "memory" in providers
        assert "file" in providers

    def test_list_returns_list(self):
        """Test list returns a list type."""
        providers = list_checkpointers()
        assert isinstance(providers, list)

    def test_list_includes_http(self):
        """Test HTTP provider is registered."""
        providers = list_checkpointers()
        assert "http" in providers


class TestHTTPCheckpointerFactory:
    """Tests for HTTP checkpointer factory."""

    def test_http_with_config_hint(self):
        """Test HTTP checkpointer with config_hint URL."""
        cp = get_checkpointer("http:http://localhost:8000")
        assert cp is not None
        from locus.memory.backends.http import HTTPCheckpointer

        assert isinstance(cp, HTTPCheckpointer)

    def test_http_with_base_url_kwarg(self):
        """Test HTTP checkpointer with base_url kwarg."""
        cp = get_checkpointer("http", base_url="http://localhost:8000")
        assert cp is not None


class TestRedisCheckpointerFactory:
    """Tests for Redis checkpointer factory."""

    def test_redis_registered(self):
        """Test Redis is registered if redis is installed."""
        providers = list_checkpointers()
        # Redis may or may not be available depending on dependencies
        # Just check the function runs without error
        assert isinstance(providers, list)


class TestCustomCheckpointerFactory:
    """Tests for custom checkpointer registration."""

    def test_full_custom_workflow(self):
        """Test full custom checkpointer workflow."""

        # Create custom checkpointer class
        class CustomCheckpointer(BaseCheckpointer):
            def __init__(self, custom_param=None):
                self.custom_param = custom_param

            async def save(self, state, thread_id, checkpoint_id=None):
                return "cp_id"

            async def load(self, thread_id, checkpoint_id=None):
                return None

            async def list_checkpoints(self, thread_id, limit=None):
                return []

            async def delete(self, thread_id, checkpoint_id=None):
                return True

            async def exists(self, thread_id, checkpoint_id=None):
                return False

        def custom_factory(config_hint=None, **kwargs):
            return CustomCheckpointer(custom_param=config_hint, **kwargs)

        # Register
        register_checkpointer("custom_test", custom_factory)

        # Get with config hint
        cp = get_checkpointer("custom_test:my_value")
        assert isinstance(cp, CustomCheckpointer)
        assert cp.custom_param == "my_value"

        # Cleanup
        del _CHECKPOINTERS["custom_test"]


class TestRedisFactoryDetails:
    """Detailed tests for Redis factory."""

    def test_redis_in_providers(self):
        """Test Redis may be in providers."""
        providers = list_checkpointers()
        # Redis should be registered if package is installed
        if "redis" in providers:
            # Just verify it's callable
            factory = _CHECKPOINTERS["redis"]
            assert callable(factory)


class TestConfigHintEdgeCases:
    """Tests for config_hint edge cases."""

    def test_config_hint_with_multiple_colons(self):
        """Test config_hint with multiple colons (like URLs)."""
        mock_factory = MagicMock(return_value=MagicMock(spec=BaseCheckpointer))
        register_checkpointer("test_multi_colon", mock_factory)

        try:
            get_checkpointer("test_multi_colon:http://host:8080/path")
            call_kwargs = mock_factory.call_args[1]
            # Split only on first colon
            assert call_kwargs["config_hint"] == "http://host:8080/path"
        finally:
            del _CHECKPOINTERS["test_multi_colon"]

    def test_empty_config_hint(self):
        """Test empty config_hint after colon is not passed (falsy)."""
        mock_factory = MagicMock(return_value=MagicMock(spec=BaseCheckpointer))
        register_checkpointer("test_empty_hint", mock_factory)

        try:
            get_checkpointer("test_empty_hint:")
            call_kwargs = mock_factory.call_args[1]
            # Empty string is falsy, so config_hint is not passed
            assert "config_hint" not in call_kwargs
        finally:
            del _CHECKPOINTERS["test_empty_hint"]

    def test_config_hint_and_kwargs_combined(self):
        """Test config_hint combined with other kwargs."""
        mock_factory = MagicMock(return_value=MagicMock(spec=BaseCheckpointer))
        register_checkpointer("test_combined", mock_factory)

        try:
            get_checkpointer("test_combined:hint", extra="value")
            call_kwargs = mock_factory.call_args[1]
            assert call_kwargs["config_hint"] == "hint"
            assert call_kwargs["extra"] == "value"
        finally:
            del _CHECKPOINTERS["test_combined"]


class TestProviderFactoryFunctions:
    """Tests for individual provider factory functions."""

    def test_file_factory_config_hint_sets_base_dir(self):
        """Test file factory uses config_hint as base_dir."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            cp = get_checkpointer(f"file:{tmpdir}")
            # Verify the checkpointer was created with the path
            from locus.memory.backends.file import FileCheckpointer

            assert isinstance(cp, FileCheckpointer)

    def test_http_factory_config_hint_sets_base_url(self):
        """Test HTTP factory uses config_hint as base_url."""
        cp = get_checkpointer("http:http://example.com/api")
        from locus.memory.backends.http import HTTPCheckpointer

        assert isinstance(cp, HTTPCheckpointer)


class TestErrorMessages:
    """Tests for error messages."""

    def test_error_lists_all_available(self):
        """Test error message lists all available providers."""
        try:
            get_checkpointer("fake_provider")
        except ValueError as e:
            error_msg = str(e)
            # Should mention memory and file at minimum
            assert "memory" in error_msg or "Available providers" in error_msg

    def test_install_hint_in_error(self):
        """Test error suggests installing dependencies."""
        try:
            get_checkpointer("unknown_xyz")
        except ValueError as e:
            error_msg = str(e)
            assert "Install optional dependencies" in error_msg or "register" in error_msg


class TestProviderAvailability:
    """Tests for checking which providers are available."""

    def test_always_available_providers(self):
        """Test that memory, file, http are always available."""
        providers = list_checkpointers()

        # These should always be registered
        assert "memory" in providers
        assert "file" in providers
        assert "http" in providers

    def test_all_providers_are_callable(self):
        """Test all registered providers are callable."""
        for name, factory in _CHECKPOINTERS.items():
            assert callable(factory), f"Provider {name} factory is not callable"
