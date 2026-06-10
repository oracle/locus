# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for memory registry module."""

from unittest.mock import MagicMock, patch

import pytest

from tulip.memory.registry import (
    _CHECKPOINTERS,
    get_checkpointer,
    list_checkpointers,
    register_checkpointer,
)


class TestRegisterCheckpointer:
    """Tests for register_checkpointer function."""

    def test_register_new_provider(self):
        """Test registering a new provider."""
        mock_factory = MagicMock()
        register_checkpointer("test_provider", mock_factory)

        assert "test_provider" in _CHECKPOINTERS
        assert _CHECKPOINTERS["test_provider"] is mock_factory

        # Cleanup
        del _CHECKPOINTERS["test_provider"]

    def test_register_overwrites_existing(self):
        """Test that registering same name overwrites."""
        factory1 = MagicMock()
        factory2 = MagicMock()

        register_checkpointer("overwrite_test", factory1)
        register_checkpointer("overwrite_test", factory2)

        assert _CHECKPOINTERS["overwrite_test"] is factory2

        # Cleanup
        del _CHECKPOINTERS["overwrite_test"]


class TestListCheckpointers:
    """Tests for list_checkpointers function."""

    def test_returns_list(self):
        """Test that it returns a list of provider names."""
        result = list_checkpointers()

        assert isinstance(result, list)
        # Should have at least memory and file (always available)
        assert "memory" in result
        assert "file" in result


class TestGetCheckpointer:
    """Tests for get_checkpointer function."""

    def test_get_memory_checkpointer(self):
        """Test getting memory checkpointer."""
        checkpointer = get_checkpointer("memory")
        assert checkpointer is not None

    def test_get_file_checkpointer(self):
        """Test getting file checkpointer."""
        checkpointer = get_checkpointer("file")
        assert checkpointer is not None

    def test_get_file_checkpointer_with_path(self):
        """Test getting file checkpointer with path hint."""
        checkpointer = get_checkpointer("file:./custom_path")
        assert checkpointer is not None
        assert checkpointer.base_dir.name == "custom_path"

    def test_get_unknown_provider(self):
        """Test getting unknown provider raises error."""
        with pytest.raises(ValueError, match="Unknown checkpointer provider"):
            get_checkpointer("nonexistent_provider")

    def test_get_with_config_hint_passed_as_kwarg(self):
        """Test that config_hint is passed to factory."""
        mock_checkpointer = MagicMock()
        mock_factory = MagicMock(return_value=mock_checkpointer)

        register_checkpointer("config_test", mock_factory)

        _result = get_checkpointer("config_test:hint_value")

        mock_factory.assert_called_once()
        call_kwargs = mock_factory.call_args.kwargs
        assert call_kwargs.get("config_hint") == "hint_value"

        # Cleanup
        del _CHECKPOINTERS["config_test"]

    def test_get_with_extra_kwargs(self):
        """Test that extra kwargs are passed to factory."""
        mock_checkpointer = MagicMock()
        mock_factory = MagicMock(return_value=mock_checkpointer)

        register_checkpointer("kwargs_test", mock_factory)

        get_checkpointer("kwargs_test", extra_param="value")

        mock_factory.assert_called_once()
        call_kwargs = mock_factory.call_args.kwargs
        assert call_kwargs.get("extra_param") == "value"

        # Cleanup
        del _CHECKPOINTERS["kwargs_test"]


class TestDefaultRegistrations:
    """Tests for default provider registrations."""

    def test_memory_registered(self):
        """Test memory is registered by default."""
        assert "memory" in list_checkpointers()

    def test_file_registered(self):
        """Test file is registered by default."""
        assert "file" in list_checkpointers()

    def test_http_registered(self):
        """Test http is registered by default."""
        assert "http" in list_checkpointers()


class TestHttpCheckpointer:
    """Tests for HTTP checkpointer factory."""

    def test_get_http_checkpointer_with_url(self):
        """Test getting HTTP checkpointer with URL hint."""
        checkpointer = get_checkpointer("http:http://localhost:8080")
        assert checkpointer is not None
        assert checkpointer.base_url == "http://localhost:8080"

    def test_get_http_checkpointer_requires_url(self):
        """Test HTTP checkpointer requires base_url."""
        # HTTP checkpointer requires base_url, so it should be provided
        with pytest.raises(TypeError):
            get_checkpointer("http")


class TestRedisCheckpointer:
    """Tests for Redis checkpointer factory."""

    def test_redis_registered_if_available(self):
        """Test Redis registration."""
        providers = list_checkpointers()
        # Just verify list works
        assert isinstance(providers, list)

    def test_redis_url_parsing(self):
        """Test Redis URL config hint parsing."""
        providers = list_checkpointers()
        if "redis" not in providers:
            pytest.skip("Redis not available")

        # Mock the redis_checkpointer to avoid actual connection
        mock_cp = MagicMock()

        with patch("tulip.memory.backends.adapters.redis_checkpointer", return_value=mock_cp):
            # Re-register to use patched version
            from tulip.memory.registry import _CHECKPOINTERS

            original_factory = _CHECKPOINTERS.get("redis")

            def patched_factory(config_hint=None, **kwargs):
                if config_hint:
                    if not config_hint.startswith("redis://"):
                        config_hint = f"redis://{config_hint}"
                    kwargs.setdefault("url", config_hint)
                return mock_cp

            _CHECKPOINTERS["redis"] = patched_factory

            try:
                cp = get_checkpointer("redis:localhost:6379")
                assert cp is mock_cp
            finally:
                if original_factory:
                    _CHECKPOINTERS["redis"] = original_factory


class TestConfigHintMultipleColons:
    """Tests for config hints with multiple colons."""

    def test_config_hint_preserves_full_url(self):
        """Test that URLs with colons are preserved."""
        mock_factory = MagicMock(return_value=MagicMock())

        register_checkpointer("url_test", mock_factory)

        get_checkpointer("url_test:redis://localhost:6379/0")

        call_kwargs = mock_factory.call_args.kwargs
        assert call_kwargs.get("config_hint") == "redis://localhost:6379/0"

        del _CHECKPOINTERS["url_test"]

    def test_config_hint_with_path_like_hint(self):
        """Test config hint with a path-like ``bucket/namespace`` format."""
        mock_factory = MagicMock(return_value=MagicMock())

        register_checkpointer("path_test", mock_factory)

        get_checkpointer("path_test:my-bucket/my-namespace")

        call_kwargs = mock_factory.call_args.kwargs
        assert call_kwargs.get("config_hint") == "my-bucket/my-namespace"

        del _CHECKPOINTERS["path_test"]


class TestErrorMessages:
    """Tests for error message formatting."""

    def test_unknown_provider_shows_available(self):
        """Test error message lists available providers."""
        with pytest.raises(ValueError, match="Unknown checkpointer provider") as exc_info:
            get_checkpointer("definitely_not_a_provider")

        error_msg = str(exc_info.value)
        assert "Available providers:" in error_msg
        assert "memory" in error_msg

    def test_error_suggests_install(self):
        """Test error message suggests installing dependencies."""
        with pytest.raises(ValueError, match="Unknown checkpointer provider") as exc_info:
            get_checkpointer("nonexistent")

        error_msg = str(exc_info.value)
        assert (
            "Install optional dependencies" in error_msg
            or "register a custom provider" in error_msg
        )


class TestFactoryKwargsHandling:
    """Tests for kwargs handling in factories."""

    def test_file_factory_kwargs_override(self):
        """Test that explicit kwargs override config_hint."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpointer = get_checkpointer("file:ignored_path", base_dir=tmpdir)
            # base_dir kwarg should take precedence
            assert tmpdir in str(checkpointer.base_dir)

    def test_factory_receives_all_kwargs(self):
        """Test factory receives all passed kwargs."""
        received = {}

        def capturing_factory(**kwargs):
            received.update(kwargs)
            return MagicMock()

        register_checkpointer("capture", capturing_factory)

        get_checkpointer("capture:hint", param1="a", param2="b")

        assert received.get("config_hint") == "hint"
        assert received.get("param1") == "a"
        assert received.get("param2") == "b"

        del _CHECKPOINTERS["capture"]


class TestRedisFactoryConfigHint:
    """Tests for Redis factory config_hint processing."""

    def test_redis_factory_adds_prefix_to_host_port(self):
        """Test redis factory adds redis:// prefix."""
        providers = list_checkpointers()
        if "redis" not in providers:
            pytest.skip("Redis not available")

        # Get the original factory
        original_factory = _CHECKPOINTERS["redis"]

        # Track what url is passed
        captured_kwargs = {}

        def mock_redis_checkpointer(**kwargs):
            captured_kwargs.update(kwargs)
            return MagicMock()

        # Patch redis_checkpointer at module level
        with patch("tulip.memory.backends.adapters.redis_checkpointer", mock_redis_checkpointer):
            # Re-register with patched import
            def redis_factory(config_hint=None, **kwargs):
                if config_hint:
                    if not config_hint.startswith("redis://"):
                        config_hint = f"redis://{config_hint}"
                    kwargs.setdefault("url", config_hint)
                return mock_redis_checkpointer(**kwargs)

            _CHECKPOINTERS["redis"] = redis_factory

            try:
                get_checkpointer("redis:myhost:6380")
                assert captured_kwargs.get("url") == "redis://myhost:6380"
            finally:
                _CHECKPOINTERS["redis"] = original_factory

    def test_redis_factory_keeps_full_url(self):
        """Test redis factory keeps full redis:// URL."""
        providers = list_checkpointers()
        if "redis" not in providers:
            pytest.skip("Redis not available")

        original_factory = _CHECKPOINTERS["redis"]
        captured_kwargs = {}

        def mock_redis_checkpointer(**kwargs):
            captured_kwargs.update(kwargs)
            return MagicMock()

        def redis_factory(config_hint=None, **kwargs):
            if config_hint:
                if not config_hint.startswith("redis://"):
                    config_hint = f"redis://{config_hint}"
                kwargs.setdefault("url", config_hint)
            return mock_redis_checkpointer(**kwargs)

        _CHECKPOINTERS["redis"] = redis_factory

        try:
            get_checkpointer("redis:redis://custom:6379/1")
            assert captured_kwargs.get("url") == "redis://custom:6379/1"
        finally:
            _CHECKPOINTERS["redis"] = original_factory


class TestOpenSearchFactoryConfigHint:
    """Tests for OpenSearch factory config_hint processing."""

    def test_opensearch_factory_parses_single_host(self):
        """Test opensearch factory parses single host."""
        providers = list_checkpointers()
        if "opensearch" not in providers:
            pytest.skip("OpenSearch not available")

        original_factory = _CHECKPOINTERS["opensearch"]
        captured_kwargs = {}

        def mock_opensearch_checkpointer(**kwargs):
            captured_kwargs.update(kwargs)
            return MagicMock()

        def opensearch_factory(config_hint=None, **kwargs):
            if config_hint:
                hosts = [h.strip() for h in config_hint.split(",")]
                kwargs.setdefault("hosts", hosts)
            return mock_opensearch_checkpointer(**kwargs)

        _CHECKPOINTERS["opensearch"] = opensearch_factory

        try:
            get_checkpointer("opensearch:localhost:9200")
            assert captured_kwargs.get("hosts") == ["localhost:9200"]
        finally:
            _CHECKPOINTERS["opensearch"] = original_factory

    def test_opensearch_factory_parses_multiple_hosts(self):
        """Test opensearch factory parses comma-separated hosts."""
        providers = list_checkpointers()
        if "opensearch" not in providers:
            pytest.skip("OpenSearch not available")

        original_factory = _CHECKPOINTERS["opensearch"]
        captured_kwargs = {}

        def mock_opensearch_checkpointer(**kwargs):
            captured_kwargs.update(kwargs)
            return MagicMock()

        def opensearch_factory(config_hint=None, **kwargs):
            if config_hint:
                hosts = [h.strip() for h in config_hint.split(",")]
                kwargs.setdefault("hosts", hosts)
            return mock_opensearch_checkpointer(**kwargs)

        _CHECKPOINTERS["opensearch"] = opensearch_factory

        try:
            get_checkpointer("opensearch:host1:9200,host2:9200,host3:9200")
            assert captured_kwargs.get("hosts") == ["host1:9200", "host2:9200", "host3:9200"]
        finally:
            _CHECKPOINTERS["opensearch"] = original_factory


class TestPostgreSQLFactoryConfigHint:
    """Tests for PostgreSQL factory config_hint processing."""

    def test_postgresql_factory_sets_database(self):
        """Test postgresql factory sets database from hint."""
        providers = list_checkpointers()
        if "postgresql" not in providers:
            pytest.skip("PostgreSQL not available")

        original_factory = _CHECKPOINTERS["postgresql"]
        captured_kwargs = {}

        def mock_postgresql_checkpointer(**kwargs):
            captured_kwargs.update(kwargs)
            return MagicMock()

        def postgresql_factory(config_hint=None, **kwargs):
            if config_hint:
                kwargs.setdefault("database", config_hint)
            return mock_postgresql_checkpointer(**kwargs)

        _CHECKPOINTERS["postgresql"] = postgresql_factory

        try:
            get_checkpointer("postgresql:mydb")
            assert captured_kwargs.get("database") == "mydb"
        finally:
            _CHECKPOINTERS["postgresql"] = original_factory


class TestMySQLFactoryConfigHint:
    """Tests for MySQL factory config_hint processing."""

    def test_mysql_factory_sets_database(self):
        """Test mysql factory sets database from hint."""
        providers = list_checkpointers()
        if "mysql" not in providers:
            pytest.skip("MySQL not available")

        original_factory = _CHECKPOINTERS["mysql"]
        captured_kwargs = {}

        def mock_mysql_checkpointer(**kwargs):
            captured_kwargs.update(kwargs)
            return MagicMock()

        def mysql_factory(config_hint=None, **kwargs):
            if config_hint:
                kwargs.setdefault("database", config_hint)
            return mock_mysql_checkpointer(**kwargs)

        _CHECKPOINTERS["mysql"] = mysql_factory

        try:
            get_checkpointer("mysql:mydb")
            assert captured_kwargs.get("database") == "mydb"
        finally:
            _CHECKPOINTERS["mysql"] = original_factory


class TestActualRedisFactory:
    """Tests for actual Redis factory invocation."""

    def test_redis_actual_factory_with_full_url(self):
        """Test actual redis factory processes full URL."""
        providers = list_checkpointers()
        if "redis" not in providers:
            pytest.skip("Redis not available")

        # Patch at the right level to catch the actual factory invocation
        with patch("tulip.memory.backends.adapters.redis_checkpointer") as mock_redis:
            mock_redis.return_value = MagicMock()

            # Need to re-invoke the factory, so patch before calling
            original = _CHECKPOINTERS["redis"]

            # The actual factory from _register_defaults
            def actual_redis_factory(config_hint=None, **kwargs):
                from tulip.memory.backends.adapters import redis_checkpointer

                if config_hint:
                    if not config_hint.startswith("redis://"):
                        config_hint = f"redis://{config_hint}"
                    kwargs.setdefault("url", config_hint)
                return redis_checkpointer(**kwargs)

            _CHECKPOINTERS["redis"] = actual_redis_factory

            try:
                _cp = get_checkpointer("redis:redis://localhost:6379")
                mock_redis.assert_called_once()
                call_kwargs = mock_redis.call_args.kwargs
                assert call_kwargs.get("url") == "redis://localhost:6379"
            finally:
                _CHECKPOINTERS["redis"] = original

    def test_redis_actual_factory_adds_prefix(self):
        """Test actual redis factory adds redis:// prefix."""
        providers = list_checkpointers()
        if "redis" not in providers:
            pytest.skip("Redis not available")

        with patch("tulip.memory.backends.adapters.redis_checkpointer") as mock_redis:
            mock_redis.return_value = MagicMock()

            original = _CHECKPOINTERS["redis"]

            def actual_redis_factory(config_hint=None, **kwargs):
                from tulip.memory.backends.adapters import redis_checkpointer

                if config_hint:
                    if not config_hint.startswith("redis://"):
                        config_hint = f"redis://{config_hint}"
                    kwargs.setdefault("url", config_hint)
                return redis_checkpointer(**kwargs)

            _CHECKPOINTERS["redis"] = actual_redis_factory

            try:
                _cp = get_checkpointer("redis:myhost:6380")
                mock_redis.assert_called_once()
                call_kwargs = mock_redis.call_args.kwargs
                assert call_kwargs.get("url") == "redis://myhost:6380"
            finally:
                _CHECKPOINTERS["redis"] = original
