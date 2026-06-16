# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for memory/checkpointer backends."""

from unittest.mock import AsyncMock

import pytest


class TestRedisBackend:
    """Tests for Redis checkpoint backend."""

    @pytest.fixture
    def mock_redis(self):
        """Create mock Redis client."""
        mock_client = AsyncMock()
        mock_client.set = AsyncMock()
        mock_client.setex = AsyncMock()
        mock_client.get = AsyncMock(return_value='{"key": "value"}')
        mock_client.delete = AsyncMock(return_value=1)
        mock_client.exists = AsyncMock(return_value=1)
        mock_client.keys = AsyncMock(
            return_value=["locus:checkpoint:thread1", "locus:checkpoint:thread2"]
        )
        mock_client.close = AsyncMock()
        return mock_client

    def test_redis_config_defaults(self):
        """Test default Redis configuration."""
        from locus.memory.backends.redis import RedisConfig

        config = RedisConfig()
        assert config.url == "redis://localhost:6379"
        assert config.prefix == "locus:checkpoint:"
        assert config.ttl_seconds is None
        assert config.db == 0

    def test_redis_config_custom(self):
        """Test custom Redis configuration."""
        from locus.memory.backends.redis import RedisConfig

        config = RedisConfig(
            url="redis://custom:6380",
            prefix="myapp:",
            ttl_seconds=3600,
            db=1,
        )
        assert config.url == "redis://custom:6380"
        assert config.prefix == "myapp:"
        assert config.ttl_seconds == 3600
        assert config.db == 1

    def test_key_generation(self):
        """Test key generation."""
        from locus.memory.backends.redis import RedisBackend

        backend = RedisBackend(url="redis://localhost")
        key = backend._key("thread123")
        assert key == "locus:checkpoint:thread123"

    @pytest.mark.asyncio
    async def test_save_without_ttl(self, mock_redis):
        """Test saving checkpoint without TTL."""
        from locus.memory.backends.redis import RedisBackend

        backend = RedisBackend()
        backend._client = mock_redis

        await backend.save("thread1", {"key": "value"})

        mock_redis.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_with_ttl(self, mock_redis):
        """Test saving checkpoint with TTL."""
        from locus.memory.backends.redis import RedisBackend

        backend = RedisBackend(ttl_seconds=3600)
        backend._client = mock_redis

        await backend.save("thread1", {"key": "value"})

        mock_redis.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_existing(self, mock_redis):
        """Test loading existing checkpoint."""
        from locus.memory.backends.redis import RedisBackend

        backend = RedisBackend()
        backend._client = mock_redis

        data = await backend.load("thread1")

        assert data == {"key": "value"}

    @pytest.mark.asyncio
    async def test_load_nonexistent(self, mock_redis):
        """Test loading nonexistent checkpoint."""
        mock_redis.get = AsyncMock(return_value=None)

        from locus.memory.backends.redis import RedisBackend

        backend = RedisBackend()
        backend._client = mock_redis

        data = await backend.load("nonexistent")
        assert data is None

    @pytest.mark.asyncio
    async def test_delete_existing(self, mock_redis):
        """Test deleting existing checkpoint."""
        from locus.memory.backends.redis import RedisBackend

        backend = RedisBackend()
        backend._client = mock_redis

        result = await backend.delete("thread1")
        assert result is True

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, mock_redis):
        """Test deleting nonexistent checkpoint."""
        mock_redis.delete = AsyncMock(return_value=0)

        from locus.memory.backends.redis import RedisBackend

        backend = RedisBackend()
        backend._client = mock_redis

        result = await backend.delete("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_exists_true(self, mock_redis):
        """Test exists returns True."""
        from locus.memory.backends.redis import RedisBackend

        backend = RedisBackend()
        backend._client = mock_redis

        result = await backend.exists("thread1")
        assert result is True

    @pytest.mark.asyncio
    async def test_exists_false(self, mock_redis):
        """Test exists returns False."""
        mock_redis.exists = AsyncMock(return_value=0)

        from locus.memory.backends.redis import RedisBackend

        backend = RedisBackend()
        backend._client = mock_redis

        result = await backend.exists("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_list_threads(self, mock_redis):
        """Test listing threads."""
        from locus.memory.backends.redis import RedisBackend

        backend = RedisBackend()
        backend._client = mock_redis

        threads = await backend.list_threads()

        assert len(threads) == 2
        assert "thread1" in threads
        assert "thread2" in threads

    @pytest.mark.asyncio
    async def test_close(self, mock_redis):
        """Test closing connection."""
        from locus.memory.backends.redis import RedisBackend

        backend = RedisBackend()
        backend._client = mock_redis

        await backend.close()

        mock_redis.close.assert_called_once()
        assert backend._client is None


class TestPostgreSQLBackend:
    """Tests for PostgreSQL checkpoint backend."""

    def test_postgresql_config_defaults(self):
        """Test default PostgreSQL configuration."""
        from locus.memory.backends.postgresql import PostgreSQLConfig

        config = PostgreSQLConfig()
        assert config.dsn is None
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.database == "locus"
        assert config.table_name == "checkpoints"

    def test_postgresql_config_with_dsn(self):
        """Test PostgreSQL configuration with DSN."""
        from locus.memory.backends.postgresql import PostgreSQLConfig

        config = PostgreSQLConfig(dsn="postgresql://user:pass@host:5432/db")
        assert config.dsn == "postgresql://user:pass@host:5432/db"
