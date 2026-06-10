# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Checkpointer registry for Tulip - provider management and discovery."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from tulip.memory.checkpointer import BaseCheckpointer


# Provider factories: name -> factory function
_CHECKPOINTERS: dict[str, Callable[..., BaseCheckpointer]] = {}


def register_checkpointer(
    name: str,
    factory: Callable[..., BaseCheckpointer],
) -> None:
    """
    Register a checkpointer provider.

    Args:
        name: Provider name (e.g., "redis", "postgresql", "s3")
        factory: Factory function that takes kwargs and returns a checkpointer

    Example:
        >>> def my_factory(**kwargs) -> BaseCheckpointer:
        ...     return MyCustomCheckpointer(**kwargs)
        >>> register_checkpointer("custom", my_factory)
    """
    _CHECKPOINTERS[name] = factory


def get_checkpointer(checkpointer_string: str, **kwargs: Any) -> BaseCheckpointer:
    """
    Get a checkpointer from a string identifier.

    Format: "provider" or "provider:config_hint"

    The config_hint is provider-specific and is passed as a keyword argument.

    Examples:
        - "memory" -> MemoryCheckpointer
        - "file:./checkpoints" -> FileCheckpointer(base_dir="./checkpoints")
        - "redis:localhost:6379" -> RedisCheckpointer(url="redis://localhost:6379")
        - "postgresql:mydb" -> PostgreSQLCheckpointer(database="mydb")
        - "mysql:mydb" -> MySQLCheckpointer(database="mydb")
        - "opensearch" -> OpenSearchCheckpointer()
        - "s3:my-bucket" -> S3Backend(bucket="my-bucket")

    Args:
        checkpointer_string: Checkpointer identifier
        **kwargs: Provider-specific configuration

    Returns:
        Checkpointer instance

    Raises:
        ValueError: If provider is unknown
    """
    if ":" in checkpointer_string:
        provider, config_hint = checkpointer_string.split(":", 1)
    else:
        provider = checkpointer_string
        config_hint = None

    if provider not in _CHECKPOINTERS:
        available = list(_CHECKPOINTERS.keys())
        raise ValueError(
            f"Unknown checkpointer provider: '{provider}'. "
            f"Available providers: {available}. "
            f"Install optional dependencies or register a custom provider."
        )

    # Pass config_hint if provided
    if config_hint:
        kwargs["config_hint"] = config_hint

    return _CHECKPOINTERS[provider](**kwargs)


def list_checkpointers() -> list[str]:
    """
    List available checkpointer providers.

    Returns:
        List of registered provider names
    """
    return list(_CHECKPOINTERS.keys())


def _register_defaults() -> None:
    """Register default checkpointers on import."""

    # Memory (always available)
    def memory_factory(**kwargs: Any) -> BaseCheckpointer:
        from tulip.memory.backends.memory import MemoryCheckpointer

        return MemoryCheckpointer()

    register_checkpointer("memory", memory_factory)

    # File (always available)
    def file_factory(config_hint: str | None = None, **kwargs: Any) -> BaseCheckpointer:
        from tulip.memory.backends.file import FileCheckpointer

        if config_hint:
            kwargs.setdefault("base_dir", config_hint)
        return FileCheckpointer(**kwargs)

    register_checkpointer("file", file_factory)

    # HTTP (always available, httpx is optional at runtime)
    def http_factory(config_hint: str | None = None, **kwargs: Any) -> BaseCheckpointer:
        from tulip.memory.backends.http import HTTPCheckpointer

        if config_hint:
            kwargs.setdefault("base_url", config_hint)
        return HTTPCheckpointer(**kwargs)

    register_checkpointer("http", http_factory)

    # Redis (optional)
    try:

        def redis_factory(config_hint: str | None = None, **kwargs: Any) -> BaseCheckpointer:
            from tulip.memory.backends.adapters import redis_checkpointer

            if config_hint:
                # Handle "host:port" format
                if not config_hint.startswith("redis://"):
                    config_hint = f"redis://{config_hint}"
                kwargs.setdefault("url", config_hint)
            return redis_checkpointer(**kwargs)

        register_checkpointer("redis", redis_factory)
    except ImportError:  # pragma: no cover - defensive parity with optional backends
        pass

    # PostgreSQL (optional)
    try:

        def postgresql_factory(config_hint: str | None = None, **kwargs: Any) -> BaseCheckpointer:
            from tulip.memory.backends.adapters import postgresql_checkpointer

            if config_hint:
                kwargs.setdefault("database", config_hint)
            return postgresql_checkpointer(**kwargs)

        register_checkpointer("postgresql", postgresql_factory)
    except ImportError:  # pragma: no cover - defensive parity with optional backends
        pass

    # MySQL (optional)
    try:

        def mysql_factory(config_hint: str | None = None, **kwargs: Any) -> BaseCheckpointer:
            from tulip.memory.backends.adapters import mysql_checkpointer

            if config_hint:
                kwargs.setdefault("database", config_hint)
            return mysql_checkpointer(**kwargs)

        register_checkpointer("mysql", mysql_factory)
    except ImportError:  # pragma: no cover - defensive parity with optional backends
        pass

    # OpenSearch (optional)
    try:

        def opensearch_factory(config_hint: str | None = None, **kwargs: Any) -> BaseCheckpointer:
            from tulip.memory.backends.adapters import opensearch_checkpointer

            if config_hint:
                # Handle "host:port" or "host:port,host:port" format
                hosts = [h.strip() for h in config_hint.split(",")]
                kwargs.setdefault("hosts", hosts)
            return opensearch_checkpointer(**kwargs)

        register_checkpointer("opensearch", opensearch_factory)
    except ImportError:  # pragma: no cover - defensive parity with optional backends
        pass

    # S3 / MinIO / R2 (optional)
    try:

        def s3_factory(config_hint: str | None = None, **kwargs: Any) -> BaseCheckpointer:
            from tulip.memory.backends.adapters import s3_checkpointer

            if config_hint:
                kwargs.setdefault("bucket", config_hint)
            return s3_checkpointer(**kwargs)

        register_checkpointer("s3", s3_factory)
    except ImportError:  # pragma: no cover - defensive parity with optional backends
        pass


# Register on import
_register_defaults()
