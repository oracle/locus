# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL

"""Integration test configuration with smart service detection.

This module auto-detects available services and credentials, skipping tests
when their requirements aren't met. No manual SKIP_* flags needed.
"""

from __future__ import annotations

import os
import socket
from functools import lru_cache

import pytest


# =============================================================================
# Service Detection Helpers
# =============================================================================


def _check_port(host: str, port: int, timeout: float = 1.0) -> bool:
    """Check if a TCP port is reachable."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, TimeoutError):
        return False


@lru_cache(maxsize=1)
def redis_available() -> bool:
    """Check if Redis is available and configured."""
    url = os.getenv("REDIS_URL")
    # Require explicit REDIS_URL to avoid connecting to random local redis
    if not url:
        return False

    # Parse host:port from redis URL
    url = url.removeprefix("redis://")
    host, _, port = url.partition(":")
    port = int(port) if port else 6379
    return _check_port(host, port)


@lru_cache(maxsize=1)
def postgres_available() -> bool:
    """Check if PostgreSQL is available and configured."""
    # Require explicit configuration to avoid connecting to random local postgres
    host = os.getenv("POSTGRES_HOST")
    user = os.getenv("POSTGRES_USER")
    database = os.getenv("POSTGRES_DB")

    # Must have explicit config
    if not (host and user and database):
        return False

    port = int(os.getenv("POSTGRES_PORT", "5432"))
    return _check_port(host, port)


@lru_cache(maxsize=1)
def mysql_available() -> bool:
    """Check if MySQL is available and configured."""
    # Require an explicit opt-in to avoid connecting to random local mysql.
    if os.getenv("TULIP_MYSQL_INTEGRATION") != "1":
        return False

    host = os.getenv("MYSQL_HOST", "localhost")
    user = os.getenv("MYSQL_USER", "tulip")
    database = os.getenv("MYSQL_DB", "tulip_test")

    if not (host and user and database):
        return False

    port = int(os.getenv("MYSQL_PORT", "3306"))
    return _check_port(host, port)


@lru_cache(maxsize=1)
def opensearch_available() -> bool:
    """Check if OpenSearch is available and credentials are set."""
    hosts = os.getenv("OPENSEARCH_HOSTS") or os.getenv("OPENSEARCH_URL")
    if not hosts:
        return False

    # Parse host:port - handle both URL and host:port formats
    host_str = hosts.replace("https://", "").replace("http://", "")
    host, _, port = host_str.partition(":")
    port_num = int(port.split("/")[0]) if port else 9200

    # For remote OpenSearch, just check if we have credentials
    user = os.getenv("OPENSEARCH_USER")
    password = os.getenv("OPENSEARCH_PASSWORD")
    if user and password:
        return True

    # For local, check port
    return _check_port(host, port_num)


@lru_cache(maxsize=1)
def openai_available() -> bool:
    """Check if OpenAI API key is set."""
    return bool(os.getenv("OPENAI_API_KEY"))


@lru_cache(maxsize=1)
def anthropic_available() -> bool:
    """Check if Anthropic API key is set."""
    return bool(os.getenv("ANTHROPIC_API_KEY"))


@lru_cache(maxsize=1)
def any_model_available() -> bool:
    """Check if any model (Anthropic or OpenAI) is available."""
    return anthropic_available() or openai_available()


# =============================================================================
# Skip Markers
# =============================================================================

# Create skip markers based on service availability
skip_without_redis = pytest.mark.skipif(
    not redis_available(), reason="Redis not available (check REDIS_URL or start Redis)"
)

skip_without_postgres = pytest.mark.skipif(
    not postgres_available(),
    reason="PostgreSQL not available (check POSTGRES_HOST/PORT or start PostgreSQL)",
)

skip_without_mysql = pytest.mark.skipif(
    not mysql_available(),
    reason=(
        "MySQL integration tests are opt-in "
        "(set TULIP_MYSQL_INTEGRATION=1 and MYSQL_HOST/PORT or start MySQL)"
    ),
)

skip_without_opensearch = pytest.mark.skipif(
    not opensearch_available(),
    reason="OpenSearch not available (set OPENSEARCH_HOSTS and credentials)",
)

skip_without_openai = pytest.mark.skipif(
    not openai_available(), reason="OpenAI API key not set (need OPENAI_API_KEY)"
)

skip_without_anthropic = pytest.mark.skipif(
    not anthropic_available(), reason="Anthropic API key not set (need ANTHROPIC_API_KEY)"
)

skip_without_model = pytest.mark.skipif(
    not any_model_available(),
    reason="No model available (need OpenAI or Anthropic API key)",
)


# =============================================================================
# Fixtures
# =============================================================================


def _build_model():
    """Build a model instance from environment variables.

    Resolution order (first match wins):

    1. OpenAI — when ``OPENAI_API_KEY`` is set.
    2. Anthropic — when ``ANTHROPIC_API_KEY`` is set.

    Model id is controlled per-provider by ``OPENAI_MODEL_ID`` /
    ``ANTHROPIC_MODEL_ID``.
    """
    # OpenAI (preferred)
    if openai_available():
        from tulip.models.native.openai import OpenAIModel

        model_id = os.getenv("OPENAI_MODEL_ID", "gpt-4o-mini")
        return OpenAIModel(model=model_id, max_tokens=8192)

    # Anthropic fallback — cheapest path for iteration.
    if anthropic_available():
        from tulip.models.native.anthropic import AnthropicModel

        model_id = os.getenv("ANTHROPIC_MODEL_ID", "claude-haiku-4-5")
        return AnthropicModel(model=model_id, max_tokens=512)

    return None


@lru_cache(maxsize=1)
def get_test_model():
    """Get the cached test model. Returns None if no model available."""
    return _build_model()


@pytest.fixture(scope="session")
def model():
    """Session-scoped model fixture for integration tests.

    Uses OpenAI if configured (OPENAI_API_KEY), falls back to Anthropic.
    Model ID from OPENAI_MODEL_ID / ANTHROPIC_MODEL_ID env vars.
    """
    m = get_test_model()
    if m is None:
        pytest.skip("No model available (need OPENAI_API_KEY or ANTHROPIC_API_KEY)")
    return m


@pytest.fixture(scope="session")
def service_status():
    """Report available services at the start of the test session."""
    return {
        "redis": redis_available(),
        "postgres": postgres_available(),
        "mysql": mysql_available(),
        "opensearch": opensearch_available(),
        "openai": openai_available(),
    }


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "requires_redis: test requires Redis")
    config.addinivalue_line("markers", "requires_postgres: test requires PostgreSQL")
    config.addinivalue_line("markers", "requires_mysql: test requires MySQL")
    config.addinivalue_line("markers", "requires_opensearch: test requires OpenSearch")
    config.addinivalue_line("markers", "requires_openai: test requires OpenAI API key")
    config.addinivalue_line(
        "markers", "requires_model: test requires any model (OpenAI or Anthropic)"
    )


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests based on marker requirements."""
    marker_map = {
        "requires_redis": skip_without_redis,
        "requires_postgres": skip_without_postgres,
        "requires_mysql": skip_without_mysql,
        "requires_opensearch": skip_without_opensearch,
        "requires_openai": skip_without_openai,
        "requires_model": skip_without_model,
    }

    for item in items:
        for marker_name, skip_marker in marker_map.items():
            if marker_name in [m.name for m in item.iter_markers()]:
                item.add_marker(skip_marker)


def pytest_report_header(config):
    """Print service availability at the start of test run."""
    lines = ["Service availability:"]
    services = [
        ("Redis", redis_available()),
        ("PostgreSQL", postgres_available()),
        ("MySQL", mysql_available()),
        ("OpenSearch", opensearch_available()),
        ("OpenAI", openai_available()),
        ("Any Model", any_model_available()),
    ]
    for name, available in services:
        status = "✓" if available else "✗"
        lines.append(f"  {status} {name}")
    return lines
