# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Integration test configuration with smart service detection.

This module auto-detects available services and credentials, skipping tests
when their requirements aren't met. No manual SKIP_* flags needed.
"""

from __future__ import annotations

import os
import socket
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

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

    # Parse robustly — handles redis://[user:pass@]host:port/db and rediss://.
    parts = urlsplit(url)
    host = parts.hostname or "localhost"
    port = parts.port or 6379
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
    if os.getenv("LOCUS_MYSQL_INTEGRATION") != "1":
        return False

    host = os.getenv("MYSQL_HOST", "localhost")
    user = os.getenv("MYSQL_USER", "locus")
    database = os.getenv("MYSQL_DB", "locus_test")

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
def oci_config_available() -> bool:
    """Check if OCI config exists and required env vars are set."""
    config_path = Path.home() / ".oci" / "config"
    if not config_path.exists():
        return False

    # Profile is required; endpoint is optional (can be derived from region)
    profile = os.getenv("OCI_PROFILE")

    return bool(profile)


@lru_cache(maxsize=1)
def anthropic_available() -> bool:
    """Check if Anthropic API key is set."""
    return bool(os.getenv("ANTHROPIC_API_KEY"))


@lru_cache(maxsize=1)
def any_model_available() -> bool:
    """Check if any model is available: ``LOCUS_TEST_MODEL``, OpenAI, Anthropic or OCI."""
    return (
        bool(os.getenv("LOCUS_TEST_MODEL"))
        or anthropic_available()
        or openai_available()
        or oci_config_available()
    )


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
        "(set LOCUS_MYSQL_INTEGRATION=1 and MYSQL_HOST/PORT or start MySQL)"
    ),
)

skip_without_opensearch = pytest.mark.skipif(
    not opensearch_available(),
    reason="OpenSearch not available (set OPENSEARCH_HOSTS and credentials)",
)

skip_without_oci = pytest.mark.skipif(
    not oci_config_available(),
    reason="OCI not configured (need ~/.oci/config + OCI_PROFILE)",
)

skip_without_openai = pytest.mark.skipif(
    not openai_available(), reason="OpenAI API key not set (need OPENAI_API_KEY)"
)

skip_without_anthropic = pytest.mark.skipif(
    not anthropic_available(), reason="Anthropic API key not set (need ANTHROPIC_API_KEY)"
)

skip_without_model = pytest.mark.skipif(
    not any_model_available(),
    reason=(
        "No model available (set LOCUS_TEST_MODEL=provider:model, "
        "OPENAI_API_KEY, ANTHROPIC_API_KEY or OCI_PROFILE)"
    ),
)


# =============================================================================
# Fixtures
# =============================================================================


def _build_model():
    """Build a model instance from environment variables.

    Resolution order (first match wins):

    0. ``LOCUS_TEST_MODEL`` — any ``provider:model`` string the model
       registry knows (``gemini:gemini-2.5-flash``, ``groq:llama-3.3-70b``,
       ``mistral:…``, ``ollama:qwen3``, ``openrouter:…``; see
       ``locus.models.registry.list_providers()``), so the generic tests run
       against any provider. The provider reads its own credentials.
       ``LOCUS_TEST_MAX_TOKENS`` (default 4096) caps each response.
    1. OpenAI — when ``OPENAI_API_KEY`` is set.
    2. Anthropic — when ``ANTHROPIC_API_KEY`` is set.
    3. OCI Generative AI — when ``OCI_PROFILE`` is set. Goes through the
       ``oci:`` string factory so the transport follows the model id
       (``OCI_MODEL_ID``, default ``meta.llama-3.3-70b-instruct``);
       ``OCI_COMPARTMENT`` / ``OCI_COMPARTMENT_ID``, ``OCI_REGION`` and
       ``OCI_ENDPOINT`` are honoured, ``OCI_AUTH_TYPE`` selects a
       principal-based mode instead of the profile.

    Model id is controlled per-provider by ``OPENAI_MODEL_ID`` /
    ``ANTHROPIC_MODEL_ID`` / ``OCI_MODEL_ID``.
    """
    spec = os.getenv("LOCUS_TEST_MODEL")
    if spec:
        from locus.models import get_model

        return get_model(spec, max_tokens=int(os.getenv("LOCUS_TEST_MAX_TOKENS", "4096")))

    # OpenAI (preferred)
    if openai_available():
        from locus.models.native.openai import OpenAIModel

        model_id = os.getenv("OPENAI_MODEL_ID", "gpt-4o-mini")
        return OpenAIModel(model=model_id, max_tokens=8192)

    # Anthropic fallback — cheapest path for iteration.
    if anthropic_available():
        from locus.models.native.anthropic import AnthropicModel

        model_id = os.getenv("ANTHROPIC_MODEL_ID", "claude-haiku-4-5")
        return AnthropicModel(model=model_id, max_tokens=512)

    # OCI Generative AI — the Oracle-native path.
    if oci_config_available():
        return build_oci_model()

    return None


def build_oci_model(max_tokens: int = 2048) -> Any:
    """OCI Generative AI model from the ``OCI_*`` environment.

    ``OCI_MODEL_ID`` (default ``meta.llama-3.3-70b-instruct``) goes through the
    ``oci:`` string factory so the transport follows the model id;
    ``OCI_PROFILE``, ``OCI_REGION``, ``OCI_COMPARTMENT`` / ``OCI_COMPARTMENT_ID``
    and ``OCI_ENDPOINT`` are honoured, ``OCI_AUTH_TYPE`` selects a
    principal-based mode instead of the profile. Shared by ``_build_model`` and
    the live test files that parametrize over provider factories.
    """
    from locus.models import get_model

    model_id = os.getenv("OCI_MODEL_ID", "meta.llama-3.3-70b-instruct")
    profile = os.getenv("OCI_PROFILE", "DEFAULT")
    region = os.getenv("OCI_REGION", "us-chicago-1")
    compartment = os.getenv("OCI_COMPARTMENT") or os.getenv("OCI_COMPARTMENT_ID")
    auth_type = os.getenv("OCI_AUTH_TYPE")
    kwargs: dict[str, Any] = {"max_tokens": max_tokens}
    if compartment:
        kwargs["compartment_id"] = compartment
    if model_id.lower().startswith(("cohere.", "ocid1.generativeaiendpoint.")):
        # SDK transport (OCIModel) argument names.
        kwargs["profile_name"] = profile
        kwargs["service_endpoint"] = os.getenv(
            "OCI_ENDPOINT",
            f"https://inference.generativeai.{region}.oci.oraclecloud.com",
        )
    elif auth_type in ("instance_principal", "resource_principal", "security_token"):
        kwargs["auth_type"] = auth_type
        kwargs["region"] = region
    else:
        # V1 transport (OCIChatCompletionsModel): profile implies api-key
        # or session auth from ~/.oci/config.
        kwargs["profile"] = profile
        kwargs["region"] = region
    return get_model(f"oci:{model_id}", **kwargs)


@lru_cache(maxsize=1)
def get_test_model():
    """Get the cached test model. Returns None if no model available."""
    return _build_model()


@pytest.fixture(scope="session")
def model():
    """Session-scoped model fixture for integration tests.

    ``LOCUS_TEST_MODEL=provider:model`` selects any registered provider;
    otherwise OpenAI, then Anthropic, then OCI — see :func:`_build_model`.
    """
    m = get_test_model()
    if m is None:
        pytest.skip(
            "No model available (set LOCUS_TEST_MODEL=provider:model, "
            "OPENAI_API_KEY, ANTHROPIC_API_KEY or OCI_PROFILE)"
        )
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
    config.addinivalue_line("markers", "requires_oci: test requires OCI config")
    config.addinivalue_line("markers", "requires_openai: test requires OpenAI API key")
    config.addinivalue_line(
        "markers", "requires_model: test requires any model (OpenAI, Anthropic, or OCI)"
    )


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests based on marker requirements."""
    marker_map = {
        "requires_redis": skip_without_redis,
        "requires_postgres": skip_without_postgres,
        "requires_mysql": skip_without_mysql,
        "requires_opensearch": skip_without_opensearch,
        "requires_oci": skip_without_oci,
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
        ("OCI GenAI", oci_config_available()),
        ("OpenAI", openai_available()),
        ("Any Model", any_model_available()),
    ]
    for name, available in services:
        status = "✓" if available else "✗"
        lines.append(f"  {status} {name}")
    return lines
