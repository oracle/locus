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
def oci_config_available() -> bool:
    """Check if OCI config exists and required env vars are set."""
    config_path = Path.home() / ".oci" / "config"
    if not config_path.exists():
        return False

    # Profile is required; endpoint is optional (can be derived from region)
    profile = os.getenv("OCI_PROFILE")

    return bool(profile)


@lru_cache(maxsize=1)
def oci_bucket_available() -> bool:
    """Check if OCI bucket credentials are available."""
    if not oci_config_available():
        return False

    bucket = os.getenv("OCI_BUCKET_NAME")
    namespace = os.getenv("OCI_NAMESPACE")

    return bool(bucket and namespace)


@lru_cache(maxsize=1)
def openai_available() -> bool:
    """Check if OpenAI API key is set."""
    return bool(os.getenv("OPENAI_API_KEY"))


@lru_cache(maxsize=1)
def anthropic_available() -> bool:
    """Check if Anthropic API key is set."""
    return bool(os.getenv("ANTHROPIC_API_KEY"))


@lru_cache(maxsize=1)
def oracle_available() -> bool:
    """Check if Oracle credentials are available."""
    # Support both direct credentials and ADB wallet-based auth
    if os.getenv("ORACLE_PASSWORD") and os.getenv("ORACLE_DSN"):
        return True
    wallet_path = os.getenv("ORACLE_WALLET")
    if wallet_path and Path(wallet_path).exists():
        return True
    return False


@lru_cache(maxsize=1)
def any_model_available() -> bool:
    """Check if any model (Anthropic, OpenAI, or OCI) is available."""
    return anthropic_available() or openai_available() or oci_config_available()


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

skip_without_opensearch = pytest.mark.skipif(
    not opensearch_available(),
    reason="OpenSearch not available (set OPENSEARCH_HOSTS and credentials)",
)

skip_without_oci = pytest.mark.skipif(
    not oci_config_available(),
    reason="OCI not configured (need ~/.oci/config + OCI_PROFILE)",
)

skip_without_oci_bucket = pytest.mark.skipif(
    not oci_bucket_available(),
    reason="OCI bucket not configured (need OCI_BUCKET_NAME + OCI_NAMESPACE)",
)

skip_without_openai = pytest.mark.skipif(
    not openai_available(), reason="OpenAI API key not set (need OPENAI_API_KEY)"
)

skip_without_anthropic = pytest.mark.skipif(
    not anthropic_available(), reason="Anthropic API key not set (need ANTHROPIC_API_KEY)"
)

skip_without_oracle = pytest.mark.skipif(
    not oracle_available(),
    reason="Oracle not configured (need ORACLE_DSN + ORACLE_PASSWORD or ORACLE_WALLET)",
)

skip_without_model = pytest.mark.skipif(
    not any_model_available(), reason="No model available (need OpenAI API key or OCI config)"
)


# =============================================================================
# Fixtures
# =============================================================================


def _build_model():
    """Build a model instance from environment variables.

    Resolution order (first match wins):

    1. OCI GenAI — when ``OCI_PROFILE`` is set and the SDK can
       construct ``OCIModel`` (needs ``OCI_ENDPOINT`` + ``OCI_COMPARTMENT``).
    2. OpenAI — when ``OPENAI_API_KEY`` is set.
    3. Anthropic — when ``ANTHROPIC_API_KEY`` is set.

    Model id is controlled per-provider by ``OCI_MODEL_ID`` /
    ``OPENAI_MODEL_ID`` / ``ANTHROPIC_MODEL_ID``.
    """
    # OCI GenAI (preferred)
    if oci_config_available():
        endpoint = os.getenv("OCI_ENDPOINT")
        compartment = os.getenv("OCI_COMPARTMENT")
        model_id = os.getenv("OCI_MODEL_ID", "openai.gpt-5.4")
        if endpoint and compartment:
            from locus.models.providers.oci import OCIModel

            # max_tokens=8192 is a ceiling, not a target. Reasoning
            # models (gpt-5.5, o-series, etc.) burn 200-2000+ tokens
            # of hidden chain-of-thought before producing any visible
            # output; at 512 they typically return empty content with
            # finish_reason='length', which surfaces in orchestrator /
            # swarm tests as ``summary=''``. 8192 leaves room for both
            # the reasoning trace and a normal-length response without
            # being wasteful on short-answer tests (the model stops
            # naturally when done).
            return OCIModel(
                model_id=model_id,
                profile_name=os.getenv("OCI_PROFILE", "DEFAULT"),
                auth_type=os.getenv("OCI_AUTH_TYPE", "api_key"),
                service_endpoint=endpoint,
                compartment_id=compartment,
                max_tokens=8192,
            )

    # OpenAI fallback
    if openai_available():
        from locus.models.native.openai import OpenAIModel

        model_id = os.getenv("OPENAI_MODEL_ID", "gpt-4o-mini")
        return OpenAIModel(model=model_id, max_tokens=8192)

    # Anthropic fallback — cheapest path for non-OCI iteration.
    if anthropic_available():
        from locus.models.native.anthropic import AnthropicModel

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

    Uses OCI GenAI if configured (OCI_PROFILE + OCI_ENDPOINT),
    falls back to OpenAI. Model ID from OCI_MODEL_ID env var.
    """
    m = get_test_model()
    if m is None:
        pytest.skip("No model available (need OCI_PROFILE+OCI_ENDPOINT or OPENAI_API_KEY)")
    return m


@pytest.fixture(scope="session")
def service_status():
    """Report available services at the start of the test session."""
    return {
        "redis": redis_available(),
        "postgres": postgres_available(),
        "opensearch": opensearch_available(),
        "oci": oci_config_available(),
        "oci_bucket": oci_bucket_available(),
        "openai": openai_available(),
        "oracle": oracle_available(),
    }


@pytest.fixture(scope="session")
def oci_bucket_config() -> dict:
    """OCI Object Storage test settings, sourced entirely from the env.

    Environment variables (all required except where noted):

    - ``OCI_BUCKET_NAME`` — target bucket (must already exist)
    - ``OCI_NAMESPACE`` — Object Storage namespace for the tenancy
    - ``OCI_BUCKET_PROFILE`` — optional override; profile to use for bucket
      access. Falls back to ``OCI_PROFILE``. Useful when GenAI tests use
      one tenancy (e.g. MY_PROFILE) and the test bucket lives in another
      (e.g. API_FREE_TIER's free-tier bucket).
    - ``OCI_BUCKET_AUTH_TYPE`` — optional; falls back to
      ``OCI_AUTH_TYPE``, then ``api_key``.
    - ``OCI_BUCKET_REGION`` — optional; falls back to ``OCI_REGION``.
    - ``OCI_BUCKET_TEST_PREFIX`` — optional; prefix under the bucket; tests
      should scope their own sub-prefix under this one

    The ``requires_oci_bucket`` marker already skips tests when the required
    values are missing, so this fixture can assume they are set.
    """
    return {
        "bucket_name": os.environ["OCI_BUCKET_NAME"],
        "namespace": os.environ["OCI_NAMESPACE"],
        "profile_name": (os.getenv("OCI_BUCKET_PROFILE") or os.environ["OCI_PROFILE"]),
        "auth_type": (os.getenv("OCI_BUCKET_AUTH_TYPE") or os.getenv("OCI_AUTH_TYPE", "api_key")),
        "region": os.getenv("OCI_BUCKET_REGION") or os.getenv("OCI_REGION"),
        "prefix": os.getenv("OCI_BUCKET_TEST_PREFIX", "locus/test/"),
    }


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "requires_redis: test requires Redis")
    config.addinivalue_line("markers", "requires_postgres: test requires PostgreSQL")
    config.addinivalue_line("markers", "requires_opensearch: test requires OpenSearch")
    config.addinivalue_line("markers", "requires_oci: test requires OCI config")
    config.addinivalue_line("markers", "requires_oci_bucket: test requires OCI bucket")
    config.addinivalue_line("markers", "requires_openai: test requires OpenAI API key")
    config.addinivalue_line("markers", "requires_oracle: test requires Oracle")
    config.addinivalue_line("markers", "requires_model: test requires any model (OpenAI or OCI)")


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests based on marker requirements."""
    marker_map = {
        "requires_redis": skip_without_redis,
        "requires_postgres": skip_without_postgres,
        "requires_opensearch": skip_without_opensearch,
        "requires_oci": skip_without_oci,
        "requires_oci_bucket": skip_without_oci_bucket,
        "requires_openai": skip_without_openai,
        "requires_oracle": skip_without_oracle,
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
        ("OpenSearch", opensearch_available()),
        ("OCI GenAI", oci_config_available()),
        ("OCI Bucket", oci_bucket_available()),
        ("OpenAI", openai_available()),
        ("Oracle", oracle_available()),
        ("Any Model", any_model_available()),
    ]
    for name, available in services:
        status = "✓" if available else "✗"
        lines.append(f"  {status} {name}")
    return lines
