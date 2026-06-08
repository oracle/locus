# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL

"""Shared fixtures and configuration for RAG integration tests.

All configuration is via environment variables - nothing is hardcoded.

Required for OpenSearch tests:
- OPENSEARCH_HOSTS: Comma-separated OpenSearch hosts (REQUIRED)
- OPENSEARCH_USER: OpenSearch username (REQUIRED)
- OPENSEARCH_PASSWORD: OpenSearch password (REQUIRED)
- OPENSEARCH_USE_SSL: Use SSL (default: true)
- OPENSEARCH_VERIFY_CERTS: Verify certs (default: false)
"""

import os

import pytest


def get_opensearch_config():
    """Get OpenSearch configuration from environment.

    Required environment variables:
    - OPENSEARCH_HOSTS: Comma-separated host list (e.g., "host1:9200,host2:9200")
    - OPENSEARCH_USER: Username
    - OPENSEARCH_PASSWORD: Password

    Optional:
    - OPENSEARCH_USE_SSL: Use SSL (default: true)
    - OPENSEARCH_VERIFY_CERTS: Verify certs (default: false)
    """
    hosts_str = os.environ.get("OPENSEARCH_HOSTS")
    if not hosts_str:
        raise ValueError(
            "OPENSEARCH_HOSTS environment variable must be set. "
            "Example: export OPENSEARCH_HOSTS=localhost:9200"
        )

    user = os.environ.get("OPENSEARCH_USER")
    password = os.environ.get("OPENSEARCH_PASSWORD")
    if not user or not password:
        raise ValueError(
            "OPENSEARCH_USER and OPENSEARCH_PASSWORD environment variables must be set."
        )

    hosts = [h.strip() for h in hosts_str.split(",")]

    return {
        "hosts": hosts,
        "http_auth": (user, password),
        "use_ssl": os.environ.get("OPENSEARCH_USE_SSL", "true").lower() == "true",
        "verify_certs": os.environ.get("OPENSEARCH_VERIFY_CERTS", "false").lower() == "true",
    }


@pytest.fixture
def opensearch_config():
    """OpenSearch configuration fixture. Skips test if env vars not set."""
    try:
        return get_opensearch_config()
    except ValueError as e:
        pytest.skip(str(e))
