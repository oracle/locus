# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Shared fixtures and configuration for RAG integration tests.

All configuration is via environment variables - nothing is hardcoded.

Required for OCI tests:
- OCI_PROFILE: OCI config profile name (REQUIRED)
- OCI_AUTH_TYPE: Auth type - api_key, security_token, etc. (default: api_key)
- OCI_COMPARTMENT_ID: Compartment OCID (optional, uses tenancy if not set)

Required for OpenSearch tests:
- OPENSEARCH_HOSTS: Comma-separated OpenSearch hosts (REQUIRED)
- OPENSEARCH_USER: OpenSearch username (REQUIRED)
- OPENSEARCH_PASSWORD: OpenSearch password (REQUIRED)
- OPENSEARCH_USE_SSL: Use SSL (default: true)
- OPENSEARCH_VERIFY_CERTS: Verify certs (default: false)
"""

import os

import pytest


def get_oci_config():
    """Get OCI configuration from environment.

    Required environment variables:
    - OCI_PROFILE: OCI config profile name (no default - must be set)
    - OCI_AUTH_TYPE: Auth type (api_key, security_token, etc.)
    - OCI_COMPARTMENT_ID: Compartment OCID (optional; also accepts OCI_COMPARTMENT)
    - OCI_ENDPOINT: Full GenAI service endpoint (optional, overrides profile region)
    """
    profile = os.environ.get("OCI_PROFILE")
    if not profile:
        raise ValueError(
            "OCI_PROFILE environment variable must be set. Example: export OCI_PROFILE=MY_PROFILE"
        )
    # Accept both OCI_COMPARTMENT_ID (historic) and OCI_COMPARTMENT (what the
    # rest of the suite uses) so a single export set drives every test file.
    compartment = os.environ.get("OCI_COMPARTMENT_ID") or os.environ.get("OCI_COMPARTMENT") or ""
    return {
        "profile_name": profile,
        "auth_type": os.environ.get("OCI_AUTH_TYPE", "api_key"),
        "compartment_id": compartment,
        "service_endpoint": os.environ.get("OCI_ENDPOINT"),
    }


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


def get_oracle_adb_config():
    """Get Oracle ADB configuration from environment.

    Required:
    - ADB_DSN: TNS alias (e.g. ``deepresearch_low``).
    - ADB_PASSWORD: ADMIN (or configured user) password.
    - ADB_WALLET_LOCATION: directory holding ``tnsnames.ora`` + ``cwallet.sso``.

    Optional:
    - ADB_USER: defaults to ``ADMIN``.
    - ADB_WALLET_PASSWORD: defaults to ``ADB_PASSWORD``.
    """
    dsn = os.environ.get("ADB_DSN")
    password = os.environ.get("ADB_PASSWORD")
    wallet = os.environ.get("ADB_WALLET_LOCATION")
    if not (dsn and password and wallet):
        raise ValueError(
            "Set ADB_DSN, ADB_PASSWORD, and ADB_WALLET_LOCATION to run "
            "Oracle ADB integration tests."
        )
    return {
        "dsn": dsn,
        "user": os.environ.get("ADB_USER", "ADMIN"),
        "password": password,
        "wallet_location": os.path.expanduser(wallet),
        "wallet_password": os.environ.get("ADB_WALLET_PASSWORD", password),
    }


@pytest.fixture
def oci_config():
    """OCI configuration fixture. Skips test if OCI_PROFILE not set."""
    try:
        return get_oci_config()
    except ValueError as e:
        pytest.skip(str(e))


@pytest.fixture
def opensearch_config():
    """OpenSearch configuration fixture. Skips test if env vars not set."""
    try:
        return get_opensearch_config()
    except ValueError as e:
        pytest.skip(str(e))


@pytest.fixture
def oracle_adb_config():
    """Oracle ADB configuration. Skips test if env vars not set."""
    try:
        return get_oracle_adb_config()
    except ValueError as e:
        pytest.skip(str(e))
