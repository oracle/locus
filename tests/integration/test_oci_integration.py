# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Integration tests for OCI GenAI.

These tests require actual OCI credentials and will be skipped if not available.

To run these tests:
1. Ensure you have ~/.oci/config with valid profiles
2. Set environment variables if needed:
   - OCI_PROFILE: Profile name (default: DEFAULT)
   - OCI_COMPARTMENT: Compartment OCID (default: from config tenancy)
   - OCI_ENDPOINT: Service endpoint (default: us-chicago-1)
"""

import os
from pathlib import Path

import pytest


def has_oci_credentials() -> bool:
    """Check if OCI credentials are available."""
    config_path = Path("~/.oci/config").expanduser()
    return config_path.exists()


def get_test_profile() -> str:
    """Get the OCI profile to use for testing."""
    profile = os.environ.get("OCI_PROFILE")
    if not profile:
        pytest.skip("OCI_PROFILE environment variable not set")
    return profile


def get_test_endpoint() -> str:
    """Get the service endpoint for testing."""
    endpoint = os.environ.get("OCI_ENDPOINT")
    if not endpoint:
        pytest.skip("OCI_ENDPOINT environment variable not set")
    return endpoint


def get_test_compartment() -> str | None:
    """Get the compartment ID for testing."""
    return os.environ.get("OCI_COMPARTMENT")


def get_test_auth_type():
    """Resolve the auth type from ``OCI_AUTH_TYPE`` env (default api_key).

    Lets the same test file work against api_key, security_token,
    instance_principal, or resource_principal profiles. Tests that
    *specifically* exercise a single auth type (e.g. SECURITY_TOKEN)
    should still hardcode that.
    """
    from locus.models.providers.oci.client import OCIAuthType

    return OCIAuthType(os.environ.get("OCI_AUTH_TYPE", "api_key"))


# Skip all tests if no OCI credentials
pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_oci,
]


class TestOCIClientIntegration:
    """Integration tests for OCIClient."""

    def test_api_key_config_loading(self):
        """Test loading OCI config with whichever auth the active profile uses."""
        from locus.models.providers.oci.client import OCIClient, OCIClientConfig

        config = OCIClientConfig(
            profile_name=get_test_profile(),
            auth_type=get_test_auth_type(),
        )
        client = OCIClient(config)

        # Should load config without error
        oci_cfg = client.oci_config
        assert "tenancy" in oci_cfg
        assert "user" in oci_cfg or "security_token_file" in oci_cfg

    def test_compartment_id_resolution(self):
        """Test compartment ID is resolved correctly."""
        from locus.models.providers.oci.client import OCIClient, OCIClientConfig

        config = OCIClientConfig(
            profile_name=get_test_profile(),
            auth_type=get_test_auth_type(),
        )
        client = OCIClient(config)

        # Should resolve to tenancy
        compartment = client.compartment_id
        assert compartment.startswith("ocid1.")

    def test_explicit_compartment_id(self):
        """Test explicit compartment ID is used."""
        from locus.models.providers.oci.client import OCIClient, OCIClientConfig

        explicit_compartment = get_test_compartment() or "ocid1.compartment.oc1..explicit"

        config = OCIClientConfig(
            profile_name=get_test_profile(),
            auth_type=get_test_auth_type(),
            compartment_id=explicit_compartment,
        )
        client = OCIClient(config)

        assert client.compartment_id == explicit_compartment

    def test_client_creation(self):
        """Test creating OCI client with whichever auth the active profile uses.

        Uses ``OCI_AUTH_TYPE`` from the environment so the test works with
        api-key, security-token, instance-principal, or resource-principal
        profiles. The previous version hardcoded ``API_KEY`` and failed with
        ``InvalidConfig: 'user': 'missing'`` against session-token profiles.
        """
        from locus.models.providers.oci.client import OCIClient, OCIClientConfig

        config = OCIClientConfig(
            profile_name=get_test_profile(),
            auth_type=get_test_auth_type(),
            service_endpoint=get_test_endpoint(),
        )
        client = OCIClient(config)

        # Should create client without error
        oci_client = client.client
        assert oci_client is not None

    def test_serving_mode_on_demand(self):
        """Test serving mode for on-demand models."""
        from oci.generative_ai_inference import models

        from locus.models.providers.oci.client import OCIClient, OCIClientConfig

        model_id = os.environ.get("OCI_MODEL_ID")
        if not model_id:
            pytest.skip("OCI_MODEL_ID environment variable not set")

        config = OCIClientConfig(profile_name=get_test_profile())
        client = OCIClient(config)

        mode = client.get_serving_mode(model_id)
        assert isinstance(mode, models.OnDemandServingMode)
        assert mode.model_id == model_id

    def test_serving_mode_dedicated(self):
        """Test serving mode for dedicated endpoints."""
        from oci.generative_ai_inference import models

        from locus.models.providers.oci.client import OCIClient, OCIClientConfig

        config = OCIClientConfig(profile_name=get_test_profile())
        client = OCIClient(config)

        endpoint_ocid = "ocid1.generativeaiendpoint.oc1.us-chicago-1.test"
        mode = client.get_serving_mode(endpoint_ocid)
        assert isinstance(mode, models.DedicatedServingMode)
        assert mode.endpoint_id == endpoint_ocid


class TestOCIModelIntegration:
    """Integration tests for OCIModel."""

    @pytest.mark.asyncio
    async def test_model_initialization(self):
        """Test OCIModel initializes correctly."""
        from locus.models.providers.oci import OCIModel

        model_id = os.environ.get("OCI_MODEL_ID")
        if not model_id:
            pytest.skip("OCI_MODEL_ID environment variable not set")

        auth = get_test_auth_type()
        model = OCIModel(
            model_id=model_id,
            profile_name=get_test_profile(),
            auth_type=auth,
            service_endpoint=get_test_endpoint(),
            compartment_id=get_test_compartment(),
        )

        assert model.config.model_id == model_id
        assert model.config.auth_type == auth

    @pytest.mark.asyncio
    async def test_model_complete_simple(self):
        """Test simple completion with OCIModel."""
        from locus.core.messages import Message
        from locus.models.providers.oci import OCIModel

        model_id = os.environ.get("OCI_MODEL_ID")
        if not model_id:
            pytest.skip("OCI_MODEL_ID environment variable not set")

        compartment = get_test_compartment()
        if not compartment:
            pytest.skip("OCI_COMPARTMENT environment variable not set")

        model = OCIModel(
            model_id=model_id,
            profile_name=get_test_profile(),
            auth_type=get_test_auth_type(),
            service_endpoint=get_test_endpoint(),
            compartment_id=compartment,
            max_tokens=50,
        )

        messages = [Message.user("What is 2+2? Answer with just the number.")]

        response = await model.complete(messages)

        assert response is not None
        assert response.message is not None
        # Response might be in content or reasoning_content
        assert (
            response.content
            or response.message.content is not None
            or hasattr(response.message, "reasoning_content")
        )

    @pytest.mark.asyncio
    async def test_model_stream(self):
        """Test streaming with OCIModel."""
        from locus.core.messages import Message
        from locus.models.providers.oci import OCIModel

        model_id = os.environ.get("OCI_MODEL_ID")
        if not model_id:
            pytest.skip("OCI_MODEL_ID environment variable not set")

        compartment = get_test_compartment()
        if not compartment:
            pytest.skip("OCI_COMPARTMENT environment variable not set")

        model = OCIModel(
            model_id=model_id,
            profile_name=get_test_profile(),
            auth_type=get_test_auth_type(),
            service_endpoint=get_test_endpoint(),
            compartment_id=compartment,
            max_tokens=50,
        )

        messages = [Message.user("Say hello.")]

        chunks = []
        async for chunk in model.stream(messages):
            chunks.append(chunk)

        # Should have at least one chunk and a done marker
        assert len(chunks) >= 1
        assert chunks[-1].done is True


class TestOCISecurityTokenAuth:
    """Integration tests for security token (session) authentication.

    These tests require a valid session created with:
        oci session authenticate --profile-name <PROFILE>
    """

    @pytest.fixture
    def session_profile(self) -> str | None:
        """Pick a profile from ~/.oci/config that has session-token auth.

        Prefers ``OCI_SESSION_PROFILE`` env var. Falls back to scanning
        the config file for any [PROFILE] block with ``security_token_file``
        set. Returns None when nothing matches (test then skips cleanly).
        """
        explicit = os.environ.get("OCI_SESSION_PROFILE")
        if explicit:
            return explicit

        config_path = Path("~/.oci/config").expanduser()
        if not config_path.exists():
            return None

        current_profile: str | None = None
        for line in config_path.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                current_profile = stripped[1:-1]
            elif stripped.startswith("security_token_file") and current_profile:
                return current_profile
        return None

    def test_session_token_config_loading(self, session_profile):
        """Test loading config with session token auth."""
        if not session_profile:
            pytest.skip("No session token profile available")

        from locus.models.providers.oci.client import OCIAuthType, OCIClient, OCIClientConfig

        config = OCIClientConfig(
            profile_name=session_profile,
            auth_type=OCIAuthType.SECURITY_TOKEN,
        )
        client = OCIClient(config)

        oci_cfg = client.oci_config
        assert "security_token_file" in oci_cfg
        assert "key_file" in oci_cfg

    def test_session_token_client_creation(self, session_profile):
        """Test creating client with session token auth."""
        if not session_profile:
            pytest.skip("No session token profile available")

        from locus.models.providers.oci.client import OCIAuthType, OCIClient, OCIClientConfig

        config = OCIClientConfig(
            profile_name=session_profile,
            auth_type=OCIAuthType.SECURITY_TOKEN,
            service_endpoint=get_test_endpoint(),
        )

        try:
            client = OCIClient(config)
            oci_client = client.client
            assert oci_client is not None
        except ValueError as e:
            if "expired" in str(e).lower() or "refresh" in str(e).lower():
                pytest.skip("Session token expired - run 'oci session authenticate'")
            raise
