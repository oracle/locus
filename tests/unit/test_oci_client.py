# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for OCI GenAI client.

Tests the OCIClient, OCIClientConfig, and OCIAuthType classes
without making actual API calls.
"""

from unittest.mock import MagicMock, mock_open, patch

import pytest

from locus.models.providers.oci.client import (
    OCIAuthType,
    OCIClient,
    OCIClientConfig,
)


class TestOCIAuthType:
    """Tests for OCIAuthType enum."""

    def test_api_key_value(self):
        """Test API_KEY auth type value."""
        assert OCIAuthType.API_KEY == "api_key"
        assert OCIAuthType.API_KEY.value == "api_key"

    def test_security_token_value(self):
        """Test SECURITY_TOKEN auth type value."""
        assert OCIAuthType.SECURITY_TOKEN == "security_token"  # noqa: S105 — enum value, not a secret

    def test_session_token_alias(self):
        """Test SESSION_TOKEN is an alias for SECURITY_TOKEN."""
        assert OCIAuthType.SESSION_TOKEN == "session_token"  # noqa: S105 — enum value, not a secret

    def test_instance_principal_value(self):
        """Test INSTANCE_PRINCIPAL auth type value."""
        assert OCIAuthType.INSTANCE_PRINCIPAL == "instance_principal"

    def test_resource_principal_value(self):
        """Test RESOURCE_PRINCIPAL auth type value."""
        assert OCIAuthType.RESOURCE_PRINCIPAL == "resource_principal"

    def test_from_string(self):
        """Test creating auth type from string."""
        assert OCIAuthType("api_key") == OCIAuthType.API_KEY
        assert OCIAuthType("security_token") == OCIAuthType.SECURITY_TOKEN
        assert OCIAuthType("instance_principal") == OCIAuthType.INSTANCE_PRINCIPAL


class TestOCIClientConfig:
    """Tests for OCIClientConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = OCIClientConfig()
        assert config.profile_name == "DEFAULT"
        assert config.config_file == "~/.oci/config"
        assert config.auth_type == OCIAuthType.API_KEY
        assert config.compartment_id is None
        assert config.service_endpoint is None

    def test_custom_values(self):
        """Test custom configuration values."""
        config = OCIClientConfig(
            profile_name="MY_PROFILE",
            config_file="/custom/path/config",
            auth_type=OCIAuthType.SECURITY_TOKEN,
            compartment_id="ocid1.compartment.oc1..test",
            service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        )
        assert config.profile_name == "MY_PROFILE"
        assert config.config_file == "/custom/path/config"
        assert config.auth_type == OCIAuthType.SECURITY_TOKEN
        assert config.compartment_id == "ocid1.compartment.oc1..test"
        assert (
            config.service_endpoint
            == "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
        )

    def test_auth_type_from_string(self):
        """Test auth_type accepts string values."""
        config = OCIClientConfig(auth_type="security_token")
        assert config.auth_type == OCIAuthType.SECURITY_TOKEN


class TestOCIClient:
    """Tests for OCIClient."""

    def test_client_initialization(self):
        """Test client initializes with config."""
        config = OCIClientConfig(profile_name="TEST")
        client = OCIClient(config)

        assert client.config == config
        assert client._client is None
        assert client._oci_config is None
        assert client._compartment_id is None

    def test_compartment_id_from_config(self):
        """Test compartment_id is taken from config when specified."""
        config = OCIClientConfig(compartment_id="ocid1.compartment.oc1..specified")
        client = OCIClient(config)

        assert client.compartment_id == "ocid1.compartment.oc1..specified"

    @patch("locus.models.providers.oci.client.Path")
    def test_compartment_id_from_oci_config(self, mock_path):
        """Test compartment_id falls back to tenancy from OCI config."""
        mock_path.return_value.expanduser.return_value = "/home/user/.oci/config"

        config = OCIClientConfig(
            profile_name="TEST",
            auth_type=OCIAuthType.API_KEY,
        )
        client = OCIClient(config)

        # Manually set the oci_config cache
        client._oci_config = {"tenancy": "ocid1.tenancy.oc1..fromconfig"}

        assert client.compartment_id == "ocid1.tenancy.oc1..fromconfig"

    def test_compartment_id_missing_raises(self):
        """Test missing compartment_id raises ValueError."""
        config = OCIClientConfig(
            auth_type=OCIAuthType.INSTANCE_PRINCIPAL,  # No config file
        )
        client = OCIClient(config)
        client._oci_config = {}  # Empty config

        with pytest.raises(ValueError, match="compartment_id required"):
            _ = client.compartment_id

    def test_get_serving_mode_on_demand(self):
        """Test get_serving_mode returns OnDemandServingMode for model IDs."""
        config = OCIClientConfig()
        client = OCIClient(config)

        with patch("locus.models.providers.oci.client.OCIClient.client"):
            from oci.generative_ai_inference import models

            mode = client.get_serving_mode("openai.gpt-oss-20b")
            assert isinstance(mode, models.OnDemandServingMode)
            assert mode.model_id == "openai.gpt-oss-20b"

    def test_get_serving_mode_dedicated(self):
        """Test get_serving_mode returns DedicatedServingMode for OCIDs."""
        config = OCIClientConfig()
        client = OCIClient(config)

        with patch("locus.models.providers.oci.client.OCIClient.client"):
            from oci.generative_ai_inference import models

            mode = client.get_serving_mode("ocid1.generativeaiendpoint.oc1..test")
            assert isinstance(mode, models.DedicatedServingMode)
            assert mode.endpoint_id == "ocid1.generativeaiendpoint.oc1..test"


class TestOCIClientAPIKeyAuth:
    """Tests for API Key authentication."""

    @patch("oci.generative_ai_inference.GenerativeAiInferenceClient")
    @patch("oci.config.from_file")
    def test_api_key_client_creation(self, mock_from_file, mock_client_class):
        """Test client creation with API key auth."""
        mock_from_file.return_value = {
            "tenancy": "ocid1.tenancy.oc1..test",
            "user": "ocid1.user.oc1..test",
            "fingerprint": "aa:bb:cc",
            "key_file": "/path/to/key.pem",
            "region": "us-chicago-1",
        }
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        config = OCIClientConfig(
            profile_name="TEST_PROFILE",
            auth_type=OCIAuthType.API_KEY,
            service_endpoint="https://test.endpoint.com",
        )
        client = OCIClient(config)

        # Access client to trigger creation
        result = client.client

        mock_from_file.assert_called_once()
        mock_client_class.assert_called_once()
        call_kwargs = mock_client_class.call_args[1]
        assert call_kwargs["service_endpoint"] == "https://test.endpoint.com"
        assert result == mock_client


class TestOCIClientSecurityTokenAuth:
    """Tests for Security Token (session) authentication."""

    @patch("oci.generative_ai_inference.GenerativeAiInferenceClient")
    @patch("oci.auth.signers.SecurityTokenSigner")
    @patch("oci.signer.load_private_key_from_file")
    @patch("oci.config.from_file")
    @patch("builtins.open", new_callable=mock_open, read_data="mock_token_content")
    @patch("locus.models.providers.oci.client.Path")
    def test_security_token_client_creation(
        self,
        mock_path,
        mock_file,
        mock_from_file,
        mock_load_key,
        mock_signer_class,
        mock_client_class,
    ):
        """Test client creation with security token auth."""
        # Setup mocks
        mock_path_instance = MagicMock()
        mock_path_instance.expanduser.return_value = mock_path_instance
        mock_path_instance.exists.return_value = True
        mock_path_instance.open = mock_file
        mock_path.return_value = mock_path_instance

        mock_from_file.return_value = {
            "tenancy": "ocid1.tenancy.oc1..test",
            "security_token_file": "/path/to/token",
            "key_file": "/path/to/key.pem",
            "region": "us-chicago-1",
        }
        mock_private_key = MagicMock()
        mock_load_key.return_value = mock_private_key
        mock_signer = MagicMock()
        mock_signer_class.return_value = mock_signer
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        config = OCIClientConfig(
            profile_name="SESSION_PROFILE",
            auth_type=OCIAuthType.SECURITY_TOKEN,
            service_endpoint="https://test.endpoint.com",
        )
        client = OCIClient(config)

        # Access client to trigger creation
        result = client.client

        mock_load_key.assert_called_once_with("/path/to/key.pem")
        mock_signer_class.assert_called_once_with(
            token="mock_token_content",  # noqa: S106 — test mock, not a real token
            private_key=mock_private_key,
        )
        assert result == mock_client

    @patch("oci.config.from_file")
    def test_security_token_missing_token_file_raises(self, mock_from_file):
        """Test missing security_token_file raises ValueError."""
        mock_from_file.return_value = {
            "tenancy": "ocid1.tenancy.oc1..test",
            "key_file": "/path/to/key.pem",
            # No security_token_file
        }

        config = OCIClientConfig(
            profile_name="BAD_PROFILE",
            auth_type=OCIAuthType.SECURITY_TOKEN,
        )
        client = OCIClient(config)

        with pytest.raises(ValueError, match="security_token_file not found"):
            _ = client.client


class TestOCIClientInstancePrincipalAuth:
    """Tests for Instance Principal authentication."""

    @patch("oci.generative_ai_inference.GenerativeAiInferenceClient")
    @patch("oci.auth.signers.InstancePrincipalsSecurityTokenSigner")
    def test_instance_principal_client_creation(self, mock_signer_class, mock_client_class):
        """Test client creation with instance principal auth."""
        mock_signer = MagicMock()
        mock_signer_class.return_value = mock_signer
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        config = OCIClientConfig(
            auth_type=OCIAuthType.INSTANCE_PRINCIPAL,
            service_endpoint="https://test.endpoint.com",
        )
        client = OCIClient(config)

        # Access client to trigger creation
        result = client.client

        mock_signer_class.assert_called_once()
        mock_client_class.assert_called_once_with(
            config={},
            signer=mock_signer,
            service_endpoint="https://test.endpoint.com",
            timeout=(10.0, 300.0),
        )
        assert result == mock_client

    def test_instance_principal_no_oci_config_needed(self):
        """Test instance principal doesn't require OCI config file."""
        config = OCIClientConfig(
            auth_type=OCIAuthType.INSTANCE_PRINCIPAL,
        )
        client = OCIClient(config)

        # Should return empty dict without reading config file
        assert client.oci_config == {}
