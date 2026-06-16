# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""OCI GenAI client wrapper.

Supports three authentication types:
- API_KEY: Uses OCI config file with API key credentials
- SECURITY_TOKEN: Uses session token from `oci session authenticate`
- INSTANCE_PRINCIPAL: Uses instance metadata (for OCI compute instances)

Example:
    ```python
    # API Key auth
    config = OCIClientConfig(
        profile_name="MY_PROFILE",
        auth_type=OCIAuthType.API_KEY,
        service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    )
    client = OCIClient(config)

    # Session token auth
    config = OCIClientConfig(
        profile_name="MY_SESSION_PROFILE",
        auth_type=OCIAuthType.SECURITY_TOKEN,
        service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    )
    client = OCIClient(config)
    ```
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field


if TYPE_CHECKING:
    from oci.generative_ai_inference import GenerativeAiInferenceClient


class OCIAuthType(StrEnum):
    """OCI authentication types."""

    API_KEY = "api_key"
    SECURITY_TOKEN = "security_token"  # noqa: S105 — OCI auth-type enum value, not a secret
    SESSION_TOKEN = "session_token"  # noqa: S105 — alias for SECURITY_TOKEN, not a secret
    INSTANCE_PRINCIPAL = "instance_principal"
    RESOURCE_PRINCIPAL = "resource_principal"


class OCIClientConfig(BaseModel):
    """Configuration for OCI GenAI client.

    Attributes:
        profile_name: OCI config profile name from ~/.oci/config
        config_file: Path to OCI config file
        auth_type: Authentication type (api_key, security_token, instance_principal)
        compartment_id: OCI compartment OCID (defaults to tenancy from config)
        service_endpoint: Full service endpoint URL (required for cross-region access)
    """

    profile_name: str = Field(default="DEFAULT", description="OCI config profile name")
    config_file: str = Field(default="~/.oci/config", description="Path to OCI config file")
    auth_type: OCIAuthType = Field(default=OCIAuthType.API_KEY, description="Auth type")
    compartment_id: str | None = Field(default=None, description="OCI compartment OCID")
    service_endpoint: str | None = Field(default=None, description="Full service endpoint URL")
    # HTTP timeouts in seconds for the underlying OCI Python SDK
    # ``GenerativeAiInferenceClient``. The SDK defaults to ``(10, 60)``
    # (connect, read); 60s read is not enough for reasoning models
    # (gpt-5.5, o-series, etc.) doing long-form summarization in
    # orchestrator/swarm flows, where the first response token can take
    # 90-180 seconds to arrive. Bump the read timeout to 300s by
    # default; callers needing tighter latency contracts can override.
    connect_timeout: float = Field(
        default=10.0,
        description="HTTP connect timeout in seconds.",
    )
    read_timeout: float = Field(
        default=300.0,
        description=(
            "HTTP read timeout in seconds. Default 300s accommodates "
            "reasoning-model summarization (gpt-5.5, o-series) which "
            "can sit on the wire for 90-180s before the first token."
        ),
    )

    model_config = {"extra": "allow"}


class OCIClient:
    """Wrapper for OCI GenerativeAiInferenceClient with auth handling.

    This client handles:
    - Multiple authentication types (API key, session token, instance principal)
    - Service endpoint configuration for cross-region access
    - Lazy client initialization
    """

    def __init__(self, config: OCIClientConfig) -> None:
        self.config = config
        self._client: GenerativeAiInferenceClient | None = None
        self._oci_config: dict[str, Any] | None = None
        self._compartment_id: str | None = None

    @property
    def oci_config(self) -> dict[str, Any]:
        """Load and cache OCI config from file."""
        if self._oci_config is None:
            # Instance/resource principal don't need config file
            if self.config.auth_type in (
                OCIAuthType.INSTANCE_PRINCIPAL,
                OCIAuthType.RESOURCE_PRINCIPAL,
            ):
                self._oci_config = {}
            else:
                from oci import config as oci_config_module

                config_path = Path(self.config.config_file).expanduser()
                self._oci_config = oci_config_module.from_file(
                    file_location=str(config_path),
                    profile_name=self.config.profile_name,
                )
        return self._oci_config

    @property
    def client(self) -> GenerativeAiInferenceClient:
        """Get or create the OCI GenAI client (lazy initialization)."""
        if self._client is None:
            self._client = self._create_client()
        return self._client

    @property
    def compartment_id(self) -> str:
        """Get compartment ID from config or OCI config file."""
        if self._compartment_id:
            return self._compartment_id

        # User-specified compartment takes priority
        if self.config.compartment_id:
            self._compartment_id = self.config.compartment_id
            return self._compartment_id

        # Fall back to tenancy from config
        tenancy = self.oci_config.get("tenancy")
        if tenancy:
            self._compartment_id = str(tenancy)
            return self._compartment_id

        raise ValueError(
            "compartment_id required - specify in config or ensure tenancy is set in OCI config"
        )

    def _create_client(self) -> GenerativeAiInferenceClient:
        """Create the OCI GenAI client based on auth type."""
        from oci.generative_ai_inference import GenerativeAiInferenceClient

        auth_type = self.config.auth_type

        # Normalize session_token alias
        if auth_type == OCIAuthType.SESSION_TOKEN:
            auth_type = OCIAuthType.SECURITY_TOKEN

        # Instance principal - for OCI compute instances
        if auth_type == OCIAuthType.INSTANCE_PRINCIPAL:
            return self._create_instance_principal_client()

        # Resource principal - for OCI functions
        if auth_type == OCIAuthType.RESOURCE_PRINCIPAL:
            return self._create_resource_principal_client()

        # Security token - for session-based auth
        if auth_type == OCIAuthType.SECURITY_TOKEN:
            return self._create_security_token_client()

        # Default: API key auth
        return GenerativeAiInferenceClient(
            config=self.oci_config,
            service_endpoint=self.config.service_endpoint,
            timeout=(self.config.connect_timeout, self.config.read_timeout),
        )

    def _create_security_token_client(self) -> GenerativeAiInferenceClient:
        """Create client with security token (session) authentication."""
        from oci.auth import signers
        from oci.generative_ai_inference import GenerativeAiInferenceClient
        from oci.signer import load_private_key_from_file

        oci_cfg = self.oci_config

        # Get token file path
        token_file = oci_cfg.get("security_token_file")
        if not token_file:
            raise ValueError(
                f"security_token_file not found in profile '{self.config.profile_name}'. "
                "Run 'oci session authenticate' to create a session."
            )

        # Load token
        token_path = Path(token_file).expanduser()
        if not token_path.exists():
            raise ValueError(
                f"Security token file not found: {token_path}. "
                "Run 'oci session authenticate' to refresh your session."
            )

        with token_path.open() as f:
            token = f.read().strip()

        # Get and load private key
        key_file = oci_cfg.get("key_file")
        if not key_file:
            raise ValueError(f"key_file not found in profile '{self.config.profile_name}'")

        private_key = load_private_key_from_file(key_file)

        # Create signer
        signer = signers.SecurityTokenSigner(
            token=token,
            private_key=private_key,
        )

        return GenerativeAiInferenceClient(
            config=oci_cfg,
            signer=signer,
            service_endpoint=self.config.service_endpoint,
            timeout=(self.config.connect_timeout, self.config.read_timeout),
        )

    def _create_instance_principal_client(self) -> GenerativeAiInferenceClient:
        """Create client with instance principal authentication.

        Used when running on OCI compute instances with proper IAM policies.
        """
        from oci.auth import signers
        from oci.generative_ai_inference import GenerativeAiInferenceClient

        signer = signers.InstancePrincipalsSecurityTokenSigner()

        return GenerativeAiInferenceClient(
            config={},
            signer=signer,
            service_endpoint=self.config.service_endpoint,
            timeout=(self.config.connect_timeout, self.config.read_timeout),
        )

    def _create_resource_principal_client(self) -> GenerativeAiInferenceClient:
        """Create client with resource principal authentication.

        Used when running in OCI Functions or other resource principal contexts.
        """
        from oci.auth import signers
        from oci.generative_ai_inference import GenerativeAiInferenceClient

        signer = signers.get_resource_principals_signer()

        return GenerativeAiInferenceClient(
            config={},
            signer=signer,
            service_endpoint=self.config.service_endpoint,
            timeout=(self.config.connect_timeout, self.config.read_timeout),
        )

    def get_serving_mode(self, model_id: str) -> Any:
        """Get the serving mode based on model_id format.

        Args:
            model_id: Model identifier or dedicated endpoint OCID

        Returns:
            OnDemandServingMode for model IDs, DedicatedServingMode for OCIDs
        """
        from oci.generative_ai_inference import models

        # OCID means dedicated endpoint
        if model_id.startswith("ocid"):
            return models.DedicatedServingMode(endpoint_id=model_id)

        # Otherwise use on-demand
        return models.OnDemandServingMode(model_id=model_id)

    def chat(self, chat_details: Any) -> Any:
        """Execute a chat request.

        Args:
            chat_details: OCI ChatDetails object

        Returns:
            OCI chat response
        """
        return self.client.chat(chat_details)
