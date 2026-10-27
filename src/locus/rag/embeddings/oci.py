# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""OCI GenAI Embeddings - Cohere models on Oracle Cloud.

Uses OCI GenAI service which hosts Cohere embedding models.
Authentication via OCI SDK (config file, instance principal, etc.).
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from locus.rag.embeddings.base import (
    BaseEmbedding,
    EmbeddingCapabilities,
    EmbeddingConfig,
    EmbeddingResult,
)


if TYPE_CHECKING:
    from oci.generative_ai_inference import GenerativeAiInferenceClient


class OCIEmbeddingModel(str, Enum):
    """Known Cohere embedding models on OCI GenAI (non-exhaustive)."""

    COHERE_EMBED_ENGLISH_V3 = "cohere.embed-english-v3.0"
    COHERE_EMBED_MULTILINGUAL_V3 = "cohere.embed-multilingual-v3.0"
    COHERE_EMBED_ENGLISH_LIGHT_V3 = "cohere.embed-english-light-v3.0"
    COHERE_EMBED_MULTILINGUAL_LIGHT_V3 = "cohere.embed-multilingual-light-v3.0"
    # Current generation, and the default. OCI retired the v3.0 family from
    # on-demand on 2026-09-30 (404 "Entity with key … not found"); the v3
    # ids stay for dedicated AI clusters that still host them.
    COHERE_EMBED_V4 = "cohere.embed-v4.0"
    COHERE_EMBED_MULTILINGUAL_V4 = "cohere.embed-multilingual-v4.0"


# Fast-path dimension hints for known models. Not exhaustive — any model_id
# not listed here is accepted, and the actual dimension is detected from the
# first successful embed response (see `_detected_dimension`).
DEFAULT_DIMENSION = 1024
MODEL_DIMENSION_HINTS = {
    OCIEmbeddingModel.COHERE_EMBED_ENGLISH_V3.value: 1024,
    OCIEmbeddingModel.COHERE_EMBED_MULTILINGUAL_V3.value: 1024,
    OCIEmbeddingModel.COHERE_EMBED_ENGLISH_LIGHT_V3.value: 384,
    OCIEmbeddingModel.COHERE_EMBED_MULTILINGUAL_LIGHT_V3.value: 384,
    # Cohere V4 returns 1536-dim vectors on OCI on-demand. The library
    # previously fell through to DEFAULT_DIMENSION=1024 and crashed any
    # downstream table-create against Oracle 26ai with ORA-51803.
    OCIEmbeddingModel.COHERE_EMBED_V4.value: 1536,
    OCIEmbeddingModel.COHERE_EMBED_MULTILINGUAL_V4.value: 1536,
}


class OCIEmbeddingConfig(BaseModel):
    """Configuration for OCI GenAI Embeddings."""

    model_id: str = Field(
        default=OCIEmbeddingModel.COHERE_EMBED_V4.value,
        description="OCI GenAI embedding model ID",
    )
    compartment_id: str = Field(
        default="",
        description="OCI compartment OCID (uses tenancy root if not specified)",
    )
    service_endpoint: str | None = Field(
        default=None,
        description="OCI GenAI service endpoint (auto-detected from region)",
    )
    profile_name: str = Field(
        default="DEFAULT",
        description="OCI config profile name",
    )
    config_file: str = Field(
        default="~/.oci/config",
        description="Path to OCI config file",
    )
    auth_type: str = Field(
        default="api_key",
        description="Auth type: api_key, security_token, instance_principal, resource_principal",
    )
    truncate: str = Field(
        default="END",
        description="Truncation strategy: NONE, START, END",
    )
    input_type: str = Field(
        default="SEARCH_DOCUMENT",
        description=(
            "Input type for embed / embed_batch / embed_documents: "
            "SEARCH_DOCUMENT, SEARCH_QUERY, CLASSIFICATION, CLUSTERING"
        ),
    )
    query_input_type: str = Field(
        default="SEARCH_QUERY",
        description=(
            "Input type used by embed_query. Defaults to SEARCH_QUERY for "
            "Cohere asymmetric retrieval (query and document embeddings live "
            "in the same space but are optimised differently). Pin it to the "
            "same value as input_type if a deployment ever ties output "
            "dimension or handling to input_type. See issue #292."
        ),
    )
    output_dimensions: int | None = Field(
        default=None,
        description=(
            "Explicit output embedding dimension (Cohere Matryoshka). "
            "cohere.embed-v4.0 supports 256/512/1024/1536 and returns the "
            "requested size; left unset, OCI's server-side default applies "
            "(1536 for v4 today). Pin this to keep every embed / embed_query "
            "/ embed_documents call — and the stored vector column — on the "
            "same dimension regardless of the server default. Models without "
            "Matryoshka support (e.g. embed-v3.0, fixed 1024) ignore it. "
            "See issue #292."
        ),
    )


class OCIEmbeddings(BaseModel, BaseEmbedding):
    """
    OCI GenAI Embeddings using Cohere models.

    Uses Oracle Cloud Infrastructure GenAI service which hosts
    Cohere embedding models with enterprise-grade reliability.

    Example:
        >>> embedder = OCIEmbeddings(
        ...     model_id="cohere.embed-v4.0",
        ...     profile_name="DEFAULT",
        ...     auth_type="security_token",
        ... )
        >>> result = await embedder.embed("Hello world")
        >>> print(len(result.embedding))  # 1536

    Example with compartment:
        >>> embedder = OCIEmbeddings(
        ...     model_id="cohere.embed-v4.0",  # multilingual as well
        ...     compartment_id="ocid1.compartment.oc1..xxx",
        ... )
    """

    oci_config: OCIEmbeddingConfig = Field(default_factory=OCIEmbeddingConfig)
    _client: GenerativeAiInferenceClient | None = None
    _oci_config_dict: dict[str, Any] | None = None
    # Populated from the first successful embed response; subsequent calls
    # short-circuit dimension lookups. Lets any OCI model work without an
    # entry in MODEL_DIMENSION_HINTS.
    _detected_dimension: int | None = None

    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        model_id: str = OCIEmbeddingModel.COHERE_EMBED_V4.value,
        compartment_id: str = "",
        profile_name: str = "DEFAULT",
        auth_type: str = "api_key",
        service_endpoint: str | None = None,
        **kwargs: Any,
    ) -> None:
        oci_config = OCIEmbeddingConfig(
            model_id=model_id,
            compartment_id=compartment_id,
            profile_name=profile_name,
            auth_type=auth_type,
            service_endpoint=service_endpoint,
            **kwargs,
        )
        super().__init__(oci_config=oci_config)

    @property
    def config(self) -> EmbeddingConfig:
        """Get embedding configuration.

        Dimension resolution: detected from first embed response if available,
        otherwise the fast-path hint for known models, otherwise a sensible
        default. The OCI Cohere family covers 384/1024/1536-dim variants.
        """
        dimension = (
            self.oci_config.output_dimensions
            or self._detected_dimension
            or MODEL_DIMENSION_HINTS.get(self.oci_config.model_id)
            or DEFAULT_DIMENSION
        )

        return EmbeddingConfig(
            dimension=dimension,
            max_tokens=8192,  # Cohere limit
            batch_size=96,  # Cohere batch limit
        )

    @property
    def capabilities(self) -> EmbeddingCapabilities:
        """OCI Cohere embeddings: native batching (96), separate
        SEARCH_QUERY vs SEARCH_DOCUMENT input types, image-capable
        variants for ``cohere.embed-*-image-v3.0``."""
        multimodal = "image" in self.oci_config.model_id
        return EmbeddingCapabilities(
            supports_query_vs_doc=True,
            supports_multimodal=multimodal,
            supports_batching=True,
            max_batch_size=96,
            max_input_tokens=8192,
        )

    async def _get_client(self) -> GenerativeAiInferenceClient:
        """Get or create the OCI client."""
        if self._client is not None:
            return self._client

        try:
            import oci
            from oci.generative_ai_inference import GenerativeAiInferenceClient
        except ImportError as e:
            raise ImportError(
                "OCIEmbeddings requires the 'oci' package. Install with: pip install oci"
            ) from e

        # Load OCI config
        config_file = self.oci_config.config_file
        if config_file.startswith("~"):
            import os

            config_file = os.path.expanduser(config_file)

        # ``oci.config.from_file`` returns the parsed config dict; the
        # ``_oci_config_dict`` field is declared as ``... | None`` to
        # represent the pre-init state, so bind to a local before use.
        config_dict: dict[str, Any] = oci.config.from_file(
            config_file,
            self.oci_config.profile_name,
        )
        self._oci_config_dict = config_dict

        # Determine service endpoint. Treat an empty-string config (which
        # notebooks pass when no env override is set) the same as None and
        # auto-derive from a region. Region preference is:
        #   1. LOCUS_OCI_REGION env (so a session-token profile whose
        #      home region is e.g. us-ashburn-1 can still hit GenAI in
        #      us-chicago-1 without code changes),
        #   2. OCI_REGION env,
        #   3. region in the OCI config profile,
        #   4. us-chicago-1 (the canonical GenAI region).
        endpoint = self.oci_config.service_endpoint
        if not endpoint:
            import os as _os

            region = (
                _os.environ.get("LOCUS_OCI_REGION")
                or _os.environ.get("OCI_REGION")
                or config_dict.get("region")
                or "us-chicago-1"
            )
            endpoint = f"https://inference.generativeai.{region}.oci.oraclecloud.com"

        # Determine auth type - respect explicit setting, only auto-detect if needed
        auth_type = self.oci_config.auth_type

        # A profile with ``security_token_file`` and no ``user`` was written by
        # ``oci session authenticate``; API-key signing cannot work for it, so
        # switch to the session token — under the default ``api_key`` setting
        # too, the same rule OCIClient and the V1 transport apply. Only the
        # explicit principal-based modes (which never read a profile) are
        # left alone.
        if (
            auth_type not in ("instance_principal", "resource_principal")
            and "security_token_file" in config_dict
            and "user" not in config_dict
        ):
            auth_type = "security_token"

        # Create client based on auth type
        if auth_type == "security_token":
            token_file = config_dict.get("security_token_file")
            if token_file:
                import os as os_module

                token_file = os_module.path.expanduser(token_file)
                with open(token_file) as f:
                    token = f.read().strip()
                key_file = os_module.path.expanduser(config_dict["key_file"])
                private_key = oci.signer.load_private_key_from_file(key_file)
                signer = oci.auth.signers.SecurityTokenSigner(token, private_key)
                self._client = GenerativeAiInferenceClient(
                    config={},
                    signer=signer,
                    service_endpoint=endpoint,
                )
            else:
                signer = oci.auth.signers.get_resource_principals_signer()
                self._client = GenerativeAiInferenceClient(
                    config={},
                    signer=signer,
                    service_endpoint=endpoint,
                )
        elif auth_type == "instance_principal":
            signer = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
            self._client = GenerativeAiInferenceClient(
                config={},
                signer=signer,
                service_endpoint=endpoint,
            )
        elif auth_type == "resource_principal":
            signer = oci.auth.signers.get_resource_principals_signer()
            self._client = GenerativeAiInferenceClient(
                config={},
                signer=signer,
                service_endpoint=endpoint,
            )
        else:
            # API key auth (default) - just pass config, SDK creates signer
            self._client = GenerativeAiInferenceClient(
                config=self._oci_config_dict,
                service_endpoint=endpoint,
            )

        return self._client

    def _get_compartment_id(self) -> str:
        """Get compartment ID, defaulting to tenancy."""
        if self.oci_config.compartment_id:
            return self.oci_config.compartment_id
        if self._oci_config_dict:
            tenancy: str = self._oci_config_dict.get("tenancy", "")
            return tenancy
        return ""

    def _record_dimension(self, embeddings: list[list[float]]) -> None:
        """Cache the detected embedding dimension from a successful response.

        Idempotent — first call wins. Lets `config.dimension` reflect the
        ground truth for whatever model OCI returned, without needing the
        model to be listed in MODEL_DIMENSION_HINTS.
        """
        if self._detected_dimension is None and embeddings:
            first = embeddings[0]
            if first:
                self._detected_dimension = len(first)

    def _build_embed_details(self, inputs: list[str], input_type: str) -> Any:
        """Build an ``EmbedTextDetails`` shared by every embed path.

        ``output_dimensions`` is only attached when the caller set it, so
        models without Matryoshka support never see the field. See #292.
        """
        from oci.generative_ai_inference.models import (
            EmbedTextDetails,
            OnDemandServingMode,
        )

        kwargs: dict[str, Any] = {
            "inputs": inputs,
            "serving_mode": OnDemandServingMode(model_id=self.oci_config.model_id),
            "compartment_id": self._get_compartment_id(),
            "truncate": self.oci_config.truncate,
            "input_type": input_type,
        }
        if self.oci_config.output_dimensions is not None:
            kwargs["output_dimensions"] = self.oci_config.output_dimensions
        return EmbedTextDetails(**kwargs)

    async def embed(self, text: str) -> EmbeddingResult:
        """Embed a single text."""
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """Embed multiple texts."""
        client = await self._get_client()

        embed_details = self._build_embed_details(texts, self.oci_config.input_type)

        response = client.embed_text(embed_details)
        embeddings = response.data.embeddings
        self._record_dimension(embeddings)

        results = []
        for i, text in enumerate(texts):
            results.append(
                EmbeddingResult(
                    embedding=embeddings[i],
                    text=text,
                    model=self.oci_config.model_id,
                    tokens=None,  # OCI doesn't return token count
                )
            )

        return results

    async def embed_query(self, query: str) -> EmbeddingResult:
        """Embed a query for retrieval.

        Uses ``query_input_type`` (default ``SEARCH_QUERY``) instead of
        hardcoding it, so callers can pin query and document embeddings to
        the same input_type when needed. See issue #292.
        """
        client = await self._get_client()

        embed_details = self._build_embed_details([query], self.oci_config.query_input_type)

        response = client.embed_text(embed_details)
        self._record_dimension(response.data.embeddings)

        return EmbeddingResult(
            embedding=response.data.embeddings[0],
            text=query,
            model=self.oci_config.model_id,
            tokens=None,
        )

    async def embed_documents(self, documents: list[str]) -> list[EmbeddingResult]:
        """Embed documents for storage.

        Uses ``input_type`` (default ``SEARCH_DOCUMENT``) from config.
        """
        client = await self._get_client()

        # Process in batches
        results = []
        batch_size = self.config.batch_size

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]

            embed_details = self._build_embed_details(batch, self.oci_config.input_type)

            response = client.embed_text(embed_details)
            self._record_dimension(response.data.embeddings)

            for j, text in enumerate(batch):
                results.append(
                    EmbeddingResult(
                        embedding=response.data.embeddings[j],
                        text=text,
                        model=self.oci_config.model_id,
                        tokens=None,
                    )
                )

        return results

    async def close(self) -> None:
        """Drop the cached OCI client.

        The OCI SDK client holds no connection pool that needs an explicit
        shutdown, so this only forgets the client; the next call rebuilds it.
        Present for parity with the other embedders, so callers can close any
        provider the same way.
        """
        self._client = None

    def __repr__(self) -> str:
        return f"OCIEmbeddings(model={self.oci_config.model_id!r})"
