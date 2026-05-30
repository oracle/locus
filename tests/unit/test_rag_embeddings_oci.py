# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for ``locus.rag.embeddings.oci`` (OCIEmbeddings).

Stubs the OCI SDK client + ``oci.config.from_file`` so the tests
never reach a real OCI endpoint. Coverage targets:

- config defaults and constructor flag forwarding
- ``capabilities`` based on model id (multimodal-image variants)
- ``_get_client`` auth-type matrix (api_key, security_token,
  instance_principal, resource_principal, security_token auto-detect,
  resource_principal fallback when no token file)
- ``_get_compartment_id`` precedence (explicit > tenancy > empty)
- ``embed`` / ``embed_batch`` / ``embed_query`` / ``embed_documents``
  including the multi-batch path (97 docs across two batches)
- missing-OCI-package import error
"""

from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock

import pytest

from locus.rag.embeddings.oci import OCIEmbeddingConfig, OCIEmbeddingModel, OCIEmbeddings


# ---------------------------------------------------------------------------
# OCI SDK stubs
# ---------------------------------------------------------------------------


def _stub_oci_modules(
    monkeypatch: pytest.MonkeyPatch,
    *,
    config_dict: dict[str, Any] | None = None,
    embeddings: list[list[float]] | None = None,
) -> dict[str, Any]:
    """Install fake ``oci`` and ``oci.generative_ai_inference`` modules.

    Returns a dict of probes the test can inspect (``client_kwargs`` is
    captured on each ``GenerativeAiInferenceClient(...)`` call).
    """
    config_dict = config_dict or {"region": "us-chicago-1", "user": "u"}
    embeddings = embeddings or [[0.1] * 1024]

    probes: dict[str, Any] = {"client_kwargs": [], "embed_text_calls": []}

    # Build the fake ``oci`` module hierarchy.
    oci_mod = types.ModuleType("oci")

    class _ConfigMod:
        @staticmethod
        def from_file(path: str, profile: str) -> dict[str, Any]:
            probes["from_file"] = (path, profile)
            return dict(config_dict)

    class _AuthSigners:
        @staticmethod
        def InstancePrincipalsSecurityTokenSigner() -> Any:  # noqa: N802
            probes["used_signer"] = "instance_principal"
            return MagicMock(name="instance_principal_signer")

        @staticmethod
        def get_resource_principals_signer() -> Any:
            probes["used_signer"] = "resource_principal"
            return MagicMock(name="resource_principal_signer")

        @staticmethod
        def SecurityTokenSigner(token: str, key: Any) -> Any:  # noqa: N802
            probes["used_signer"] = "security_token"
            probes["security_token"] = token
            return MagicMock(name="security_token_signer")

    class _Auth:
        signers = _AuthSigners

    class _Signer:
        @staticmethod
        def load_private_key_from_file(path: str) -> Any:
            probes["loaded_key_from"] = path
            return MagicMock(name="private_key")

    oci_mod.config = _ConfigMod  # type: ignore[attr-defined]
    oci_mod.auth = _Auth  # type: ignore[attr-defined]
    oci_mod.signer = _Signer  # type: ignore[attr-defined]

    # ``oci.generative_ai_inference`` — provides ``GenerativeAiInferenceClient``.
    gai_mod = types.ModuleType("oci.generative_ai_inference")

    class _Response:
        def __init__(self) -> None:
            self.data = types.SimpleNamespace(embeddings=embeddings)

    class _Client:
        def __init__(self, **kwargs: Any) -> None:
            probes["client_kwargs"].append(kwargs)

        def embed_text(self, details: Any) -> Any:
            probes["embed_text_calls"].append(details)
            return _Response()

    gai_mod.GenerativeAiInferenceClient = _Client  # type: ignore[attr-defined]

    # ``oci.generative_ai_inference.models`` — provides EmbedTextDetails + OnDemandServingMode.
    models_mod = types.ModuleType("oci.generative_ai_inference.models")

    class _EmbedTextDetails:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            for k, v in kwargs.items():
                setattr(self, k, v)

    class _OnDemandServingMode:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    models_mod.EmbedTextDetails = _EmbedTextDetails  # type: ignore[attr-defined]
    models_mod.OnDemandServingMode = _OnDemandServingMode  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "oci", oci_mod)
    monkeypatch.setitem(sys.modules, "oci.generative_ai_inference", gai_mod)
    monkeypatch.setitem(sys.modules, "oci.generative_ai_inference.models", models_mod)

    return probes


# ---------------------------------------------------------------------------
# Config + capabilities
# ---------------------------------------------------------------------------


class TestConfigAndCapabilities:
    def test_defaults(self) -> None:
        cfg = OCIEmbeddingConfig()
        assert cfg.model_id == OCIEmbeddingModel.COHERE_EMBED_ENGLISH_V3.value
        assert cfg.profile_name == "DEFAULT"
        assert cfg.input_type == "SEARCH_DOCUMENT"
        assert cfg.query_input_type == "SEARCH_QUERY"
        assert cfg.output_dimensions is None
        assert cfg.truncate == "END"
        assert cfg.compartment_id == ""

    def test_config_dimension_for_known_model(self) -> None:
        m = OCIEmbeddings(model_id=OCIEmbeddingModel.COHERE_EMBED_ENGLISH_V3.value)
        assert m.config.dimension == 1024

    def test_config_dimension_for_light_variant(self) -> None:
        m = OCIEmbeddings(model_id=OCIEmbeddingModel.COHERE_EMBED_ENGLISH_LIGHT_V3.value)
        assert m.config.dimension == 384

    def test_config_dimension_unknown_model_defaults(self) -> None:
        m = OCIEmbeddings(model_id="cohere.embed-totally-new")
        assert m.config.dimension == 1024

    def test_capabilities_supports_query_vs_doc(self) -> None:
        m = OCIEmbeddings()
        caps = m.capabilities
        assert caps.supports_query_vs_doc is True
        assert caps.max_batch_size == 96
        assert caps.supports_batching is True
        assert caps.supports_multimodal is False

    def test_capabilities_multimodal_for_image_variant(self) -> None:
        m = OCIEmbeddings(model_id="cohere.embed-english-image-v3.0")
        assert m.capabilities.supports_multimodal is True


# ---------------------------------------------------------------------------
# _get_client auth-type matrix
# ---------------------------------------------------------------------------


class TestGetClientAuthMatrix:
    @pytest.mark.asyncio
    async def test_api_key_auth_passes_config_dict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        probes = _stub_oci_modules(monkeypatch)
        m = OCIEmbeddings(auth_type="api_key")
        await m._get_client()
        kwargs = probes["client_kwargs"][0]
        # The API-key path passes the parsed config dict, no signer.
        assert kwargs.get("config") is not None
        assert "signer" not in kwargs

    @pytest.mark.asyncio
    async def test_caches_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        probes = _stub_oci_modules(monkeypatch)
        m = OCIEmbeddings()
        c1 = await m._get_client()
        c2 = await m._get_client()
        assert c1 is c2
        # ``GenerativeAiInferenceClient`` only constructed once.
        assert len(probes["client_kwargs"]) == 1

    @pytest.mark.asyncio
    async def test_security_token_auto_detected(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        # Profile has ``security_token_file`` but no ``user`` → auto upgrade
        # to security_token auth even when caller passed a different type.
        token_file = tmp_path / "token.txt"
        token_file.write_text("FAKE_TOKEN")
        probes = _stub_oci_modules(
            monkeypatch,
            config_dict={
                "region": "us-chicago-1",
                "security_token_file": str(token_file),
                "key_file": str(tmp_path / "key.pem"),
            },
        )
        m = OCIEmbeddings(auth_type="security_token_explicit_other")
        await m._get_client()
        assert probes["used_signer"] == "security_token"
        assert probes["security_token"] == "FAKE_TOKEN"  # noqa: S105

    @pytest.mark.asyncio
    async def test_security_token_without_file_falls_back_to_resource_principal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Caller asked for security_token but no ``security_token_file``
        # in the config dict → fall back to resource principal.
        probes = _stub_oci_modules(monkeypatch, config_dict={"region": "us-chicago-1"})
        m = OCIEmbeddings(auth_type="security_token")
        await m._get_client()
        assert probes["used_signer"] == "resource_principal"

    @pytest.mark.asyncio
    async def test_instance_principal_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        probes = _stub_oci_modules(monkeypatch)
        m = OCIEmbeddings(auth_type="instance_principal")
        await m._get_client()
        assert probes["used_signer"] == "instance_principal"

    @pytest.mark.asyncio
    async def test_resource_principal_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        probes = _stub_oci_modules(monkeypatch)
        m = OCIEmbeddings(auth_type="resource_principal")
        await m._get_client()
        assert probes["used_signer"] == "resource_principal"

    @pytest.mark.asyncio
    async def test_explicit_endpoint_overrides_region(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        probes = _stub_oci_modules(monkeypatch)
        m = OCIEmbeddings(service_endpoint="https://override.example.com")
        await m._get_client()
        assert probes["client_kwargs"][0]["service_endpoint"] == "https://override.example.com"

    @pytest.mark.asyncio
    async def test_endpoint_inferred_from_region(self, monkeypatch: pytest.MonkeyPatch) -> None:
        probes = _stub_oci_modules(monkeypatch, config_dict={"region": "uk-london-1", "user": "u"})
        m = OCIEmbeddings()
        await m._get_client()
        assert "uk-london-1" in probes["client_kwargs"][0]["service_endpoint"]

    @pytest.mark.asyncio
    async def test_missing_oci_package_raises_clear_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Hide ``oci`` so the import inside ``_get_client`` raises.
        monkeypatch.setitem(sys.modules, "oci", None)
        m = OCIEmbeddings()
        with pytest.raises(ImportError, match="OCIEmbeddings requires the 'oci' package"):
            await m._get_client()


# ---------------------------------------------------------------------------
# _get_compartment_id
# ---------------------------------------------------------------------------


class TestCompartmentId:
    def test_explicit_compartment_takes_precedence(self) -> None:
        m = OCIEmbeddings(compartment_id="ocid1.compartment.oc1..xxx")
        assert m._get_compartment_id() == "ocid1.compartment.oc1..xxx"

    def test_falls_back_to_tenancy(self) -> None:
        m = OCIEmbeddings()
        m._oci_config_dict = {"tenancy": "ocid1.tenancy.oc1..yyy"}
        assert m._get_compartment_id() == "ocid1.tenancy.oc1..yyy"

    def test_returns_empty_when_neither_set(self) -> None:
        m = OCIEmbeddings()
        assert m._get_compartment_id() == ""


# ---------------------------------------------------------------------------
# embed / embed_batch / embed_query / embed_documents
# ---------------------------------------------------------------------------


class TestEmbedSingle:
    @pytest.mark.asyncio
    async def test_embed_returns_first_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_oci_modules(monkeypatch, embeddings=[[0.5] * 1024])
        m = OCIEmbeddings()
        result = await m.embed("hello")
        assert result.text == "hello"
        assert len(result.embedding) == 1024


class TestEmbedBatch:
    @pytest.mark.asyncio
    async def test_returns_one_result_per_input(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_oci_modules(monkeypatch, embeddings=[[0.0] * 1024, [1.0] * 1024])
        m = OCIEmbeddings()
        results = await m.embed_batch(["alpha", "beta"])
        assert [r.text for r in results] == ["alpha", "beta"]
        assert results[0].model == m.oci_config.model_id


class TestEmbedQuery:
    @pytest.mark.asyncio
    async def test_uses_search_query_input_type(self, monkeypatch: pytest.MonkeyPatch) -> None:
        probes = _stub_oci_modules(monkeypatch, embeddings=[[0.0] * 1024])
        m = OCIEmbeddings()
        await m.embed_query("what is locus")
        # Default ``query_input_type`` is ``SEARCH_QUERY``.
        assert probes["embed_text_calls"][-1].input_type == "SEARCH_QUERY"

    @pytest.mark.asyncio
    async def test_query_input_type_is_configurable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # issue #292: embed_query must honor query_input_type, not hardcode
        # SEARCH_QUERY. Pinning it to SEARCH_DOCUMENT lets cohere.embed-v4.0
        # queries match a SEARCH_DOCUMENT-indexed (1536-dim) table.
        probes = _stub_oci_modules(monkeypatch, embeddings=[[0.0] * 1536])
        m = OCIEmbeddings(query_input_type="SEARCH_DOCUMENT")
        await m.embed_query("luxury hotels in Greece")
        assert probes["embed_text_calls"][-1].input_type == "SEARCH_DOCUMENT"


class TestEmbedDocuments:
    @pytest.mark.asyncio
    async def test_uses_search_document_input_type(self, monkeypatch: pytest.MonkeyPatch) -> None:
        probes = _stub_oci_modules(monkeypatch, embeddings=[[0.0] * 1024])
        m = OCIEmbeddings()
        await m.embed_documents(["doc1"])
        assert probes["embed_text_calls"][-1].input_type == "SEARCH_DOCUMENT"

    @pytest.mark.asyncio
    async def test_input_type_is_configurable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # issue #292: embed_documents reads input_type from config rather than
        # hardcoding SEARCH_DOCUMENT.
        probes = _stub_oci_modules(monkeypatch, embeddings=[[0.0] * 1024])
        m = OCIEmbeddings(input_type="CLASSIFICATION")
        await m.embed_documents(["doc1"])
        assert probes["embed_text_calls"][-1].input_type == "CLASSIFICATION"


class TestOutputDimensions:
    """issue #292: Cohere v4 is Matryoshka — output_dimensions controls the
    returned dimension. Locus must let callers pin it so every embed path
    (and the stored vector column) stays on one dimension regardless of OCI's
    server-side default."""

    @pytest.mark.asyncio
    async def test_omitted_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Unset → field never attached, so non-Matryoshka models never see it.
        probes = _stub_oci_modules(monkeypatch, embeddings=[[0.0] * 1536])
        m = OCIEmbeddings(model_id="cohere.embed-v4.0")
        await m.embed("hello")
        assert not hasattr(probes["embed_text_calls"][-1], "output_dimensions")

    @pytest.mark.asyncio
    async def test_forwarded_on_every_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        probes = _stub_oci_modules(monkeypatch, embeddings=[[0.0] * 1024])
        m = OCIEmbeddings(model_id="cohere.embed-v4.0", output_dimensions=1024)
        await m.embed("a")
        await m.embed_query("q")
        await m.embed_documents(["d"])
        # embed (via embed_batch), embed_query, embed_documents all pin 1024.
        assert [c.output_dimensions for c in probes["embed_text_calls"]] == [1024, 1024, 1024]

    def test_config_dimension_reflects_output_dimensions(self) -> None:
        # The vector store sizes its column from config.dimension; an explicit
        # output_dimensions must win over the model hint (v4 hint is 1536).
        m = OCIEmbeddings(model_id="cohere.embed-v4.0", output_dimensions=1024)
        assert m.config.dimension == 1024

    @pytest.mark.asyncio
    async def test_batches_when_inputs_exceed_batch_size(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Provide enough embeddings to satisfy each batch's response.
        # The stubs are reused across calls; each call returns one fixed
        # response — each test embed has the same shape, so we just need
        # >= longest batch size.
        _stub_oci_modules(monkeypatch, embeddings=[[0.0] * 1024] * 96)
        m = OCIEmbeddings()
        # 97 docs → batch_size=96 → two calls (96 + 1).
        results = await m.embed_documents([f"doc{i}" for i in range(97)])
        assert len(results) == 97
