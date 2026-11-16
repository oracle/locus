# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""``locus._oci_auth``: one resource-principal signer for every platform, and every switch that uses it.

On an OKE pod with workload identity the generic ``get_resource_principals_signer``
fails (``session_token was not provided``); the OKE signer exchanges the pod's
service-account token. ``oke_workload_identity`` names that path outright and
``resource_principal`` detects it from the token file, in the entrypoint, the
GenAI clients, the OCI tool, the Object Storage backend and the litellm bridge.
"""

from __future__ import annotations

import sys
import types
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from locus import _oci_auth


if TYPE_CHECKING:
    from pathlib import Path


class _Signer:
    def __init__(self, kind: str, region: str = "eu-frankfurt-1") -> None:
        self.kind = kind
        self.region = region


def _fake_oci(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    oci = types.ModuleType("oci")
    signers = types.SimpleNamespace(
        get_resource_principals_signer=lambda: _Signer("generic"),
        get_oke_workload_identity_resource_principal_signer=lambda: _Signer("oke"),
    )
    oci.auth = types.SimpleNamespace(signers=signers)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "oci", oci)
    return oci


@pytest.fixture
def no_token(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A machine without a Kubernetes service-account token."""
    absent = tmp_path / "absent"
    monkeypatch.setattr(_oci_auth, "SERVICE_ACCOUNT_TOKEN_PATH", absent)
    return absent


@pytest.fixture
def pod_token(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A pod: the service-account token is mounted."""
    token = tmp_path / "token"
    token.write_text("eyJhbGciOi...")
    monkeypatch.setattr(_oci_auth, "SERVICE_ACCOUNT_TOKEN_PATH", token)
    return token


# --------------------------------------------------------------------------
# The helper
# --------------------------------------------------------------------------


def test_explicit_oke_workload_identity_always_uses_the_oke_signer(
    monkeypatch: pytest.MonkeyPatch, no_token: Path
) -> None:
    oci = _fake_oci(monkeypatch)
    signer = _oci_auth.resource_principal_signer("oke_workload_identity", oci_module=oci)
    assert signer.kind == "oke"


def test_resource_principal_detects_a_pod_from_the_token_file(
    monkeypatch: pytest.MonkeyPatch, pod_token: Path
) -> None:
    oci = _fake_oci(monkeypatch)
    pod = {"OCI_RESOURCE_PRINCIPAL_VERSION": "2.2"}
    assert _oci_auth.on_oke_workload_identity(pod)
    assert _oci_auth.resource_principal_signer(oci_module=oci, environ=pod).kind == "oke"
    # An explicit token path wins over the module default.
    assert not _oci_auth.on_oke_workload_identity(pod, pod_token.with_name("elsewhere"))


def test_a_session_token_means_the_generic_signer_even_with_a_token_file(
    monkeypatch: pytest.MonkeyPatch, pod_token: Path
) -> None:
    oci = _fake_oci(monkeypatch)
    for variable in ("OCI_RESOURCE_PRINCIPAL_RPST", "OCI_RESOURCE_PRINCIPAL_PRIVATE_PEM"):
        environ = {variable: "set"}
        assert not _oci_auth.on_oke_workload_identity(environ)
        signer = _oci_auth.resource_principal_signer(oci_module=oci, environ=environ)
        assert signer.kind == "generic"


def test_without_a_token_file_resource_principal_is_the_generic_signer(
    monkeypatch: pytest.MonkeyPatch, no_token: Path
) -> None:
    oci = _fake_oci(monkeypatch)
    assert not _oci_auth.on_oke_workload_identity({})
    assert _oci_auth.resource_principal_signer(oci_module=oci, environ={}).kind == "generic"


def test_the_helper_imports_oci_itself_and_reads_the_process_environment(
    monkeypatch: pytest.MonkeyPatch, pod_token: Path
) -> None:
    _fake_oci(monkeypatch)
    monkeypatch.delenv("OCI_RESOURCE_PRINCIPAL_RPST", raising=False)
    monkeypatch.delenv("OCI_RESOURCE_PRINCIPAL_PRIVATE_PEM", raising=False)
    assert _oci_auth.resource_principal_signer().kind == "oke"
    monkeypatch.setenv("OCI_RESOURCE_PRINCIPAL_RPST", "token")
    assert _oci_auth.resource_principal_signer().kind == "generic"


def test_other_auth_types_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    oci = _fake_oci(monkeypatch)
    with pytest.raises(ValueError, match="resource_principal or oke_workload_identity"):
        _oci_auth.resource_principal_signer("instance_principal", oci_module=oci)


# --------------------------------------------------------------------------
# The SDK switches
# --------------------------------------------------------------------------


def test_the_genai_sdk_client_accepts_oke_workload_identity(no_token: Path) -> None:
    pytest.importorskip("oci")
    from locus.models.providers.oci.client import OCIAuthType, OCIClient, OCIClientConfig

    assert OCIAuthType("oke_workload_identity") is OCIAuthType.OKE_WORKLOAD_IDENTITY
    config = OCIClientConfig(auth_type=OCIAuthType.OKE_WORKLOAD_IDENTITY)
    client = OCIClient(config)
    assert client.oci_config == {}  # no profile is read
    oke_signer = MagicMock(name="oke-signer")
    with (
        patch(
            "oci.auth.signers.get_oke_workload_identity_resource_principal_signer",
            return_value=oke_signer,
        ) as oke,
        patch("oci.auth.signers.get_resource_principals_signer") as generic,
        patch("oci.generative_ai_inference.GenerativeAiInferenceClient") as client_class,
    ):
        _ = client.client
    oke.assert_called_once_with()
    generic.assert_not_called()
    assert client_class.call_args.kwargs["signer"] is oke_signer


def test_the_genai_sdk_client_detects_a_pod_under_resource_principal(pod_token: Path) -> None:
    pytest.importorskip("oci")
    from locus.models.providers.oci.client import OCIAuthType, OCIClient, OCIClientConfig

    client = OCIClient(OCIClientConfig(auth_type=OCIAuthType.RESOURCE_PRINCIPAL))
    with (
        patch.dict("os.environ", {}, clear=False) as environ,
        patch("oci.auth.signers.get_oke_workload_identity_resource_principal_signer") as oke,
        patch("oci.auth.signers.get_resource_principals_signer") as generic,
        patch("oci.generative_ai_inference.GenerativeAiInferenceClient"),
    ):
        environ.pop("OCI_RESOURCE_PRINCIPAL_RPST", None)
        environ.pop("OCI_RESOURCE_PRINCIPAL_PRIVATE_PEM", None)
        _ = client.client
    oke.assert_called_once_with()
    generic.assert_not_called()


def test_the_oci_tool_builds_the_oke_signer(no_token: Path) -> None:
    pytest.importorskip("oci")
    from locus.tools import oci as oci_tool

    oci_mod = oci_tool._load_oci()
    with patch.object(
        oci_mod.auth.signers, "get_oke_workload_identity_resource_principal_signer"
    ) as factory:
        factory.return_value.region = "uk-london-1"
        config, signer = oci_tool._build_config_and_signer(
            oci_mod, profile="ignored", auth_type="oke_workload_identity", region=None
        )
    assert config == {"region": "uk-london-1"}
    assert signer is factory.return_value


def test_the_object_storage_backend_builds_the_oke_signer(no_token: Path) -> None:
    pytest.importorskip("oci")
    from locus.memory.backends.oci_bucket import OCIBucketConfig, build_object_storage_client

    cfg = OCIBucketConfig(bucket_name="b", namespace="ns", auth_type="oke_workload_identity")
    with (
        patch("oci.auth.signers.get_oke_workload_identity_resource_principal_signer") as oke,
        patch("oci.object_storage.ObjectStorageClient") as client_class,
    ):
        build_object_storage_client(cfg)
    oke.assert_called_once_with()
    client_class.assert_called_once_with(config={}, signer=oke.return_value)


def test_the_litellm_bridge_reads_oke_workload_identity_from_the_environment(
    monkeypatch: pytest.MonkeyPatch, no_token: Path
) -> None:
    pytest.importorskip("oci")
    from locus.memory.managers.oracle_agent_memory import oci_litellm_auth

    for name in ("OCI_USER", "OCI_KEY_FILE", "OCI_KEY_CONTENT", "OCI_FINGERPRINT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OCI_AUTH_TYPE", "oke_workload_identity")
    monkeypatch.setenv("OCI_COMPARTMENT_ID", "ocid1.compartment.oc1..example")
    monkeypatch.delenv("OCI_REGION", raising=False)
    with patch("oci.auth.signers.get_oke_workload_identity_resource_principal_signer") as oke:
        oke.return_value.region = "uk-london-1"
        kwargs = oci_litellm_auth("oci/cohere.embed-v4.0")
    assert kwargs["oci_signer"] is oke.return_value
    assert kwargs["oci_region"] == "uk-london-1"


def test_the_v1_and_responses_transports_accept_oke_workload_identity(no_token: Path) -> None:
    pytest.importorskip("oci")
    from locus.models.providers.oci import openai_compat, responses

    sentinel = MagicMock(name="oke-signer")
    with patch(
        "oci.auth.signers.get_oke_workload_identity_resource_principal_signer",
        return_value=sentinel,
    ) as oke:
        assert openai_compat._build_oke_workload_identity_signer() is sentinel
        chat = openai_compat.OCIChatCompletionsModel(
            model="openai.gpt-4.1",
            auth_type="oke_workload_identity",
            compartment_id="ocid1.compartment.oc1..example",
        )
        assert chat._build_signer() is sentinel
        model = responses.OCIResponsesModel(
            model="openai.gpt-5.5",
            auth_type="oke_workload_identity",
            compartment_id="ocid1.compartment.oc1..example",
        )
        assert model._build_signer() is sentinel
    assert oke.call_count == 3
    refresh = openai_compat._refresh_callable_for(MagicMock(), auth_type="oke_workload_identity")
    assert refresh is openai_compat._build_oke_workload_identity_signer
    with pytest.raises(ValueError, match="auth_type must be one of"):
        openai_compat.OCIChatCompletionsModel(
            model="openai.gpt-4.1", auth_type="workload", compartment_id="ocid1.compartment.oc1..x"
        )


def test_the_generic_resource_principal_builder_detects_a_pod(pod_token: Path) -> None:
    """``OCIChatCompletionsModel(auth_type="resource_principal")`` on OKE, the generated server's path."""
    pytest.importorskip("oci")
    from locus.models.providers.oci import openai_compat

    with (
        patch.dict("os.environ", {}, clear=False) as environ,
        patch("oci.auth.signers.get_oke_workload_identity_resource_principal_signer") as oke,
        patch("oci.auth.signers.get_resource_principals_signer") as generic,
    ):
        environ.pop("OCI_RESOURCE_PRINCIPAL_RPST", None)
        environ.pop("OCI_RESOURCE_PRINCIPAL_PRIVATE_PEM", None)
        assert openai_compat._build_resource_principal_signer() is oke.return_value
        environ["OCI_RESOURCE_PRINCIPAL_RPST"] = "token"
        assert openai_compat._build_resource_principal_signer() is generic.return_value
