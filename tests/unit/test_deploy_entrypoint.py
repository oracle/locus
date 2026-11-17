# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""The container entrypoint's Vault client: which principal it signs with and how it reads a bundle."""

from __future__ import annotations

import base64
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from locus import _oci_auth
from locus.deploy import entrypoint


class _Signer:
    def __init__(self, region: str = "eu-frankfurt-1") -> None:
        self.region = region


class _Bundle:
    def __init__(self, text: str) -> None:
        content = types.SimpleNamespace(content=base64.b64encode(text.encode()).decode())
        self.data = types.SimpleNamespace(secret_bundle_content=content)


class _SecretsClient:
    instances: list[_SecretsClient] = []

    def __init__(self, config: dict[str, Any], signer: Any = None) -> None:
        self.config = config
        self.signer = signer
        self.requested: list[str] = []
        _SecretsClient.instances.append(self)

    def get_secret_bundle(self, secret_id: str) -> _Bundle:
        self.requested.append(secret_id)
        return _Bundle(f"content of {secret_id}")


def _fake_oci(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """A stand-in ``oci`` package with the four signer factories and the secrets client."""
    oci = types.ModuleType("oci")
    signers = types.SimpleNamespace(
        get_resource_principals_signer=lambda: _Signer("us-ashburn-1"),
        get_oke_workload_identity_resource_principal_signer=lambda: _Signer("uk-london-1"),
        InstancePrincipalsSecurityTokenSigner=lambda: _Signer("eu-frankfurt-1"),
    )
    oci.auth = types.SimpleNamespace(signers=signers)  # type: ignore[attr-defined]
    oci.config = types.SimpleNamespace(  # type: ignore[attr-defined]
        from_file=lambda profile_name: {"profile": profile_name, "region": "us-chicago-1"}
    )
    oci.signer = types.SimpleNamespace(  # type: ignore[attr-defined]
        Signer=types.SimpleNamespace(from_config=lambda config: _Signer(config["region"]))
    )
    oci.secrets = types.SimpleNamespace(SecretsClient=_SecretsClient)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "oci", oci)
    _SecretsClient.instances.clear()
    return oci


def test_resource_principal_signer_carries_its_region(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_oci(monkeypatch)
    # No service-account token on this machine: the generic signer.
    monkeypatch.setattr(_oci_auth, "SERVICE_ACCOUNT_TOKEN_PATH", tmp_path / "absent")
    config, signer = entrypoint._signer("resource_principal", {})
    assert config == {"region": "us-ashburn-1"}
    assert signer.region == "us-ashburn-1"


def test_oke_workload_identity_is_its_own_auth_type(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_oci(monkeypatch)
    config, signer = entrypoint._signer("oke_workload_identity", {})
    assert config == {"region": "uk-london-1"}
    assert signer.region == "uk-london-1"


def test_resource_principal_on_an_oke_pod_uses_the_workload_identity_signer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The chart's default, and what a user who wrote ``resource_principal`` gets on OKE."""
    _fake_oci(monkeypatch)
    token = tmp_path / "token"
    token.write_text("eyJ...")
    monkeypatch.setattr(_oci_auth, "SERVICE_ACCOUNT_TOKEN_PATH", token)
    pod = {"OCI_RESOURCE_PRINCIPAL_VERSION": "2.2", "OCI_RESOURCE_PRINCIPAL_REGION": "uk-london-1"}
    config, signer = entrypoint._signer("resource_principal", pod)
    assert config == {"region": "uk-london-1"}
    # A Container Instance or Function carries the session token: generic signer, token file or not.
    with_rpst = {**pod, "OCI_RESOURCE_PRINCIPAL_RPST": "/path/or/token"}
    config, signer = entrypoint._signer("resource_principal", with_rpst)
    assert config == {"region": "us-ashburn-1"}
    # vault_fetcher's default on a pod goes the same way.
    entrypoint.vault_fetcher(pod)
    assert _SecretsClient.instances[-1].signer.region == "uk-london-1"


def test_instance_principal_signer_carries_its_region(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_oci(monkeypatch)
    config, signer = entrypoint._signer("instance_principal", {})
    assert config == {"region": "eu-frankfurt-1"}
    assert signer.region == "eu-frankfurt-1"


def test_api_key_signer_reads_the_named_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_oci(monkeypatch)
    config, signer = entrypoint._signer("api_key", {"OCI_PROFILE": "LOCAL"})
    assert config == {"profile": "LOCAL", "region": "us-chicago-1"}
    assert signer.region == "us-chicago-1"
    default_config, _ = entrypoint._signer("api_key", {})
    assert default_config["profile"] == "DEFAULT"


def test_unknown_auth_type_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_oci(monkeypatch)
    with pytest.raises(ValueError, match=r"LOCUS_OCI_AUTH_TYPE must be .*oke_workload_identity"):
        entrypoint._signer("workload", {})


def test_vault_fetcher_decodes_the_bundle_content(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_oci(monkeypatch)
    fetch = entrypoint.vault_fetcher({"LOCUS_OCI_AUTH_TYPE": "instance_principal"})
    assert fetch("ocid1.vaultsecret.oc1..key") == "content of ocid1.vaultsecret.oc1..key"
    (client,) = _SecretsClient.instances
    assert client.config == {"region": "eu-frankfurt-1"}
    assert client.requested == ["ocid1.vaultsecret.oc1..key"]


def test_vault_fetcher_picks_the_platform_principal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without ``LOCUS_OCI_AUTH_TYPE`` the resource principal wins when the platform offers one."""
    _fake_oci(monkeypatch)
    entrypoint.vault_fetcher({"OCI_RESOURCE_PRINCIPAL_VERSION": "2.2"})
    entrypoint.vault_fetcher({})
    first, second = _SecretsClient.instances
    assert first.signer.region == "us-ashburn-1"  # resource principal
    assert second.signer.region == "eu-frankfurt-1"  # instance principal


def test_resolve_secrets_uses_the_vault_fetcher_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_oci(monkeypatch)
    environ = {
        "LOCUS_OCI_AUTH_TYPE": "resource_principal",
        "LOCUS_APM_DATA_KEY_SECRET_OCID": "ocid1.vaultsecret.oc1..apm",
    }
    assert entrypoint.resolve_secrets(environ) == ["LOCUS_APM_DATA_KEY"]
    assert environ["LOCUS_APM_DATA_KEY"] == "content of ocid1.vaultsecret.oc1..apm"
    assert environ["OTEL_EXPORTER_OTLP_HEADERS"].startswith("Authorization=dataKey%20content of")
