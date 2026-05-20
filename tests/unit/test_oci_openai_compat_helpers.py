# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Coverage tests for the *helper* functions in
``locus.models.providers.oci.openai_compat``.

The existing ``test_oci_openai_compat.py`` exercises the public API
(``OCIChatCompletionsModel.__init__``, ``client``) but stubs the helper functions
out, so the helper bodies are never executed. This file invokes each
helper directly and patches the real ``oci`` SDK calls they depend on.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest


pytest.importorskip("oci")

from locus.models.providers.oci import openai_compat as oc  # noqa: E402


# ---------------------------------------------------------------------------
# _load_profile_config
# ---------------------------------------------------------------------------


class TestLoadProfileConfig:
    def test_calls_oci_config_from_file(self) -> None:
        fake_cfg = {
            "tenancy": "ocid1.tenancy.oc1..t",
            "user": "ocid1.user.oc1..u",
        }
        with patch("oci.config.from_file", return_value=fake_cfg) as mock_from_file:
            cfg = oc._load_profile_config("MYPROFILE", "~/.oci/config")
        assert cfg["tenancy"] == "ocid1.tenancy.oc1..t"
        # Path is expanduser'd before being passed in.
        call_path = mock_from_file.call_args.args[0]
        assert call_path.endswith(".oci/config")
        # Profile name forwarded as kwarg
        assert mock_from_file.call_args.kwargs["profile_name"] == "MYPROFILE"


# ---------------------------------------------------------------------------
# _build_signer_from_profile — picks user vs session
# ---------------------------------------------------------------------------


class TestBuildSignerFromProfile:
    def test_picks_user_principal_when_no_token_file(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg_no_token = {
            "tenancy": "ocid1.tenancy.oc1..t",
            "user": "ocid1.user.oc1..u",
            "fingerprint": "fp",
            "key_file": "/tmp/key.pem",
        }
        sentinel = MagicMock(name="user-signer")
        with (
            patch.object(oc, "_load_profile_config", return_value=cfg_no_token),
            patch.object(oc, "_build_user_principal_signer", return_value=sentinel) as mock_user,
            patch.object(oc, "_build_session_signer") as mock_session,
        ):
            out = oc._build_signer_from_profile("P", "~/.oci/config")
        assert out is sentinel
        mock_user.assert_called_once_with(cfg_no_token)
        mock_session.assert_not_called()

    def test_picks_session_when_token_file_present(self) -> None:
        cfg_with_token = {
            "tenancy": "ocid1.tenancy.oc1..t",
            "key_file": "/tmp/key.pem",
            "security_token_file": "/tmp/token",
        }
        sentinel = MagicMock(name="session-signer")
        with (
            patch.object(oc, "_load_profile_config", return_value=cfg_with_token),
            patch.object(oc, "_build_session_signer", return_value=sentinel) as mock_session,
            patch.object(oc, "_build_user_principal_signer") as mock_user,
        ):
            out = oc._build_signer_from_profile("P", "~/.oci/config")
        assert out is sentinel
        mock_session.assert_called_once_with(cfg_with_token)
        mock_user.assert_not_called()


# ---------------------------------------------------------------------------
# _build_user_principal_signer / _build_session_signer (direct)
# ---------------------------------------------------------------------------


class TestDirectUserPrincipalSigner:
    def test_calls_oci_signer_with_correct_kwargs(self) -> None:
        cfg = {
            "tenancy": "ocid1.tenancy.oc1..t",
            "user": "ocid1.user.oc1..u",
            "fingerprint": "fp",
            "key_file": "/tmp/key.pem",
            "pass_phrase": "secret",
        }
        with patch("oci.signer.Signer") as mock_signer:
            mock_signer.return_value = MagicMock(name="user-signer-instance")
            signer = oc._build_user_principal_signer(cfg)
        mock_signer.assert_called_once_with(
            tenancy="ocid1.tenancy.oc1..t",
            user="ocid1.user.oc1..u",
            fingerprint="fp",
            private_key_file_location="/tmp/key.pem",
            pass_phrase="secret",  # noqa: S106
        )
        assert signer is mock_signer.return_value


class TestDirectSessionSigner:
    def test_loads_token_and_private_key(self, tmp_path: Any) -> None:
        token_path = tmp_path / "token"
        token_path.write_text("session-token-bytes")
        cfg = {
            "key_file": "/tmp/key.pem",
            "security_token_file": str(token_path),
            "pass_phrase": None,
        }
        with (
            patch("oci.auth.signers.SecurityTokenSigner") as mock_session,
            patch(
                "oci.signer.load_private_key_from_file", return_value="pk-loaded"
            ) as mock_load_pk,
        ):
            mock_session.return_value = MagicMock(name="session-instance")
            signer = oc._build_session_signer(cfg)
        mock_load_pk.assert_called_once_with("/tmp/key.pem", None)
        # SecurityTokenSigner gets ``token=...`` and ``private_key=...``
        kwargs = mock_session.call_args.kwargs
        assert kwargs["token"] == "session-token-bytes"  # noqa: S105
        assert kwargs["private_key"] == "pk-loaded"
        assert signer is mock_session.return_value


# ---------------------------------------------------------------------------
# Instance / resource principal builders
# ---------------------------------------------------------------------------


class TestPrincipalBuilders:
    def test_instance_principal(self) -> None:
        with patch("oci.auth.signers.InstancePrincipalsSecurityTokenSigner") as mock_inst:
            mock_inst.return_value = MagicMock(name="ip-signer")
            signer = oc._build_instance_principal_signer()
        mock_inst.assert_called_once_with()
        assert signer is mock_inst.return_value

    def test_resource_principal(self) -> None:
        with patch("oci.auth.signers.get_resource_principals_signer") as mock_rp:
            mock_rp.return_value = MagicMock(name="rp-signer")
            signer = oc._build_resource_principal_signer()
        mock_rp.assert_called_once_with()
        assert signer is mock_rp.return_value
