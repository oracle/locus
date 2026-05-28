# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for the OCI auto-detection in ``examples/config.py``.

Covers the three transports (V1 ``OCIChatCompletionsModel``, SDK
``OCIModel``, Responses ``OCIResponsesModel``) indirectly: the resolver
returns a profile-name string that is contract-compatible with all
three constructors, so we test the resolver in isolation rather than
constructing each model class.

Closes #276.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _REPO_ROOT / "examples" / "config.py"


def _load_examples_config():
    """Load ``examples/config.py`` as a module without putting examples/ on
    ``sys.path`` (which would shadow real packages when running other tests).
    """
    spec = importlib.util.spec_from_file_location("_examples_config_under_test", _CONFIG_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


examples_config = _load_examples_config()


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Point ``$HOME`` at an empty tmp dir and clear OCI-related env vars."""
    monkeypatch.setenv("HOME", str(tmp_path))
    for var in (
        "LOCUS_MODEL_PROVIDER",
        "LOCUS_OCI_PROFILE",
        "OCI_PROFILE",
        "LOCUS_OCI_AUTH_TYPE",
        "OCI_AUTH_TYPE",
    ):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


def _write_oci_config(home: Path, content: str) -> Path:
    oci_dir = home / ".oci"
    oci_dir.mkdir(parents=True, exist_ok=True)
    cfg = oci_dir / "config"
    cfg.write_text(content)
    return cfg


# --- _oci_config_profiles probe ---------------------------------------------


class TestProbeProfiles:
    def test_missing_file_returns_empty(self, isolated_home):
        assert examples_config._oci_config_profiles(isolated_home / ".oci" / "config") == []

    def test_default_section_surfaced(self, isolated_home):
        cfg = _write_oci_config(
            isolated_home,
            "[DEFAULT]\n"
            "user=ocid1.user.oc1..aaa\n"
            "fingerprint=xx\n"
            "tenancy=ocid1.tenancy.oc1..bbb\n"
            "region=us-chicago-1\n"
            "key_file=/tmp/x.pem\n",
        )
        assert examples_config._oci_config_profiles(cfg) == ["DEFAULT"]

    def test_multiple_named_sections(self, isolated_home):
        cfg = _write_oci_config(
            isolated_home,
            "[PROFILE_A]\nuser=u\n[PROFILE_B]\nuser=u\n",
        )
        assert examples_config._oci_config_profiles(cfg) == ["PROFILE_A", "PROFILE_B"]

    def test_default_and_named(self, isolated_home):
        cfg = _write_oci_config(
            isolated_home,
            "[DEFAULT]\nuser=u\n[NAMED]\nuser=u\n",
        )
        # DEFAULT always sorts first.
        assert examples_config._oci_config_profiles(cfg) == ["DEFAULT", "NAMED"]

    def test_malformed_file_returns_empty(self, isolated_home):
        cfg = _write_oci_config(isolated_home, "this is not ini syntax\n")
        # configparser raises MissingSectionHeaderError on bare lines; the
        # probe must swallow it and return [].
        assert examples_config._oci_config_profiles(cfg) == []


# --- _auto_detect_provider --------------------------------------------------


class TestAutoDetectProvider:
    def test_no_config_returns_mock(self, isolated_home):
        assert examples_config._auto_detect_provider() == "mock"

    def test_workload_identity_via_locus_env(self, isolated_home, monkeypatch):
        monkeypatch.setenv("LOCUS_OCI_AUTH_TYPE", "instance_principal")
        assert examples_config._auto_detect_provider() == "oci"

    def test_workload_identity_via_oci_env(self, isolated_home, monkeypatch):
        monkeypatch.setenv("OCI_AUTH_TYPE", "resource_principal")
        assert examples_config._auto_detect_provider() == "oci"

    def test_workload_identity_ignores_non_workload_value(self, isolated_home, monkeypatch):
        # api_key/security_token are *not* workload-identity hints; without
        # a config file we still fall back to mock.
        monkeypatch.setenv("OCI_AUTH_TYPE", "api_key")
        assert examples_config._auto_detect_provider() == "mock"

    def test_default_profile_returns_oci(self, isolated_home):
        _write_oci_config(isolated_home, "[DEFAULT]\nuser=u\n")
        assert examples_config._auto_detect_provider() == "oci"

    def test_named_only_profile_returns_oci(self, isolated_home):
        _write_oci_config(isolated_home, "[MY_PROFILE]\nuser=u\n")
        assert examples_config._auto_detect_provider() == "oci"

    def test_empty_config_falls_back_to_mock(self, isolated_home):
        _write_oci_config(isolated_home, "")
        assert examples_config._auto_detect_provider() == "mock"

    def test_malformed_config_falls_back_to_mock(self, isolated_home):
        _write_oci_config(isolated_home, "garbage with no sections\n")
        assert examples_config._auto_detect_provider() == "mock"


# --- _resolve_oci_profile ---------------------------------------------------


class TestResolveOciProfile:
    def test_oci_profile_env_wins_without_file(self, isolated_home, monkeypatch):
        monkeypatch.setenv("OCI_PROFILE", "FROM_ENV")
        assert examples_config._resolve_oci_profile() == "FROM_ENV"

    def test_locus_oci_profile_env_wins(self, isolated_home, monkeypatch):
        monkeypatch.setenv("LOCUS_OCI_PROFILE", "LOCUS_PICK")
        assert examples_config._resolve_oci_profile() == "LOCUS_PICK"

    def test_env_takes_precedence_over_default_section(self, isolated_home, monkeypatch):
        _write_oci_config(isolated_home, "[DEFAULT]\nuser=u\n[OTHER]\nuser=u\n")
        monkeypatch.setenv("OCI_PROFILE", "OTHER")
        assert examples_config._resolve_oci_profile() == "OTHER"

    def test_default_section_picked(self, isolated_home):
        _write_oci_config(isolated_home, "[DEFAULT]\nuser=u\n[B]\nuser=u\n")
        assert examples_config._resolve_oci_profile() == "DEFAULT"

    def test_single_named_profile_picked(self, isolated_home):
        _write_oci_config(isolated_home, "[ONLY_ONE]\nuser=u\n")
        assert examples_config._resolve_oci_profile() == "ONLY_ONE"

    def test_multiple_no_default_raises_clear_error(self, isolated_home):
        _write_oci_config(
            isolated_home,
            "[A]\nuser=u\n[B]\nuser=u\n[C]\nuser=u\n",
        )
        with pytest.raises(RuntimeError) as exc:
            examples_config._resolve_oci_profile()
        msg = str(exc.value)
        # Every candidate is named so the user can pick.
        assert "A" in msg
        assert "B" in msg
        assert "C" in msg
        # And the user is told how to fix it.
        assert "OCI_PROFILE" in msg

    def test_no_config_raises_clearly(self, isolated_home):
        with pytest.raises(RuntimeError) as exc:
            examples_config._resolve_oci_profile()
        assert "No usable profiles" in str(exc.value)

    def test_returns_string_compatible_with_all_transports(self, isolated_home):
        """The resolver's return value must be a plain ``str`` — that's the
        contract every transport's ``profile=`` / ``profile_name=`` kwarg
        relies on (V1 ``OCIChatCompletionsModel``, SDK ``OCIModel``,
        Responses ``OCIResponsesModel``).
        """
        _write_oci_config(isolated_home, "[DEFAULT]\nuser=u\n")
        result = examples_config._resolve_oci_profile()
        assert isinstance(result, str)
        assert result == "DEFAULT"
