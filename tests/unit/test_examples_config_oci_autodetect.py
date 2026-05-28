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


# --- Integration: get_model() end-to-end ------------------------------------
#
# These exercise the full harness flow — auto-detect provider, route to the
# OCI V1 transport builder, call the resolver, hand the resolved profile name
# to ``OCIChatCompletionsModel(...)``. The SDK constructor is mocked so the
# test stays offline; the assertion is on the kwargs that flow into it.


class TestGetModelIntegration:
    def _patch_oci_v1_model(self, monkeypatch):
        """Replace ``OCIChatCompletionsModel`` so we capture its kwargs without
        triggering the real OCI SDK file-parsing path.
        """
        captured: dict[str, object] = {}

        class _StubOCIChatCompletionsModel:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def __repr__(self):
                return "<StubOCIChatCompletionsModel>"

        # Patch at the import site used by _get_oci_v1_model.
        import locus.models.providers.oci as oci_pkg

        monkeypatch.setattr(
            oci_pkg, "OCIChatCompletionsModel", _StubOCIChatCompletionsModel, raising=True
        )
        return captured

    def test_default_section_routes_to_v1_with_default_profile(self, isolated_home, monkeypatch):
        _write_oci_config(isolated_home, "[DEFAULT]\nuser=u\n")
        captured = self._patch_oci_v1_model(monkeypatch)
        examples_config.get_model()
        assert captured["profile"] == "DEFAULT"
        # And the auto-derived call shape stays intact.
        assert "region" in captured
        assert "model" in captured

    def test_single_named_profile_routes_to_v1_with_that_profile(self, isolated_home, monkeypatch):
        _write_oci_config(isolated_home, "[BOAT-OC1]\nuser=u\n")
        captured = self._patch_oci_v1_model(monkeypatch)
        examples_config.get_model()
        assert captured["profile"] == "BOAT-OC1"

    def test_oci_profile_env_overrides_default_section(self, isolated_home, monkeypatch):
        _write_oci_config(isolated_home, "[DEFAULT]\nuser=u\n[OTHER]\nuser=u\n")
        monkeypatch.setenv("OCI_PROFILE", "OTHER")
        captured = self._patch_oci_v1_model(monkeypatch)
        examples_config.get_model()
        assert captured["profile"] == "OTHER"

    def test_multi_profile_no_default_raises_friendly_error(self, isolated_home, monkeypatch):
        _write_oci_config(
            isolated_home,
            "[PROFILE_A]\nuser=u\n[PROFILE_B]\nuser=u\n",
        )
        # OCI_PROFILE unset (isolated_home cleared it). Auto-detect picks OCI
        # because the file has parseable profiles, then the resolver fails
        # because there's no [DEFAULT] and no env override.
        self._patch_oci_v1_model(monkeypatch)
        with pytest.raises(RuntimeError) as exc:
            examples_config.get_model()
        msg = str(exc.value)
        assert "PROFILE_A" in msg
        assert "PROFILE_B" in msg
        assert "OCI_PROFILE" in msg

    def test_no_config_falls_back_to_mock(self, isolated_home, monkeypatch):
        # No ~/.oci/config and no workload-identity env → auto-detect returns
        # mock, so OCIChatCompletionsModel is never touched.
        result = examples_config.get_model()
        # MockModel is a pydantic BaseModel defined in examples/config.py.
        assert type(result).__name__ == "MockModel"

    def test_workbench_env_export_pattern_is_honoured(self, isolated_home, monkeypatch):
        """The workbench backend exports ``LOCUS_OCI_PROFILE`` and
        ``OCI_PROFILE`` into the notebook subprocess env (see
        ``workbench/backend/runner.py``). Confirm both names take precedence
        over file-based resolution so the workbench path is unaffected by
        the new detector.
        """
        _write_oci_config(isolated_home, "[A]\nuser=u\n[B]\nuser=u\n")
        monkeypatch.setenv("LOCUS_OCI_PROFILE", "WORKBENCH_PICK")
        captured = self._patch_oci_v1_model(monkeypatch)
        examples_config.get_model()
        assert captured["profile"] == "WORKBENCH_PICK"


# --- Integration: all three OCI transports honour the resolver --------------
#
# The previous TestGetModelIntegration class only exercises the V1 path. The
# resolver is documented as transport-agnostic (V1, SDK, Responses) — that's
# the load-bearing claim, so it gets its own targeted tests.


class TestAllThreeTransportsHonourResolver:
    def test_v1_transport_gets_resolved_profile(self, isolated_home, monkeypatch):
        """Cohere-free, non-DAC model id → V1 builder
        (``OCIChatCompletionsModel``)."""
        _write_oci_config(isolated_home, "[ONE]\nuser=u\n")
        monkeypatch.setenv("LOCUS_MODEL_ID", "openai.gpt-5.5-2026-04-23")

        captured: dict[str, object] = {}

        class _StubV1:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        import locus.models.providers.oci as oci_pkg

        monkeypatch.setattr(oci_pkg, "OCIChatCompletionsModel", _StubV1, raising=True)

        examples_config.get_model()
        assert captured["profile"] == "ONE"

    def test_sdk_transport_gets_resolved_profile(self, isolated_home, monkeypatch):
        """Cohere R-series model id → SDK builder (``OCIModel``).

        ``_pick_oci_transport`` routes ``cohere.command-r-*`` through
        ``_get_oci_sdk_model``, which now also calls ``_resolve_oci_profile``
        instead of hard-coding ``"DEFAULT"`` on the api-key path.
        """
        _write_oci_config(isolated_home, "[SDK_PROFILE]\nuser=u\n")
        monkeypatch.setenv("LOCUS_MODEL_ID", "cohere.command-r-plus-08-2024")

        captured: dict[str, object] = {}

        class _StubSdk:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        # ``_get_oci_sdk_model`` does ``from locus.models import OCIModel``
        # so we patch at that import location.
        import locus.models as locus_models

        monkeypatch.setattr(locus_models, "OCIModel", _StubSdk, raising=True)

        examples_config.get_model()
        # SDK transport uses ``profile_name=`` (not ``profile=``) — confirm
        # the resolved name reaches that exact kwarg.
        assert captured["profile_name"] == "SDK_PROFILE"

    def test_responses_transport_accepts_resolver_output(self, isolated_home):
        """Responses transport contract check.

        ``OCIResponsesModel`` is not auto-routed by the harness — it's an
        opt-in via direct construction. But its ``profile=`` kwarg has the
        same str-or-None contract as V1, so the resolver's output must be
        a usable input to it.

        We can't import OCIResponsesModel and construct it for real (it
        would try to read the (fake) config and fail on missing keys), so
        we assert the contract: the resolver returns a non-empty ``str``
        that matches the type expected by all three transports'
        ``__init__`` signatures (which we verified by inspection at
        src/locus/models/providers/oci/responses.py:148).
        """
        _write_oci_config(isolated_home, "[RESP_PROFILE]\nuser=u\n")
        result = examples_config._resolve_oci_profile()
        assert isinstance(result, str)
        assert result == "RESP_PROFILE"
        # The Responses constructor's ``profile`` kwarg is ``str | None``;
        # the resolver never returns None.
        assert result is not None
        assert len(result) > 0

    def test_sdk_workload_identity_keeps_default_fallback(self, isolated_home, monkeypatch):
        """When ``LOCUS_OCI_AUTH_TYPE=instance_principal``, the SDK builder
        does NOT call the resolver — the profile_name field is unused at
        signing time, so we keep the literal ``"DEFAULT"`` fallback. This
        test pins that compatibility shim.
        """
        # No ~/.oci/config on disk — workload identity doesn't need one.
        monkeypatch.setenv("LOCUS_OCI_AUTH_TYPE", "instance_principal")
        monkeypatch.setenv("LOCUS_OCI_COMPARTMENT", "ocid1.compartment.oc1..test")
        monkeypatch.setenv("LOCUS_MODEL_ID", "cohere.command-r-plus-08-2024")

        captured: dict[str, object] = {}

        class _StubSdk:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        import locus.models as locus_models

        monkeypatch.setattr(locus_models, "OCIModel", _StubSdk, raising=True)

        examples_config.get_model()
        # Resolver was not invoked — no file to parse and no env override,
        # but the builder still produced a model with ``profile_name``
        # populated by the workload-identity fallback path.
        assert captured["profile_name"] == "DEFAULT"
