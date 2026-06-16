# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for the model registry's ``oci:`` factory.

Verifies the OCI_PROFILE env-var fallback, family-based transport
routing, and that explicit kwargs take precedence over the env var.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from locus.models.providers.oci.openai_compat import OCIChatCompletionsModel
from locus.models.registry import get_model


COMPARTMENT_OCID = "ocid1.tenancy.oc1..registrytest"


@pytest.fixture
def mock_profile_load():
    """Mock the OCI profile loader so init never touches ``~/.oci/config``."""
    with patch(
        "locus.models.providers.oci.openai_compat._load_profile_config",
        return_value={"tenancy": COMPARTMENT_OCID},
    ) as m:
        yield m


class TestOCIRegistryEnvFallback:
    def test_env_profile_used_when_no_kwargs(self, monkeypatch, mock_profile_load):
        monkeypatch.setenv("OCI_PROFILE", "FROM_ENV")
        model = get_model("oci:openai.gpt-5.5")
        assert isinstance(model, OCIChatCompletionsModel)
        assert model.config.profile == "FROM_ENV"

    def test_explicit_profile_overrides_env(self, monkeypatch, mock_profile_load):
        monkeypatch.setenv("OCI_PROFILE", "FROM_ENV")
        model = get_model("oci:openai.gpt-5.5", profile="EXPLICIT")
        assert model.config.profile == "EXPLICIT"

    def test_explicit_auth_type_skips_env(self, monkeypatch):
        monkeypatch.setenv("OCI_PROFILE", "FROM_ENV")
        model = get_model(
            "oci:openai.gpt-5.5",
            auth_type="instance_principal",
            compartment_id=COMPARTMENT_OCID,
        )
        # auth_type path: profile must remain unset.
        assert model.config.profile is None
        assert model.config.auth_type == "instance_principal"

    def test_no_env_no_kwargs_raises_clean_value_error(self, monkeypatch):
        monkeypatch.delenv("OCI_PROFILE", raising=False)
        with pytest.raises(ValueError, match="specify exactly one"):
            get_model("oci:openai.gpt-5.5")

    def test_empty_env_treated_as_unset(self, monkeypatch):
        monkeypatch.setenv("OCI_PROFILE", "")
        with pytest.raises(ValueError, match="specify exactly one"):
            get_model("oci:openai.gpt-5.5")


class TestOCIRegistryFamilyRouting:
    def test_cohere_r_routes_to_sdk_transport(self, monkeypatch):
        # SDK transport accepts ``profile_name`` (not ``profile``) and
        # defaults to ``DEFAULT``, so the env-var fallback intentionally
        # does *not* fire on this branch — the SDK path stays as it was.
        monkeypatch.setenv("OCI_PROFILE", "FROM_ENV")
        from locus.models.providers.oci import OCIModel

        model = get_model("oci:cohere.command-r-plus")
        assert isinstance(model, OCIModel)
        # Confirm OCI_PROFILE was *not* injected as a kwarg (it would have
        # been rejected by OCIConfig). Default profile_name remains.
        assert model.config.profile_name == "DEFAULT"

    def test_openai_family_routes_to_v1(self, monkeypatch, mock_profile_load):
        monkeypatch.setenv("OCI_PROFILE", "FROM_ENV")
        model = get_model("oci:openai.gpt-5.5")
        assert isinstance(model, OCIChatCompletionsModel)

    def test_meta_family_routes_to_v1(self, monkeypatch, mock_profile_load):
        monkeypatch.setenv("OCI_PROFILE", "FROM_ENV")
        model = get_model("oci:meta.llama-3.3-70b-instruct")
        assert isinstance(model, OCIChatCompletionsModel)

    def test_non_r_cohere_routes_to_v1(self, monkeypatch, mock_profile_load):
        # cohere.command-a-* is not an R-series model — V1 transport.
        monkeypatch.setenv("OCI_PROFILE", "FROM_ENV")
        model = get_model("oci:cohere.command-a-03-2025")
        assert isinstance(model, OCIChatCompletionsModel)
