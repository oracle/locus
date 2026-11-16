# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""``auth_type="api_key"`` on the V1 OCI transports means the profile path.

It is the default of every other OCI auth switch in the SDK and of
``LOCUS_OCI_AUTH_TYPE``; the two V1 models rejected it, so the image
``locus deploy`` builds could not run locally with a profile.
"""

from __future__ import annotations

from typing import Any

import pytest

from locus.models.providers.oci.openai_compat import (
    PROFILE_AUTH_TYPES,
    OCIChatCompletionsModel,
    resolve_profile_auth,
)
from locus.models.providers.oci.responses import OCIResponsesModel


COMPARTMENT = "ocid1.compartment.oc1..aaaaaaaatest"


@pytest.fixture(autouse=True)
def _no_profile_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OCI_PROFILE", raising=False)
    monkeypatch.delenv("LOCUS_OCI_PROFILE", raising=False)
    monkeypatch.delenv("OCI_COMPARTMENT", raising=False)
    monkeypatch.delenv("OCI_COMPARTMENT_ID", raising=False)


def test_resolve_profile_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    assert {"api_key", "security_token", "session_token"} == PROFILE_AUTH_TYPES
    assert resolve_profile_auth(None, "api_key") == ("DEFAULT", None)
    monkeypatch.setenv("OCI_PROFILE", "FROM_ENV")
    assert resolve_profile_auth(None, "api_key") == ("FROM_ENV", None)
    monkeypatch.setenv("LOCUS_OCI_PROFILE", "LOCUS_FIRST")
    assert resolve_profile_auth(None, "security_token") == ("LOCUS_FIRST", None)
    assert resolve_profile_auth("MINE", "api_key") == ("MINE", None)
    # Principals and the plain profile path are left as given.
    assert resolve_profile_auth(None, "resource_principal") == (None, "resource_principal")
    assert resolve_profile_auth("MINE", None) == ("MINE", None)
    assert resolve_profile_auth(None, None) == (None, None)


@pytest.mark.parametrize("model_cls", [OCIChatCompletionsModel, OCIResponsesModel])
class TestApiKeyAuthType:
    def test_api_key_uses_the_env_profile(
        self, model_cls: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OCI_PROFILE", "FROM_ENV")
        model = model_cls(model="openai.gpt-4.1", auth_type="api_key", compartment_id=COMPARTMENT)
        assert model.config.profile == "FROM_ENV"
        assert model.config.auth_type is None
        assert model.config.compartment_id == COMPARTMENT

    def test_api_key_without_a_profile_anywhere_is_default(self, model_cls: Any) -> None:
        model = model_cls(model="openai.gpt-4.1", auth_type="api_key", compartment_id=COMPARTMENT)
        assert model.config.profile == "DEFAULT"
        assert model.config.auth_type is None

    def test_api_key_together_with_profile_is_one_mode(self, model_cls: Any) -> None:
        model = model_cls(
            model="openai.gpt-4.1",
            profile="MINE",
            auth_type="security_token",
            compartment_id=COMPARTMENT,
        )
        assert model.config.profile == "MINE"
        assert model.config.auth_type is None

    def test_principals_and_unknown_values_are_unchanged(self, model_cls: Any) -> None:
        model = model_cls(
            model="openai.gpt-4.1", auth_type="resource_principal", compartment_id=COMPARTMENT
        )
        assert model.config.auth_type == "resource_principal"
        assert model.config.profile is None
        with pytest.raises(ValueError, match="auth_type must be one of"):
            model_cls(model="openai.gpt-4.1", auth_type="bogus", compartment_id=COMPARTMENT)
        with pytest.raises(ValueError, match="exactly one"):
            model_cls(model="openai.gpt-4.1", profile="MINE", auth_type="resource_principal")


def test_the_oci_factory_passes_api_key_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """What the deploy template does: ``model_kwargs={"auth_type": <LOCUS_OCI_AUTH_TYPE>, ...}``."""
    from locus.models import get_model

    monkeypatch.setenv("OCI_PROFILE", "FROM_ENV")
    model = get_model("oci:openai.gpt-4.1", auth_type="api_key", compartment_id=COMPARTMENT)
    assert isinstance(model, OCIChatCompletionsModel)
    assert model.config.profile == "FROM_ENV"
    assert model.config.auth_type is None
