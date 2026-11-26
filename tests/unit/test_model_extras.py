# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""A provider whose package is missing says what to install.

Model providers are extras, so ``pip install locus-sdk`` alone has no provider
client. The error for that has to name the fix, not a module.
"""

from __future__ import annotations

import importlib.util
import sys
from unittest.mock import patch

import pytest

from locus.models import registry
from locus.models._extras import missing_extra, require_packages
from locus.models.registry import get_model, list_providers


def test_missing_extra_names_the_package_and_the_install_command() -> None:
    error = missing_extra("OpenAIModel", "openai", "openai")
    assert isinstance(error, ImportError)
    assert error.name == "openai"
    assert str(error) == 'OpenAIModel needs the openai package: pip install "locus-sdk[openai]"'
    unnamed = missing_extra("The OCI provider", "oci", None)
    assert 'pip install "locus-sdk[oci]"' in str(unnamed)


def test_require_packages_raises_for_the_first_one_missing() -> None:
    require_packages("Anything", "oci", ("json", "sys"))
    with pytest.raises(ModuleNotFoundError, match=r'pip install "locus-sdk\[oci\]"') as raised:
        require_packages("The OCI provider", "oci", ("json", "no_such_package_for_locus"))
    assert raised.value.name == "no_such_package_for_locus"


def test_oci_stays_registered_and_says_what_to_install(monkeypatch: pytest.MonkeyPatch) -> None:
    real_find_spec = importlib.util.find_spec

    def without_requests(name: str, *args: object, **kwargs: object) -> object:
        return None if name == "requests" else real_find_spec(name, *args, **kwargs)  # type: ignore[arg-type]

    for name in [m for m in sys.modules if m.startswith("locus.models.providers.oci")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(importlib.util, "find_spec", without_requests)
    try:
        registry._register_defaults()
        assert "oci" in list_providers()
        with pytest.raises(ModuleNotFoundError) as raised:
            get_model("oci:openai.gpt-5.5", profile="DEFAULT")
        assert str(raised.value) == (
            'The OCI provider needs the requests package: pip install "locus-sdk[oci]"'
        )
        assert raised.value.name == "requests"
    finally:
        monkeypatch.undo()
        registry._register_defaults()
    from locus.models.providers.oci import OCIChatCompletionsModel

    assert isinstance(get_model("oci:openai.gpt-5.5", profile="DEFAULT"), OCIChatCompletionsModel)


def test_openai_model_without_the_client_says_what_to_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from locus.models.native.openai import OpenAIModel

    model = OpenAIModel(model="gpt-4o", api_key="test")
    monkeypatch.setitem(sys.modules, "openai", None)
    with pytest.raises(ModuleNotFoundError) as raised:
        _ = model.client
    assert str(raised.value) == (
        'OpenAIModel needs the openai package: pip install "locus-sdk[openai]"'
    )


def test_oci_v1_model_without_the_client_says_what_to_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from locus.models.providers.oci import OCIChatCompletionsModel

    with patch(
        "locus.models.providers.oci.openai_compat._load_profile_config",
        return_value={"tenancy": "ocid1.tenancy.oc1..example"},
    ):
        model = OCIChatCompletionsModel(model="openai.gpt-5.5", profile="MY_PROFILE")
    monkeypatch.setitem(sys.modules, "openai", None)
    with pytest.raises(ModuleNotFoundError) as raised:
        _ = model.client
    assert str(raised.value) == (
        'OCIChatCompletionsModel needs the openai package: pip install "locus-sdk[oci]"'
    )


def test_anthropic_model_without_the_client_says_what_to_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from locus.models.native.anthropic import AnthropicModel

    model = AnthropicModel(model="claude-sonnet-5-5", api_key="test")
    monkeypatch.setitem(sys.modules, "anthropic", None)
    with pytest.raises(ModuleNotFoundError) as raised:
        _ = model.client
    assert str(raised.value) == (
        'AnthropicModel needs the anthropic package: pip install "locus-sdk[anthropic]"'
    )
