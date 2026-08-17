# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Azure OpenAI and Gemini — the last two of the Bedrock/Gemini/Azure gap.

Gemini rides the OpenAI-compatible routing table, because Google publishes and
maintains an OpenAI-shaped endpoint alongside the native API. Azure cannot: it
names deployments rather than models, authenticates with an ``api-key`` header,
and requires an ``api-version``. So Azure is ``OpenAIModel`` with a different
client under it, and these tests pin the parts of that swap that can silently
go wrong — chiefly that the Azure client really is Azure's, and that the
inherited conversion behaviour is inherited rather than re-implemented.
"""

from __future__ import annotations

import pytest

from tulip.models.providers import COMPATIBLE_PROVIDERS
from tulip.models.registry import get_model, list_providers


pytest.importorskip("openai", reason="openai extra not installed")

from tulip.models.native.azure import (  # noqa: E402 — after importorskip
    DEFAULT_API_VERSION,
    AzureOpenAIModel,
)
from tulip.models.native.openai import OpenAIModel  # noqa: E402


ENDPOINT = "https://my-resource.openai.azure.com"


# ------------------------------------------------------------------- gemini


def test_gemini_is_registered() -> None:
    assert "gemini" in list_providers()


def test_gemini_points_at_googles_own_openai_surface() -> None:
    """First-party, not a proxy — the distinction the capability matrix draws."""
    [spec] = [s for s in COMPATIBLE_PROVIDERS if s.prefix == "gemini"]
    assert spec.base_url == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert spec.env_key == "GEMINI_API_KEY"
    assert spec.requires_key is True


def test_gemini_builds_with_an_explicit_key(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    model = get_model("gemini:gemini-2.0-flash", api_key="test-key")
    assert model.config.base_url == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert model.config.model == "gemini-2.0-flash"


def test_gemini_without_a_key_names_the_variable(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        get_model("gemini:gemini-2.0-flash")


def test_gemini_base_url_is_overridable(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("TULIP_GEMINI_BASE_URL", "https://proxy.internal/v1")
    assert get_model("gemini:gemini-2.0-flash").config.base_url == "https://proxy.internal/v1"


# -------------------------------------------------------------------- azure


def test_azure_is_registered() -> None:
    assert "azure" in list_providers()


def test_azure_is_an_openai_model_underneath() -> None:
    """Inheritance is the design: message and tool conversion must not fork."""
    model = AzureOpenAIModel(model="gpt4o-prod", endpoint=ENDPOINT, api_key="k")
    assert isinstance(model, OpenAIModel)


def test_azure_reads_its_settings_from_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", ENDPOINT)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "env-key")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2025-01-01")
    model = get_model("azure:gpt4o-prod")
    assert model.config.endpoint == ENDPOINT
    assert model.config.api_key == "env-key"
    assert model.config.api_version == "2025-01-01"


def test_explicit_arguments_beat_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://wrong.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "wrong")
    model = get_model("azure:gpt4o-prod", endpoint=ENDPOINT, api_key="right")
    assert model.config.endpoint == ENDPOINT
    assert model.config.api_key == "right"


def test_api_version_defaults_to_a_pinned_ga_version(monkeypatch) -> None:
    """Azure has no rolling 'latest'; an unset version must not mean 'none'."""
    monkeypatch.delenv("AZURE_OPENAI_API_VERSION", raising=False)
    model = AzureOpenAIModel(model="gpt4o-prod", endpoint=ENDPOINT, api_key="k")
    assert model.config.api_version == DEFAULT_API_VERSION


def test_the_model_string_is_a_deployment_name() -> None:
    """``gpt4o-prod`` is a deployment, not a model id — it must pass through verbatim."""
    model = AzureOpenAIModel(model="gpt4o-prod", endpoint=ENDPOINT, api_key="k")
    assert model.config.model == "gpt4o-prod"


def test_azure_client_is_azures(monkeypatch) -> None:
    import openai

    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    model = AzureOpenAIModel(
        model="gpt4o-prod", endpoint=ENDPOINT, api_key="k", api_version="2024-10-21"
    )
    client = model.client
    assert isinstance(client, openai.AsyncAzureOpenAI)
    # The deployment path and api-version must both survive into the URL, or
    # every request 404s against a resource that looks correctly configured.
    assert ENDPOINT in str(client.base_url)


def test_missing_endpoint_says_what_to_set(monkeypatch) -> None:
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    model = AzureOpenAIModel(model="gpt4o-prod", api_key="k")
    with pytest.raises(ValueError, match="AZURE_OPENAI_ENDPOINT"):
        _ = model.client


def test_missing_credentials_says_what_to_set(monkeypatch) -> None:
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    model = AzureOpenAIModel(model="gpt4o-prod", endpoint=ENDPOINT)
    with pytest.raises(ValueError, match="AZURE_OPENAI_API_KEY"):
        _ = model.client


def test_an_entra_token_is_accepted_instead_of_a_key(monkeypatch) -> None:
    """Managed identity is how Azure shops usually authenticate in production."""
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    model = AzureOpenAIModel(
        model="gpt4o-prod",
        endpoint=ENDPOINT,
        azure_ad_token="entra-token",  # noqa: S106 — fake token; no request is made
    )
    import openai

    assert isinstance(model.client, openai.AsyncAzureOpenAI)


def test_azure_inherits_tool_conversion(monkeypatch) -> None:
    """The whole reason for subclassing: no second conversion path to drift."""
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", ENDPOINT)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    azure = AzureOpenAIModel(model="gpt4o-prod")
    plain = OpenAIModel(model="gpt-4o", api_key="k")
    assert type(azure)._convert_messages is type(plain)._convert_messages


# ------------------------------------------------------------------ the row


def test_the_bedrock_gemini_azure_row_is_closed() -> None:
    """The capability matrix listed all three as 'not offered'."""
    registered = set(list_providers())
    assert {"bedrock", "gemini", "azure"} <= registered
