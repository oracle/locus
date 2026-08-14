# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the OpenAI-compatible provider table.

Before this table the registry knew two prefixes, ``openai`` and
``anthropic``, so anyone standardised on Ollama, vLLM, Groq, Together,
OpenRouter, DeepSeek, Mistral or xAI could not address their endpoint by
name at all. Each of those speaks the OpenAI wire protocol and differs only
in base URL and key variable, so they are all :class:`OpenAIModel` with a
different ``base_url``.

These tests guard the parts that are easy to get wrong:

- every table entry is actually registered, and the prefixes are unique;
- the closure binds per iteration (a bare closure over the loop variable
  gives every provider the *last* table entry — the classic late-binding
  bug, and it would be invisible until someone used a non-final provider);
- resolution precedence is explicit > TULIP_ override > vendor env > default;
- local servers work with no key, hosted ones fail with a message naming the
  variable to set;
- ``/v1/responses`` is never auto-selected against a custom base URL, which
  would break every gateway that serves only chat-completions.
"""

from __future__ import annotations

import sys

import pytest

from tulip.models.providers import (
    COMPATIBLE_PROVIDERS,
    CompatibleProvider,
    provider_table,
    register_compatible_providers,
)
from tulip.models.registry import get_model, list_providers


def test_every_table_entry_is_registered() -> None:
    registered = set(list_providers())
    for spec in COMPATIBLE_PROVIDERS:
        assert spec.prefix in registered, f"{spec.prefix} missing from the registry"


def test_native_providers_survive_registration() -> None:
    """The compatible table must not shadow the native prefixes."""
    registered = set(list_providers())
    assert {"openai", "anthropic"} <= registered


def test_prefixes_are_unique() -> None:
    prefixes = [s.prefix for s in COMPATIBLE_PROVIDERS]
    assert len(prefixes) == len(set(prefixes))


def test_each_provider_gets_its_own_base_url() -> None:
    """Guards the late-binding closure bug.

    If the factory closed over the loop variable instead of binding a
    default argument, every prefix would resolve to the last entry in the
    table. Constructing two different providers and comparing base URLs is
    the cheapest way to catch that.
    """
    seen: dict[str, str | None] = {}
    for spec in COMPATIBLE_PROVIDERS:
        if spec.base_url is None:  # the escape hatch needs an explicit URL
            continue
        model = get_model(f"{spec.prefix}:some-model", api_key="k")
        seen[spec.prefix] = model.config.base_url
        assert model.config.base_url == spec.base_url

    # Distinct providers must not collapse onto one endpoint.
    assert len(set(seen.values())) == len(seen)


def test_explicit_base_url_wins_over_default() -> None:
    model = get_model("groq:llama-3.3-70b", api_key="k", base_url="https://example.test/v1")
    assert model.config.base_url == "https://example.test/v1"


def test_tulip_env_override_wins_over_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TULIP_OLLAMA_BASE_URL", "http://gpu-box:11434/v1")
    model = get_model("ollama:qwen3")
    assert model.config.base_url == "http://gpu-box:11434/v1"


def test_vendor_env_override_is_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TULIP_VLLM_BASE_URL", raising=False)
    monkeypatch.setenv("VLLM_BASE_URL", "http://serving:8000/v1")
    model = get_model("vllm:mistral-7b")
    assert model.config.base_url == "http://serving:8000/v1"


def test_tulip_override_beats_vendor_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_BASE_URL", "http://vendor:8000/v1")
    monkeypatch.setenv("TULIP_VLLM_BASE_URL", "http://tulip:8000/v1")
    model = get_model("vllm:mistral-7b")
    assert model.config.base_url == "http://tulip:8000/v1"


def test_local_provider_needs_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    model = get_model("ollama:qwen3")
    assert model.config.api_key  # a placeholder, but present


def test_hosted_provider_without_key_names_the_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        get_model("groq:llama-3.3-70b")


def test_provider_env_key_is_read(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-from-env")
    model = get_model("deepseek:deepseek-chat")
    assert model.config.api_key == "ds-from-env"


def test_explicit_key_wins_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-from-env")
    model = get_model("deepseek:deepseek-chat", api_key="explicit")
    assert model.config.api_key == "explicit"


def test_escape_hatch_requires_a_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TULIP_OPENAI_COMPATIBLE_BASE_URL", raising=False)
    with pytest.raises(ValueError, match="base_url"):
        get_model("openai-compatible:whatever")


def test_escape_hatch_accepts_an_explicit_base_url() -> None:
    model = get_model("openai-compatible:my-model", base_url="https://host.test/v1")
    assert model.config.base_url == "https://host.test/v1"


def test_responses_api_never_auto_selected_on_custom_base_url() -> None:
    """A gateway serves chat-completions, not ``/v1/responses``.

    ``api="auto"`` must stay on chat-completions whenever a base URL is set,
    including for model ids whose name would otherwise route to Responses.
    """
    for spec in COMPATIBLE_PROVIDERS:
        if spec.base_url is None:
            continue
        model = get_model(f"{spec.prefix}:gpt-5.6-turbo", api_key="k")
        assert model._use_responses_api() is False, spec.prefix


def test_model_id_with_a_colon_survives(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ollama tags look like ``qwen3:4b``; only the first colon splits."""
    monkeypatch.delenv("TULIP_OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    model = get_model("ollama:qwen3:4b")
    assert model.config.model == "qwen3:4b"


def test_registration_is_idempotent() -> None:
    before = len(list_providers())
    register_compatible_providers()
    assert len(list_providers()) == before


def test_provider_table_lists_every_prefix() -> None:
    table = provider_table()
    for spec in COMPATIBLE_PROVIDERS:
        assert f"`{spec.prefix}`" in table
    assert "`openai`" in table
    assert "`anthropic`" in table


def test_tulip_base_url_env_name_is_shell_safe() -> None:
    """``openai-compatible`` must not produce a variable with a dash in it."""
    spec = CompatibleProvider(prefix="openai-compatible", base_url=None, env_key="X", label="X")
    assert spec.tulip_base_url_env == "TULIP_OPENAI_COMPATIBLE_BASE_URL"


# --------------------------------------------------------------------------
# Registration degrades gracefully when an optional dependency is absent.
# Each provider block is wrapped in try/except ImportError precisely so that
# `pip install tulip-agents` with no extras still imports; these tests
# exercise those branches instead of leaving them as untested prose.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("blocked", "absent_prefix", "still_present"),
    [
        ("tulip.models.native.openai", "openai", "anthropic"),
        ("tulip.models.native.anthropic", "anthropic", "openai"),
    ],
)
def test_a_missing_provider_dep_does_not_break_registration(
    monkeypatch: pytest.MonkeyPatch,
    blocked: str,
    absent_prefix: str,
    still_present: str,
) -> None:
    """One uninstalled provider must not take the registry down with it."""
    from tulip.models import registry

    monkeypatch.setattr(registry, "_PROVIDERS", {})
    # ``None`` in sys.modules makes ``import`` raise ImportError, which is
    # what an uninstalled extra looks like from inside the try block.
    monkeypatch.setitem(sys.modules, blocked, None)

    registry._register_defaults()

    assert absent_prefix not in registry._PROVIDERS
    assert still_present in registry._PROVIDERS


def test_missing_compatible_table_does_not_break_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The native providers survive even if the compatible table is gone."""
    from tulip.models import registry

    monkeypatch.setattr(registry, "_PROVIDERS", {})
    monkeypatch.setitem(sys.modules, "tulip.models.providers", None)

    registry._register_defaults()

    assert {"openai", "anthropic"} <= set(registry._PROVIDERS)
    assert "groq" not in registry._PROVIDERS


def test_registry_is_restored_after_the_degradation_tests() -> None:
    """Guard against the monkeypatched registry leaking into other tests."""
    from tulip.models.registry import list_providers

    assert len(list_providers()) >= 18
