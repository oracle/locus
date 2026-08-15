# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Provider configuration travelling with the one-string model form.

``model="provider:name"`` is the documented, headline way to pick a model, but
``AgentConfig`` sets ``extra="forbid"``, so nothing could ride along with it.
That was fine while every provider read its whole configuration from the
environment and stopped being fine the moment ``openai-compatible:`` shipped —
that prefix *requires* a ``base_url``, so through ``Agent(model=...)`` it was
reachable only via ``TULIP_OPENAI_COMPATIBLE_BASE_URL``. Same for pointing
``vllm:`` or ``ollama:`` at another host, or using a per-agent key.

``model_kwargs`` closes that without giving up ``extra="forbid"``, which is
worth keeping: it turns a misspelled field into an error rather than a setting
that silently does nothing.
"""

from __future__ import annotations

from typing import Any

import pytest

from tulip.agent import Agent
from tulip.agent.config import AgentConfig
from tulip.testing import ScriptedModel, text


# --------------------------------------------------------------------------
# The gap the field exists to close
# --------------------------------------------------------------------------


def test_provider_config_reaches_the_provider() -> None:
    agent = Agent(
        model="openai-compatible:my-model",
        model_kwargs={"base_url": "https://host.example/v1", "api_key": "k"},
    )
    agent._initialize()

    assert agent._model.config.base_url == "https://host.example/v1"
    assert agent._model.config.model == "my-model"


def test_an_openai_compatible_endpoint_is_usable_without_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The concrete regression: this prefix has no usable default base_url."""
    monkeypatch.delenv("TULIP_OPENAI_COMPATIBLE_BASE_URL", raising=False)

    agent = Agent(
        model="openai-compatible:qwen3.6-35b",
        model_kwargs={"base_url": "http://127.0.0.1:8000/v1"},
    )
    agent._initialize()

    assert agent._model.config.base_url == "http://127.0.0.1:8000/v1"


def test_arbitrary_provider_settings_are_forwarded() -> None:
    """Not an allow-list — whatever the provider accepts, it receives."""
    agent = Agent(
        model="openai:gpt-4o",
        model_kwargs={"api_key": "k", "temperature": 0.0, "max_tokens": 64},
    )
    agent._initialize()

    assert agent._model.config.temperature == 0.0
    assert agent._model.config.max_tokens == 64


def test_the_default_is_empty_and_changes_nothing() -> None:
    assert AgentConfig(model="openai:gpt-4o").model_kwargs == {}


# --------------------------------------------------------------------------
# The two ways to get it wrong
# --------------------------------------------------------------------------


def test_kwargs_alongside_a_built_model_is_an_error_not_a_silent_no_op() -> None:
    """An already-built model carries its own config, so these would vanish.

    Dropping them quietly is the worst outcome available: the agent runs
    against the wrong endpoint and every log line looks healthy.
    """
    with pytest.raises(ValueError, match="only when `model` is a provider string"):
        Agent(
            model=ScriptedModel([text("hi")]),
            model_kwargs={"base_url": "https://wrong.example/v1"},
        )


def test_an_unknown_field_says_where_provider_config_goes() -> None:
    """``extra="forbid"`` alone answers the wrong question.

    A reader who writes the documented string form and adds the ``base_url``
    that provider requires got "Extra inputs are not permitted" and no hint
    that the setting is supported one field over — which reads as a missing
    feature rather than a wrong spelling.
    """
    with pytest.raises(ValueError, match="model_kwargs"):
        Agent(model="openai-compatible:m", base_url="https://host.example/v1")


def test_a_plain_typo_still_fails() -> None:
    """The reason to keep ``extra="forbid"`` in the first place."""
    with pytest.raises(ValueError, match="max_iteration"):
        AgentConfig(model="openai:gpt-4o", max_iteration=3)


def test_validation_of_an_already_built_config_is_left_alone() -> None:
    """Pydantic revalidates model instances too; the check only reads dicts."""
    original = AgentConfig(model="openai:gpt-4o")
    assert AgentConfig.model_validate(original).model == "openai:gpt-4o"


def test_a_non_mapping_input_falls_through_to_pydantic() -> None:
    """The check must not swallow the error pydantic already words well.

    Reporting "[] are not fields of AgentConfig" for a string input would be
    both wrong and less helpful than what pydantic says on its own.
    """
    with pytest.raises(ValueError, match="valid dictionary or instance"):
        AgentConfig.model_validate("not a config")


def test_a_known_field_is_unaffected() -> None:
    assert AgentConfig(model="openai:gpt-4o", max_iterations=3).max_iterations == 3


# --------------------------------------------------------------------------
# It must not disturb the paths that already worked
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_model_instance_still_runs() -> None:
    agent = Agent(model=ScriptedModel([text("done")]))
    assert (await agent.arun("hello")).message == "done"


def test_the_auxiliary_model_is_left_alone() -> None:
    """``model_kwargs`` is scoped to the primary, and deliberately so.

    An auxiliary is usually a cheap model at a different provider; forwarding
    a self-hosted ``base_url`` to it would point it at an endpoint that has
    never heard of it.
    """
    agent = Agent(
        model="openai-compatible:big",
        model_kwargs={"base_url": "http://127.0.0.1:8000/v1", "api_key": "k"},
        auxiliary_model="openai:gpt-4o-mini",
    )
    agent._initialize()

    assert agent._model.config.base_url == "http://127.0.0.1:8000/v1"
    assert agent._auxiliary_model.config.base_url != "http://127.0.0.1:8000/v1"


def test_config_round_trips_through_a_dict() -> None:
    """Serialised config is how a deployment carries this, so it has to survive."""
    original = AgentConfig(
        model="vllm:qwen3.6-35b", model_kwargs={"base_url": "http://gpu-1:8000/v1"}
    )
    restored: Any = AgentConfig(**original.model_dump())

    assert restored.model_kwargs == {"base_url": "http://gpu-1:8000/v1"}
