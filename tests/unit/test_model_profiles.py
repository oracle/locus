# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-model capability profiles: what a harness adapts its tools and prompt to.

The resolution order is the contract: an override beats metadata, metadata
beats the family heuristics, and an unknown id gets a conservative default.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tulip.models.metadata import ModelMetadata, register_metadata
from tulip.models.profiles import (
    FAMILIES,
    PROFILES_ENV,
    family_of,
    load_overrides,
    profile_for,
)


@pytest.fixture(autouse=True)
def _no_env_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(PROFILES_ENV, raising=False)


@pytest.mark.parametrize(
    ("model", "family"),
    [
        ("openai:gpt-5.5", "gpt"),
        ("gpt-4o-mini", "gpt"),
        ("o3", "gpt"),
        ("openrouter:openai/gpt-oss-120b", "gpt"),
        ("anthropic:claude-sonnet-5-5", "claude"),
        ("claude-3-5-sonnet-20241022", "claude"),
        ("gemini:gemini-3-pro", "gemini"),
        ("vllm:Qwen/Qwen3-Coder-30B", "qwen"),
        ("deepseek:deepseek-v4-pro", "deepseek"),
        ("groq:llama-3.3-70b-versatile", "llama"),
        ("mistral:devstral-small-2507", "mistral"),
        ("ollama:phi4", "default"),
    ],
)
def test_family_is_read_from_metadata_or_the_id(model: str, family: str) -> None:
    assert family_of(model) == family
    assert profile_for(model).family == family
    assert family in FAMILIES


def test_gpt_family_prefers_patches_and_reasoning_effort() -> None:
    profile = profile_for("openai:gpt-5.5")
    assert profile.edit_format == "apply_patch"
    assert profile.prompt_variant == "gpt"
    assert profile.reasoning == "reasoning_effort"
    assert profile.parallel_tool_calls
    assert profile.vision
    # From metadata, not from a second table.
    assert profile.context_window == 400_000
    assert profile.sources[0] == "metadata"


def test_non_reasoning_and_text_only_gpt_models_are_told_apart() -> None:
    assert profile_for("gpt-4o").reasoning is None
    assert not profile_for("o3-mini").vision
    oss = profile_for("vllm:gpt-oss-120b")
    assert not oss.parallel_tool_calls
    assert not oss.prompt_caching


@pytest.mark.parametrize(
    ("model", "reasoning"),
    [
        ("claude-opus-5-5", "adaptive"),
        ("claude-sonnet-5", "adaptive"),
        ("claude-sonnet-4-6", "adaptive"),
        ("claude-fable-5-1", "adaptive"),
        ("anthropic:claude-sonnet-4-5-20250929", "budget_tokens"),
        ("claude-haiku-4-5", "budget_tokens"),
        ("claude-3-7-sonnet-latest", "budget_tokens"),
    ],
)
def test_claude_thinking_follows_the_generation(model: str, reasoning: str) -> None:
    profile = profile_for(model)
    assert profile.reasoning == reasoning
    assert profile.edit_format == "str_replace"
    assert profile.prompt_variant == "claude"
    assert profile.vision
    assert profile.prompt_caching


def test_open_weight_models_are_conservative_until_told_otherwise() -> None:
    qwen = profile_for("vllm:qwen3.6-35b")
    assert not qwen.parallel_tool_calls
    assert not qwen.vision
    assert qwen.reasoning == "enable_thinking"
    assert qwen.context_window == 262_000
    assert profile_for("ollama:qwen2.5-vl-7b").vision
    assert profile_for("mistral:pixtral-large").vision
    assert profile_for("deepseek:deepseek-reasoner").reasoning == "always"


def test_an_unknown_model_gets_the_default() -> None:
    profile = profile_for("acme:mystery-1")
    assert profile.family == "default"
    assert profile.tool_calling
    assert not profile.parallel_tool_calls
    assert not profile.vision
    assert profile.context_window is None
    assert profile.sources == ("family:default",)


def test_metadata_caching_flag_reaches_the_profile() -> None:
    register_metadata(
        ModelMetadata(
            model_id="acme-cached-1",
            family="acme",
            context_length=32_000,
            max_output_tokens=4_000,
            supports_prompt_caching=True,
        )
    )
    profile = profile_for("acme-cached-1")
    assert profile.prompt_caching
    assert profile.max_output_tokens == 4_000


def test_an_override_mapping_beats_everything() -> None:
    profile = profile_for(
        "vllm:qwen3.6-35b",
        overrides={"vllm:qwen3.6-*": {"parallel_tool_calls": True, "vision": True}},
    )
    assert profile.parallel_tool_calls
    assert profile.vision
    assert profile.sources[0] == "override:vllm:qwen3.6-*"


def test_an_exact_override_wins_over_a_glob() -> None:
    profile = profile_for(
        "my-model",
        overrides={
            "my-*": {"edit_format": "whole_file"},
            "my-model": {"edit_format": "apply_patch"},
        },
    )
    assert profile.edit_format == "apply_patch"


def test_an_override_can_move_a_model_to_another_family() -> None:
    profile = profile_for("acme:mystery-1", overrides={"mystery-1": {"family": "gpt"}})
    assert profile.family == "gpt"


def test_the_override_file_from_the_environment_applies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps({"models": {"acme:*": {"vision": True, "context_window": 9}}}))
    monkeypatch.setenv(PROFILES_ENV, str(path))
    profile = profile_for("acme:mystery-1")
    assert profile.vision
    assert profile.context_window == 9


def test_an_override_file_can_be_passed_directly(tmp_path: Path) -> None:
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps({"mystery-1": {"prompt_variant": "gpt"}}))
    assert profile_for("acme:mystery-1", overrides=path).prompt_variant == "gpt"
    assert profile_for("acme:mystery-1", overrides=str(path)).prompt_variant == "gpt"


@pytest.mark.parametrize(
    "content",
    [
        "[1, 2]",
        json.dumps({"m": "not an object"}),
        json.dumps({"m": {"visoin": True}}),
    ],
)
def test_a_malformed_override_file_fails_loudly(tmp_path: Path, content: str) -> None:
    path = tmp_path / "profiles.json"
    path.write_text(content)
    with pytest.raises(ValueError):
        load_overrides(path)


def test_a_misspelt_override_field_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown field"):
        profile_for("x", overrides={"x": {"paralel_tool_calls": True}})


def test_the_profile_is_immutable() -> None:
    profile = profile_for("gpt-5.5")
    with pytest.raises(Exception):  # noqa: B017, PT011 — pydantic's frozen error
        profile.vision = False  # type: ignore[misc]
