# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""What a model can do, so a harness can adapt to it instead of assuming.

A coding agent built for one vendor's models breaks quietly on the others: it
sends parallel tool calls to a server that serialises them badly, asks for
``budget_tokens`` thinking on a model that takes an effort level, offers a
search-and-replace edit tool to a model trained on patches, or attaches an
image to a model that cannot see. Each of those is a fact about the model,
and every harness on top of the SDK needs the same facts.

:func:`profile_for` answers with a :class:`ModelProfile`. It is resolved in
layers, most specific first:

1. **Overrides** — a JSON file named by ``TULIP_MODEL_PROFILES``, or a mapping
   passed as ``overrides=``. Keys are model ids or ``fnmatch`` globs
   (``"vllm:my-finetune-*"``); values are any subset of the profile's fields.
   This is how a self-hosted fine-tune says it can call tools in parallel.
2. **Metadata** — :mod:`tulip.models.metadata` already knows the context
   window, the output cap and prompt caching for the models it seeds. The
   profile reads them from there rather than keeping a second table.
3. **Family** — the rest comes from the family the id belongs to (claude,
   gpt, gemini, qwen, deepseek, kimi, glm, llama, mistral), with a few per-generation
   refinements such as which Claude models take adaptive thinking.
4. **Default** — an OpenAI-compatible model nobody has described: tool
   calling yes, everything else conservative.

The profile describes; it does not configure. The SDK acts on two of its
facts — ``vision``, for where a tool result's images go, and
``leaked_tool_call_formats``, the call markup the agent loop recognises in a
message body — and a harness reads the rest and decides.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, Field

from tulip.models.metadata import metadata_for


__all__ = [
    "FAMILIES",
    "PROFILES_ENV",
    "EditFormat",
    "LeakedToolCallFormat",
    "ModelProfile",
    "family_of",
    "load_overrides",
    "profile_for",
]

#: The environment variable naming a JSON override file.
PROFILES_ENV: Final[str] = "TULIP_MODEL_PROFILES"

#: Families the heuristics know. ``default`` is everything else.
FAMILIES: Final[tuple[str, ...]] = (
    "claude",
    "gpt",
    "gemini",
    "qwen",
    "deepseek",
    "kimi",
    "glm",
    "llama",
    "mistral",
    "default",
)

#: How a model edits files best.
#:
#: ``str_replace``  exact old/new string replacement (the edit tool most
#:                  harnesses ship)
#: ``apply_patch``  a multi-file patch in the ``*** Begin Patch`` envelope —
#:                  what GPT-family models are trained to emit
#: ``whole_file``   rewriting the file; for models that fumble both of the
#:                  above
EditFormat = Literal["str_replace", "apply_patch", "whole_file"]


#: Tool-call markup a model may leak into its message body as text; the
#: parsers live in :mod:`tulip.agent.leaked_tool_calls`.
#:
#: ``dsml``      DeepSeek V3.2 / V4 (``<｜DSML｜tool_calls>``)
#: ``deepseek``  DeepSeek V3 / V3.1 (``<｜tool▁calls▁begin｜>``)
#: ``hermes``    ``<tool_call>{json}</tool_call>`` (Hermes, Qwen 2.5 / 3)
#: ``qwen_xml``  Qwen3-Coder (``<tool_call><function=…>``)
#: ``kimi``      Kimi K2 (``<|tool_calls_section_begin|>``)
#: ``glm``       GLM-4.5 / 4.6 (``<tool_call>name<arg_key>…``)
LeakedToolCallFormat = Literal["dsml", "deepseek", "hermes", "qwen_xml", "kimi", "glm"]


class ModelProfile(BaseModel):
    """The capabilities a harness adapts to, for one model."""

    model_config = {"frozen": True}

    model: str = Field(description="The model id the profile was resolved for.")
    family: str = Field(description="One of :data:`FAMILIES`.")
    context_window: int | None = Field(
        default=None, description="Input window in tokens; None when unknown."
    )
    max_output_tokens: int | None = Field(
        default=None, description="Output cap in tokens; None when unknown."
    )
    tool_calling: bool = Field(default=True, description="Whether the model calls tools natively.")
    parallel_tool_calls: bool = Field(
        default=False, description="Whether one response may carry several tool calls."
    )
    reasoning: str | None = Field(
        default=None,
        description=(
            "How extended reasoning is requested: 'adaptive' (Claude 4.6+ "
            "thinking), 'budget_tokens' (earlier Claude), 'reasoning_effort' "
            "(OpenAI reasoning models), 'thinking_budget' (Gemini), "
            "'enable_thinking' (Qwen3 chat template), 'always' (a model that "
            "always reasons, nothing to set), or None."
        ),
    )
    prompt_caching: bool = Field(
        default=False, description="Whether the provider caches prompt prefixes."
    )
    vision: bool = Field(default=False, description="Whether the model accepts images.")
    edit_format: EditFormat = Field(
        default="str_replace", description="The edit tool the model handles best."
    )
    leaked_tool_call_formats: tuple[LeakedToolCallFormat, ...] = Field(
        default=(),
        description=(
            "Tool-call markup the model writes into its message body even though "
            "it calls tools natively — its own training format, which a router "
            "or server sometimes fails to lift into structured calls. The agent "
            "loop recognises these as calls (see "
            ":mod:`tulip.agent.leaked_tool_calls`)."
        ),
    )
    prompt_variant: str = Field(
        default="default",
        description="Which system-prompt variant suits it; a harness maps this to text.",
    )
    sources: tuple[str, ...] = Field(
        default=(),
        description="Which layers contributed, most specific first, for debugging.",
    )


# ---------------------------------------------------------------------------
# Families
# ---------------------------------------------------------------------------

#: Metadata's vendor names, as profile families.
_METADATA_FAMILIES: Final[dict[str, str]] = {
    "anthropic": "claude",
    "openai": "gpt",
    "google": "gemini",
    "gemini": "gemini",
    "qwen": "qwen",
    "deepseek": "deepseek",
    "moonshot": "kimi",
    "kimi": "kimi",
    "zhipu": "glm",
    "glm": "glm",
    "meta": "llama",
    "llama": "llama",
    "mistral": "mistral",
}

#: Id patterns, checked in order, against the lower-cased id without its
#: routing prefix.
_FAMILY_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    ("claude", re.compile(r"claude|anthropic")),
    ("gpt", re.compile(r"(^|[:/])(gpt-|o\d(-|$)|codex|chatgpt)|openai/")),
    ("gemini", re.compile(r"gemini|gemma")),
    ("qwen", re.compile(r"qwen|qwq")),
    ("deepseek", re.compile(r"deepseek")),
    ("kimi", re.compile(r"kimi|moonshot")),
    ("glm", re.compile(r"glm|zhipu|z-ai/|chatglm")),
    ("llama", re.compile(r"llama")),
    ("mistral", re.compile(r"mistral|mixtral|codestral|devstral|pixtral|magistral")),
)

#: What each family is unless something more specific says otherwise.
_FAMILY_DEFAULTS: Final[dict[str, dict[str, Any]]] = {
    "claude": {
        "parallel_tool_calls": True,
        "prompt_caching": True,
        "vision": True,
        "reasoning": "budget_tokens",
        "prompt_variant": "claude",
    },
    "gpt": {
        "parallel_tool_calls": True,
        "prompt_caching": True,
        "vision": True,
        "edit_format": "apply_patch",
        "prompt_variant": "gpt",
    },
    "gemini": {
        "parallel_tool_calls": True,
        "prompt_caching": True,
        "vision": True,
        "reasoning": "thinking_budget",
        "prompt_variant": "gemini",
    },
    # Open-weight families are usually served by vLLM, llama.cpp or Ollama,
    # where parallel calls depend on the server's tool parser — off until an
    # override says the deployment handles them.
    #
    # ``leaked_tool_call_formats`` is each family's own call markup, which
    # reaches the message body as text when the server does not parse it —
    # observed with DeepSeek V4 through OpenRouter.
    "qwen": {
        "reasoning": "enable_thinking",
        "prompt_variant": "qwen",
        "leaked_tool_call_formats": ("hermes", "qwen_xml"),
    },
    "deepseek": {"prompt_variant": "deepseek", "leaked_tool_call_formats": ("dsml", "deepseek")},
    "kimi": {"prompt_variant": "default", "leaked_tool_call_formats": ("kimi",)},
    "glm": {"prompt_variant": "default", "leaked_tool_call_formats": ("glm",)},
    "llama": {"prompt_variant": "default"},
    "mistral": {"prompt_variant": "default"},
    "default": {"prompt_variant": "default"},
}

#: A Claude id's tier and version: ``claude-sonnet-4-6`` → (4, 6),
#: ``claude-opus-5`` → (5, None), ``claude-sonnet-4-5-20250929`` → (4, 5).
_CLAUDE_VERSION: Final[re.Pattern[str]] = re.compile(
    r"claude-(?:opus|sonnet|haiku|fable|mythos)-(\d+)(?:-(\d{1,2}))?(?:$|[-@.])"
)


def _claude_adaptive(model: str) -> bool:
    """Whether a Claude id takes adaptive thinking: 4.6 and every later model.

    Earlier ones (Haiku 4.5, the 4.5 and 4.1 Opus and Sonnet, the 3.x line)
    take ``budget_tokens``; an id this cannot read is treated as earlier,
    which is the form older models accept.
    """
    match = _CLAUDE_VERSION.search(model)
    if match is None:
        return False
    major = int(match.group(1))
    minor = int(match.group(2) or 0)
    return major >= 5 or (major == 4 and minor >= 6)


#: OpenAI models that reason and take ``reasoning_effort``.
_GPT_REASONING: Final[re.Pattern[str]] = re.compile(r"(^|[:/])(o\d|gpt-5)")

#: Text-only GPT-family ids.
_GPT_NO_VISION: Final[re.Pattern[str]] = re.compile(r"gpt-3\.5|o1-mini|o3-mini|gpt-oss")

#: Open-weight ids that carry a vision tower.
_OPEN_VISION: Final[re.Pattern[str]] = re.compile(r"-vl\b|vl-|vision|pixtral|llava|gemma-3")

#: DeepSeek ids that always reason.
_DEEPSEEK_REASONER: Final[re.Pattern[str]] = re.compile(r"reasoner|-r1\b|deepseek-r1")


def _bare(model: str) -> str:
    """``model`` without a routing prefix such as ``vllm:`` or ``openrouter:``."""
    text = model.strip()
    return text.split(":", 1)[1] if ":" in text else text


def family_of(model: str) -> str:
    """The family ``model`` belongs to, by metadata record or by its id."""
    meta = metadata_for(model)
    if meta is not None and meta.family in _METADATA_FAMILIES:
        return _METADATA_FAMILIES[meta.family]
    # The routing prefix is dropped (``ollama:`` contains "llama"); an
    # organisation after it is kept, since ``meta-llama/...`` is often the
    # only clue.
    lowered = _bare(model).lower()
    for family, pattern in _FAMILY_PATTERNS:
        if pattern.search(lowered):
            return family
    return "default"


def _refine(family: str, model: str) -> dict[str, Any]:
    """Per-generation corrections to a family's defaults."""
    lowered = model.strip().lower()
    out: dict[str, Any] = {}
    if family == "claude" and _claude_adaptive(lowered):
        out["reasoning"] = "adaptive"
    elif family == "gpt":
        if _GPT_REASONING.search(lowered):
            out["reasoning"] = "reasoning_effort"
        if _GPT_NO_VISION.search(lowered):
            out["vision"] = False
        if "gpt-oss" in lowered:
            # Served like any open-weight model, not by OpenAI's API.
            out["parallel_tool_calls"] = False
            out["prompt_caching"] = False
    elif family in {"qwen", "llama", "mistral", "deepseek", "kimi", "glm", "default"}:
        if _OPEN_VISION.search(lowered):
            out["vision"] = True
        if family == "deepseek" and _DEEPSEEK_REASONER.search(lowered):
            out["reasoning"] = "always"
    return out


# ---------------------------------------------------------------------------
# Overrides
# ---------------------------------------------------------------------------

_FIELDS: Final[frozenset[str]] = frozenset(ModelProfile.model_fields) - {"model", "sources"}


def load_overrides(path: str | os.PathLike[str]) -> dict[str, dict[str, Any]]:
    """Read an override file: ``{"<id or glob>": {<profile fields>}, ...}``.

    A top-level ``"models"`` key holding that mapping is accepted too, so the
    file can grow other sections without breaking this one.

    Raises:
        ValueError: The file is not a JSON object of objects, or names a
            field the profile does not have — a typo in an override should
            fail loudly, not be ignored.
        OSError: The file cannot be read.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("models"), dict):
        data = data["models"]
    if not isinstance(data, dict):
        # ValueError for every malformed file, so a caller handles one kind.
        raise ValueError(  # noqa: TRY004
            f"{path}: expected a JSON object of model ids to profile fields"
        )
    out: dict[str, dict[str, Any]] = {}
    for key, fields in data.items():
        if not isinstance(fields, dict):
            raise ValueError(f"{path}: the entry for {key!r} is not an object")  # noqa: TRY004
        unknown = set(fields) - _FIELDS
        if unknown:
            raise ValueError(f"{path}: {key!r} sets unknown field(s) {sorted(unknown)}")
        out[str(key)] = dict(fields)
    return out


def _env_overrides() -> dict[str, dict[str, Any]]:
    path = os.environ.get(PROFILES_ENV, "").strip()
    if not path:
        return {}
    return load_overrides(path)


def _matching(
    model: str, overrides: Mapping[str, Mapping[str, Any]]
) -> list[tuple[str, Mapping[str, Any]]]:
    """Override entries that apply to ``model``: globs first, exact ids last.

    Applied in that order, so an exact id beats a glob that also matches it.
    """
    bare = model.split(":", 1)[1] if ":" in model else model
    names = {model, bare, model.lower(), bare.lower()}
    globs = [
        (k, v)
        for k, v in overrides.items()
        if k not in names and any(fnmatch.fnmatchcase(n, k) for n in names)
    ]
    exact = [(k, v) for k, v in overrides.items() if k in names]
    return [*globs, *exact]


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def profile_for(
    model: str,
    *,
    overrides: Mapping[str, Mapping[str, Any]] | str | os.PathLike[str] | None = None,
) -> ModelProfile:
    """The capability profile for ``model``.

    Args:
        model: A model id, with or without a provider prefix
            (``"openai:gpt-5.5"``, ``"claude-sonnet-5-5"``).
        overrides: Profile fields by model id or glob, or the path of a JSON
            file holding them. Applied over the ``TULIP_MODEL_PROFILES`` file,
            which is read on every call so an edit takes effect without a
            restart.

    Raises:
        ValueError: An override names a field the profile does not have.
    """
    family = family_of(model)
    fields: dict[str, Any] = dict(_FAMILY_DEFAULTS[family])
    sources = [f"family:{family}"]

    refined = _refine(family, model)
    if refined:
        fields.update(refined)
        sources.insert(0, "heuristics")

    meta = metadata_for(model)
    if meta is not None:
        fields["context_window"] = meta.context_length
        fields["max_output_tokens"] = meta.max_output_tokens
        fields["prompt_caching"] = meta.supports_prompt_caching or fields.get(
            "prompt_caching", False
        )
        sources.insert(0, "metadata")

    layers = _env_overrides()
    if isinstance(overrides, (str, os.PathLike)):
        layers = {**layers, **load_overrides(overrides)}
    elif overrides is not None:
        for key, value in overrides.items():
            unknown = set(value) - _FIELDS
            if unknown:
                raise ValueError(f"override {key!r} sets unknown field(s) {sorted(unknown)}")
        layers = {**layers, **{k: dict(v) for k, v in overrides.items()}}
    for key, value in _matching(model, layers):
        fields.update(value)
        sources.insert(0, f"override:{key}")

    return ModelProfile(
        model=model, family=fields.pop("family", family), sources=tuple(sources), **fields
    )
