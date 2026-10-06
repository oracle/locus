# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``tulip.models.metadata``."""

from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from tulip.models.metadata import (
    ModelMetadata,
    known_models,
    metadata_for,
    register_metadata,
)


# ---------------------------------------------------------------------------
# Record validation.
# ---------------------------------------------------------------------------


class TestModelMetadata:
    def test_frozen(self) -> None:
        md = ModelMetadata(
            model_id="x",
            family="test",
            context_length=128_000,
            max_output_tokens=4_096,
        )
        with pytest.raises(ValidationError, match="frozen"):
            md.context_length = 99

    def test_context_length_positive(self) -> None:
        with pytest.raises(ValidationError):
            ModelMetadata(
                model_id="x",
                family="test",
                context_length=0,
                max_output_tokens=4_096,
            )

    def test_empty_model_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ModelMetadata(
                model_id="",
                family="test",
                context_length=100,
                max_output_tokens=10,
            )


# ---------------------------------------------------------------------------
# Seed table lookups.
# ---------------------------------------------------------------------------


class TestSeedLookups:
    @pytest.mark.parametrize(
        ("model_id", "expected_family", "expected_window"),
        [
            ("gpt-4o", "openai", 128_000),
            ("gpt-4.1", "openai", 1_000_000),
            ("gpt-5", "openai", 400_000),
            ("o3", "openai", 200_000),
            ("claude-opus-4", "anthropic", 1_000_000),
            ("claude-haiku-4", "anthropic", 200_000),
            ("qwen3.6-35b", "qwen", 262_000),
            ("qwen3.6-35b-a3b", "qwen", 262_000),
        ],
    )
    def test_known_model(self, model_id: str, expected_family: str, expected_window: int) -> None:
        md = metadata_for(model_id)
        assert md is not None
        assert md.family == expected_family
        assert md.context_length == expected_window

    def test_unknown_returns_none(self) -> None:
        assert metadata_for("nonexistent-model-999") is None

    def test_pricing_parsed_as_decimal(self) -> None:
        md = metadata_for("claude-opus-4")
        assert md is not None
        assert md.input_price_per_mtok == Decimal("15.00")
        assert md.output_price_per_mtok == Decimal("75.00")

    def test_prompt_caching_flag(self) -> None:
        assert metadata_for("gpt-4o").supports_prompt_caching is True  # type: ignore[union-attr]
        assert metadata_for("o1").supports_prompt_caching is False  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Provider-prefix stripping.
# ---------------------------------------------------------------------------


class TestPrefixStripping:
    @pytest.mark.parametrize(
        "input_id",
        [
            "openai:gpt-4o",
            "OPENAI:gpt-4o",
            " openai : gpt-4o ",  # whitespace tolerance (stripped + partition)
            "gpt-4o",
        ],
    )
    def test_openai_prefix(self, input_id: str) -> None:
        md = metadata_for(input_id)
        assert md is not None
        assert md.model_id == "gpt-4o"

    def test_unrecognised_prefix_not_stripped(self) -> None:
        # Prefix isn't in _PROVIDER_PREFIXES — entire string must match,
        # which it won't.
        assert metadata_for("bogus:gpt-4o") is None


# ---------------------------------------------------------------------------
# register_metadata extension point.
# ---------------------------------------------------------------------------


class TestRegisterMetadata:
    def test_register_custom_model(self) -> None:
        custom = ModelMetadata(
            model_id="custom-finetune-v1",
            family="custom",
            context_length=32_000,
            max_output_tokens=4_000,
        )
        register_metadata(custom)
        md = metadata_for("custom-finetune-v1")
        assert md is custom

    def test_register_overwrites(self) -> None:
        first = ModelMetadata(
            model_id="overwrite-test",
            family="v1",
            context_length=1_000,
            max_output_tokens=100,
        )
        second = ModelMetadata(
            model_id="overwrite-test",
            family="v2",
            context_length=2_000,
            max_output_tokens=200,
        )
        register_metadata(first)
        register_metadata(second)
        md = metadata_for("overwrite-test")
        assert md is not None
        assert md.family == "v2"
        assert md.context_length == 2_000


# ---------------------------------------------------------------------------
# known_models snapshot.
# ---------------------------------------------------------------------------


class TestKnownModels:
    def test_returns_sorted_list(self) -> None:
        names = known_models()
        assert names == sorted(names)
        assert "gpt-4o" in names
        assert "claude-opus-4" in names


# ---------------------------------------------------------------------------
# DeepSeek V4 seeds and routing prefixes.
# ---------------------------------------------------------------------------


class TestDeepSeekV4:
    @pytest.mark.parametrize(
        ("model_id", "inp", "out", "window"),
        [
            # OpenRouter's listed prices (GET /api/v1/models, 2026-09-28).
            ("openrouter:deepseek/deepseek-v4-flash", "0.05152", "0.10304", 1_048_576),
            ("deepseek/deepseek-v4-flash", "0.05152", "0.10304", 1_048_576),
            ("openrouter:deepseek/deepseek-v4.1-flash", "0.30", "1.20", 1_048_576),
            ("openrouter:deepseek/deepseek-v4-pro", "0.783", "1.566", 1_048_576),
            # DeepSeek's own API, peak rates.
            ("deepseek:deepseek-flash", "0.30", "1.20", 1_000_000),
            ("deepseek:deepseek-v4-flash", "0.30", "1.20", 1_000_000),
            ("deepseek-v4-pro", "1.32", "3.96", 1_000_000),
        ],
    )
    def test_priced(self, model_id: str, inp: str, out: str, window: int) -> None:
        md = metadata_for(model_id)
        assert md is not None
        assert md.family == "deepseek"
        assert md.input_price_per_mtok == Decimal(inp)
        assert md.output_price_per_mtok == Decimal(out)
        assert md.context_length == window

    def test_routes_price_separately(self) -> None:
        # The OpenRouter slug and DeepSeek's own slug are different entries.
        routed = metadata_for("openrouter:deepseek/deepseek-v4-flash")
        direct = metadata_for("deepseek:deepseek-v4-flash")
        assert routed is not None
        assert direct is not None
        assert routed.model_id != direct.model_id

    def test_every_compatible_provider_prefix_is_stripped(self) -> None:
        from tulip.models.metadata import _PROVIDER_PREFIXES
        from tulip.models.providers import COMPATIBLE_PROVIDERS

        missing = {p.prefix for p in COMPATIBLE_PROVIDERS} - _PROVIDER_PREFIXES
        assert not missing, f"add {sorted(missing)} to _PROVIDER_PREFIXES"

    def test_self_hosted_prefix_resolves_the_seed(self) -> None:
        md = metadata_for("vllm:qwen3.6-35b")
        assert md is not None
        assert md.model_id == "qwen3.6-35b"


# ---------------------------------------------------------------------------
# model_id_of: model objects and proxies.
# ---------------------------------------------------------------------------


class _Config:
    def __init__(self, model: str) -> None:
        self.model = model


class _Model:
    def __init__(self, model: str) -> None:
        self.config = _Config(model)


class _ProxyView:
    """A per-turn view that forwards attribute access to the model it wraps."""

    def __init__(self, inner: object) -> None:
        self.inner = inner

    def __getattr__(self, name: str) -> object:
        return getattr(self.inner, name)


class TestModelIdOf:
    def test_string_passes_through(self) -> None:
        from tulip.models.metadata import model_id_of

        assert model_id_of("openrouter:deepseek/deepseek-v4-flash") == (
            "openrouter:deepseek/deepseek-v4-flash"
        )

    def test_config_model(self) -> None:
        from tulip.models.metadata import model_id_of

        assert model_id_of(_Model("gpt-4o")) == "gpt-4o"

    def test_bare_model_attribute(self) -> None:
        from types import SimpleNamespace

        from tulip.models.metadata import model_id_of

        assert model_id_of(SimpleNamespace(model="gpt-4o")) == "gpt-4o"

    def test_proxy_over_fallback_chain_resolves_the_primary(self) -> None:
        from tulip.models.fallback import FallbackChain
        from tulip.models.metadata import model_id_of

        chain = FallbackChain(
            [_Model("deepseek/deepseek-v4-flash"), _Model("gpt-4o-mini")], names=["t1", "t2"]
        )
        view = _ProxyView(chain)
        assert model_id_of(view) == "deepseek/deepseek-v4-flash"
        md = metadata_for(model_id_of(view) or "")
        assert md is not None
        assert md.input_price_per_mtok == Decimal("0.05152")

    def test_nothing_names_it(self) -> None:
        from tulip.models.metadata import model_id_of

        assert model_id_of(object()) is None


class TestDiscoverContextLength:
    """The window a vLLM (or OpenRouter-style) server lists for a model."""

    _BASE = "http://vllm.test:8000/v1"

    @pytest.mark.asyncio
    async def test_reads_max_model_len_and_registers_it(self) -> None:
        import httpx
        import respx

        from tulip.models.metadata import discover_context_length

        with respx.mock:
            route = respx.get(f"{self._BASE}/models").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "data": [
                            {"id": "other", "max_model_len": 4096},
                            {"id": "discover-test-qwen", "max_model_len": 65_536},
                        ]
                    },
                )
            )
            window = await discover_context_length(
                self._BASE, "vllm:discover-test-qwen", api_key="k"
            )

        assert window == 65_536
        assert route.calls.last.request.headers["Authorization"] == "Bearer k"
        md = metadata_for("vllm:discover-test-qwen")
        assert md is not None
        assert md.context_length == 65_536

    @pytest.mark.asyncio
    async def test_keeps_registered_prices(self) -> None:
        import httpx
        import respx

        from tulip.models.metadata import discover_context_length

        register_metadata(
            ModelMetadata(
                model_id="discover-test-priced",
                family="test",
                context_length=1_000,
                max_output_tokens=500,
                input_price_per_mtok=Decimal(1),
                output_price_per_mtok=Decimal(2),
            )
        )
        with respx.mock:
            respx.get(f"{self._BASE}/models").mock(
                return_value=httpx.Response(
                    200, json={"data": [{"id": "discover-test-priced", "context_length": 32_768}]}
                )
            )
            window = await discover_context_length(self._BASE, "discover-test-priced")

        md = metadata_for("discover-test-priced")
        assert window == 32_768
        assert md is not None
        assert md.context_length == 32_768
        assert md.input_price_per_mtok == Decimal(1)

    @pytest.mark.asyncio
    async def test_register_false_leaves_the_registry_alone(self) -> None:
        import httpx
        import respx

        from tulip.models.metadata import discover_context_length

        with respx.mock:
            respx.get(f"{self._BASE}/models").mock(
                return_value=httpx.Response(
                    200, json={"data": [{"id": "discover-test-dry", "max_model_len": 8192}]}
                )
            )
            window = await discover_context_length(self._BASE, "discover-test-dry", register=False)

        assert window == 8192
        assert metadata_for("discover-test-dry") is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "response",
        [
            {"status_code": 500, "json": {}},
            {"status_code": 200, "json": {"data": [{"id": "x", "max_model_len": 1}]}},
            {"status_code": 200, "json": {"data": [{"id": "discover-test-none"}]}},
            {
                "status_code": 200,
                "json": {"data": [{"id": "discover-test-none", "max_model_len": True}]},
            },
            {"status_code": 200, "text": "not json"},
            {"status_code": 200, "json": ["not", "an", "object"]},
        ],
    )
    async def test_unlisted_or_unreadable_returns_none(self, response: dict[str, object]) -> None:
        import httpx
        import respx

        from tulip.models.metadata import discover_context_length

        with respx.mock:
            respx.get(f"{self._BASE}/models").mock(return_value=httpx.Response(**response))  # type: ignore[arg-type]
            assert await discover_context_length(self._BASE, "discover-test-none") is None
        assert metadata_for("discover-test-none") is None

    @pytest.mark.asyncio
    async def test_unreachable_server_returns_none(self) -> None:
        import httpx
        import respx

        from tulip.models.metadata import discover_context_length

        with respx.mock:
            respx.get(f"{self._BASE}/models").mock(side_effect=httpx.ConnectError("refused"))
            assert await discover_context_length(self._BASE, "discover-test-none") is None
