# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Tests for ``locus.models.metadata``."""

from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from locus.models.metadata import (
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
            ("cohere.command-r-plus", "oci-cohere", 128_000),
            ("meta.llama-3.3-70b-instruct", "oci-meta", 128_000),
            ("google.gemini-2.5-pro", "oci-google", 2_000_000),
            ("xai.grok-4", "oci-xai", 256_000),
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
        assert metadata_for("meta.llama-3.3-70b-instruct").supports_prompt_caching is False  # type: ignore[union-attr]


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

    def test_oci_prefix_on_cohere(self) -> None:
        md = metadata_for("oci:cohere.command-r-plus")
        assert md is not None
        assert md.family == "oci-cohere"

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
