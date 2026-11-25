# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Current models are priced, and dated snapshots find their base record.

A cost budget refuses a model it cannot price, so a missing seed is not a
cosmetic gap: ``max_cost_usd`` on ``gpt-5.5`` or ``claude-sonnet-5-5`` used to
refuse to start.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from locus.models.metadata import metadata_for


@pytest.mark.parametrize(
    ("model", "price_in", "price_out", "window"),
    [
        ("openai:gpt-5.5", "1.25", "10.00", 400_000),
        ("gpt-5.5-mini", "0.25", "2.00", 400_000),
        ("gpt-5.5-nano", "0.05", "0.40", 400_000),
        ("gpt-5.6-luna", "0.20", "1.20", 1_050_000),
        ("gpt-5.6-sol", "4.00", "20.00", 1_050_000),
        ("gpt-5.6-terra", "2.00", "12.00", 1_050_000),
        ("gpt-6-astra", "10.00", "50.00", 1_050_000),
        ("gpt-6-sol", "2.00", "10.00", 1_050_000),
        ("openai:gpt-6-luna", "0.10", "0.50", 1_050_000),
        ("gpt-6.1-sol", "2.00", "10.00", 1_050_000),
        ("gpt-6.1-terra", "1.00", "5.00", 1_050_000),
        ("gpt-6.2-astra", "8.00", "40.00", 1_050_000),
        ("gpt-6.2-sol", "2.00", "10.00", 1_050_000),
        ("gpt-5", "1.25", "10.00", 400_000),
        ("gpt-5-mini", "0.25", "2.00", 400_000),
        ("gpt-5.1", "1.25", "10.00", 400_000),
        ("o1", "15.00", "60.00", 200_000),
        ("o3", "2.00", "8.00", 200_000),
        ("o4-mini", "1.10", "4.40", 200_000),
        ("gpt-5.2", "1.75", "14.00", 400_000),
        ("gpt-5.2-2025-12-11", "1.75", "14.00", 400_000),
        ("gpt-5.2-pro", "21.00", "168.00", 400_000),
        ("gpt-5.3-codex", "1.75", "14.00", 400_000),
        ("gpt-5.4", "2.50", "15.00", 1_050_000),
        ("gpt-5.4-mini-2026-03-17", "0.75", "4.50", 400_000),
        ("gpt-5.4-nano", "0.20", "1.25", 400_000),
        ("gpt-5.4-pro", "30.00", "180.00", 1_050_000),
        ("gpt-5.5-pro", "30.00", "180.00", 1_050_000),
        ("gpt-5.5-pro-2026-04-23", "30.00", "180.00", 1_050_000),
        ("gpt-4.1-nano", "0.10", "0.40", 1_047_576),
        ("gpt-5-codex", "1.25", "10.00", 400_000),
        ("gpt-5-nano", "0.05", "0.40", 400_000),
        ("gpt-5.1-codex-mini", "0.25", "2.00", 400_000),
        ("o3-mini", "1.10", "4.40", 200_000),
        ("gpt-audio-mini", "0.60", "2.40", 128_000),
        ("anthropic:claude-opus-5-5", "4.00", "20.00", 1_000_000),
        ("claude-opus-5", "5.00", "25.00", 1_000_000),
        ("claude-sonnet-5-5", "2.00", "10.00", 1_000_000),
        ("claude-sonnet-5", "2.00", "10.00", 1_000_000),
        ("claude-sonnet-4-6", "3.00", "15.00", 1_000_000),
        ("claude-opus-4-8", "5.00", "25.00", 1_000_000),
        ("claude-fable-5-1", "10.00", "50.00", 1_000_000),
        ("claude-haiku-4-5", "1.00", "5.00", 200_000),
    ],
)
def test_current_models_are_seeded(model: str, price_in: str, price_out: str, window: int) -> None:
    meta = metadata_for(model)
    assert meta is not None
    assert meta.input_price_per_mtok == Decimal(price_in)
    assert meta.output_price_per_mtok == Decimal(price_out)
    assert meta.context_length == window


@pytest.mark.parametrize(
    ("model", "base"),
    [
        ("claude-haiku-4-5-20251001", "claude-haiku-4-5"),
        ("anthropic:claude-sonnet-4-6-2026-02-17", "claude-sonnet-4-6"),
        ("claude-opus-4-6@20260205", "claude-opus-4-6"),
        ("gpt-5.5-2026-04-23", "gpt-5.5"),
        ("gpt-4o-latest", "gpt-4o"),
    ],
)
def test_a_dated_snapshot_resolves_to_its_base(model: str, base: str) -> None:
    meta = metadata_for(model)
    assert meta is not None
    assert meta.model_id == base


@pytest.mark.parametrize("model", ["gpt-5.5-turbo", "gpt-5.6", "claude-opus-5-9", "gpt-4o-2024"])
def test_only_snapshot_suffixes_are_stripped(model: str) -> None:
    # ``gpt-5.6`` must not borrow ``gpt-5``'s or ``gpt-5.5``'s record: a near
    # miss priced as its neighbour is a silently wrong budget.
    assert metadata_for(model) is None


@pytest.mark.parametrize(
    ("oci_id", "slug"),
    [
        ("oci:openai.gpt-5.6-luna", "gpt-5.6-luna"),
        ("oci:openai.gpt-6-luna", "gpt-6-luna"),
        ("openai.gpt-6.2-astra", "gpt-6.2-astra"),
        ("oci:openai.gpt-5.2-2025-12-11", "gpt-5.2"),
        ("oci:openai.gpt-5.5-pro", "gpt-5.5-pro"),
        ("oci-responses:openai.gpt-5.4", "gpt-5.4"),
    ],
)
def test_an_oci_id_carries_the_window_but_not_the_list_price(oci_id: str, slug: str) -> None:
    # OCI Generative AI bills at its own rates, which the catalogue does not
    # carry, so the served model's list price must not leak onto an OCI id.
    served = metadata_for(slug)
    through_oci = metadata_for(oci_id)
    assert served is not None
    assert through_oci is not None
    assert served.input_price_per_mtok is not None
    assert through_oci.context_length == served.context_length
    assert through_oci.max_output_tokens == served.max_output_tokens
    assert through_oci.input_price_per_mtok is None
    assert through_oci.output_price_per_mtok is None
