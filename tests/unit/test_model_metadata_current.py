# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Current models are priced, and dated snapshots find their base record.

A cost budget refuses a model it cannot price, so a missing seed is not a
cosmetic gap: ``max_cost_usd`` on ``gpt-5.5`` or ``claude-sonnet-5-5`` used to
refuse to start.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from tulip.models.metadata import metadata_for


@pytest.mark.parametrize(
    ("model", "price_in", "price_out", "window"),
    [
        ("openai:gpt-5.5", "1.25", "10.00", 400_000),
        ("gpt-5.5-mini", "0.25", "2.00", 400_000),
        ("gpt-5.5-nano", "0.05", "0.40", 400_000),
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
