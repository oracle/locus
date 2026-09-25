# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""``tulip.testing.check_model_conformance`` — verify a model before trusting it."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from tulip.core.events import ModelChunkEvent
from tulip.core.messages import Message
from tulip.models.fallback import FallbackChain
from tulip.testing import FunctionModel, check_model_conformance, text, tool_call


def _well_behaved(messages: list[Message], tools: list[dict[str, Any]]) -> Any:
    if any(m.role == "tool" for m in messages):
        return text("It is 18C with light rain in Lisbon.")
    if tools:
        return tool_call("get_weather", call_id="call_1", city="Lisbon")
    return text("pong")


async def test_a_conformant_model_passes() -> None:
    report = await check_model_conformance(FunctionModel(_well_behaved))
    assert report.ok, report.failures
    assert [name for name, _, _ in report.checks] == [
        "complete_text",
        "complete_tool_call",
        "tool_result_round_trip",
        "stream_text",
        "stream_tool_call",
    ]


async def test_every_tier_of_a_chain_can_be_checked() -> None:
    chain = FallbackChain([FunctionModel(_well_behaved), FunctionModel(_well_behaved)])
    for tier in chain.tiers:
        await check_model_conformance(tier.model)
    await check_model_conformance(chain)


async def test_schema_violations_are_reported() -> None:
    def wrong_args(messages: list[Message], tools: list[dict[str, Any]]) -> Any:
        if any(m.role == "tool" for m in messages):
            return text("ok")
        if tools:
            return tool_call("get_weather", call_id="c", town="Lisbon", unit="kelvin")
        return text("pong")

    report = await check_model_conformance(FunctionModel(wrong_args), raise_on_failure=False)
    assert not report.ok
    detail = " ".join(report.failures)
    assert "missing required 'city'" in detail
    assert "unexpected argument 'town'" in detail
    assert "not in ['celsius', 'fahrenheit']" in detail


async def test_a_stream_that_drops_tool_calls_fails() -> None:
    class TextOnlyStream(FunctionModel):
        async def stream(
            self, messages: list[Message], tools: Any = None, **kwargs: Any
        ) -> AsyncIterator[ModelChunkEvent]:
            response = await self.complete(messages, tools, **kwargs)
            yield ModelChunkEvent(content=response.content or "")
            yield ModelChunkEvent(done=True)

    with pytest.raises(AssertionError, match="stream_tool_call: no get_weather call"):
        await check_model_conformance(TextOnlyStream(_well_behaved))


async def test_a_raising_model_is_reported_not_crashing_the_check() -> None:
    class Down:
        async def complete(self, *a: Any, **k: Any) -> Any:
            raise ConnectionError("down")

        async def stream(self, *a: Any, **k: Any) -> AsyncIterator[ModelChunkEvent]:
            raise ConnectionError("down")
            yield  # pragma: no cover

    report = await check_model_conformance(Down(), raise_on_failure=False)
    assert not report.ok
    assert all("ConnectionError" in f or "skipped" in f for f in report.failures)
    assert repr(report) == "ConformanceReport(0/5 passed)"
