# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""``locus.testing.check_model_conformance`` — verify a model before trusting it."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest
from pydantic import SecretStr

from locus.core.events import ModelChunkEvent
from locus.core.messages import Message
from locus.models.credentials import Credential, CredentialPool
from locus.models.fallback import FallbackChain
from locus.models.pooled import CredentialPoolModel
from locus.testing import FunctionModel, ScriptedModel, check_model_conformance, text, tool_call


def _well_behaved(messages: list[Message], tools: list[dict[str, Any]]) -> Any:
    if any(m.role == "tool" for m in messages):
        return text("It is 18C with light rain in Lisbon.")
    if tools:
        return tool_call("get_weather", call_id="call_1", city="Lisbon")
    return text("pong")


def _scripted() -> ScriptedModel:
    """The five turns the check makes: text, tool call, text, then the streamed pair."""
    return ScriptedModel(
        [
            text("pong"),
            tool_call("get_weather", call_id="call_1", city="Lisbon"),
            text("It is 18C with light rain in Lisbon."),
            text("pong"),
            tool_call("get_weather", call_id="call_2", city="Lisbon"),
        ]
    )


def _two_credentials() -> CredentialPool:
    return CredentialPool(
        [
            Credential(label="primary", api_key=SecretStr("key-a")),
            Credential(label="backup", api_key=SecretStr("key-b")),
        ]
    )


@pytest.mark.parametrize(
    "make",
    [
        lambda: FunctionModel(_well_behaved),
        _scripted,
        lambda: CredentialPoolModel(
            pool=_two_credentials(), build_model=lambda c: FunctionModel(_well_behaved)
        ),
    ],
    ids=["FunctionModel", "ScriptedModel", "CredentialPoolModel"],
)
async def test_a_conformant_model_passes(make: Callable[[], Any]) -> None:
    report = await check_model_conformance(make())
    assert report.ok, report.failures
    assert [name for name, _, _ in report.checks] == [
        "complete_text",
        "complete_tool_call",
        "tool_result_round_trip",
        "stream_text",
        "stream_tool_call",
    ]


async def test_a_scripted_model_is_consumed_turn_by_turn() -> None:
    model = _scripted()
    await check_model_conformance(model)
    assert model.call_count == 5
    assert model.offered_tools == [[], ["get_weather"], ["get_weather"], [], ["get_weather"]]


async def test_a_credential_pool_builds_one_model_per_credential() -> None:
    built: list[str] = []

    def build(credential: Credential) -> FunctionModel:
        built.append(credential.label)
        return FunctionModel(_well_behaved)

    model = CredentialPoolModel(pool=_two_credentials(), build_model=build)
    report = await check_model_conformance(model)
    assert report.ok, report.failures
    assert built == ["primary", "backup"], "one model per credential, cached by label"
    assert model.attempts == 5
    assert model.last_credential is not None


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
