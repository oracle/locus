# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for partial JSON parsing and structured streaming."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel, Field

from locus.core.events import LocusEvent, ModelChunkEvent, TerminateEvent
from locus.core.structured import _close_partial_json, parse_partial
from locus.streaming.structured import StructuredStream, stream_structured


class Vendor(BaseModel):
    name: str
    score: float


class VendorList(BaseModel):
    vendors: list[Vendor] = Field(default_factory=list)


# =============================================================================
# _close_partial_json
# =============================================================================


class TestClosePartialJson:
    def test_balanced_object_unchanged(self):
        assert _close_partial_json('{"a": 1}') == '{"a": 1}'

    def test_closes_open_object(self):
        completed = _close_partial_json('{"a": 1')
        assert completed == '{"a": 1}'

    def test_closes_nested_arrays_and_objects(self):
        completed = _close_partial_json('{"vendors": [{"name": "Acme"')
        # name string is a complete value here (closing quote present),
        # so we should close the inner object then the array then the outer.
        assert completed == '{"vendors": [{"name": "Acme"}]}'

    def test_drops_dangling_unterminated_string(self):
        completed = _close_partial_json('{"a": "abc')
        # ``"a": "abc`` cannot be closed safely — drop back to the opening
        # brace and close it.
        assert completed == "{}"

    def test_handles_escaped_quote_in_string(self):
        # The closing quote is real, so the buffer is balanced once we
        # close the outer object.
        completed = _close_partial_json('{"a": "with \\" quote"')
        assert completed == '{"a": "with \\" quote"}'


# =============================================================================
# parse_partial
# =============================================================================


class TestParsePartial:
    def test_returns_none_for_empty(self):
        assert parse_partial("", VendorList) is None

    def test_returns_instance_for_complete(self):
        result = parse_partial('{"vendors": [{"name": "Acme", "score": 0.9}]}', VendorList)
        assert result is not None
        assert result.vendors[0].name == "Acme"

    def test_returns_partial_when_outer_is_completable(self):
        # Inner array opened but no items yet; with default vendors=[] this
        # validates fine.
        result = parse_partial('{"vendors": [', VendorList)
        # ``{"vendors": []}`` is a valid VendorList.
        assert result is not None
        assert result.vendors == []

    def test_returns_none_when_required_field_missing(self):
        # Vendor needs name + score; only ``name`` is present.
        result = parse_partial('{"vendors": [{"name": "Acme"}]}', VendorList)
        assert result is None


# =============================================================================
# StructuredStream
# =============================================================================


async def _emit(events: list[LocusEvent]) -> AsyncIterator[LocusEvent]:
    for ev in events:
        yield ev


class TestStructuredStream:
    async def test_yields_progressive_partials(self):
        chunks = [
            ModelChunkEvent(content='{"vendors": ['),
            ModelChunkEvent(content='{"name": "Acme", "score": 0.9}'),
            ModelChunkEvent(content="]}"),
            TerminateEvent(
                reason="complete",
                iterations_used=1,
                final_confidence=0.9,
                total_tool_calls=0,
                final_message='{"vendors": [{"name": "Acme", "score": 0.9}]}',
            ),
        ]

        stream = StructuredStream(_emit(chunks), schema=VendorList)
        seen: list[VendorList] = []
        async for partial in stream:
            seen.append(partial)

        # First emit: empty vendor list (after first chunk closes to ``{"vendors": []}``).
        # Second emit: 1-element list.
        assert len(seen) >= 1
        assert seen[-1].vendors[0].name == "Acme"
        assert stream.final is not None
        assert stream.final.vendors[0].score == 0.9
        assert stream.terminate_reason == "complete"

    async def test_no_yield_when_unparseable(self):
        chunks = [ModelChunkEvent(content="prose only, no JSON yet")]
        stream = StructuredStream(_emit(chunks), schema=VendorList)
        seen: list[VendorList] = []
        async for partial in stream:
            seen.append(partial)
        # Buffer is "prose only, no JSON yet" — no JSON object, no emit.
        assert seen == []

    async def test_dedupes_identical_emits(self):
        # Two chunks that complete to the same partial — only one yield.
        chunks = [
            ModelChunkEvent(content='{"vendors":'),
            ModelChunkEvent(content=" ["),
            ModelChunkEvent(content="]}"),
            TerminateEvent(
                reason="complete",
                iterations_used=1,
                final_confidence=0.0,
                total_tool_calls=0,
                final_message='{"vendors": []}',
            ),
        ]
        stream = StructuredStream(_emit(chunks), schema=VendorList)
        seen: list[VendorList] = []
        async for partial in stream:
            seen.append(partial)
        # All three chunks parse to the same VendorList(vendors=[]) once
        # auto-closed; we should see exactly 1 unique snapshot.
        assert len(seen) == 1
        assert seen[0].vendors == []

    async def test_emit_unchanged_yields_each_parseable_chunk(self):
        # First chunk (``{"vendors":``) is unparseable; the following two
        # both auto-close to ``{"vendors": []}`` and emit_unchanged=True
        # forces both to surface.
        chunks = [
            ModelChunkEvent(content='{"vendors":'),
            ModelChunkEvent(content=" ["),
            ModelChunkEvent(content="]}"),
        ]
        stream = StructuredStream(_emit(chunks), schema=VendorList, emit_unchanged=True)
        seen: list[VendorList] = []
        async for partial in stream:
            seen.append(partial)
        assert len(seen) == 2

    async def test_factory_function(self):
        chunks = [
            ModelChunkEvent(content='{"vendors": [{"name": "Acme", "score": 0.5}]}'),
        ]
        stream = stream_structured(_emit(chunks), schema=VendorList)
        seen: list[VendorList] = []
        async for partial in stream:
            seen.append(partial)
        assert seen[0].vendors[0].name == "Acme"


# =============================================================================
# AgentResult.parsed_as
# =============================================================================


class TestParsedAs:
    def _result_with_parsed(self, parsed: BaseModel | None, parse_error: str | None = None) -> Any:
        from locus.agent.result import AgentResult
        from locus.core.state import AgentState

        state = AgentState()
        return AgentResult(
            message="",
            state=state,
            stop_reason="complete",
            parsed=parsed,
            parse_error=parse_error,
        )

    def test_returns_parsed_when_type_matches(self):
        parsed = VendorList(vendors=[Vendor(name="Acme", score=0.5)])
        result = self._result_with_parsed(parsed)
        out = result.parsed_as(VendorList)
        assert out is parsed
        assert out.vendors[0].name == "Acme"

    def test_raises_value_error_when_parsed_is_none(self):
        import pytest

        result = self._result_with_parsed(None)
        with pytest.raises(ValueError, match=r"no parsed output"):
            result.parsed_as(VendorList)

    def test_includes_parse_error_in_value_error(self):
        import pytest

        result = self._result_with_parsed(None, parse_error="missing fields x, y")
        with pytest.raises(ValueError, match=r"missing fields x, y"):
            result.parsed_as(VendorList)

    def test_raises_type_error_on_wrong_schema(self):
        import pytest

        parsed = Vendor(name="Acme", score=0.5)
        result = self._result_with_parsed(parsed)
        with pytest.raises(TypeError, match=r"Expected VendorList, got Vendor"):
            result.parsed_as(VendorList)
