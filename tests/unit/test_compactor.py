# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Tests for ``locus.memory.compactor.LLMCompactor``."""

from __future__ import annotations

from typing import Any

import pytest

from locus.core.messages import Message, Role, ToolResult
from locus.memory.compactor import LLMCompactor


def _asst(content: str) -> Message:
    return Message(role=Role.ASSISTANT, content=content)


def _user(content: str) -> Message:
    return Message(role=Role.USER, content=content)


def _system(content: str = "you are a helpful assistant.") -> Message:
    return Message(role=Role.SYSTEM, content=content)


def _tool_result(call_id: str, content: str, name: str = "fake_tool") -> Message:
    return Message.tool(ToolResult(tool_call_id=call_id, name=name, content=content))


class TestInitialisation:
    def test_default_params_ok(self) -> None:
        c = LLMCompactor()
        assert c.context_length == 128_000
        assert c.trigger_fraction == 0.8

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("context_length", 0),
            ("trigger_fraction", 0.0),
            ("trigger_fraction", 1.1),
            ("head_turns", -1),
            ("tail_token_fraction", 0.0),
            ("tail_token_fraction", 1.0),
            ("tool_output_ttl_turns", -1),
        ],
    )
    def test_validates_bounds(self, field: str, value: Any) -> None:
        kwargs: dict[str, Any] = {field: value}
        with pytest.raises(ValueError):
            LLMCompactor(**kwargs)


class TestUnderBudget:
    def test_noop_when_under_trigger(self) -> None:
        c = LLMCompactor(context_length=1_000_000, trigger_fraction=0.8)
        msgs = [_system(), _user("hi"), _asst("hello")]
        # Sync path
        assert c.apply(msgs) == msgs

    @pytest.mark.asyncio
    async def test_async_noop_when_under_trigger(self) -> None:
        c = LLMCompactor(context_length=1_000_000, trigger_fraction=0.8)
        msgs = [_system(), _user("hi"), _asst("hello")]
        out = await c.async_apply(msgs)
        assert out == msgs


class TestToolOutputPruning:
    @pytest.mark.asyncio
    async def test_stale_tool_outputs_replaced(self) -> None:
        # Set a tiny context so we definitely trip the threshold.
        c = LLMCompactor(
            context_length=200,
            trigger_fraction=0.5,
            tool_output_ttl_turns=2,
            head_turns=0,
            tail_token_fraction=0.9,
        )
        big = "x" * 800
        msgs = [
            _user("ask1"),
            _asst("ok"),
            _tool_result("c1", big),  # idx 2 — stale
            _asst("ok again"),
            _tool_result("c2", big),  # idx 4 — stale
            _asst("done"),
            _user("recent q"),
            _tool_result("c3", "fresh tool output"),  # idx 7 — fresh
        ]
        # No LLM wired — expect sync path fallback.
        out = await c.async_apply(msgs)
        # Stale tool outputs must be replaced with the placeholder text.
        stale_placeholders = [
            m for m in out if m.role == Role.TOOL and "compacted" in (m.content or "")
        ]
        assert len(stale_placeholders) >= 1
        # Fresh tool output survives unless it got trimmed by tail budget.
        tool_msgs = [m for m in out if m.role == Role.TOOL]
        fresh = [m for m in tool_msgs if "fresh tool output" in (m.content or "")]
        assert len(fresh) == 1


class TestHeadTailLLMPath:
    @pytest.mark.asyncio
    async def test_llm_path_keeps_system_head_tail_and_inserts_summary(self) -> None:
        calls: list[tuple[int, str | None]] = []

        async def _summarize(middle, prev):  # type: ignore[no-untyped-def]
            calls.append((len(middle), prev))
            return "SUMMARY: resolved=a pending=b remaining=c"

        c = LLMCompactor(
            context_length=400,
            trigger_fraction=0.5,
            head_turns=2,
            tail_token_fraction=0.3,
            tool_output_ttl_turns=0,
            summarize_fn=_summarize,
        )

        # Build a ~large conversation.
        msgs: list[Message] = [_system("sys")]
        for i in range(20):
            msgs.append(_user(f"q{i} " * 20))
            msgs.append(_asst(f"a{i} " * 20))

        out = await c.async_apply(msgs)

        # Expect: system first, then summary (system role), then head, then tail.
        assert out[0].role == Role.SYSTEM
        assert out[0].content == "sys"
        assert out[1].role == Role.SYSTEM
        assert "SUMMARY:" in (out[1].content or "")
        assert "REFERENCE ONLY" in (out[1].content or "")

        # Head preserved (the first two non-system messages).
        assert out[2].content is not None
        assert out[2].content.startswith("q0")
        assert out[3].content is not None
        assert out[3].content.startswith("a0")

        # Tail includes the last message.
        assert out[-1].content is not None
        assert out[-1].content.startswith("a19")

        # Middle was summarised exactly once.
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_previous_summary_passed_on_second_compaction(self) -> None:
        seen_prev: list[str | None] = []

        async def _summarize(middle, prev):  # type: ignore[no-untyped-def]
            seen_prev.append(prev)
            return f"summary-{len(seen_prev)}"

        c = LLMCompactor(
            context_length=200,
            trigger_fraction=0.5,
            head_turns=1,
            tail_token_fraction=0.3,
            tool_output_ttl_turns=0,
            summarize_fn=_summarize,
        )

        msgs = [_system("s")]
        for i in range(30):
            msgs.append(_user(f"q{i} " * 40))

        out1 = await c.async_apply(msgs)
        _ = await c.async_apply(out1 + [_user("newer " * 50)])
        assert seen_prev[0] is None
        assert seen_prev[1] == "summary-1"


class TestLLMFailureFallback:
    @pytest.mark.asyncio
    async def test_summarize_fn_exception_falls_back_to_sync_path(self) -> None:
        async def _boom(middle, prev):  # type: ignore[no-untyped-def]
            raise RuntimeError("provider down")

        c = LLMCompactor(
            context_length=200,
            trigger_fraction=0.5,
            head_turns=1,
            tail_token_fraction=0.3,
            tool_output_ttl_turns=0,
            summarize_fn=_boom,
        )
        msgs = [_system("s")] + [_user(f"q{i} " * 40) for i in range(20)]
        out = await c.async_apply(msgs)
        # Must not raise, must return *something*.
        assert out
        # No summary block inserted (fallback path).
        assert not any("REFERENCE ONLY" in (m.content or "") for m in out)


class TestRepr:
    def test_repr_has_key_fields(self) -> None:
        c = LLMCompactor(context_length=999)
        r = repr(c)
        assert "999" in r
        assert "LLMCompactor" in r


class TestReplayedMetadataCounting:
    """The default counter charges for the metadata a transport replays on the wire.

    Native compaction leaves an encrypted ``compaction`` item on an assistant
    message's metadata; every later request carries it, so ``tokens_after``
    has to see it or the report shrinks while the request grows.
    """

    @staticmethod
    def _carrying(key: str, item: dict[str, Any]) -> Message:
        return Message(
            role=Role.ASSISTANT,
            content="[Context compacted by the model.]",
            metadata={key: [item]},
        )

    def test_a_compaction_item_counts(self) -> None:
        import json

        from locus.memory.compactor import _char_count_tokens

        item = {"type": "compaction", "id": "cmp_1", "encrypted_content": "x" * 4000}
        plain = _asst("[Context compacted by the model.]")
        carrying = self._carrying("oci_responses_items", item)

        extra = _char_count_tokens(carrying) - _char_count_tokens(plain)
        assert extra == len(json.dumps([item], separators=(",", ":"))) // 4
        assert extra >= 1000

    def test_the_openai_replay_key_counts_too(self) -> None:
        from locus.memory.compactor import _char_count_tokens

        item = {"type": "reasoning", "id": "rs_1", "encrypted_content": "y" * 2000}
        plain = _asst("[Context compacted by the model.]")
        assert _char_count_tokens(self._carrying("openai_responses_items", item)) > (
            _char_count_tokens(plain) + 400
        )

    def test_unrelated_metadata_is_free(self) -> None:
        from locus.memory.compactor import _char_count_tokens

        plain = _asst("hello")
        other = Message(
            role=Role.ASSISTANT,
            content="hello",
            metadata={"pin": True, "trace": {"big": "z" * 5000}, "oci_responses_items": []},
        )
        assert _char_count_tokens(other) == _char_count_tokens(plain)

    def test_the_keys_are_the_providers_constants(self) -> None:
        from locus.memory.compactor import REPLAYED_METADATA_KEYS
        from locus.models.native.openai import RESPONSES_ITEMS_METADATA_KEY
        from locus.models.providers.oci import OCI_RESPONSES_ITEMS_METADATA_KEY

        assert set(REPLAYED_METADATA_KEYS) == {
            OCI_RESPONSES_ITEMS_METADATA_KEY,
            RESPONSES_ITEMS_METADATA_KEY,
        }

    async def test_native_compaction_reports_the_item_it_keeps(self) -> None:
        """``tokens_after`` of the native stage includes the compaction item."""
        from locus.memory.compaction import ContextCompactor
        from locus.models.providers.oci._responses_parse import parse_compaction_response

        item = {"type": "compaction", "id": "cmp_1", "encrypted_content": "e" * 3600}
        kept, _usage = parse_compaction_response({"output": [item], "usage": {}})

        class _Model:
            async def compact_history(  # noqa: PLR6301
                self, messages: list[Message], *, instructions: str | None = None
            ) -> tuple[list[Message], dict[str, int]]:
                return kept, {"total_tokens": 1}

        compactor = ContextCompactor(context_length=100_000, native_compaction=True)
        history = [_system(), _user("task"), *[_user(f"note {i}: " + "n" * 40) for i in range(20)]]
        outcome = await compactor._compact_natively(_Model(), history, 0, 5_000, 1.0, None)

        assert outcome is not None
        assert outcome.stage == "native"
        # The head stays (system + task); the body is the one compaction item,
        # charged for its encrypted content rather than for its placeholder text.
        assert outcome.tokens_after >= compactor.tokens(history[:2]) + 3600 // 4
