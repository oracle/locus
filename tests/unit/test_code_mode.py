# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""#176 — code mode: in-program tool calls clear the same seam as loop calls.

The market's versions fork the paths — code-mediated calls bypass hooks and
guardrails. These tests pin the property that makes ours different: a gate or
hook that would stop a loop-issued call stops the identical call made from
inside the model's program, and the program observes a refusal, not the
effect.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from tulip.agent.hook_orchestrator import HookOrchestrator
from tulip.tools.code_mode import create_code_tool
from tulip.tools.decorator import tool
from tulip.tools.registry import create_registry


@tool
def lookup_price(sku: str) -> str:
    """Price for a SKU."""
    return f"{len(sku) * 10}.00"


EFFECTS: list[str] = []


@tool(labels={"payment"})
def charge_card(amount: float) -> str:
    """Charge the customer's card."""
    EFFECTS.append(f"charged {amount}")
    return f"charged {amount}"


def _code_tool(tools: list[Any], hooks: list[Any] | None = None) -> Any:
    registry = create_registry(*tools)
    return create_code_tool(registry, HookOrchestrator(hooks or []), timeout=20.0)


class TestBasics:
    @pytest.mark.asyncio
    async def test_program_calls_tool_and_returns_result(self) -> None:
        run_code = _code_tool([lookup_price])
        out = json.loads(
            await run_code.execute(
                code=(
                    "prices = [tools.call('lookup_price', sku=s) for s in ['ab', 'abcd']]\n"
                    "print('checked', len(prices))\n"
                    "result = ','.join(prices)"
                )
            )
        )
        assert out["result"] == "'20.00,40.00'"
        assert "checked 2" in out["stdout"]
        assert [c["name"] for c in out["tool_calls"]] == ["lookup_price", "lookup_price"]

    @pytest.mark.asyncio
    async def test_unknown_tool_is_an_error_and_recursion_is_refused(self) -> None:
        run_code = _code_tool([lookup_price])
        out = json.loads(await run_code.execute(code="tools.call('nope')"))
        assert "unknown tool" in out["error"]
        out2 = json.loads(await run_code.execute(code="tools.call('run_code', code='1')"))
        assert "unknown tool" in out2["error"]

    @pytest.mark.asyncio
    async def test_timeout_kills_the_program(self) -> None:
        registry = create_registry(lookup_price)
        run_code = create_code_tool(registry, HookOrchestrator([]), timeout=2.0)
        out = json.loads(await run_code.execute(code="while True:\n    pass"))
        assert "timed out" in out["error"]


class TestSeamParity:
    @pytest.mark.asyncio
    async def test_gate_holds_inside_the_program(self) -> None:
        """The headline property: admit() refuses the in-program call, the
        effect never runs, and the program gets a catchable refusal."""
        from tulip.control import ControlPolicy, gate_tool

        EFFECTS.clear()
        gated = gate_tool(
            charge_card, policy=ControlPolicy(deny_for={"payment"}), on_refusal="raise"
        )
        run_code = _code_tool([lookup_price, gated])
        out = json.loads(
            await run_code.execute(
                code=(
                    "try:\n"
                    "    tools.call('charge_card', amount=99.0)\n"
                    "    result = 'charged!'\n"
                    "except ToolRefused as e:\n"
                    "    result = 'refused: ' + str(e)[:40]\n"
                )
            )
        )
        assert EFFECTS == [], "the gate must hold for an in-program call"
        assert "refused:" in out["result"]
        assert out["tool_calls"] == [{"name": "charge_card", "refused": True}]

    @pytest.mark.asyncio
    async def test_before_hook_cancel_is_a_refusal(self) -> None:
        EFFECTS.clear()

        class _Veto:
            async def on_before_tool_call(self, event: Any) -> None:
                if event.tool_name == "charge_card":
                    event.cancel = "blocked by hook"

        run_code = _code_tool([charge_card], hooks=[_Veto()])
        out = json.loads(
            await run_code.execute(
                code=(
                    "try:\n"
                    "    tools.call('charge_card', amount=5.0)\n"
                    "except ToolRefused as e:\n"
                    "    result = str(e)\n"
                )
            )
        )
        assert EFFECTS == []
        assert out["result"] == "'blocked by hook'"

    @pytest.mark.asyncio
    async def test_after_hook_replacement_reaches_the_program(self) -> None:
        class _Redact:
            async def on_after_tool_call(self, event: Any) -> None:
                if event.result and "0.00" in str(event.result):
                    event.result = "[REDACTED]"

        run_code = _code_tool([lookup_price], hooks=[_Redact()])
        out = json.loads(
            await run_code.execute(code="result = tools.call('lookup_price', sku='ab')")
        )
        assert out["result"] == "'[REDACTED]'"

    @pytest.mark.asyncio
    async def test_hooks_see_every_inner_call(self) -> None:
        seen: list[str] = []

        class _Watch:
            async def on_before_tool_call(self, event: Any) -> None:
                seen.append(event.tool_name)

        run_code = _code_tool([lookup_price], hooks=[_Watch()])
        await run_code.execute(code="[tools.call('lookup_price', sku=s) for s in 'abc']")
        assert seen == ["lookup_price"] * 3


class TestAgentWiring:
    def test_run_code_registered_only_under_code_mode(self) -> None:
        from tulip.agent import Agent

        class _Stub:
            async def complete(self, messages: Any, **kwargs: Any) -> Any:
                raise AssertionError  # pragma: no cover

        assert "run_code" in Agent(model=_Stub(), tools=[lookup_price], code_mode=True).tools
        assert "run_code" not in Agent(model=_Stub(), tools=[lookup_price]).tools


class TestLooseAddressing:
    """Models address tools loosely; the host meets them (#176)."""

    @pytest.mark.asyncio
    async def test_positional_arguments_map_by_schema_order(self) -> None:
        run_code = _code_tool([lookup_price])
        out = json.loads(await run_code.execute(code="result = tools.call('lookup_price', 'abc')"))
        assert out["result"] == "'30.00'"

    @pytest.mark.asyncio
    async def test_functions_prefix_is_stripped(self) -> None:
        run_code = _code_tool([lookup_price])
        out = json.loads(
            await run_code.execute(code="result = tools.call('functions.lookup_price', sku='ab')")
        )
        assert out["result"] == "'20.00'"

    @pytest.mark.asyncio
    async def test_too_many_positional_arguments_is_an_error(self) -> None:
        run_code = _code_tool([lookup_price])
        out = json.loads(await run_code.execute(code="tools.call('lookup_price', 'a', 'b', 'c')"))
        assert "at most 1 argument" in out["error"]


class TestRpcRobustness:
    @pytest.mark.asyncio
    async def test_garbage_on_the_rpc_channel_is_ignored(self) -> None:
        """A library printing to the real fd must not kill the run."""
        run_code = _code_tool([lookup_price])
        out = json.loads(
            await run_code.execute(
                code=(
                    "import sys\n"
                    "sys.__stdout__.write('not json\\n'); sys.__stdout__.flush()\n"
                    'sys.__stdout__.write(\'{"rpc": "noise"}\\n\'); sys.__stdout__.flush()\n'
                    "result = tools.call('lookup_price', sku='ab')"
                )
            )
        )
        assert out["result"] == "'20.00'"

    @pytest.mark.asyncio
    async def test_program_dying_without_done_frame_is_reported(self) -> None:
        run_code = _code_tool([lookup_price])
        out = json.loads(await run_code.execute(code="import os\nos._exit(0)"))
        assert "without a done frame" in out["error"]
