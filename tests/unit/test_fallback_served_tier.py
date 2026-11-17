# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""``FallbackChain`` reports which tier served EACH call.

One attribute on a chain shared by concurrent calls would describe whichever
call finished last; ``last_tier`` was that attribute and left in 6.0.0. Each
call carries its own :class:`ServedTier` — on the response and in the
calling context.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from locus.core.events import ModelChunkEvent
from locus.core.messages import Message
from locus.models.fallback import FallbackChain, ServedTier, served_tier
from locus.testing import text


class OverloadedError(Exception):
    def __init__(self) -> None:
        super().__init__("overloaded")
        self.status_code = 529
        self.headers: dict[str, str] = {}


class Primary:
    """Fails fast for prompt "fail", answers slowly for anything else."""

    async def complete(self, messages: list[Message], tools: Any = None, **kw: Any) -> Any:
        if messages[-1].content == "fail":
            await asyncio.sleep(0.01)
            raise OverloadedError
        await asyncio.sleep(0.1)
        return text("primary")

    async def stream(
        self, messages: list[Message], tools: Any = None, **kw: Any
    ) -> AsyncIterator[ModelChunkEvent]:
        if messages[-1].content == "fail":
            await asyncio.sleep(0.01)
            raise OverloadedError
        await asyncio.sleep(0.1)
        yield ModelChunkEvent(content="primary")


class Backup:
    async def complete(self, messages: list[Message], tools: Any = None, **kw: Any) -> Any:
        return text("backup")

    async def stream(
        self, messages: list[Message], tools: Any = None, **kw: Any
    ) -> AsyncIterator[ModelChunkEvent]:
        yield ModelChunkEvent(content="backup")


def _chain() -> FallbackChain:
    # A high threshold keeps the primary's breaker closed for the "ok" call.
    return FallbackChain([Primary(), Backup()], names=["primary", "backup"], failure_threshold=99)


async def test_concurrent_completes_each_report_their_own_tier() -> None:
    chain = _chain()

    async def call(prompt: str) -> tuple[Any, ServedTier | None]:
        response = await chain.complete([Message.user(prompt)])
        return response, served_tier()

    (failed_over, failed_ctx), (primary, primary_ctx) = await asyncio.gather(
        call("fail"), call("ok")
    )

    assert failed_over.content == "backup"
    assert failed_over.metadata["fallback"] == {"tier": 1, "model": "backup", "attempts": 2}
    assert failed_ctx == ServedTier(index=1, model="backup", attempts=2)
    assert failed_ctx.fell_back

    assert primary.content == "primary"
    assert primary.metadata["fallback"] == {"tier": 0, "model": "primary", "attempts": 1}
    assert primary_ctx == ServedTier(index=0, model="primary", attempts=1)

    # The shared attribute that would have described whichever call finished
    # last — the race the per-call report exists to avoid — is gone.
    assert not hasattr(chain, "last_tier")


async def test_concurrent_streams_each_report_their_own_tier() -> None:
    chain = _chain()

    async def call(prompt: str) -> tuple[str, ServedTier | None]:
        out = "".join([c.content or "" async for c in chain.stream([Message.user(prompt)])])
        return out, served_tier()

    (a_text, a_tier), (b_text, b_tier) = await asyncio.gather(call("fail"), call("ok"))
    assert (a_text, a_tier) == ("backup", ServedTier(index=1, model="backup", attempts=2))
    assert (b_text, b_tier) == ("primary", ServedTier(index=0, model="primary", attempts=1))


async def test_the_served_annotation_does_not_mutate_the_tier_response() -> None:
    shared = text("cached")

    class Cached:
        async def complete(self, *a: Any, **kw: Any) -> Any:
            return shared

    response = await FallbackChain([Cached()]).complete([Message.user("q")])
    assert response.metadata["fallback"]["tier"] == 0
    assert shared.metadata == {}
