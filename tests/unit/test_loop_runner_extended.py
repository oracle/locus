# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Coverage tests for ``tulip.loop.runner``.

The existing ``test_loop_runner.py`` builds the runner but doesn't drive
its ``run()`` / ``run_to_completion()`` / ``BatchRunner`` / streaming
helpers. This file uses a stub :class:`ReActLoop` so we can exercise the
full execution paths without a real model.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock

import pytest

from tulip.core.events import (
    LoopEvent,
    TerminateEvent,
    ThinkEvent,
)
from tulip.core.state import AgentState
from tulip.loop.react import ReActLoop
from tulip.loop.runner import (
    BatchRunner,
    LoopRunner,
    StreamingCollector,
    create_runner,
)


# ---------------------------------------------------------------------------
# Stub ReActLoop subclass
# ---------------------------------------------------------------------------


class _StubLoop(ReActLoop):
    """Real :class:`ReActLoop` subclass that overrides run methods.

    Pydantic accepts the type because we extend the real class. The model
    and registry attributes are required by the parent so we feed it
    MagicMocks; they're never invoked because we override ``run``.
    """

    _events_seq: list[LoopEvent] = []
    _raise_first_n_val: int = 0
    _call_counter: list[int] = [0]

    def __init__(
        self,
        events: list[LoopEvent] | None = None,
        raise_first_n: int = 0,
    ) -> None:
        super().__init__(model=MagicMock(), registry=MagicMock())
        # Stash sequence on the instance via object dict — these attribute
        # names are not declared as fields, so use ``object.__setattr__``
        # to bypass Pydantic validation.
        evs = events or [
            ThinkEvent(iteration=1, reasoning="r"),
            TerminateEvent(
                reason="complete",
                iterations_used=1,
                final_confidence=1.0,
                total_tool_calls=0,
            ),
        ]
        object.__setattr__(self, "_events_seq", evs)
        object.__setattr__(self, "_raise_first_n_val", raise_first_n)
        object.__setattr__(self, "_call_counter", [0])

    async def run(
        self, prompt: str, initial_state: AgentState | None = None, **kwargs: Any
    ) -> AsyncIterator[LoopEvent]:
        self._call_counter[0] += 1
        if self._call_counter[0] <= self._raise_first_n_val:
            raise RuntimeError(f"transient {self._call_counter[0]}")
        for ev in self._events_seq:
            yield ev

    async def run_to_completion(
        self, prompt: str, initial_state: AgentState | None = None, **kwargs: Any
    ) -> tuple[AgentState, list[LoopEvent]]:
        return (AgentState(agent_id="x"), list(self._events_seq))


# ---------------------------------------------------------------------------
# LoopRunner.run — happy path + callbacks
# ---------------------------------------------------------------------------


class TestLoopRunnerRun:
    @pytest.mark.asyncio
    async def test_yields_events_and_invokes_event_callback(self) -> None:
        seen: list[LoopEvent] = []
        runner = LoopRunner(loop=_StubLoop(), on_event=seen.append)
        events = []
        async for ev in runner.run("hi"):
            events.append(ev)
        assert any(isinstance(e, TerminateEvent) for e in events)
        # event callback received every event
        assert len(seen) == len(events)

    @pytest.mark.asyncio
    async def test_on_complete_invoked_when_final_state_set(self) -> None:
        complete_seen: dict[str, Any] = {}
        # Set _final_state during iteration via the on_event callback so
        # the on_complete check at the end of run() is truthy.
        runner_holder: dict[str, LoopRunner] = {}

        def _on_event(_: LoopEvent) -> None:
            runner = runner_holder["r"]
            runner._final_state = AgentState(agent_id="x")

        runner = LoopRunner(
            loop=_StubLoop(),
            on_event=_on_event,
            on_complete=lambda state, events: complete_seen.update(called=True, n=len(events)),
        )
        runner_holder["r"] = runner
        async for _ in runner.run("hi"):
            pass
        assert complete_seen.get("called") is True
        assert complete_seen.get("n", 0) > 0

    @pytest.mark.asyncio
    async def test_error_without_retry_propagates(self) -> None:
        runner = LoopRunner(
            loop=_StubLoop(raise_first_n=1),
            retry_on_error=False,
        )
        with pytest.raises(RuntimeError, match="transient 1"):
            async for _ in runner.run("hi"):
                pass

    @pytest.mark.asyncio
    async def test_error_callback_invoked_then_re_raised(self) -> None:
        captured: list[Any] = []
        runner = LoopRunner(
            loop=_StubLoop(raise_first_n=1),
            retry_on_error=False,
            on_error=lambda err, state: captured.append((err, state)),
        )
        with pytest.raises(RuntimeError):
            async for _ in runner.run("hi"):
                pass
        assert captured
        assert isinstance(captured[0][0], RuntimeError)

    @pytest.mark.asyncio
    async def test_retries_then_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Skip the exponential-backoff sleep so the test runs fast.
        async def fake_sleep(_: float) -> None:
            return None

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)
        runner = LoopRunner(
            loop=_StubLoop(raise_first_n=2),
            retry_on_error=True,
            max_retries=5,
        )
        events = []
        async for ev in runner.run("hi"):
            events.append(ev)
        assert any(isinstance(e, TerminateEvent) for e in events)


# ---------------------------------------------------------------------------
# _run_with_timeout
# ---------------------------------------------------------------------------


class TestRunWithTimeout:
    @pytest.mark.asyncio
    async def test_timeout_yields_timeout_terminate(self) -> None:
        # Build a stub loop whose ``run`` sleeps longer than the timeout.

        class _SlowLoop(ReActLoop):
            def __init__(self) -> None:
                super().__init__(model=MagicMock(), registry=MagicMock())

            async def run(self, *a: Any, **kw: Any) -> AsyncIterator[LoopEvent]:
                await asyncio.sleep(1.0)
                yield TerminateEvent(
                    reason="complete",
                    iterations_used=0,
                    final_confidence=0.0,
                    total_tool_calls=0,
                )

        runner = LoopRunner(loop=_SlowLoop(), timeout=0.05)
        events: list[LoopEvent] = []
        async for ev in runner.run("hi"):
            events.append(ev)
        # The terminate event has reason="timeout"
        terms = [e for e in events if isinstance(e, TerminateEvent)]
        assert terms
        assert any(e.reason == "timeout" for e in terms)


# ---------------------------------------------------------------------------
# run_to_completion
# ---------------------------------------------------------------------------


class TestRunToCompletion:
    @pytest.mark.asyncio
    async def test_returns_state_and_events(self) -> None:
        runner = LoopRunner(loop=_StubLoop())
        state, events = await runner.run_to_completion("hi")
        assert isinstance(state, AgentState)
        assert any(isinstance(e, TerminateEvent) for e in events)


# ---------------------------------------------------------------------------
# BatchRunner
# ---------------------------------------------------------------------------


class TestBatchRunner:
    @pytest.mark.asyncio
    async def test_runs_each_prompt(self) -> None:
        runner = BatchRunner(loop=_StubLoop(), max_concurrency=2)
        captured: list[str] = []
        results = await runner.run_batch(
            ["a", "b", "c"],
            on_result=lambda p, state, events: captured.append(p),
        )
        assert {r[0] for r in results} == {"a", "b", "c"}
        assert set(captured) == {"a", "b", "c"}


# ---------------------------------------------------------------------------
# StreamingCollector
# ---------------------------------------------------------------------------


class TestStreamingCollector:
    def test_collects_into_buckets(self) -> None:
        coll = StreamingCollector()
        coll.collect(ThinkEvent(iteration=1, reasoning="r"))
        coll.collect(
            TerminateEvent(
                reason="complete",
                iterations_used=2,
                final_confidence=0.9,
                total_tool_calls=0,
            )
        )
        assert len(coll.events) == 2
        assert len(coll.think_events) == 1
        assert coll.terminate_event is not None
        assert coll.is_complete is True
        assert coll.iterations == 2
        assert coll.final_confidence == 0.9

    def test_reset_clears_state(self) -> None:
        coll = StreamingCollector()
        coll.collect(ThinkEvent(iteration=1, reasoning="r"))
        coll.reset()
        assert coll.events == []
        assert coll.think_events == []
        assert coll.terminate_event is None

    def test_unset_metrics_default_to_zero(self) -> None:
        coll = StreamingCollector()
        assert coll.iterations == 0
        assert coll.final_confidence == 0.0


# ---------------------------------------------------------------------------
# create_runner factory
# ---------------------------------------------------------------------------


class TestCreateRunner:
    def test_returns_configured_runner(self) -> None:
        from unittest.mock import MagicMock

        runner = create_runner(
            model=MagicMock(),
            registry=MagicMock(),
            max_iterations=50,
            confidence_threshold=0.7,
            enable_reflection=False,
            system_prompt="be terse",
            timeout=30.0,
        )
        assert isinstance(runner, LoopRunner)
        assert runner.timeout == 30.0


# ---------------------------------------------------------------------------
# Timeout must only ever interrupt the loop, never the consumer
# ---------------------------------------------------------------------------


def _term(reason: str = "complete") -> TerminateEvent:
    return TerminateEvent(
        reason=reason, iterations_used=0, final_confidence=1.0, total_tool_calls=0
    )


class _ScriptLoop(ReActLoop):
    """Yields a fixed list of events and records whether it was closed."""

    def __init__(self, events: list[LoopEvent]) -> None:
        super().__init__(model=MagicMock(), registry=MagicMock())
        # ReActLoop is a pydantic model; keep test bookkeeping off its fields.
        object.__setattr__(self, "_track", {"script": events, "closed": False})

    @property
    def closed(self) -> bool:
        return bool(self._track["closed"])

    async def run(self, *a: Any, **kw: Any) -> AsyncIterator[LoopEvent]:
        try:
            for ev in self._track["script"]:
                yield ev
        finally:
            self._track["closed"] = True


class TestTimeoutScope:
    @pytest.mark.asyncio
    async def test_slow_consumer_is_not_cancelled_by_the_loop_timeout(self) -> None:
        # The timer used to stay armed across ``yield``: a consumer still busy
        # with the terminate event when the deadline passed had its own
        # ``await`` cancelled with a bare CancelledError.
        runner = LoopRunner(loop=_ScriptLoop([_term(), _term("extra")]), timeout=0.05)
        seen: list[str] = []
        async for ev in runner.run("hi"):
            assert isinstance(ev, TerminateEvent)
            seen.append(ev.reason)
            await asyncio.sleep(0.2)  # outlives the deadline
        assert seen == ["complete"]

    @pytest.mark.asyncio
    async def test_inner_loop_is_closed_when_run_stops_on_terminate(self) -> None:
        loop = _ScriptLoop([_term(), _term("never consumed")])
        runner = LoopRunner(loop=loop, timeout=5.0)
        async for _ev in runner.run("hi"):
            # Checked while ``run`` is still suspended at its yield: the
            # inner generator is still open here...
            assert loop.closed is False
        # ...and closed deterministically once ``run`` breaks, not at GC.
        assert loop.closed is True

    @pytest.mark.asyncio
    async def test_loop_ending_without_terminate_under_timeout(self) -> None:
        think = ThinkEvent(iteration=1, reasoning="r")
        loop = _ScriptLoop([think])
        runner = LoopRunner(loop=loop, timeout=5.0)
        events = [ev async for ev in runner.run("hi")]
        assert events == [think]
        assert loop.closed is True

    @pytest.mark.asyncio
    async def test_loop_ending_without_terminate_no_timeout(self) -> None:
        think = ThinkEvent(iteration=1, reasoning="r")
        loop = _ScriptLoop([think])
        runner = LoopRunner(loop=loop)
        events = [ev async for ev in runner.run("hi")]
        assert events == [think]
        assert loop.closed is True

    @pytest.mark.asyncio
    async def test_plain_async_iterator_without_aclose(self) -> None:
        # ``ReActLoop.run`` is typed as an AsyncIterator, not a generator; an
        # override returning one without ``aclose`` must still work.
        class _Iter:
            def __init__(self) -> None:
                self._items = [_term()]

            def __aiter__(self) -> _Iter:
                return self

            async def __anext__(self) -> LoopEvent:
                if not self._items:
                    raise StopAsyncIteration
                return self._items.pop(0)

        class _IterLoop(ReActLoop):
            def __init__(self) -> None:
                super().__init__(model=MagicMock(), registry=MagicMock())

            def run(self, *a: Any, **kw: Any) -> AsyncIterator[LoopEvent]:  # type: ignore[override]
                return _Iter()

        runner = LoopRunner(loop=_IterLoop(), timeout=5.0)
        events = [ev async for ev in runner.run("hi")]
        assert [e.reason for e in events if isinstance(e, TerminateEvent)] == ["complete"]
