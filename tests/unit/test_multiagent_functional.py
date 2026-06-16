# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for ``locus.multiagent.functional`` (``@task`` + ``@entrypoint``).

The functional API is a decorator-based alternative to the
``StateGraph`` builder. Coverage was 0% — these tests exercise both
decorator forms (bare and parametrised), the retry/cache machinery,
the entrypoint result tracking, and the failure paths.
"""

from __future__ import annotations

import asyncio

import pytest

from locus.multiagent.functional import (
    EntrypointResult,
    TaskResult,
    entrypoint,
    task,
)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


class TestResultDataclasses:
    def test_task_result_success_when_no_error(self) -> None:
        result = TaskResult(value=42)
        assert result.success is True

    def test_task_result_failure_when_error_set(self) -> None:
        result = TaskResult(error="boom")
        assert result.success is False

    def test_entrypoint_result_success_when_no_error(self) -> None:
        result = EntrypointResult(value="ok")
        assert result.success is True

    def test_entrypoint_result_failure_when_error_set(self) -> None:
        result = EntrypointResult(error="boom")
        assert result.success is False


# ---------------------------------------------------------------------------
# @task — bare form, async/sync, retries, caching
# ---------------------------------------------------------------------------


class TestTaskDecorator:
    @pytest.mark.asyncio
    async def test_bare_decorator_on_async_fn(self) -> None:
        @task
        async def fetch(x: int) -> int:
            return x * 2

        result = await fetch(5)
        assert result == 10
        assert getattr(fetch, "_is_task", False) is True
        assert getattr(fetch, "_task_name", None) == "fetch"

    @pytest.mark.asyncio
    async def test_decorator_on_sync_fn(self) -> None:
        # Sync user functions still get wrapped — the wrapper is async
        # but invokes them synchronously.
        @task
        def double(x: int) -> int:
            return x * 2

        result = await double(3)
        assert result == 6

    @pytest.mark.asyncio
    async def test_explicit_name_overrides_default(self) -> None:
        @task(name="custom-name")
        async def fetch() -> str:
            return "ok"

        await fetch()
        assert fetch._task_name == "custom-name"

    @pytest.mark.asyncio
    async def test_retry_on_failure(self) -> None:
        attempts = {"n": 0}

        @task(retry_attempts=3)
        async def flaky() -> str:
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise RuntimeError("transient")
            return "ok"

        result = await flaky()
        assert result == "ok"
        assert attempts["n"] == 3

    @pytest.mark.asyncio
    async def test_retry_exhausted_raises_last_error(self) -> None:
        @task(retry_attempts=2)
        async def always_fails() -> str:
            raise RuntimeError("permanent")

        with pytest.raises(RuntimeError, match="permanent"):
            await always_fails()

    @pytest.mark.asyncio
    async def test_cache_hits_skip_re_execution(self) -> None:
        attempts = {"n": 0}

        @task(cache=True)
        async def expensive(x: int) -> int:
            attempts["n"] += 1
            return x * x

        await expensive(5)
        await expensive(5)  # Cache hit.
        await expensive(6)  # Cache miss.

        # ``5`` ran once, ``6`` ran once.
        assert attempts["n"] == 2

    @pytest.mark.asyncio
    async def test_cache_disabled_re_executes(self) -> None:
        attempts = {"n": 0}

        @task  # cache=False default
        async def expensive(x: int) -> int:
            attempts["n"] += 1
            return x * x

        await expensive(5)
        await expensive(5)
        assert attempts["n"] == 2


# ---------------------------------------------------------------------------
# @entrypoint
# ---------------------------------------------------------------------------


class TestEntrypointDecorator:
    @pytest.mark.asyncio
    async def test_bare_decorator_returns_value(self) -> None:
        @entrypoint
        async def workflow(x: int) -> int:
            return x + 1

        result = await workflow(41)
        assert result == 42
        assert getattr(workflow, "_is_entrypoint", False) is True

    @pytest.mark.asyncio
    async def test_explicit_name(self) -> None:
        @entrypoint(name="my-workflow")
        async def wf() -> str:
            return "ok"

        await wf()
        assert wf._entrypoint_name == "my-workflow"

    @pytest.mark.asyncio
    async def test_sync_user_fn_supported(self) -> None:
        @entrypoint
        def sync_wf(x: int) -> int:
            return x * 2

        # Wrapper is still async even when wrapping a sync body.
        result = await sync_wf(3)
        assert result == 6

    @pytest.mark.asyncio
    async def test_get_result_records_tasks_run(self) -> None:
        @task
        async def step(x: int) -> int:
            return x + 1

        @entrypoint
        async def wf() -> int:
            a = await step(1)
            b = await step(a)
            return b

        await wf()
        result = wf.get_result()
        assert result is not None
        assert result.success is True
        assert len(result.tasks) == 2
        assert result.tasks[0].task_name == "step"

    @pytest.mark.asyncio
    async def test_failure_records_error_and_re_raises(self) -> None:
        @entrypoint
        async def wf() -> str:
            raise ValueError("nope")

        with pytest.raises(ValueError, match="nope"):
            await wf()

        result = wf.get_result()
        assert result is not None
        assert result.success is False
        assert "nope" in (result.error or "")

    @pytest.mark.asyncio
    async def test_get_result_returns_none_before_first_call(self) -> None:
        @entrypoint
        async def wf() -> str:
            return "ok"

        # Before any invocation, ``get_result()`` returns the initial slot.
        assert wf.get_result() is None


# ---------------------------------------------------------------------------
# task + entrypoint integration
# ---------------------------------------------------------------------------


class TestTaskInsideEntrypoint:
    @pytest.mark.asyncio
    async def test_task_failure_recorded_in_entrypoint_tasks(self) -> None:
        @task(retry_attempts=1)
        async def boom() -> None:
            raise RuntimeError("kaboom")

        @entrypoint
        async def wf() -> None:
            try:
                await boom()
            except RuntimeError:
                pass

        await wf()
        result = wf.get_result()
        assert result is not None
        assert len(result.tasks) == 1
        assert result.tasks[0].error is not None
        assert "kaboom" in result.tasks[0].error

    @pytest.mark.asyncio
    async def test_parallel_tasks_all_recorded(self) -> None:
        @task
        async def one() -> int:
            return 1

        @task
        async def two() -> int:
            return 2

        @entrypoint
        async def wf() -> tuple[int, int]:
            a, b = await asyncio.gather(one(), two())
            return a, b

        result = await wf()
        assert result == (1, 2)
        ep_result = wf.get_result()
        assert ep_result is not None
        names = sorted(t.task_name for t in ep_result.tasks)
        assert names == ["one", "two"]
