# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Functional API for graph workflows — imperative-style graphs.

Write workflows as decorated functions instead of building StateGraph
objects. Lower barrier to entry for developers who prefer imperative code.

Example:
    from locus.multiagent.functional import entrypoint, task

    @task
    async def fetch_data(url: str) -> dict:
        return {"data": f"fetched from {url}"}

    @task
    async def process(data: dict) -> str:
        return f"processed: {data}"

    @entrypoint
    async def pipeline(url: str) -> str:
        data = await fetch_data(url)
        result = await process(data)
        return result

    # Run it
    result = await pipeline("https://example.com")
"""

from __future__ import annotations

import asyncio
import functools
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskResult:
    """Result from a task execution."""

    value: Any = None
    duration_ms: float = 0.0
    task_name: str = ""
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None


@dataclass
class EntrypointResult:
    """Result from an entrypoint execution."""

    value: Any = None
    tasks: list[TaskResult] = field(default_factory=list)
    duration_ms: float = 0.0
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None


# Track tasks within an entrypoint
_current_tasks: list[TaskResult] = []


def task(
    fn: Callable | None = None,
    *,
    name: str | None = None,
    retry_attempts: int = 1,
    cache: bool = False,
) -> Any:
    """Decorator that marks a function as a parallelizable task.

    Tasks are tracked within an entrypoint for monitoring and can be
    configured with retry and caching.

    Args:
        fn: The function to decorate.
        name: Task name (defaults to function name).
        retry_attempts: Number of retry attempts on failure.
        cache: If True, cache results for identical arguments.

    Example:
        @task
        async def fetch(url: str) -> dict:
            return await httpx.get(url).json()

        @task(retry_attempts=3)
        async def unreliable_api(query: str) -> str:
            return await call_api(query)
    """

    def decorator(func: Callable) -> Callable:
        task_name = name or func.__name__
        _cache: dict[str, Any] = {}

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()

            # Check cache
            if cache:
                cache_key = f"{args}:{kwargs}"
                if cache_key in _cache:
                    return _cache[cache_key]

            last_error = None
            for attempt in range(retry_attempts):
                try:
                    if asyncio.iscoroutinefunction(func):
                        result = await func(*args, **kwargs)
                    else:
                        result = func(*args, **kwargs)

                    duration = (time.perf_counter() - start) * 1000
                    _current_tasks.append(
                        TaskResult(
                            value=result,
                            duration_ms=duration,
                            task_name=task_name,
                        )
                    )

                    if cache:
                        _cache[cache_key] = result

                    return result

                except Exception as e:  # noqa: BLE001 — retry-any-failure semantics for user task bodies
                    last_error = e
                    if attempt < retry_attempts - 1:
                        await asyncio.sleep(0.1 * (attempt + 1))

            duration = (time.perf_counter() - start) * 1000
            _current_tasks.append(
                TaskResult(
                    duration_ms=duration,
                    task_name=task_name,
                    error=str(last_error),
                )
            )
            raise last_error  # type: ignore[misc]

        wrapper._is_task = True  # type: ignore[attr-defined]  # noqa: SLF001
        wrapper._task_name = task_name  # type: ignore[attr-defined]  # noqa: SLF001
        return wrapper

    if fn is not None:
        return decorator(fn)
    return decorator


def entrypoint(
    fn: Callable | None = None,
    *,
    name: str | None = None,
) -> Any:
    """Decorator that marks a function as a workflow entrypoint.

    The entrypoint is the top-level function that orchestrates tasks.
    It tracks all task executions and returns an EntrypointResult.

    Args:
        fn: The function to decorate.
        name: Entrypoint name (defaults to function name).

    Example:
        @entrypoint
        async def my_workflow(input_data: str) -> str:
            step1 = await fetch(input_data)
            step2 = await process(step1)
            return step2

        result = await my_workflow("hello")
        # result is the raw return value
        # Access metadata via my_workflow.last_result
    """

    def decorator(func: Callable) -> Callable:
        ep_name = name or func.__name__
        last_result: list[EntrypointResult] = [None]  # type: ignore[list-item]

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            global _current_tasks
            _current_tasks = []

            start = time.perf_counter()

            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)

                duration = (time.perf_counter() - start) * 1000
                last_result[0] = EntrypointResult(
                    value=result,
                    tasks=list(_current_tasks),
                    duration_ms=duration,
                )
                return result

            except Exception as e:
                duration = (time.perf_counter() - start) * 1000
                last_result[0] = EntrypointResult(
                    tasks=list(_current_tasks),
                    duration_ms=duration,
                    error=str(e),
                )
                raise

        wrapper.last_result = property(lambda self: last_result[0])  # type: ignore[attr-defined]  # noqa: ARG005
        wrapper._last_result = last_result  # type: ignore[attr-defined]  # noqa: SLF001
        wrapper._is_entrypoint = True  # type: ignore[attr-defined]  # noqa: SLF001
        wrapper._entrypoint_name = ep_name  # type: ignore[attr-defined]  # noqa: SLF001

        def get_result() -> EntrypointResult | None:
            return last_result[0]

        wrapper.get_result = get_result  # type: ignore[attr-defined]
        return wrapper

    if fn is not None:
        return decorator(fn)
    return decorator
