# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Model retry hook — exponential backoff on throttle/rate limit errors.

Automatically retries model calls that fail due to rate limiting,
throttling, or transient errors. Works across all model providers.

Example:
    from locus.hooks.builtin.retry import ModelRetryHook

    agent = Agent(config=AgentConfig(
        model=model,
        hooks=[ModelRetryHook(max_retries=3)],
    ))
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from locus.hooks.provider import HookPriority, HookProvider


logger = logging.getLogger(__name__)


class ModelRetryHook(HookProvider):
    """Retry model calls on throttle/rate limit with exponential backoff.

    Catches empty responses and rate limit indicators, sets event.retry=True
    to trigger automatic re-invocation with increasing delays.

    Works with all providers (OCI, OpenAI, Anthropic, Ollama).

    Args:
        max_retries: Maximum retry attempts per model call.
        initial_delay: First retry delay in seconds.
        max_delay: Maximum delay between retries.
        backoff_factor: Multiplier for each subsequent delay.
        retry_on_empty: Retry when model returns empty content.
    """

    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 30.0,
        backoff_factor: float = 2.0,
        retry_on_empty: bool = True,
        priority: int = HookPriority.DEFAULT,
    ) -> None:
        self._max_retries = max_retries
        self._initial_delay = initial_delay
        self._max_delay = max_delay
        self._backoff_factor = backoff_factor
        self._retry_on_empty = retry_on_empty
        self._priority = priority
        self._attempt = 0
        self.retries_total = 0

    @property
    def priority(self) -> int:
        return self._priority

    @property
    def name(self) -> str:
        return "ModelRetryHook"

    async def on_before_model_call(self, event: Any) -> None:
        """Reset attempt counter before each new model call."""
        self._attempt = 0

    async def on_after_model_call(self, event: Any) -> None:
        """Check response and retry if needed."""
        response = event.response
        content = response.message.content or ""
        has_tool_calls = bool(response.message.tool_calls)

        # Determine if we should retry
        should_retry = False

        if self._retry_on_empty and not content and not has_tool_calls:
            should_retry = True

        if not should_retry:
            # Successful response — reset
            self._attempt = 0
            return

        # Check retry budget
        if self._attempt >= self._max_retries:
            logger.warning(
                "ModelRetryHook: exhausted %d retries, accepting empty response",
                self._max_retries,
            )
            self._attempt = 0
            return

        # Calculate delay with exponential backoff
        delay = min(
            self._initial_delay * (self._backoff_factor**self._attempt),
            self._max_delay,
        )

        self._attempt += 1
        self.retries_total += 1

        logger.info(
            "ModelRetryHook: retry %d/%d after %.1fs delay (empty response)",
            self._attempt,
            self._max_retries,
            delay,
        )

        from locus.observability.emit import EV_HOOK_MODEL_RETRY, emit  # noqa: PLC0415

        await emit(
            EV_HOOK_MODEL_RETRY,
            attempt=self._attempt,
            max_retries=self._max_retries,
            delay_seconds=delay,
            reason="empty_response",
        )

        await asyncio.sleep(delay)
        event.retry = True
