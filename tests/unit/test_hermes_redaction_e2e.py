# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Integration test for redaction (A.1) wired through the tool executor.

Verifies the secret-redaction patterns added in milestone A actually
fire on real exception messages thrown by tool execution, end to
end through :class:`~tulip.tools.executor.SequentialExecutor`.
"""

from __future__ import annotations

import pytest

from tulip.core.messages import ToolCall
from tulip.tools.decorator import tool
from tulip.tools.executor import (
    SequentialExecutor,
    ToolContextFactory,
    redact_sensitive_text,
)
from tulip.tools.registry import ToolRegistry


def _make_executor() -> tuple[SequentialExecutor, ToolRegistry, ToolContextFactory]:
    registry = ToolRegistry()
    executor = SequentialExecutor()
    factory = ToolContextFactory(run_id="run-x", agent_id="agent-1", iteration=0)
    return executor, registry, factory


def _make_call(tool_name: str, arguments: dict | None = None) -> ToolCall:
    return ToolCall(
        id="call-1",
        name=tool_name,
        arguments=arguments or {},
    )


async def _run_one(
    executor: SequentialExecutor,
    registry: ToolRegistry,
    factory: ToolContextFactory,
    tool_name: str,
) -> object:
    """Execute one tool call and return the single ToolResult."""
    results = await executor.execute([_make_call(tool_name)], registry, factory)
    assert len(results) == 1
    return results[0]


# ---------------------------------------------------------------------------
# Tool raises with embedded vendor key — error message comes back redacted.
# ---------------------------------------------------------------------------


class TestExecutorRedactsVendorKeys:
    @pytest.mark.asyncio
    async def test_anthropic_key_in_error_redacted(self) -> None:
        executor, registry, factory = _make_executor()

        @tool
        def leak_anthropic() -> str:
            raise RuntimeError(
                "upstream auth failed for sk-ant-api03-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            )

        registry.register(leak_anthropic)
        result = await _run_one(executor, registry, factory, "leak_anthropic")

        assert not result.success
        # The full key value must not survive in the error message.
        assert "sk-ant-api03-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" not in (result.error or "")

    @pytest.mark.asyncio
    async def test_openai_key_in_error_redacted(self) -> None:
        executor, registry, factory = _make_executor()

        @tool
        def leak_openai() -> str:
            fake = (
                "sk-proj-" + "FAKEFIXTUREabcdefghijklmnopqrstuvwxyz01"
            )  # gitleaks:allow (test fixture)
            raise RuntimeError(f"401 from API: {fake}")

        registry.register(leak_openai)
        result = await _run_one(executor, registry, factory, "leak_openai")

        assert not result.success
        assert "FAKEFIXTUREabcdefghijklmnopqrstuvwxyz01" not in (result.error or "")


# ---------------------------------------------------------------------------
# URL with sensitive query params: URL preserved, value redacted.
# ---------------------------------------------------------------------------


class TestExecutorRedactsUrlQuery:
    @pytest.mark.asyncio
    async def test_url_access_token_value_only_redacted(self) -> None:
        executor, registry, factory = _make_executor()

        @tool
        def leak_url() -> str:
            raise RuntimeError(
                "fetch failed: https://api.example.com/v1/x?access_token=ZZZSECRETZZZ&user=42"
            )

        registry.register(leak_url)
        result = await _run_one(executor, registry, factory, "leak_url")

        assert not result.success
        err = result.error or ""
        assert "ZZZSECRETZZZ" not in err
        # The host + path must survive.
        assert "https://api.example.com/v1/x" in err
        # Non-sensitive params survive too.
        assert "user=42" in err


# ---------------------------------------------------------------------------
# Multi-line redaction: redact_sensitive_text is the public helper used
# anywhere log lines (not just tool errors) need scrubbing.
# ---------------------------------------------------------------------------


class TestRedactSensitiveTextHelper:
    def test_multiline_log_line(self) -> None:
        log = (
            "2026-04-25 INFO request done\n"
            "Authorization: Bearer abcdefghijklmnopqrstuvwxyz0123456789\n"
            "Authentication failed for sk-proj-aaaaaaaaaaaaaaaaaaaaaaaaaaa"
        )
        out = redact_sensitive_text(log)
        # Multi-line stays multi-line.
        assert "\n" in out
        # All three lines preserved (1 raw, 2 redacted).
        assert "request done" in out
        # Bearer value masked, header preserved.
        assert "Authorization: Bearer" in out
        assert "abcdefghijklmnopqrstuvwxyz0123456789" not in out
        assert "sk-proj-aaaaaaaaaaaaaaaaaaaaaaaaaaa" not in out
