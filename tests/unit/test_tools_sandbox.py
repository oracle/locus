# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for sandboxed tool execution (`tulip.tools.sandbox`).

The ``SubprocessSandbox`` tests mirror the tulip-sandbox conformance
invariants: code runs and is contained, timeouts report instead of hanging,
the host environment never reaches the box, runs don't share a filesystem,
and ``run_tool`` recovers the function's return value.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

import pytest

from tulip.agent.hook_orchestrator import HookOrchestrator
from tulip.hooks.provider import BeforeToolCallEvent, HookPriority
from tulip.observability import get_event_bus, reset_event_bus, run_context
from tulip.security.policy import ControlPolicy
from tulip.tools import tool
from tulip.tools.sandbox import (
    SandboxEnforcerHook,
    SandboxError,
    SandboxExecutionError,
    SandboxManifest,
    SandboxResult,
    SandboxSpec,
    SubprocessSandbox,
    ToolSandbox,
    _compose_code,
    failure_message,
    normalize_sandbox,
    provider_label,
    resolve_sandbox,
    run_tool_sandboxed,
    source_for_sandbox,
    validate_sandboxable,
)


@tool(sandbox=True, labels={"code-exec"})
def boxed_add(a: int, b: int) -> dict:
    """Add two numbers in a box."""
    return {"total": a + b}


@tool(labels={"code-exec"})
def bare_labelled(x: int) -> int:
    """Labelled but not sandboxed."""
    return x


@tool
def plain(x: int) -> int:
    """Neither labelled nor sandboxed."""
    return x


def free_fn(a: int) -> int:
    """A plain module-level function (source-shippable)."""
    return a * 2


class _FakeOutcome:
    """A duck result that is NOT a SandboxResult — exercises normalization."""

    ok = True
    stdout = "out"
    stderr = ""
    value = {"faked": True}
    exit_code = 0
    timed_out = False
    duration_ms = 1.5


class _FakeProvider:
    """Minimal ToolSandbox: records the call, returns a duck outcome."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def run_tool(
        self,
        code: str,
        func: str,
        args: dict[str, Any] | None = None,
        *,
        manifest: Any | None = None,
        timeout: float = 30.0,
    ) -> _FakeOutcome:
        self.calls.append(
            {"code": code, "func": func, "args": args, "manifest": manifest, "timeout": timeout}
        )
        return _FakeOutcome()


class TestSubprocessSandbox:
    """The conformance invariants, on the built-in zero-infra provider."""

    def test_runs_code_and_reports_success(self) -> None:
        res = SubprocessSandbox().run_code("print('hello box')")
        assert res.ok
        assert "hello box" in res.stdout
        assert res.exit_code == 0
        assert res.duration_ms >= 0

    def test_failing_code_is_contained_and_not_ok(self) -> None:
        res = SubprocessSandbox().run_code("raise ValueError('nope')")
        assert not res.ok
        assert res.exit_code != 0
        assert "ValueError" in res.stderr

    def test_timeout_is_reported_not_hung(self) -> None:
        res = SubprocessSandbox().run_code("import time; time.sleep(30)", timeout=0.5)
        assert not res.ok
        assert res.timed_out
        assert res.exit_code == 124

    def test_manifest_files_are_seeded(self) -> None:
        manifest = SandboxManifest(files={"data/in.txt": "seeded!"})
        res = SubprocessSandbox().run_code("print(open('data/in.txt').read())", manifest=manifest)
        assert res.ok
        assert "seeded!" in res.stdout

    def test_manifest_file_escaping_the_workspace_raises(self) -> None:
        manifest = SandboxManifest(files={"../evil.txt": "x"})
        with pytest.raises(SandboxError, match="escapes the workspace"):
            SubprocessSandbox().run_code("pass", manifest=manifest)

    def test_host_env_is_never_inherited(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TULIP_TEST_HOST_SECRET", "s3cr3t")
        res = SubprocessSandbox().run_code(
            "import os; print(os.environ.get('TULIP_TEST_HOST_SECRET', 'ABSENT'))"
        )
        assert res.ok
        assert "ABSENT" in res.stdout
        assert "s3cr3t" not in res.stdout

    def test_only_manifest_env_reaches_the_box(self) -> None:
        manifest = SandboxManifest(env={"GRANTED": "yes"})
        res = SubprocessSandbox().run_code(
            "import os; print(os.environ.get('GRANTED', 'ABSENT'))", manifest=manifest
        )
        assert res.ok
        assert "yes" in res.stdout

    def test_runs_do_not_share_a_filesystem(self) -> None:
        box = SubprocessSandbox()
        first = box.run_code("open('marker.txt', 'w').write('here')")
        assert first.ok
        second = box.run_code("import os; print(os.path.exists('marker.txt'))")
        assert second.ok
        assert "False" in second.stdout

    def test_workspace_persists_across_runs(self, tmp_path: Any) -> None:
        manifest = SandboxManifest(workspace=str(tmp_path))
        box = SubprocessSandbox()
        first = box.run_code("open('marker.txt', 'w').write('here')", manifest=manifest)
        assert first.ok
        second = box.run_code("import os; print(os.path.exists('marker.txt'))", manifest=manifest)
        assert second.ok
        assert "True" in second.stdout

    def test_run_tool_returns_a_recoverable_value(self) -> None:
        code = "def f(a, b):\n    return {'total': a + b}\n"
        res = SubprocessSandbox().run_tool(code, "f", {"a": 2, "b": 3})
        assert res.ok
        assert res.value == {"total": 5}

    def test_run_tool_failure_surfaces_without_a_value(self) -> None:
        code = "def f():\n    raise RuntimeError('kaboom')\n"
        res = SubprocessSandbox().run_tool(code, "f")
        assert not res.ok
        assert res.value is None
        assert "kaboom" in res.stderr

    def test_run_tool_does_not_mutate_the_caller_manifest(self) -> None:
        manifest = SandboxManifest()
        code = "def f():\n    return 1\n"
        SubprocessSandbox().run_tool(code, "f", manifest=manifest)
        assert "_tool_args.json" not in manifest.files

    def test_satisfies_the_tool_sandbox_protocol(self) -> None:
        assert isinstance(SubprocessSandbox(), ToolSandbox)

    def test_compose_code_prepends_dep_preamble(self) -> None:
        composed = _compose_code("x = 1", ["left-pad"])
        assert "pip" in composed
        assert "'left-pad'" in composed
        assert composed.endswith("x = 1")
        assert _compose_code("x = 1", []) == "x = 1"


class TestNormalizeAndResolve:
    def test_none_and_false_mean_off(self) -> None:
        assert normalize_sandbox(None) is None
        assert normalize_sandbox(False) is None

    def test_true_means_default_spec(self) -> None:
        spec = normalize_sandbox(True)
        assert isinstance(spec, SandboxSpec)
        assert spec.provider is None

    def test_name_and_spec_and_provider_forms(self) -> None:
        named = normalize_sandbox("docker")
        assert named is not None
        assert named.provider == "docker"

        spec = SandboxSpec(timeout=5.0)
        assert normalize_sandbox(spec) is spec

        provider = _FakeProvider()
        wrapped = normalize_sandbox(provider)
        assert wrapped is not None
        assert wrapped.provider is provider

    def test_garbage_is_rejected(self) -> None:
        with pytest.raises(SandboxError, match="not a bool, provider name"):
            normalize_sandbox(42)

    def test_default_resolves_to_subprocess(self) -> None:
        assert isinstance(resolve_sandbox(SandboxSpec()), SubprocessSandbox)
        assert isinstance(resolve_sandbox(SandboxSpec(provider="local")), SubprocessSandbox)

    def test_env_var_picks_the_default_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TULIP_SANDBOX", "subprocess")
        assert isinstance(resolve_sandbox(SandboxSpec()), SubprocessSandbox)

    def test_named_provider_without_tulip_sandbox_fails_closed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "tulip_sandbox", None)
        with pytest.raises(SandboxError, match="needs the tulip-sandbox package"):
            resolve_sandbox(SandboxSpec(provider="docker"))

    def test_provider_object_passes_through(self) -> None:
        provider = _FakeProvider()
        assert resolve_sandbox(SandboxSpec(provider=provider)) is provider

    def test_non_provider_object_is_rejected(self) -> None:
        with pytest.raises(SandboxError, match="not a sandbox provider"):
            resolve_sandbox(SandboxSpec(provider=42))

    def test_provider_label_forms(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TULIP_SANDBOX", raising=False)
        assert provider_label(SandboxSpec()) == "subprocess"
        assert provider_label(SandboxSpec(provider="docker")) == "docker"
        assert provider_label(SandboxSpec(provider=_FakeProvider())) == "_FakeProvider"


class TestSourceAndValidation:
    def test_source_is_dedented_and_decorator_stripped(self) -> None:
        source = source_for_sandbox(boxed_add.fn)
        assert source.startswith("def boxed_add(")
        assert "@tool" not in source

    def test_unreadable_source_is_rejected(self) -> None:
        with pytest.raises(SandboxError, match="cannot read source"):
            source_for_sandbox(len)

    def test_lambda_is_rejected(self) -> None:
        with pytest.raises(SandboxError, match="no function definition"):
            source_for_sandbox(lambda x: x)

    def test_async_function_is_rejected(self) -> None:
        async def afn() -> None:
            """Async."""

        with pytest.raises(SandboxError, match="must be a synchronous function"):
            validate_sandboxable(afn)

    def test_context_parameter_is_rejected(self) -> None:
        def with_ctx(ctx: object, x: int) -> int:
            """Takes a context."""
            return x

        with pytest.raises(SandboxError, match="ctx/context parameter"):
            validate_sandboxable(with_ctx)

    def test_plain_function_passes(self) -> None:
        validate_sandboxable(free_fn)


class TestFailureMessage:
    def test_timeout_message(self) -> None:
        res = SandboxResult(ok=False, timed_out=True, exit_code=124)
        assert "timed out" in failure_message("t", res)

    def test_stderr_tail_is_included(self) -> None:
        res = SandboxResult(ok=False, stderr="x" * 1000 + "the actual error", exit_code=1)
        msg = failure_message("t", res)
        assert "the actual error" in msg
        assert len(msg) < 700

    def test_exit_code_when_stderr_empty(self) -> None:
        res = SandboxResult(ok=False, exit_code=7)
        assert "exit code 7" in failure_message("t", res)


class TestSandboxedToolExecution:
    async def test_execute_runs_in_the_box_and_formats_the_value(self) -> None:
        result = await boxed_add.execute(a=2, b=3)
        assert '"total": 5' in result

    async def test_execute_does_not_see_host_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TULIP_TEST_HOST_SECRET", "s3cr3t")

        @tool(sandbox=True)
        def peek() -> dict:
            """Peek at the environment."""
            import os  # noqa: PLC0415

            return {"secret": os.environ.get("TULIP_TEST_HOST_SECRET", "ABSENT")}

        result = await peek.execute()
        assert "ABSENT" in result
        assert "s3cr3t" not in result

    async def test_failure_raises_sandbox_execution_error(self) -> None:
        @tool(sandbox=True)
        def boom() -> None:
            """Always fails."""
            raise RuntimeError("kaboom")

        with pytest.raises(SandboxExecutionError, match="kaboom") as excinfo:
            await boom.execute()
        assert excinfo.value.result.ok is False

    async def test_timeout_surfaces_as_timed_out(self) -> None:
        @tool(sandbox=SandboxSpec(timeout=0.5))
        def sleepy() -> None:
            """Sleeps past the timeout."""
            import time  # noqa: PLC0415

            time.sleep(30)

        with pytest.raises(SandboxExecutionError, match="timed out") as excinfo:
            await sleepy.execute()
        assert excinfo.value.result.timed_out

    async def test_injected_provider_and_duck_result_normalization(self) -> None:
        provider = _FakeProvider()
        spec = SandboxSpec(provider=provider, timeout=9.0, env={"K": "V"}, deps=["pkg"])

        result = run_tool_sandboxed(spec, free_fn, {"a": 3})
        assert isinstance(result, SandboxResult)
        assert result.value == {"faked": True}

        call = provider.calls[0]
        assert call["func"] == "free_fn"
        assert call["args"] == {"a": 3}
        assert call["timeout"] == 9.0
        assert call["manifest"].env == {"K": "V"}
        assert call["manifest"].deps == ["pkg"]
        assert call["code"].startswith("def free_fn(")

    async def test_dunder_call_is_sandboxed_too(self) -> None:
        assert boxed_add(2, b=3) == {"total": 5}

    async def test_dunder_call_failure_raises(self) -> None:
        @tool(sandbox=True)
        def boom() -> None:
            """Always fails."""
            raise RuntimeError("kaboom")

        with pytest.raises(SandboxExecutionError, match="kaboom"):
            boom()

    async def test_unsandboxed_tool_untouched(self) -> None:
        assert plain.sandbox is None
        assert await plain.execute(x=1) == "1"
        assert plain(5) == 5

    def test_decorator_stores_labels_and_spec(self) -> None:
        assert boxed_add.labels == frozenset({"code-exec"})
        assert isinstance(boxed_add.sandbox, SandboxSpec)
        assert bare_labelled.sandbox is None

    def test_decorator_sandbox_false_means_off(self) -> None:
        @tool(sandbox=False)
        def off(x: int) -> int:
            """Off."""
            return x

        assert off.sandbox is None

    def test_decorator_rejects_async_sandboxed_tool(self) -> None:
        with pytest.raises(SandboxError, match="synchronous"):

            @tool(sandbox=True)
            async def nope() -> None:
                """Async tool."""

    def test_decorator_rejects_ctx_sandboxed_tool(self) -> None:
        with pytest.raises(SandboxError, match="ctx/context"):

            @tool(sandbox=True)
            def with_ctx(ctx: object) -> None:
                """Context tool."""

    async def test_events_are_published_on_the_bus(self) -> None:
        reset_event_bus()
        try:
            bus = get_event_bus()
            received: list[Any] = []

            async def consumer() -> None:
                async for ev in bus.subscribe("sandbox-run"):
                    received.append(ev)
                    if ev.event_type == "tool.sandbox.completed":
                        return

            consumer_task = asyncio.create_task(consumer())
            await asyncio.sleep(0)

            provider = _FakeProvider()

            @tool(sandbox=SandboxSpec(provider=provider))
            def fast(x: int) -> int:
                """Fake-run."""
                return x

            async with run_context("sandbox-run"):
                await fast.execute(x=1)

            await asyncio.wait_for(consumer_task, timeout=2.0)
            types = [ev.event_type for ev in received]
            assert types == ["tool.sandbox.started", "tool.sandbox.completed"]
            assert received[0].data["tool"] == "fast"
            assert received[1].data["ok"] is True
        finally:
            reset_event_bus()


class TestSandboxEnforcerHook:
    POLICY = ControlPolicy(require_sandbox_for=frozenset({"code-exec"}))

    def _event(self, tool_name: str) -> BeforeToolCallEvent:
        return BeforeToolCallEvent(tool_name=tool_name, tool_call_id="tc-1", arguments={})

    async def test_cancels_unsandboxed_tool_with_required_label(self) -> None:
        hook = SandboxEnforcerHook(self.POLICY, [boxed_add, bare_labelled, plain])
        event = self._event("bare_labelled")
        await hook.on_before_tool_call(event)
        assert isinstance(event.cancel, str)
        assert "require sandboxed execution" in event.cancel
        assert "code-exec" in event.cancel

    async def test_sandboxed_tool_passes(self) -> None:
        hook = SandboxEnforcerHook(self.POLICY, [boxed_add, bare_labelled, plain])
        event = self._event("boxed_add")
        await hook.on_before_tool_call(event)
        assert event.cancel is False

    async def test_unlabelled_tool_passes(self) -> None:
        hook = SandboxEnforcerHook(self.POLICY, [boxed_add, bare_labelled, plain])
        event = self._event("plain")
        await hook.on_before_tool_call(event)
        assert event.cancel is False

    async def test_unknown_tool_is_left_alone(self) -> None:
        hook = SandboxEnforcerHook(self.POLICY, [boxed_add])
        event = self._event("mystery")
        await hook.on_before_tool_call(event)
        assert event.cancel is False

    async def test_default_priority_is_security_band(self) -> None:
        hook = SandboxEnforcerHook(self.POLICY, [])
        assert hook.priority == HookPriority.SECURITY_DEFAULT
        assert hook.name == "SandboxEnforcerHook"

    async def test_through_the_orchestrator(self) -> None:
        hook = SandboxEnforcerHook(self.POLICY, [bare_labelled])
        orchestrator = HookOrchestrator([hook])
        event = await orchestrator.run_before_tool("bare_labelled", "tc-2", {"x": 1})
        assert isinstance(event.cancel, str)
