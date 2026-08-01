# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Sandboxed tool execution — run a tool's code in an isolated box, not the host.

``@tool(sandbox=True)`` ships the tool function's *source* into an isolated
environment and executes it there; the host process never runs the body. The
zero-infra default is :class:`SubprocessSandbox` — a fresh working directory,
``python -I``, and a scrubbed environment (``PATH``/``LANG`` plus whatever the
manifest explicitly grants) per call. Stronger boundaries plug in through the
same structural :class:`ToolSandbox` protocol: any provider from the
``tulip-sandbox`` package (Docker, Firecracker, SSH, Lambda) satisfies it
unchanged, without either package importing the other.

The subprocess tier is process + environment isolation only — it is NOT a
network or filesystem boundary. Escalate the provider (``TULIP_SANDBOX=docker``,
a provider object, or a :class:`SandboxSpec`) when the code is truly untrusted.

A sandboxed tool must be self-contained: a synchronous, module-level function
whose imports live inside its body, taking JSON-serializable arguments and
returning a JSON-serializable value. Closures, globals, and ``ToolContext``
do not cross the boundary.

Governance: :class:`~tulip.security.policy.ControlPolicy` can *require* the
sandbox — ``require_sandbox_for`` names the labels whose actions must carry
the ``sandboxed`` tag or be denied by
:func:`~tulip.security.policy.approve`, and :class:`SandboxEnforcerHook`
enforces the same rule at the agent loop's admission seam by cancelling an
un-sandboxed call to a matching tool.
"""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

from pydantic import BaseModel, Field

from tulip.hooks.provider import HookPriority, HookProvider
from tulip.observability.emit import EV_TOOL_SANDBOX_DENIED, emit


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from tulip.hooks.provider import BeforeToolCallEvent
    from tulip.security.policy import ControlPolicy
    from tulip.tools.decorator import Tool


# Same marker the tulip-sandbox package prints, so results round-trip through
# either implementation of the run_tool protocol.
TOOL_RESULT_MARKER = "__TULIP_TOOL_RESULT__"

_ARGS_FILE = "_tool_args.json"

# The only environment a box inherits; everything else must be granted
# explicitly via the manifest.
_BASE_ENV = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}

_LOCAL_PROVIDER_NAMES = frozenset({"subprocess", "local"})

_STDERR_TAIL = 500


class SandboxError(Exception):
    """The sandbox itself failed (bad provider, escaped path, unshippable
    function) — distinct from the sandboxed *code* failing, which is reported
    as :class:`SandboxExecutionError` / ``SandboxResult(ok=False)``."""


class SandboxExecutionError(Exception):
    """The sandboxed code ran and failed; carries the box's diagnostics."""

    def __init__(self, message: str, *, result: SandboxResult) -> None:
        super().__init__(message)
        self.result = result


class SandboxManifest(BaseModel):
    """What the box receives — nothing else reaches it.

    Duck-compatible with ``tulip_sandbox.Manifest`` (same field names), so it
    can be handed to any provider from that package unchanged.
    """

    files: dict[str, str] = Field(default_factory=dict)
    """Relative path → content, seeded into the box's working directory."""

    env: dict[str, str] = Field(default_factory=dict)
    """The ONLY environment variables the code sees (beyond ``PATH``/``LANG``)."""

    deps: list[str] = Field(default_factory=list)
    """pip packages installed into a workspace-local ``.deps`` before the run."""

    workspace: str | None = None
    """Persistent working directory; ``None`` means a fresh box per run."""


class SandboxResult(BaseModel):
    """Outcome of one sandboxed run (same shape as ``tulip_sandbox.RunResult``)."""

    ok: bool
    stdout: str = ""
    stderr: str = ""
    value: Any = None
    exit_code: int = 0
    timed_out: bool = False
    duration_ms: float = 0.0


@runtime_checkable
class ToolSandbox(Protocol):
    """Structural contract for a sandbox provider — the ``run_tool`` verb.

    Providers from the ``tulip-sandbox`` package satisfy this protocol without
    either package importing the other; a custom provider only needs the one
    method. ``isinstance`` checks method presence (``runtime_checkable``), not
    signatures.
    """

    def run_tool(
        self,
        code: str,
        func: str,
        args: dict[str, Any] | None = None,
        *,
        manifest: Any | None = None,
        timeout: float = 30.0,
    ) -> Any:
        """Run ``func(**args)`` defined in ``code``; return an object exposing
        ``ok`` / ``stdout`` / ``stderr`` / ``value`` / ``exit_code`` /
        ``timed_out`` / ``duration_ms``."""
        ...  # pragma: no cover - protocol signature


class SandboxSpec(BaseModel):
    """Per-tool sandbox configuration, normalized from ``@tool(sandbox=...)``.

    ``provider`` may be ``None`` (resolve the default: ``$TULIP_SANDBOX``,
    falling back to the built-in subprocess box), a provider *name*
    (``"subprocess"``/``"local"`` are built in; other names resolve through
    the optional ``tulip-sandbox`` package), or a ready provider object
    satisfying :class:`ToolSandbox`.
    """

    provider: Any = None
    timeout: float = 30.0
    env: dict[str, str] = Field(default_factory=dict)
    deps: list[str] = Field(default_factory=list)
    workspace: str | None = None

    model_config = {"arbitrary_types_allowed": True}


def normalize_sandbox(sandbox: Any) -> SandboxSpec | None:
    """Normalize the ``@tool(sandbox=...)`` argument into a spec (or ``None``).

    Accepts ``None``/``False`` (off), ``True`` (default provider), a provider
    name, a :class:`SandboxSpec`, or a provider object.
    """
    if sandbox is None or sandbox is False:
        return None
    if sandbox is True:
        return SandboxSpec()
    if isinstance(sandbox, SandboxSpec):
        return sandbox
    if isinstance(sandbox, str):
        return SandboxSpec(provider=sandbox)
    if isinstance(sandbox, ToolSandbox):
        return SandboxSpec(provider=sandbox)
    raise SandboxError(
        f"sandbox={sandbox!r} is not a bool, provider name, SandboxSpec, "
        "or provider object with a run_tool method"
    )


def resolve_sandbox(spec: SandboxSpec) -> ToolSandbox:
    """Resolve a spec to a live provider.

    Names other than the built-in ``subprocess``/``local`` are delegated to
    the ``tulip-sandbox`` package's factory when it is installed.
    """
    provider = spec.provider
    if provider is None:
        provider = os.environ.get("TULIP_SANDBOX", "subprocess")
    if isinstance(provider, str):
        name = provider.lower()
        if name in _LOCAL_PROVIDER_NAMES:
            return SubprocessSandbox()
        try:
            from tulip_sandbox import build_provider  # noqa: PLC0415
        except ImportError as exc:
            raise SandboxError(
                f"sandbox provider {provider!r} needs the tulip-sandbox package "
                "(only 'subprocess' is built in)"
            ) from exc
        return cast("ToolSandbox", build_provider(name))
    if isinstance(provider, ToolSandbox):
        return provider
    raise SandboxError(f"not a sandbox provider: {provider!r} (needs a run_tool method)")


def provider_label(spec: SandboxSpec) -> str:
    """A short human/event label for the spec's provider."""
    provider = spec.provider
    if provider is None:
        return os.environ.get("TULIP_SANDBOX", "subprocess")
    if isinstance(provider, str):
        return provider
    return type(provider).__name__


def validate_sandboxable(fn: Callable[..., Any]) -> None:
    """Fail at decoration time when ``fn`` cannot run in a box.

    Raises :class:`SandboxError` for async functions (the box calls
    ``func(**args)`` synchronously), functions taking a ``ctx``/``context``
    parameter (``ToolContext`` cannot cross the boundary), and functions whose
    source cannot be recovered.
    """
    name = getattr(fn, "__name__", repr(fn))
    if inspect.iscoroutinefunction(fn):
        raise SandboxError(
            f"sandboxed tool {name!r} must be a synchronous function — "
            "it runs in a separate process that calls it directly"
        )
    if any(p in ("ctx", "context") for p in inspect.signature(fn).parameters):
        raise SandboxError(
            f"sandboxed tool {name!r} cannot take a ctx/context parameter — "
            "ToolContext does not cross the sandbox boundary"
        )
    source_for_sandbox(fn)


def source_for_sandbox(fn: Callable[..., Any]) -> str:
    """The function's source, dedented and decorator-stripped — what ships
    into the box.

    The box sees only this text, so a sandboxed tool must keep its imports
    inside the function body; closures and module globals are not carried
    over.
    """
    name = getattr(fn, "__name__", repr(fn))
    try:
        raw = inspect.getsource(fn)
    except (OSError, TypeError) as exc:
        raise SandboxError(
            f"cannot read source for {name!r}: a sandboxed tool must be a "
            "plain function defined in a source file"
        ) from exc
    lines = textwrap.dedent(raw).splitlines()
    for i, line in enumerate(lines):
        if line.startswith(("def ", "async def ")):
            return "\n".join(lines[i:])
    raise SandboxError(f"no function definition found in the source of {name!r}")


def failure_message(tool_name: str, result: SandboxResult) -> str:
    """One-line diagnosis of a failed sandboxed run, safe to show the model."""
    if result.timed_out:
        return f"sandboxed tool {tool_name!r} timed out"
    tail = result.stderr.strip()[-_STDERR_TAIL:]
    detail = tail or f"exit code {result.exit_code}"
    return f"sandboxed tool {tool_name!r} failed: {detail}"


def run_tool_sandboxed(
    spec: SandboxSpec,
    fn: Callable[..., Any],
    arguments: dict[str, Any],
) -> SandboxResult:
    """Resolve the provider, ship ``fn``'s source, run it, return the outcome.

    Blocking — async callers run it in a worker thread. Provider results are
    normalized onto :class:`SandboxResult` by attribute (duck) access so any
    :class:`ToolSandbox` implementation works.
    """
    provider = resolve_sandbox(spec)
    source = source_for_sandbox(fn)
    manifest = SandboxManifest(env=dict(spec.env), deps=list(spec.deps), workspace=spec.workspace)
    raw = provider.run_tool(source, fn.__name__, arguments, manifest=manifest, timeout=spec.timeout)
    if isinstance(raw, SandboxResult):
        return raw
    return SandboxResult(
        ok=bool(getattr(raw, "ok", False)),
        stdout=str(getattr(raw, "stdout", "")),
        stderr=str(getattr(raw, "stderr", "")),
        value=getattr(raw, "value", None),
        exit_code=int(getattr(raw, "exit_code", 1)),
        timed_out=bool(getattr(raw, "timed_out", False)),
        duration_ms=float(getattr(raw, "duration_ms", 0.0)),
    )


def _as_text(value: str | bytes | None) -> str:
    """Coerce subprocess output (typed ``str | bytes | None``) to text."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _seed_files(root: Path, files: dict[str, str]) -> None:
    """Write manifest files under ``root``, refusing paths that escape it."""
    base = root.resolve()
    for rel, content in files.items():
        target = (base / rel).resolve()
        if not target.is_relative_to(base):
            raise SandboxError(f"manifest file escapes the workspace: {rel!r}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def _deps_preamble(deps: list[str]) -> str:
    """In-box preamble that pip-installs ``deps`` into a local ``.deps`` dir."""
    pkgs = ", ".join(repr(d) for d in deps)
    return (
        "import os as _os, subprocess as _sp, sys as _sys\n"
        "_target = _os.path.join(_os.getcwd(), '.deps')\n"
        "_proc = _sp.run(\n"
        f"    [_sys.executable, '-m', 'pip', 'install', '--quiet', '--target', _target, {pkgs}],\n"
        "    capture_output=True, text=True, check=False,\n"
        ")\n"
        "if _proc.returncode != 0:\n"
        f"    raise RuntimeError('dep install failed: ' + _proc.stderr[-{_STDERR_TAIL}:])\n"
        "_sys.path.insert(0, _target)\n"
    )


def _compose_code(code: str, deps: list[str]) -> str:
    """The exact program the box runs: optional dep preamble, then the code."""
    if not deps:
        return code
    return _deps_preamble(deps) + "\n" + code


def _tool_wrapper(code: str, func: str) -> str:
    """Append the loader that calls ``func`` with the seeded args and prints
    the marker line the result is recovered from."""
    return (
        f"{code}\n\n"
        "import json as _json\n"
        f"with open({_ARGS_FILE!r}, encoding='utf-8') as _f:\n"
        "    _args = _json.load(_f)\n"
        f"_result = {func}(**_args)\n"
        f"print({TOOL_RESULT_MARKER!r} + _json.dumps(_result))\n"
    )


def _extract_value(stdout: str) -> Any:
    """JSON-decode the last marker line; ``None`` when the run printed none."""
    value: Any = None
    for line in stdout.splitlines():
        if line.startswith(TOOL_RESULT_MARKER):
            value = json.loads(line[len(TOOL_RESULT_MARKER) :])
    return value


class SubprocessSandbox:
    """Zero-infra sandbox: fresh cwd, ``python -I``, scrubbed env, per-call
    timeout.

    Process + environment isolation only — no network or filesystem boundary.
    The development default; escalate the provider for truly untrusted code.
    """

    def __init__(self, python: str | None = None) -> None:
        self._python = python or sys.executable

    def run_code(
        self,
        code: str,
        *,
        manifest: Any | None = None,
        timeout: float = 30.0,
    ) -> SandboxResult:
        """Run a program in a fresh box; never raises for in-box failures."""
        m = manifest if manifest is not None else SandboxManifest()
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="tulip-sandbox-") as tmp:
            root = Path(m.workspace) if m.workspace else Path(tmp)
            root.mkdir(parents=True, exist_ok=True)
            _seed_files(root, dict(m.files))
            env = {**_BASE_ENV, **m.env}
            argv = [self._python, "-I", "-c", _compose_code(code, list(m.deps))]
            try:
                proc = subprocess.run(  # noqa: S603 — executing model/tool code in a scrubbed box is this class's purpose
                    argv,
                    cwd=root,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                return SandboxResult(
                    ok=False,
                    stdout=_as_text(exc.stdout),
                    stderr=_as_text(exc.stderr),
                    exit_code=124,
                    timed_out=True,
                    duration_ms=(time.perf_counter() - started) * 1000,
                )
        return SandboxResult(
            ok=proc.returncode == 0,
            stdout=proc.stdout,
            stderr=proc.stderr,
            exit_code=proc.returncode,
            duration_ms=(time.perf_counter() - started) * 1000,
        )

    def run_tool(
        self,
        code: str,
        func: str,
        args: dict[str, Any] | None = None,
        *,
        manifest: Any | None = None,
        timeout: float = 30.0,
    ) -> SandboxResult:
        """Run ``func(**args)`` defined in ``code``; recover its return value."""
        base = manifest if manifest is not None else SandboxManifest()
        m = base.model_copy(deep=True)
        m.files[_ARGS_FILE] = json.dumps(args or {})
        result = self.run_code(_tool_wrapper(code, func), manifest=m, timeout=timeout)
        return result.model_copy(update={"value": _extract_value(result.stdout)})


class SandboxEnforcerHook(HookProvider):
    """Cancel tool calls that policy requires sandboxed but aren't.

    :class:`~tulip.security.policy.ControlPolicy.require_sandbox_for` names
    the labels whose tools must run sandboxed. When the model calls a tool
    carrying one of those labels and the tool has no ``sandbox`` configured,
    the call is cancelled via ``event.cancel`` — the body never runs and the
    model sees the reason as the tool result. This is the loop-seam
    counterpart of the ``sandboxed``-tag rule in
    :func:`~tulip.security.policy.approve`.

    Args:
        policy: The governing policy; only ``require_sandbox_for`` is read.
        tools: The agent's tools, used to look up labels and sandbox config
            by name. Tools not in this collection are left alone.
        priority: Hook priority; defaults to the security band so the gate
            fires before observability/business hooks.

    Example:
        from tulip import Agent
        from tulip.security.policy import ControlPolicy
        from tulip.tools.sandbox import SandboxEnforcerHook

        policy = ControlPolicy(require_sandbox_for=frozenset({"code-exec"}))
        agent = Agent(
            model="openai:gpt-4o",
            tools=[run_snippet, lookup],
            hooks=[SandboxEnforcerHook(policy, [run_snippet, lookup])],
        )
    """

    def __init__(
        self,
        policy: ControlPolicy,
        tools: Iterable[Tool],
        *,
        priority: int = HookPriority.SECURITY_DEFAULT,
    ) -> None:
        self._policy = policy
        self._tools = {t.name: t for t in tools}
        self._priority = priority

    @property
    def priority(self) -> int:
        return self._priority

    async def on_before_tool_call(self, event: BeforeToolCallEvent) -> None:
        """Cancel the call when its labels require a sandbox it doesn't have."""
        tool = self._tools.get(event.tool_name)
        if tool is None or tool.sandbox is not None:
            return
        required = tool.labels & self._policy.require_sandbox_for
        if not required:
            return
        await emit(EV_TOOL_SANDBOX_DENIED, tool=tool.name, labels=sorted(required))
        event.cancel = (
            f"ControlPolicy blocked: tool {tool.name!r} carries labels "
            f"{sorted(required)}, which require sandboxed execution — "
            "declare it with @tool(sandbox=...)."
        )


__all__ = [
    "TOOL_RESULT_MARKER",
    "SandboxEnforcerHook",
    "SandboxError",
    "SandboxExecutionError",
    "SandboxManifest",
    "SandboxResult",
    "SandboxSpec",
    "SubprocessSandbox",
    "ToolSandbox",
    "failure_message",
    "normalize_sandbox",
    "provider_label",
    "resolve_sandbox",
    "run_tool_sandboxed",
    "source_for_sandbox",
    "validate_sandboxable",
]
