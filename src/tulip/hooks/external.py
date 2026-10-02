# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Hooks an operator writes as programs, not Python: command and HTTP hooks.

:class:`~tulip.hooks.provider.HookProvider` is how code that ships *with* an
agent observes and steers it. An operator running somebody else's agent needs
the same leverage without writing a provider: *run the formatter after every
edit*, *refuse any ``git push`` to main*, *do not let it stop until the tests
pass*. Claude Code and Codex answer that with hooks configured in a settings
file — a program run at a lifecycle event, handed the event as JSON on stdin,
answering with an exit code and optional JSON on stdout. This module is that
contract, mapped onto Tulip's hook system::

    {
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": "bash",
                    "hooks": [
                        {
                            "type": "command",
                            "command": "./scripts/guard.sh",
                            "timeout": 10,
                        }
                    ],
                }
            ],
            "PostToolUse": [
                {
                    "matcher": "edit|write",
                    "hooks": [
                        {"type": "command", "command": "ruff format --quiet ."}
                    ],
                }
            ],
            "Stop": [
                {
                    "hooks": [
                        {"type": "command", "command": "./scripts/tests-pass.sh"}
                    ]
                }
            ],
        }
    }

    hooks = ExternalHooks(HookConfig.from_settings(settings["hooks"]), cwd=project)
    agent = Agent(model=..., hooks=[hooks], final_answer_verifier=hooks.verifier())

**How each event maps.**

``PreToolUse``         :meth:`ExternalHooks.on_before_tool_call` — deny cancels the
                       call with the reason; ``updatedInput`` replaces the arguments;
                       allow / ask are handed to the host's gate (``on_permission``).
``PostToolUse``        :meth:`ExternalHooks.on_after_tool_call` — a block reason or
                       ``additionalContext`` is appended to the result the model reads.
``UserPromptSubmit``   :meth:`ExternalHooks.on_before_invocation` — a block raises
                       :class:`HookBlockedError` before the model is called;
                       context (or plain stdout) is appended to the prompt.
``Stop`` /             :meth:`ExternalHooks.verifier` — a final-answer verifier: a
``SubagentStop``       block sends its reason back to the model and the loop goes on.
                       That is how *verify before finishing* becomes enforced rather
                       than requested. ``stop_hook_active`` is set from the second
                       attempt; ``final_answer_verifier_max_replans`` bounds it.
``SessionStart``,      the host fires these with :meth:`ExternalHooks.run`; the SDK
``SessionEnd``,        has no session concept of its own. ``SessionStart`` context is
``PreCompact``,        returned for the host to add to the conversation.
``Notification``

**The protocol.** Exit 0: success; stdout that is a JSON object is read for
``decision`` / ``reason``, ``continue`` / ``stopReason``, ``systemMessage`` and
``hookSpecificOutput`` (``permissionDecision``, ``permissionDecisionReason``,
``updatedInput``, ``additionalContext``). Exit 2: a blocking error — stderr is
the reason. Any other exit, a timeout or a crash: a non-blocking error, reported
and otherwise ignored, so a broken hook costs its own effect and nothing else.
An HTTP hook POSTs the same JSON and reads its response body like stdout; a
non-2xx status is a non-blocking error.

**Nothing runs unseen.** Every execution — including one the host's ``guard``
refused — is reported to ``on_run`` as a :class:`HookRun` with its exit code,
duration, output and decision, so a host can stream it and audit it. A hook is
a side effect the operator configured, and it is held to the same bar as any
other: gated (``guard``) and recorded (``on_run``).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Awaitable, Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from tulip.hooks.provider import HookPriority, HookProvider


if TYPE_CHECKING:
    from tulip.agent.verification import FinalAnswerContext
    from tulip.core.state import AgentState
    from tulip.hooks.provider import AfterToolCallEvent, BeforeToolCallEvent


__all__ = [
    "EVENTS",
    "ExternalHooks",
    "HookBlockedError",
    "HookCommand",
    "HookConfig",
    "HookOutcome",
    "HookRun",
]

#: Events a configuration may name.
EVENTS: tuple[str, ...] = (
    "PreToolUse",
    "PostToolUse",
    "UserPromptSubmit",
    "Stop",
    "SubagentStop",
    "SessionStart",
    "SessionEnd",
    "PreCompact",
    "Notification",
)

#: Events whose exit-2 / ``decision: "block"`` changes what happens next.
#: Elsewhere a block is reported to the operator and changes nothing.
_BLOCKABLE = frozenset({"PreToolUse", "PostToolUse", "UserPromptSubmit", "Stop", "SubagentStop"})

#: Events whose plain stdout (not JSON) is context for the model.
_STDOUT_IS_CONTEXT = frozenset({"UserPromptSubmit", "SessionStart"})

DEFAULT_TIMEOUT = 60.0

#: Characters of stdout / stderr kept per run. A hook that prints a build log
#: must not put a megabyte into the conversation or the audit trail.
MAX_OUTPUT = 20_000

_STRENGTH = {None: 0, "allow": 1, "ask": 2, "block": 3, "deny": 3}


class HookBlockedError(RuntimeError):
    """A hook refused to let the run proceed (``UserPromptSubmit``)."""

    def __init__(self, event: str, reason: str) -> None:
        self.event = event
        self.reason = reason
        super().__init__(f"blocked by {event} hook: {reason}")


@dataclass(frozen=True)
class HookCommand:
    """One configured hook.

    Attributes:
        type: ``"command"`` (a shell command line) or ``"http"`` (a URL POSTed to).
        target: The command line or the URL.
        timeout: Seconds before it is killed and reported as timed out.
        matcher: The matcher it was configured under, for the record.
        source: Where it was configured: a settings file path.
        headers: Extra HTTP headers, for ``http`` hooks.
    """

    type: Literal["command", "http"]
    target: str
    timeout: float = DEFAULT_TIMEOUT
    matcher: str = ""
    source: str = ""
    headers: Mapping[str, str] = field(default_factory=dict)

    def matches(self, subject: str | None) -> bool:
        """Whether this hook's matcher covers ``subject`` (a tool name, a source…).

        Empty and ``*`` match everything; anything else is a regex that must
        match the whole subject, case-insensitively — so ``Bash`` matches the
        ``bash`` tool and ``edit|write`` matches either. A matcher that is not
        a valid regex is compared as plain text.
        """
        if self.matcher in ("", "*") or subject is None:
            return True
        try:
            return re.fullmatch(self.matcher, subject, re.IGNORECASE) is not None
        except re.error:
            return self.matcher.lower() == subject.lower()


@dataclass(frozen=True)
class HookConfig:
    """Hooks per event, in configuration order.

    Attributes:
        hooks: ``event -> hooks``.
        problems: Entries that were skipped and why — an unknown event, a hook
            type this runtime does not run. Surfaced so a typo is visible.
    """

    hooks: Mapping[str, tuple[HookCommand, ...]] = field(default_factory=dict)
    problems: tuple[str, ...] = ()

    @classmethod
    def from_settings(cls, block: Mapping[str, Any] | None, *, source: str = "") -> HookConfig:
        """Read a ``hooks`` settings block (Claude Code's shape).

        ``{"Event": [{"matcher": "...", "hooks": [{"type": "command",
        "command": "...", "timeout": 30}]}]}``. Malformed entries are skipped
        and listed in :attr:`problems`; a block that is not a mapping raises
        ``TypeError``.
        """
        if block is None:
            return cls()
        if not isinstance(block, Mapping):
            raise TypeError("hooks must be an object of event -> matcher groups")
        out: dict[str, list[HookCommand]] = {}
        problems: list[str] = []
        for event, groups in block.items():
            if event not in EVENTS:
                problems.append(f"{source}: unknown hook event {event!r}")
                continue
            if not isinstance(groups, list):
                problems.append(f"{source}: hooks.{event} must be a list of matcher groups")
                continue
            for group in groups:
                if not isinstance(group, Mapping):
                    problems.append(f"{source}: hooks.{event} has an entry that is not an object")
                    continue
                matcher = str(group.get("matcher") or "")
                for raw in group.get("hooks") or []:
                    hook, problem = _hook_from(raw, matcher, source, event)
                    if hook is not None:
                        out.setdefault(event, []).append(hook)
                    if problem:
                        problems.append(problem)
        return cls(hooks={k: tuple(v) for k, v in out.items()}, problems=tuple(problems))

    def merged(self, other: HookConfig) -> HookConfig:
        """Both configurations' hooks: ``self``'s first, then ``other``'s."""
        hooks = {k: tuple(v) for k, v in self.hooks.items()}
        for event, more in other.hooks.items():
            hooks[event] = hooks.get(event, ()) + tuple(more)
        return HookConfig(hooks=hooks, problems=self.problems + other.problems)

    def for_event(self, event: str, subject: str | None = None) -> list[HookCommand]:
        """The hooks that run for ``event`` on ``subject``."""
        return [h for h in self.hooks.get(event, ()) if h.matches(subject)]

    def has(self, event: str) -> bool:
        return bool(self.hooks.get(event))

    def __len__(self) -> int:
        return sum(len(v) for v in self.hooks.values())


def _hook_from(
    raw: Any, matcher: str, source: str, event: str
) -> tuple[HookCommand | None, str | None]:
    if not isinstance(raw, Mapping):
        return None, f"{source}: hooks.{event} has a hook that is not an object"
    kind = str(raw.get("type") or "command")
    timeout = raw.get("timeout", DEFAULT_TIMEOUT)
    try:
        seconds = float(timeout)
    except (TypeError, ValueError):
        return None, f"{source}: hooks.{event} timeout {timeout!r} is not a number"
    if seconds <= 0:
        return None, f"{source}: hooks.{event} timeout must be positive"
    if kind == "command" and raw.get("command"):
        return HookCommand("command", str(raw["command"]), seconds, matcher, source), None
    if kind == "http" and raw.get("url"):
        headers = {str(k): str(v) for k, v in (raw.get("headers") or {}).items()}
        return HookCommand("http", str(raw["url"]), seconds, matcher, source, headers), None
    if kind in ("command", "http"):
        return (
            None,
            f"{source}: hooks.{event} {kind} hook has no {'command' if kind == 'command' else 'url'}",
        )
    return None, f"{source}: hooks.{event} hook type {kind!r} is not supported (command, http)"


@dataclass(frozen=True)
class HookRun:
    """One hook execution, as it is reported and audited."""

    event: str
    hook: HookCommand
    exit_code: int | None
    duration_ms: int
    stdout: str = ""
    stderr: str = ""
    #: Why it did not succeed: ``"timed out after 5s"``, ``"exit 1"``,
    #: ``"refused: ..."``. ``None`` when it ran and exited 0 or 2.
    error: str | None = None
    #: What it decided: ``allow`` / ``ask`` / ``deny`` / ``block`` or ``None``.
    decision: str | None = None
    reason: str = ""

    def as_dict(self, limit: int = 400) -> dict[str, Any]:
        """A JSON-ready summary, output cut to ``limit`` characters each."""
        return {
            "event": self.event,
            "type": self.hook.type,
            "target": self.hook.target,
            "matcher": self.hook.matcher,
            "source": self.hook.source,
            "exit_code": self.exit_code,
            "duration_ms": self.duration_ms,
            "decision": self.decision,
            "reason": self.reason,
            "error": self.error,
            "stdout": _cut(self.stdout, limit),
            "stderr": _cut(self.stderr, limit),
        }


@dataclass
class HookOutcome:
    """Every hook of one event, folded into one answer. Strongest decision wins."""

    event: str
    runs: list[HookRun] = field(default_factory=list)
    decision: str | None = None
    reasons: list[str] = field(default_factory=list)
    context: list[str] = field(default_factory=list)
    updated_input: dict[str, Any] | None = None
    system_messages: list[str] = field(default_factory=list)
    #: A hook said ``"continue": false``.
    stop: bool = False
    stop_reason: str = ""

    @property
    def reason(self) -> str:
        return "\n".join(r for r in self.reasons if r)

    @property
    def blocked(self) -> bool:
        return self.decision in ("block", "deny")

    @property
    def additional_context(self) -> str:
        return "\n\n".join(c for c in self.context if c)


class ExternalHooks(HookProvider):
    """Runs configured command and HTTP hooks at Tulip's lifecycle points.

    Args:
        config: What to run, per event.
        cwd: Working directory for command hooks, and ``cwd`` in the payload.
        session_id: ``session_id`` in the payload when the run has no thread.
        env: Extra environment variables for command hooks.
        on_run: Called with every :class:`HookRun` — stream it, audit it.
        guard: ``(hook, event) -> refusal or None``, consulted before each
            execution. A refusal is reported and the hook does not run.
        on_permission: ``(tool, arguments, decision, reason)`` for a
            ``PreToolUse`` allow or ask, so the host's own gate can honour it.
            A deny never reaches it: it cancels the call outright.
        payload: Extra fields for every payload, computed per event — the
            host's permission mode, a transcript path.
    """

    def __init__(  # noqa: PLR0913 — every one is an optional, independent seam
        self,
        config: HookConfig,
        *,
        cwd: str | os.PathLike[str] | None = None,
        session_id: str = "",
        env: Mapping[str, str] | None = None,
        on_run: Callable[[HookRun], None] | None = None,
        guard: Callable[[HookCommand, str], str | None] | None = None,
        on_permission: Callable[[str, dict[str, Any], str, str], None] | None = None,
        payload: Callable[[str], Mapping[str, Any]] | None = None,
    ) -> None:
        self.config = config
        self.cwd = Path(cwd) if cwd is not None else Path.cwd()
        self.session_id = session_id
        self._env = dict(env or {})
        self._on_run = on_run
        self._guard = guard
        self._on_permission = on_permission
        self._payload = payload

    @property
    def priority(self) -> int:
        # A PreToolUse deny is a security decision: it runs with the guardrails.
        return HookPriority.SECURITY_DEFAULT

    # -------------------------------------------------------------- running --

    def run(
        self,
        event: str,
        payload: Mapping[str, Any] | None = None,
        *,
        subject: str | None = None,
        session_id: str | None = None,
    ) -> HookOutcome:
        """Run every hook for ``event`` matching ``subject``; fold their answers.

        Hooks of one event run concurrently, as each is an independent
        program. Blocking: call from a thread, or use :meth:`arun`.
        """
        hooks = self.config.for_event(event, subject)
        outcome = HookOutcome(event=event)
        if not hooks:
            return outcome
        body = {
            "session_id": session_id or self.session_id,
            "cwd": str(self.cwd),
            "hook_event_name": event,
            **(dict(self._payload(event)) if self._payload is not None else {}),
            **dict(payload or {}),
        }
        stdin = json.dumps(body, default=str)
        if len(hooks) == 1:
            runs = [self._execute(event, hooks[0], stdin)]
        else:
            with ThreadPoolExecutor(max_workers=len(hooks)) as pool:
                runs = list(pool.map(lambda h: self._execute(event, h, stdin), hooks))
        for hook_run, parsed in runs:
            _fold(outcome, hook_run, parsed)
            if self._on_run is not None:
                self._on_run(hook_run)
        return outcome

    async def arun(
        self,
        event: str,
        payload: Mapping[str, Any] | None = None,
        *,
        subject: str | None = None,
        session_id: str | None = None,
    ) -> HookOutcome:
        """:meth:`run` on a worker thread, so the event loop keeps turning."""
        if not self.config.for_event(event, subject):
            return HookOutcome(event=event)
        return await asyncio.to_thread(
            self.run, event, payload, subject=subject, session_id=session_id
        )

    def _execute(self, event: str, hook: HookCommand, stdin: str) -> tuple[HookRun, dict[str, Any]]:
        started = time.monotonic()
        refusal = self._guard(hook, event) if self._guard is not None else None
        if refusal:
            return HookRun(event, hook, None, 0, error=f"refused: {refusal}"), {}
        if hook.type == "http":
            code, out, err, error = _post(hook, stdin)
        else:
            code, out, err, error = _spawn(hook, stdin, self.cwd, self._environment(event))
        elapsed = int((time.monotonic() - started) * 1000)
        return _interpret(event, hook, code, out, err, error, elapsed)

    def _environment(self, event: str) -> dict[str, str]:
        return {
            **os.environ,
            **self._env,
            "TULIP_PROJECT_DIR": str(self.cwd),
            "TULIP_HOOK_EVENT": event,
        }

    # ----------------------------------------------------- lifecycle points --

    async def on_before_invocation(self, prompt: str, state: AgentState) -> AgentState:
        """``UserPromptSubmit``: refuse the prompt, or add context to it."""
        if not self.config.has("UserPromptSubmit"):
            return state
        out = await self.arun("UserPromptSubmit", {"prompt": prompt})
        if out.blocked or out.stop:
            raise HookBlockedError("UserPromptSubmit", out.reason or out.stop_reason or "refused")
        if not out.additional_context:
            return state
        return _with_context(state, out.additional_context)

    async def on_before_tool_call(self, event: BeforeToolCallEvent) -> None:
        """``PreToolUse``: deny, rewrite the arguments, or pass a verdict to the gate."""
        if not self.config.for_event("PreToolUse", event.tool_name):
            return
        out = await self.arun(
            "PreToolUse",
            {
                "tool_name": event.tool_name,
                "tool_input": dict(event.arguments),
                "tool_use_id": event.tool_call_id,
            },
            subject=event.tool_name,
            session_id=_thread_of(event),
        )
        if out.blocked or out.stop:
            why = out.reason or out.stop_reason or "no reason given"
            event.cancel = f"Refused by a PreToolUse hook: {why}"
            return
        if out.updated_input is not None:
            event.arguments = dict(out.updated_input)
        if out.decision in ("allow", "ask") and self._on_permission is not None:
            self._on_permission(event.tool_name, dict(event.arguments), out.decision, out.reason)

    async def on_after_tool_call(self, event: AfterToolCallEvent) -> None:
        """``PostToolUse``: hand the model a hook's objection or added context."""
        if event.error is not None or not self.config.for_event("PostToolUse", event.tool_name):
            return
        out = await self.arun(
            "PostToolUse",
            {
                "tool_name": event.tool_name,
                "tool_input": dict(event.arguments),
                "tool_use_id": event.tool_call_id,
                "tool_response": event.result,
            },
            subject=event.tool_name,
            session_id=_thread_of(event),
        )
        notes: list[str] = []
        if out.blocked and out.reason:
            notes.append(f"[PostToolUse hook] {out.reason}")
        if out.additional_context:
            notes.append(f"[PostToolUse hook context] {out.additional_context}")
        if notes:
            result = event.result if isinstance(event.result, str) else json.dumps(event.result)
            event.result = result + "\n\n" + "\n".join(notes)

    def verifier(
        self, event: str = "Stop"
    ) -> Callable[[str, FinalAnswerContext], Awaitable[str | None]]:
        """``Stop`` (or ``SubagentStop``) as a final-answer verifier.

        Pass it as ``final_answer_verifier``. A hook that blocks hands its
        reason to the model, which keeps working; ``"continue": false`` lets
        the run end regardless. ``SubagentStop`` goes on the subagent's own
        agent.
        """

        async def verify(draft: str, ctx: FinalAnswerContext) -> str | None:
            out = await self.arun(
                event,
                {"stop_hook_active": ctx.attempt > 0, "last_assistant_message": draft},
                session_id=getattr(ctx.run, "thread_id", None),
            )
            if out.stop or not out.blocked:
                return None
            return out.reason or f"a {event} hook asked for more work before finishing"

        return verify


# ----------------------------------------------------------------- helpers --


def _thread_of(event: Any) -> str | None:
    run = getattr(event, "run", None)
    return getattr(run, "thread_id", None) if run is not None else None


def _with_context(state: AgentState, context: str) -> AgentState:
    """``state`` with ``context`` added to the prompt it is about to send."""
    from tulip.core.messages import Message, Role  # noqa: PLC0415 — keeps hooks import-light

    note = f"[context from a UserPromptSubmit hook]\n{context}"
    messages = list(state.messages)
    last = messages[-1] if messages else None
    if last is not None and last.role == Role.USER and isinstance(last.content, str):
        messages[-1] = last.model_copy(update={"content": f"{last.content}\n\n{note}"})
        return state.model_copy(update={"messages": tuple(messages)})
    return state.with_message(Message.user(note))


def _cut(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]} … {text[-half:]}"


def _spawn(
    hook: HookCommand, stdin: str, cwd: Path, env: Mapping[str, str]
) -> tuple[int | None, str, str, str | None]:
    """Run a command hook. Its whole process group dies with it on a timeout."""
    try:
        proc = subprocess.Popen(  # noqa: S602 — a hook is a command line the operator wrote
            hook.target,
            shell=True,
            cwd=cwd,
            env=dict(env),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=os.name == "posix",
        )
    except OSError as exc:
        return None, "", "", f"could not start: {exc}"
    try:
        out, err = proc.communicate(stdin, timeout=hook.timeout)
    except subprocess.TimeoutExpired:
        _kill(proc)
        out, err = proc.communicate()
        return None, out or "", err or "", f"timed out after {hook.timeout:g}s"
    return proc.returncode, out or "", err or "", None


def _kill(proc: subprocess.Popen[str]) -> None:
    if os.name == "posix":
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:  # pragma: no cover - already gone
            return
        return
    proc.kill()  # pragma: no cover - not POSIX


def _post(hook: HookCommand, body: str) -> tuple[int | None, str, str, str | None]:
    """POST an HTTP hook. The response body is read like a command's stdout."""
    if not hook.target.startswith(("http://", "https://")):
        return None, "", "", "an http hook needs an http(s) URL"
    request = urllib.request.Request(  # noqa: S310 — scheme checked above
        hook.target,
        data=body.encode("utf-8"),
        headers={"Content-Type": "application/json", **hook.headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=hook.timeout) as response:  # noqa: S310
            return 0, response.read().decode("utf-8", "replace"), "", None
    except urllib.error.HTTPError as exc:
        return None, "", exc.read().decode("utf-8", "replace"), f"HTTP {exc.code}"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return None, "", "", f"request failed: {getattr(exc, 'reason', exc)}"


def _interpret(  # noqa: PLR0913 — one execution's whole result
    event: str,
    hook: HookCommand,
    code: int | None,
    out: str,
    err: str,
    error: str | None,
    elapsed: int,
) -> tuple[HookRun, dict[str, Any]]:
    """The run as reported, and what it asked for."""
    out, err = out[:MAX_OUTPUT], err[:MAX_OUTPUT]
    asked: dict[str, Any] = {}
    decision: str | None = None
    reason = ""
    if error is None and code == 2:
        reason = err.strip() or "blocked (exit 2)"
        if event in _BLOCKABLE:
            decision = "deny" if event == "PreToolUse" else "block"
    elif error is None and code == 0:
        asked, parse_error = _read_output(event, out)
        if parse_error:
            error = parse_error
        decision = asked.get("decision")
        reason = str(asked.get("reason") or "")
    elif error is None:
        error = f"exit {code}"
    run = HookRun(event, hook, code, elapsed, out, err, error, decision, reason)
    return run, asked


def _read_output(event: str, stdout: str) -> tuple[dict[str, Any], str | None]:
    """What an exit-0 hook asked for, from its stdout."""
    text = stdout.strip()
    if not text.startswith("{"):
        if text and event in _STDOUT_IS_CONTEXT:
            return {"context": text}, None
        return {}, None
    try:
        data = json.loads(text)
    except ValueError:
        return {}, "stdout looked like JSON and was not"
    if not isinstance(data, dict):  # pragma: no cover - a "{" that parses is an object
        return {}, None
    asked: dict[str, Any] = {}
    raw_specific = data.get("hookSpecificOutput")
    specific: dict[str, Any] = raw_specific if isinstance(raw_specific, dict) else {}
    if data.get("continue") is False:
        asked["stop"] = True
        asked["stop_reason"] = str(data.get("stopReason") or "")
    if data.get("systemMessage"):
        asked["system_message"] = str(data["systemMessage"])
    legacy = data.get("decision")
    if legacy == "block" and event in _BLOCKABLE:
        asked["decision"] = "deny" if event == "PreToolUse" else "block"
        asked["reason"] = str(data.get("reason") or "")
    elif legacy == "approve" and event == "PreToolUse":
        asked["decision"] = "allow"
        asked["reason"] = str(data.get("reason") or "")
    permission = specific.get("permissionDecision")
    if event == "PreToolUse" and permission in ("allow", "ask", "deny"):
        asked["decision"] = permission
        asked["reason"] = str(specific.get("permissionDecisionReason") or asked.get("reason") or "")
    if event == "PreToolUse" and isinstance(specific.get("updatedInput"), dict):
        asked["updated_input"] = specific["updatedInput"]
    if specific.get("additionalContext"):
        asked["context"] = str(specific["additionalContext"])
    return asked, None


def _fold(outcome: HookOutcome, run: HookRun, asked: Mapping[str, Any]) -> None:
    outcome.runs.append(run)
    if _STRENGTH.get(run.decision, 0) > _STRENGTH.get(outcome.decision, 0):
        outcome.decision = run.decision
    if run.decision in ("deny", "block", "ask") or (run.decision and run.reason):
        outcome.reasons.append(run.reason)
    if asked.get("context"):
        outcome.context.append(str(asked["context"]))
    if asked.get("updated_input") is not None:
        outcome.updated_input = dict(asked["updated_input"])
    if asked.get("system_message"):
        outcome.system_messages.append(str(asked["system_message"]))
    if asked.get("stop"):
        outcome.stop = True
        outcome.stop_reason = outcome.stop_reason or str(asked.get("stop_reason") or "")
