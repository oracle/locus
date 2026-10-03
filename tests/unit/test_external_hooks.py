# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Command and HTTP hooks: the protocol, and what each event does to a run.

Hooks here are real programs — small Python scripts run through the shell —
because the contract under test is the process boundary: JSON on stdin, an
exit code and stdout back, a timeout that kills the whole process group.
"""

from __future__ import annotations

import json
import shlex
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

from tulip.agent import Agent
from tulip.core.events import FinalAnswerVerificationEvent, TerminateEvent, TulipEvent
from tulip.hooks import (
    ExternalHooks,
    HookBlockedError,
    HookCommand,
    HookConfig,
    HookRun,
)
from tulip.hooks.provider import AfterToolCallEvent, BeforeToolCallEvent
from tulip.testing import ScriptedModel, text, tool_call
from tulip.tools.decorator import tool


def _script(tmp_path: Path, name: str, body: str) -> str:
    """A hook command running ``body`` as Python, with the event on ``event``."""
    path = tmp_path / f"hook_{name}.py"
    path.write_text(
        "import json, os, sys\nevent = json.load(sys.stdin)\n" + body,
        encoding="utf-8",
    )
    return f"{shlex.quote(sys.executable)} {shlex.quote(str(path))}"


def _config(event: str, command: str, matcher: str = "", **extra: Any) -> HookConfig:
    return HookConfig.from_settings(
        {
            event: [
                {"matcher": matcher, "hooks": [{"type": "command", "command": command, **extra}]}
            ]
        },
        source="test.json",
    )


def _hooks(
    config: HookConfig, tmp_path: Path, **kwargs: Any
) -> tuple[ExternalHooks, list[HookRun]]:
    runs: list[HookRun] = []
    hooks = ExternalHooks(config, cwd=tmp_path, session_id="s1", on_run=runs.append, **kwargs)
    return hooks, runs


@tool
def bash(command: str) -> str:
    """Run a command (a test double: it only echoes)."""
    return f"ran: {command}"


async def _collect(agent: Agent, prompt: str) -> list[TulipEvent]:
    return [e async for e in agent.run(prompt)]


# ------------------------------------------------------------ configuration --


def test_configuration_is_read_and_problems_are_kept() -> None:
    config = HookConfig.from_settings(
        {
            "PreToolUse": [
                {"matcher": "bash", "hooks": [{"type": "command", "command": "a", "timeout": 5}]},
                {"hooks": [{"type": "http", "url": "http://x.test/h", "headers": {"A": 1}}]},
                {"hooks": [{"type": "prompt", "prompt": "judge it"}]},
                {"hooks": [{"type": "command"}]},
                {"hooks": [{"type": "http"}]},
                {"hooks": [{"command": "b", "timeout": "soon"}]},
                {"hooks": [{"command": "b", "timeout": 0}]},
                {"hooks": ["not an object"]},
                "not a group",
            ],
            "OnStop": [],
            "Stop": "not a list",
        },
        source="settings.json",
    )
    (first, second) = config.hooks["PreToolUse"]
    assert (first.type, first.target, first.timeout, first.matcher) == ("command", "a", 5.0, "bash")
    assert second.type == "http"
    assert second.headers == {"A": "1"}
    assert first.source == "settings.json"
    assert len(config) == 2
    problems = "\n".join(config.problems)
    for expected in (
        "'prompt' is not supported",
        "command hook has no command",
        "http hook has no url",
        "is not a number",
        "must be positive",
        "hook that is not an object",
        "entry that is not an object",
        "unknown hook event 'OnStop'",
        "hooks.Stop must be a list",
    ):
        assert expected in problems
    with pytest.raises(TypeError):
        HookConfig.from_settings(["nope"])  # type: ignore[arg-type]
    assert len(HookConfig.from_settings(None)) == 0


def test_layers_merge_in_order() -> None:
    a = _config("Stop", "first")
    b = _config("Stop", "second").merged(_config("PreToolUse", "third"))
    merged = a.merged(b)
    assert [h.target for h in merged.for_event("Stop")] == ["first", "second"]
    assert merged.has("PreToolUse")
    assert not merged.has("SessionEnd")


@pytest.mark.parametrize(
    ("matcher", "subject", "hit"),
    [
        ("", "bash", True),
        ("*", "anything", True),
        ("Bash", "bash", True),
        ("edit|write", "write", True),
        ("edit|write", "edit_notes", False),
        ("mcp__.*", "mcp__github__x", True),
        ("[unclosed", "[unclosed", True),
        ("[unclosed", "bash", False),
        ("bash", None, True),
    ],
)
def test_matchers(matcher: str, subject: str | None, hit: bool) -> None:
    assert HookCommand("command", "x", matcher=matcher).matches(subject) is hit


# ---------------------------------------------------------------- protocol --


def test_the_event_arrives_on_stdin_and_the_environment_names_the_project(tmp_path: Path) -> None:
    seen = tmp_path / "seen.json"
    command = _script(
        tmp_path,
        "record",
        f"event['env'] = os.environ['TULIP_PROJECT_DIR'], os.environ['TULIP_HOOK_EVENT'], "
        f"os.environ['EXTRA']\nopen({str(seen)!r}, 'w').write(json.dumps(event))\n",
    )
    hooks, runs = _hooks(
        _config("Notification", command),
        tmp_path,
        env={"EXTRA": "yes"},
        payload=lambda event: {"permission_mode": "ask"},
    )
    outcome = hooks.run("Notification", {"message": "needs you"})
    event = json.loads(seen.read_text())
    assert event["hook_event_name"] == "Notification"
    assert event["session_id"] == "s1"
    assert event["cwd"] == str(tmp_path)
    assert event["message"] == "needs you"
    assert event["permission_mode"] == "ask"
    assert event["env"] == [str(tmp_path), "Notification", "yes"]
    assert outcome.decision is None
    assert runs[0].exit_code == 0
    assert runs[0].error is None


def test_exit_two_blocks_with_stderr_as_the_reason(tmp_path: Path) -> None:
    command = _script(tmp_path, "no", "print('tests are red', file=sys.stderr)\nsys.exit(2)\n")
    hooks, runs = _hooks(_config("Stop", command), tmp_path)
    outcome = hooks.run("Stop", {})
    assert outcome.blocked
    assert outcome.decision == "block"
    assert outcome.reason == "tests are red"
    assert runs[0].exit_code == 2
    assert runs[0].decision == "block"


def test_exit_two_on_an_event_that_cannot_block_is_only_reported(tmp_path: Path) -> None:
    command = _script(tmp_path, "no", "sys.exit(2)\n")
    hooks, runs = _hooks(_config("SessionEnd", command), tmp_path)
    outcome = hooks.run("SessionEnd", {"reason": "exit"})
    assert not outcome.blocked
    assert runs[0].reason == "blocked (exit 2)"


def test_any_other_exit_is_a_non_blocking_error(tmp_path: Path) -> None:
    command = _script(tmp_path, "boom", "print('oops', file=sys.stderr)\nsys.exit(1)\n")
    hooks, runs = _hooks(_config("Stop", command), tmp_path)
    assert not hooks.run("Stop", {}).blocked
    assert runs[0].error == "exit 1"
    assert runs[0].stderr.strip() == "oops"


def test_a_slow_hook_is_killed_with_everything_it_started(tmp_path: Path) -> None:
    marker = tmp_path / "child-survived"
    command = (
        f"( sleep 2; touch {shlex.quote(str(marker))} ) & "
        f"{shlex.quote(sys.executable)} -c 'import time; time.sleep(30)'"
    )
    hooks, runs = _hooks(_config("Stop", command, timeout=0.5), tmp_path)
    started = time.monotonic()
    outcome = hooks.run("Stop", {})
    assert time.monotonic() - started < 10
    assert not outcome.blocked
    assert runs[0].error == "timed out after 0.5s"
    assert runs[0].exit_code is None
    time.sleep(2.5)
    assert not marker.exists(), "the timeout must kill the hook's process group"


def test_a_hook_that_cannot_start_is_reported(tmp_path: Path) -> None:
    hooks, runs = _hooks(_config("Stop", "true"), tmp_path / "missing-dir")
    hooks.run("Stop", {})
    assert runs[0].error is not None
    assert runs[0].error.startswith("could not start")


def test_json_output_is_read(tmp_path: Path) -> None:
    command = _script(
        tmp_path,
        "answer",
        "print(json.dumps({'decision': 'block', 'reason': 'run the tests',"
        " 'systemMessage': 'heads up'}))\n",
    )
    hooks, runs = _hooks(_config("Stop", command), tmp_path)
    outcome = hooks.run("Stop", {})
    assert outcome.blocked
    assert outcome.reason == "run the tests"
    assert outcome.system_messages == ["heads up"]
    assert runs[0].as_dict()["decision"] == "block"


def test_continue_false_stops(tmp_path: Path) -> None:
    command = _script(
        tmp_path, "halt", "print(json.dumps({'continue': False, 'stopReason': 'enough'}))\n"
    )
    hooks, _ = _hooks(_config("Stop", command), tmp_path)
    outcome = hooks.run("Stop", {})
    assert outcome.stop
    assert outcome.stop_reason == "enough"


def test_output_that_looks_like_json_and_is_not_is_reported(tmp_path: Path) -> None:
    command = _script(tmp_path, "bad", "print('{not json')\n")
    hooks, runs = _hooks(_config("Stop", command), tmp_path)
    assert not hooks.run("Stop", {}).blocked
    assert runs[0].error == "stdout looked like JSON and was not"


def test_plain_stdout_is_context_only_where_it_is_meant_to_be(tmp_path: Path) -> None:
    command = _script(tmp_path, "say", "print('on branch main')\n")
    for event, expected in (("SessionStart", "on branch main"), ("Stop", "")):
        hooks, _ = _hooks(_config(event, command), tmp_path)
        assert hooks.run(event, {}).additional_context == expected


def test_the_strongest_decision_of_several_hooks_wins(tmp_path: Path) -> None:
    allow = _script(
        tmp_path,
        "allow",
        "print(json.dumps({'hookSpecificOutput': {'permissionDecision': 'allow'}}))\n",
    )
    deny = _script(tmp_path, "deny", "print('not on main', file=sys.stderr)\nsys.exit(2)\n")
    config = _config("PreToolUse", allow).merged(_config("PreToolUse", deny))
    hooks, runs = _hooks(config, tmp_path)
    outcome = hooks.run("PreToolUse", {"tool_name": "bash"}, subject="bash")
    assert outcome.decision == "deny"
    assert outcome.reason == "not on main"
    assert len(runs) == 2


def test_a_guard_refusal_is_reported_and_nothing_runs(tmp_path: Path) -> None:
    marker = tmp_path / "ran"
    hooks, runs = _hooks(
        _config("Stop", f"touch {shlex.quote(str(marker))}"),
        tmp_path,
        guard=lambda hook, event: "not on my watch" if "touch" in hook.target else None,
    )
    assert not hooks.run("Stop", {}).blocked
    assert not marker.exists()
    assert runs[0].error == "refused: not on my watch"
    assert runs[0].exit_code is None


def test_long_output_is_cut(tmp_path: Path) -> None:
    command = _script(tmp_path, "loud", "print('x' * 50000)\n")
    hooks, runs = _hooks(_config("SessionEnd", command), tmp_path)
    hooks.run("SessionEnd", {})
    assert len(runs[0].stdout) == 20_000
    assert len(runs[0].as_dict(limit=100)["stdout"]) < 110


def test_no_hooks_no_work(tmp_path: Path) -> None:
    hooks, runs = _hooks(HookConfig(), tmp_path)
    assert hooks.run("Stop", {}).runs == []
    assert runs == []


# -------------------------------------------------------------------- http --


class _Handler(BaseHTTPRequestHandler):
    received: list[dict[str, Any]] = []  # noqa: RUF012 — shared by the test server
    reply: tuple[int, str] = (200, "{}")

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        type(self).received.append(
            {"body": json.loads(self.rfile.read(length)), "header": self.headers.get("X-Hook")}
        )
        status, body = type(self).reply
        self.send_response(status)
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, *args: Any) -> None:
        pass


@pytest.fixture
def server() -> Any:
    _Handler.received = []
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/hook"
    httpd.shutdown()


def _http(event: str, url: str) -> HookConfig:
    return HookConfig.from_settings(
        {event: [{"hooks": [{"type": "http", "url": url, "headers": {"X-Hook": "h"}}]}]}
    )


def test_an_http_hook_posts_the_event_and_reads_the_reply(tmp_path: Path, server: str) -> None:
    _Handler.reply = (200, json.dumps({"decision": "block", "reason": "lint first"}))
    hooks, runs = _hooks(_http("Stop", server), tmp_path)
    outcome = hooks.run("Stop", {"stop_hook_active": False})
    assert outcome.blocked
    assert outcome.reason == "lint first"
    assert _Handler.received[0]["body"]["hook_event_name"] == "Stop"
    assert _Handler.received[0]["header"] == "h"
    assert runs[0].exit_code == 0


def test_an_http_error_is_non_blocking(tmp_path: Path, server: str) -> None:
    _Handler.reply = (500, "broken")
    hooks, runs = _hooks(_http("Stop", server), tmp_path)
    assert not hooks.run("Stop", {}).blocked
    assert runs[0].error == "HTTP 500"
    assert runs[0].stderr == "broken"


def test_an_unreachable_or_odd_url_is_non_blocking(tmp_path: Path) -> None:
    hooks, runs = _hooks(_http("Stop", "http://127.0.0.1:9/nothing"), tmp_path)
    hooks.run("Stop", {})
    assert runs[0].error is not None
    assert runs[0].error.startswith("request failed")
    hooks, runs = _hooks(_http("Stop", "file:///etc/passwd"), tmp_path)
    hooks.run("Stop", {})
    assert runs[0].error == "an http hook needs an http(s) URL"


# ------------------------------------------------------------ in the loop --


async def test_pre_tool_use_denies_and_the_tool_never_runs(tmp_path: Path) -> None:
    command = _script(
        tmp_path,
        "guard",
        "cmd = event['tool_input']['command']\n"
        "if 'push' in cmd:\n"
        "    print('no pushing from the agent', file=sys.stderr); sys.exit(2)\n",
    )
    hooks, runs = _hooks(_config("PreToolUse", command, matcher="bash"), tmp_path)
    model = ScriptedModel([tool_call("bash", command="git push"), text("ok")])
    agent = Agent(model=model, tools=[bash], hooks=[hooks], reflexion=False, grounding=False)
    await _collect(agent, "push it")
    seen = [m.content for m in model.received_messages[-1] if m.role == "tool"]
    assert seen == ["Refused by a PreToolUse hook: no pushing from the agent"]
    assert runs[0].event == "PreToolUse"
    assert runs[0].decision == "deny"


async def test_pre_tool_use_rewrites_arguments_and_reports_a_verdict(tmp_path: Path) -> None:
    command = _script(
        tmp_path,
        "rewrite",
        "print(json.dumps({'hookSpecificOutput': {'hookEventName': 'PreToolUse',"
        " 'permissionDecision': 'allow', 'permissionDecisionReason': 'read-only',"
        " 'updatedInput': {'command': event['tool_input']['command'] + ' --dry-run'}}}))\n",
    )
    verdicts: list[tuple[str, dict[str, Any], str, str]] = []
    hooks, _ = _hooks(
        _config("PreToolUse", command),
        tmp_path,
        on_permission=lambda *args: verdicts.append(args),
    )
    model = ScriptedModel([tool_call("bash", command="make deploy"), text("ok")])
    agent = Agent(model=model, tools=[bash], hooks=[hooks], reflexion=False, grounding=False)
    await _collect(agent, "deploy")
    seen = [m.content for m in model.received_messages[-1] if m.role == "tool"]
    assert seen == ["ran: make deploy --dry-run"]
    assert verdicts == [("bash", {"command": "make deploy --dry-run"}, "allow", "read-only")]


async def test_a_matcher_keeps_a_hook_off_other_tools(tmp_path: Path) -> None:
    hooks, runs = _hooks(_config("PreToolUse", "exit 2", matcher="edit|write"), tmp_path)
    event = BeforeToolCallEvent("bash", "c1", {"command": "ls"})
    await hooks.on_before_tool_call(event)
    assert not event.cancel
    assert runs == []


async def test_post_tool_use_adds_what_the_model_should_know(tmp_path: Path) -> None:
    command = _script(
        tmp_path,
        "post",
        "assert event['tool_response'] == 'ran: ls'\n"
        "print(json.dumps({'decision': 'block', 'reason': 'formatting changed 2 files',"
        " 'hookSpecificOutput': {'additionalContext': 'ruff reformatted a.py'}}))\n",
    )
    hooks, _ = _hooks(_config("PostToolUse", command), tmp_path)
    event = AfterToolCallEvent("bash", "ran: ls", None, arguments={"command": "ls"})
    await hooks.on_after_tool_call(event)
    assert event.result == (
        "ran: ls\n\n[PostToolUse hook] formatting changed 2 files\n"
        "[PostToolUse hook context] ruff reformatted a.py"
    )
    failed = AfterToolCallEvent("bash", None, "boom", arguments={})
    await hooks.on_after_tool_call(failed)
    assert failed.result is None


async def test_post_tool_use_serialises_a_structured_result(tmp_path: Path) -> None:
    command = _script(
        tmp_path, "ctx", "print(json.dumps({'hookSpecificOutput': {'additionalContext': 'n'}}))\n"
    )
    hooks, _ = _hooks(_config("PostToolUse", command), tmp_path)
    event = AfterToolCallEvent("bash", {"a": 1}, None)
    await hooks.on_after_tool_call(event)
    assert event.result == '{"a": 1}\n\n[PostToolUse hook context] n'


async def test_user_prompt_submit_adds_context_or_refuses(tmp_path: Path) -> None:
    context = _script(tmp_path, "ctx", "print('today is release day')\n")
    hooks, _ = _hooks(_config("UserPromptSubmit", context), tmp_path)
    model = ScriptedModel([text("ok")])
    agent = Agent(model=model, hooks=[hooks], reflexion=False, grounding=False)
    await _collect(agent, "what now?")
    prompt = [m.content for m in model.received_messages[0] if m.role == "user"][-1]
    assert prompt.startswith("what now?")
    assert "today is release day" in prompt

    refuse = _script(
        tmp_path, "refuse", "print('no secrets in prompts', file=sys.stderr); sys.exit(2)\n"
    )
    hooks, _ = _hooks(_config("UserPromptSubmit", refuse), tmp_path)
    agent = Agent(model=ScriptedModel([]), hooks=[hooks], reflexion=False, grounding=False)
    with pytest.raises(HookBlockedError, match="no secrets in prompts"):
        await _collect(agent, "my key is sk-...")


async def test_user_prompt_submit_context_after_a_non_text_prompt(tmp_path: Path) -> None:
    from tulip.core.messages import Message
    from tulip.core.state import AgentState

    hooks, _ = _hooks(_config("UserPromptSubmit", "echo extra"), tmp_path)
    state = AgentState(messages=(Message.assistant("hi"),))
    out = await hooks.on_before_invocation("p", state)
    assert out.messages[-1].role == "user"
    assert "extra" in str(out.messages[-1].content)
    assert await _hooks(HookConfig(), tmp_path)[0].on_before_invocation("p", state) is state


async def test_stop_blocks_until_the_hook_is_satisfied(tmp_path: Path) -> None:
    counter = tmp_path / "count"
    command = _script(
        tmp_path,
        "stop",
        f"path = {str(counter)!r}\n"
        "n = int(open(path).read()) if os.path.exists(path) else 0\n"
        "open(path, 'w').write(str(n + 1))\n"
        "assert event['stop_hook_active'] == (n > 0)\n"
        "if 'tests pass' not in event['last_assistant_message']:\n"
        "    print('run the tests before you finish', file=sys.stderr); sys.exit(2)\n",
    )
    hooks, runs = _hooks(_config("Stop", command), tmp_path)
    model = ScriptedModel(
        [text("done"), tool_call("bash", command="pytest"), text("done, tests pass")]
    )
    agent = Agent(
        model=model,
        tools=[bash],
        hooks=[hooks],
        final_answer_verifier=hooks.verifier(),
        reflexion=False,
        grounding=False,
    )
    events = await _collect(agent, "fix it")
    final = [e for e in events if isinstance(e, TerminateEvent)][-1]
    assert final.final_message == "done, tests pass"
    verdicts = [e for e in events if isinstance(e, FinalAnswerVerificationEvent)]
    assert [v.passed for v in verdicts] == [False, True]
    assert "run the tests before you finish" in (verdicts[0].feedback or "")
    assert [r.decision for r in runs] == ["block", None]


async def test_stop_with_continue_false_lets_the_run_end(tmp_path: Path) -> None:
    command = _script(
        tmp_path,
        "enough",
        "print(json.dumps({'decision': 'block', 'reason': 'x', 'continue': False}))\n",
    )
    hooks, _ = _hooks(_config("SubagentStop", command), tmp_path)
    verify = hooks.verifier("SubagentStop")

    class _Ctx:
        attempt = 0
        run = None

    assert await verify("done", _Ctx()) is None  # type: ignore[arg-type]


async def test_a_stop_block_without_a_reason_still_says_why(tmp_path: Path) -> None:
    command = _script(tmp_path, "terse", "print(json.dumps({'decision': 'block'}))\n")
    hooks, _ = _hooks(_config("Stop", command), tmp_path)

    class _Ctx:
        attempt = 1
        run = None

    feedback = await hooks.verifier()("done", _Ctx())  # type: ignore[arg-type]
    assert feedback == "a Stop hook asked for more work before finishing"


async def test_the_small_cases(tmp_path: Path) -> None:
    from tulip.hooks import HookPriority

    legacy = _script(
        tmp_path, "legacy", "print(json.dumps({'decision': 'approve', 'reason': 'ok'}))\n"
    )
    hooks, _ = _hooks(_config("PreToolUse", legacy), tmp_path)
    assert hooks.priority == HookPriority.SECURITY_DEFAULT
    outcome = hooks.run("PreToolUse", {}, subject="bash")
    assert outcome.decision == "allow"
    assert outcome.reason == "ok"
    assert (await hooks.arun("Stop", {})).runs == []

    quiet = _script(tmp_path, "quiet", "pass\n")
    hooks, _ = _hooks(_config("UserPromptSubmit", quiet), tmp_path)
    from tulip.core.state import AgentState

    state = AgentState()
    assert await hooks.on_before_invocation("p", state) is state


# ------------------------------------------------- a call inside another call --


def test_pre_tool_use_for_a_nested_call_sees_it_under_its_own_name(tmp_path: Path) -> None:
    command = _script(
        tmp_path,
        "nested",
        "assert event['tool_name'] == 'bash'\n"
        "assert event['tool_use_id'] == 'c1:then_run'\n"
        "print(json.dumps({'hookSpecificOutput': {'hookEventName': 'PreToolUse',"
        " 'permissionDecision': 'allow', 'permissionDecisionReason': 'tests are fine',"
        " 'updatedInput': {'command': event['tool_input']['command'] + ' -q'}}}))\n",
    )
    verdicts: list[tuple[str, dict[str, Any], str, str]] = []
    hooks, runs = _hooks(
        _config("PreToolUse", command, matcher="bash"),
        tmp_path,
        on_permission=lambda *args: verdicts.append(args),
    )
    verdict = hooks.pre_tool_use("bash", {"command": "pytest"}, tool_use_id="c1:then_run")
    assert verdict.cancel is None
    assert verdict.arguments == {"command": "pytest -q"}
    assert verdicts == [("bash", {"command": "pytest -q"}, "allow", "tests are fine")]
    assert runs[0].event == "PreToolUse"


def test_pre_tool_use_for_a_nested_call_can_refuse_it(tmp_path: Path) -> None:
    command = _script(tmp_path, "deny", "print('not that', file=sys.stderr); sys.exit(2)\n")
    hooks, _ = _hooks(_config("PreToolUse", command, matcher="bash"), tmp_path)
    verdict = hooks.pre_tool_use("bash", {"command": "make deploy"})
    assert verdict.cancel == "Refused by a PreToolUse hook: not that"
    assert verdict.arguments == {"command": "make deploy"}


def test_nested_calls_skip_hooks_that_do_not_match(tmp_path: Path) -> None:
    hooks, runs = _hooks(_config("PreToolUse", "exit 2", matcher="edit"), tmp_path)
    assert hooks.pre_tool_use("bash", {"command": "ls"}).cancel is None
    assert hooks.post_tool_use("bash", {"command": "ls"}, "exit 0") == ""
    assert runs == []


def test_post_tool_use_for_a_nested_call_returns_its_notes(tmp_path: Path) -> None:
    command = _script(
        tmp_path,
        "post",
        "assert event['tool_response'] == 'exit 0'\n"
        "print(json.dumps({'decision': 'block', 'reason': 'coverage dropped'}))\n",
    )
    hooks, _ = _hooks(_config("PostToolUse", command, matcher="bash"), tmp_path)
    notes = hooks.post_tool_use("bash", {"command": "pytest"}, "exit 0", tool_use_id="c2")
    assert notes == "[PostToolUse hook] coverage dropped"
