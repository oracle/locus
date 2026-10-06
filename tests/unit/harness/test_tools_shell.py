# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""bash and its handles — bash_output, write_stdin, kill_shell — on every shell.

Ported from tulip-code's shell tests. The interesting cases are the ones a
model hits at night with nobody watching: a command that waits for input, a
command that never ends, output too long to read, a server it started and
forgot.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import pytest

from tests.unit.harness.conftest import FILE_BACKENDS, make_backend
from tulip.harness import ExecRecord, HarnessConfig, WorkspaceBackend, build_harness
from tulip.harness.toolset import Harness
from tulip.tools.context import ToolContext
from tulip.tools.result_storage import ToolResultStore


def harness(backend: WorkspaceBackend, **config: object) -> Harness:
    return build_harness(backend, config=HarnessConfig(**config))  # type: ignore[arg-type]


async def call(h: Harness, name: str, **kwargs: object) -> str:
    result: str = await h.tool(name).execute(**kwargs)
    return result


async def test_bash_returns_the_exit_status_with_the_output(shell: WorkspaceBackend) -> None:
    out = await call(harness(shell), "bash", command="echo hello")
    assert out == "exit 0\nhello"


async def test_a_non_zero_exit_is_information_not_an_exception(shell: WorkspaceBackend) -> None:
    assert (await call(harness(shell), "bash", command="exit 3")).startswith("exit 3")


async def test_bash_captures_stderr_in_order(shell: WorkspaceBackend) -> None:
    out = await call(harness(shell), "bash", command="echo one; echo two >&2; echo three")
    assert "two" in out
    assert out.index("one") < out.index("three")


async def test_bash_reports_no_output_rather_than_an_empty_string(shell: WorkspaceBackend) -> None:
    assert "(no output)" in await call(harness(shell), "bash", command="true")
    killing = harness(shell, on_timeout="kill")
    assert "(no output)" in await call(killing, "bash", command="true")


async def test_bash_closes_stdin_so_a_prompt_cannot_hang_it(shell: WorkspaceBackend) -> None:
    # `cat` with no file reads stdin. Inherited from an open pipe it would
    # wait for ever; closed, it sees end of input and exits at once.
    for mode in ("background", "kill"):
        out = await call(
            harness(shell, on_timeout=mode), "bash", command="cat; echo done", timeout=5
        )
        assert out.startswith("exit 0")
        assert "done" in out


async def test_bash_truncates_output_too_large_to_hand_a_model(shell: WorkspaceBackend) -> None:
    out = await call(harness(shell), "bash", command="head -c 60000 /dev/zero | tr '\\0' 'x'")
    assert "truncated from the middle" in out
    assert len(out) < 31_000


async def test_the_whole_output_is_stored_when_it_was_cut(shell: WorkspaceBackend) -> None:
    saved: dict[str, str] = {}
    store = ToolResultStore(
        save=saved.__setitem__, load=saved.get, threshold_chars=100, preview_chars=50
    )
    h = harness(shell, result_store=store, output_chars=200, output_head_chars=50)
    out = await call(h, "bash", command="seq 1 500")
    assert "the full output, " in out
    assert "key=" in out
    (key,) = saved
    assert saved[key].endswith("500")
    ctx = ToolContext(tool_call_id="c1", tool_name="bash", run_id="r9", iteration=4)
    await h.tool("bash").execute(ctx=ctx, command="seq 1 500")
    assert any(k.startswith("tulip:result:r9:4") for k in saved)


async def test_a_store_that_declines_leaves_the_capped_output(shell: WorkspaceBackend) -> None:
    store = ToolResultStore(save=lambda k, v: None, load=lambda k: None, threshold_chars=10**6)
    h = harness(shell, result_store=store, output_chars=200, output_head_chars=50)
    out = await call(h, "bash", command="seq 1 500")
    assert "truncated" in out
    assert "key=" not in out


async def test_bash_timeout_kills_what_the_command_started(
    shell: WorkspaceBackend, tmp_path: Path
) -> None:
    pidfile = tmp_path / "child.pid"
    h = harness(shell, on_timeout="kill")
    out = await call(
        h, "bash", command=f"echo started; sleep 30 & echo $! > {pidfile}; wait", timeout=1
    )
    assert "timed out after 1s — killed it and everything it started" in out
    assert "started" in out
    await asyncio.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        os.kill(int(pidfile.read_text()), 0)


async def test_bash_caps_the_timeout_a_model_asks_for(shell: WorkspaceBackend) -> None:
    h = harness(shell, bash_max_timeout=1, on_timeout="kill")
    out = await call(h, "bash", command="sleep 30", timeout=3600)
    assert "timed out after 1s (the limit is 1s)" in out


async def test_a_nonsense_timeout_falls_back_to_the_default(shell: WorkspaceBackend) -> None:
    out = await call(harness(shell), "bash", command="echo ok", timeout="soon")
    assert out == "exit 0\nok"


async def test_a_timed_out_command_moves_to_the_background(shell: WorkspaceBackend) -> None:
    h = harness(shell)
    out = await call(h, "bash", command="echo first; sleep 2; echo second", timeout=1)
    assert "still running in the background as sh1" in out
    assert "first" in out
    assert not shell.wait_background("sh1", 10).running
    read = await call(h, "bash_output", handle="sh1")
    # Only what is new since the timeout's own read.
    assert read.endswith("\nsecond\n[next since=13]")


async def test_a_command_that_finishes_leaves_no_handle_behind(shell: WorkspaceBackend) -> None:
    h = harness(shell)
    await call(h, "bash", command="echo hi")
    assert shell.list_background() == []


async def test_background_returns_a_handle_at_once(shell: WorkspaceBackend) -> None:
    seen: list[ExecRecord] = []
    h = harness(shell, on_exec=seen.append)
    started = time.monotonic()
    out = await call(h, "bash", command="echo $((40 + 2)); sleep 30", background=True)
    assert time.monotonic() - started < 5
    assert out.startswith("started sh1 in the background")
    assert 'bash_output(handle="sh1")' in out
    assert seen[-1].background
    assert seen[-1].exit_code is None
    deadline = time.monotonic() + 5
    while "\n42\n" not in (read := await call(h, "bash_output", handle="sh1")):
        assert time.monotonic() < deadline
        await asyncio.sleep(0.05)
    assert "sh1 is running" in read
    assert "[next since=3]" in read
    assert "(no new output)" in await call(h, "bash_output", handle="sh1")
    assert "\n42\n" in await call(h, "bash_output", handle="sh1", since=0)
    killed = await call(h, "kill_shell", handle="sh1")
    assert killed.startswith("killed sh1: echo $((40 + 2)); sleep 30")
    assert "nothing to stop" in await call(h, "kill_shell", handle="sh1")


async def test_kill_shell_reports_the_last_output(shell: WorkspaceBackend) -> None:
    h = harness(shell)
    await call(h, "bash", command="echo last words; sleep 30", background=True)
    deadline = time.monotonic() + 5
    while not shell.read_background("sh1").data:
        assert time.monotonic() < deadline
        await asyncio.sleep(0.05)
    assert "last output:\nlast words" in await call(h, "kill_shell", handle="sh1")


async def test_bash_output_reports_the_exit_code_once_it_ends(shell: WorkspaceBackend) -> None:
    h = harness(shell)
    await call(h, "bash", command="exit 5", background=True)
    shell.wait_background("sh1", 5)
    assert "sh1 exited 5" in await call(h, "bash_output", handle="sh1")


async def test_an_unknown_handle_lists_the_ones_that_exist(shell: WorkspaceBackend) -> None:
    h = harness(shell)
    assert "none have been started" in await call(h, "bash_output", handle="sh9")
    await call(h, "bash", command="sleep 30", background=True)
    for name, extra in [("bash_output", {}), ("kill_shell", {}), ("write_stdin", {"text": "x"})]:
        out = await call(h, name, handle="sh9", **extra)
        assert "These exist:" in out, name
        assert "sh1" in out, name
    await call(h, "kill_shell", handle="sh1")


async def test_write_stdin_feeds_a_command_waiting_for_input(shell: WorkspaceBackend) -> None:
    h = harness(shell)
    await call(h, "bash", command="read line; echo got $line", background=True)
    assert await call(h, "write_stdin", handle="sh1", text="hello\n") == "wrote 6 characters to sh1"
    shell.wait_background("sh1", 5)
    assert "got hello" in await call(h, "bash_output", handle="sh1")


async def test_write_stdin_can_close_the_input(shell: WorkspaceBackend) -> None:
    h = harness(shell)
    await call(h, "bash", command="cat; echo eof", background=True)
    out = await call(h, "write_stdin", handle="sh1", text="x\n", close=True)
    assert out.endswith("and closed its stdin")
    shell.wait_background("sh1", 5)
    assert "eof" in await call(h, "bash_output", handle="sh1")


async def test_write_to_a_finished_command_says_so(shell: WorkspaceBackend) -> None:
    h = harness(shell)
    await call(h, "bash", command="true", background=True)
    shell.wait_background("sh1", 5)
    assert "cannot take input" in await call(h, "write_stdin", handle="sh1", text="x")


async def test_a_timed_out_foreground_command_has_no_stdin_to_write(
    shell: WorkspaceBackend,
) -> None:
    h = harness(shell)
    await call(h, "bash", command="sleep 30", timeout=1)
    assert "has no stdin" in await call(h, "write_stdin", handle="sh1", text="x")
    await call(h, "kill_shell", handle="sh1")


async def test_too_many_background_commands_is_a_message(root: Path) -> None:
    backend = make_backend("local", root)
    backend._max_background = 1  # type: ignore[attr-defined]
    h = harness(backend)
    await call(h, "bash", command="sleep 30", background=True)
    out = await call(h, "bash", command="sleep 30", background=True)
    assert "already running" in out
    backend.close()  # type: ignore[attr-defined]


async def test_every_command_leaves_an_exec_record(shell: WorkspaceBackend) -> None:
    seen: list[ExecRecord] = []
    for mode in ("background", "kill"):
        h = harness(shell, on_exec=seen.append, on_timeout=mode)
        await call(h, "bash", command="echo evidence; exit 2")
    assert [r.exit_code for r in seen] == [2, 2]
    assert all(r.backend_label == shell.capabilities.label for r in seen)
    assert all(r.output_bytes == len(b"evidence\n") for r in seen)


@pytest.mark.parametrize("kind", [p for p in FILE_BACKENDS if p == "memory"])
async def test_shell_tools_on_a_backend_without_a_shell_say_so(kind: str, root: Path) -> None:
    from tulip.harness.ledger import ReadLedger
    from tulip.harness.tools import FACTORIES
    from tulip.harness.tools.common import HarnessContext

    backend = make_backend(kind, root)
    context = HarnessContext(backend=backend, config=HarnessConfig(), ledger=ReadLedger())
    for name, kwargs in [
        ("bash", {"command": "ls"}),
        ("bash", {"command": "ls", "background": True}),
        ("bash_output", {"handle": "sh1"}),
        ("kill_shell", {"handle": "sh1"}),
        ("write_stdin", {"handle": "sh1", "text": "x"}),
    ]:
        out = await FACTORIES[name](context).execute(**kwargs)
        assert "has no shell" in out, name
