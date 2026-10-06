# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""The shell: bash, and the handles of what it left running.

One command line per call, run by the workspace's shell with standard input
closed. Output comes back as a head and a tail — the summary of a test run
is at the end — and when it is too long to show, the whole of it can be
kept in a :class:`~tulip.tools.result_storage.ToolResultStore` and fetched
back by key.

A command that outlives its timeout is, by default, not lost: it keeps
running under a handle, and ``bash_output`` / ``kill_shell`` read or stop it
like any command started with ``background=true``. Killing a build that was
nearly done throws away minutes of work, and the model can kill it in one
call. ``HarnessConfig.on_timeout="kill"`` stops it instead.

Every command produces an :class:`~tulip.harness.evidence.ExecRecord`.
"""

from __future__ import annotations

from tulip.core.messages import ToolResult
from tulip.harness.backend import ExecResult, ExecUnsupportedError, JobError, JobOutput
from tulip.harness.evidence import exec_record, publish
from tulip.harness.tools.common import HarnessContext, cap_output
from tulip.tools.context import ToolContext
from tulip.tools.decorator import Tool, tool
from tulip.tools.result_storage import extract_reference_key


__all__ = ["make_bash", "make_bash_output", "make_kill_shell", "make_write_stdin"]


def _text(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def _record(h: HarnessContext, command: str, result: ExecResult, *, background: bool) -> None:
    label = h.backend.capabilities.label
    publish(
        exec_record(command, result, backend_label=label, background=background), h.config.on_exec
    )


def _shown(h: HarnessContext, text: str, ctx: ToolContext | None) -> str:
    """``text`` capped for the model, with the whole of it stored when it was cut."""
    cfg = h.config
    capped = cap_output(text, cfg.output_chars, cfg.output_head_chars)
    store = cfg.result_store
    if store is None or capped == text:
        return capped
    if ctx is not None:
        run_id, iteration, call_id = ctx.run_id, ctx.iteration, ctx.tool_call_id
    else:
        h.spills += 1
        run_id, iteration, call_id = "harness", h.spills, f"bash-{h.spills}"
    stored = store.maybe_offload(
        ToolResult(tool_call_id=call_id, name="bash", content=text),
        run_id=run_id,
        iteration=iteration,
    )
    key = extract_reference_key(stored.content)
    if key is None:
        return capped
    return f"{capped}\n[the full output, {len(text):,} characters, is stored: key={key}]"


def make_bash(h: HarnessContext) -> Tool:
    """The ``bash`` tool over ``h``."""
    cfg = h.config

    def foreground(command: str, limit: int, asked: int, ctx: ToolContext | None) -> str:
        note = f" (the limit is {cfg.bash_max_timeout}s)" if limit < asked else ""
        if cfg.on_timeout == "kill":
            result = h.backend.exec(command, timeout=limit)
            _record(h, command, result, background=False)
            out = _shown(h, _text(result.output).strip(), ctx)
            if result.timed_out:
                # Whatever it printed before it was stopped is usually why it hung.
                return f"timed out after {limit}s{note} — killed it and everything it started\n" + (
                    out or "(no output)"
                )
            return f"exit {result.exit_code}\n{out or '(no output)'}"

        # One mechanism for both outcomes: started as a job, waited on, and
        # either collected or left running under its handle.
        job = h.backend.start_background(command, foreground=True)
        status = h.backend.wait_background(job.handle, limit)
        read = h.backend.read_background(job.handle, 0)
        result = ExecResult(
            exit_code=None if status.running else status.exit_code,
            output=read.data,
            truncated=read.lost > 0,
            timed_out=status.running,
            duration_s=status.elapsed_s,
        )
        _record(h, command, result, background=False)
        out = _shown(h, _text(read.data).strip(), ctx)
        if not status.running:
            h.backend.release_background(job.handle)
            return f"exit {status.exit_code}\n{out or '(no output)'}"
        h.cursors[job.handle] = read.offset
        return (
            f"timed out after {limit}s{note} — still running in the background as {job.handle}. "
            f'Check on it with bash_output(handle="{job.handle}"), or stop it with '
            f'kill_shell(handle="{job.handle}").\n' + (out or "(no output yet)")
        )

    def bash(
        command: str,
        timeout: int = cfg.bash_timeout,
        background: bool = False,
        ctx: ToolContext | None = None,
    ) -> str:
        """Run a shell command in the workspace.

        Stdin is closed, so a command that waits for input fails fast instead
        of hanging. Long output keeps its beginning and its end.

        For anything long-lived — a dev server, a watcher, a slow build you
        want to check on — pass ``background=true``: it returns a handle at
        once, and ``bash_output`` reads what it has printed since you last
        looked. A command that outlives its timeout is moved to the
        background the same way rather than lost. Stop either with
        ``kill_shell``.

        Args:
            command: The command line to run.
            timeout: Seconds to wait before handing the command back as a
                background handle (capped by the operator's limit). Ignored
                with ``background``.
            background: Start it and return a handle without waiting. It gets
                a stdin pipe, which ``write_stdin`` writes to.
        """
        try:
            if background:
                try:
                    job = h.backend.start_background(command)
                except JobError as exc:
                    return str(exc)
                _record(h, command, ExecResult(exit_code=None, output=b""), background=True)
                pid = f" (pid {job.pid})" if job.pid else ""
                return (
                    f"started {job.handle} in the background{pid}. "
                    f'Read its output with bash_output(handle="{job.handle}"); '
                    f'stop it with kill_shell(handle="{job.handle}").'
                )
            try:
                asked = int(timeout)
            except (TypeError, ValueError):
                asked = cfg.bash_timeout
            limit = max(1, min(asked, cfg.bash_max_timeout))
            return foreground(command, limit, asked, ctx)
        except ExecUnsupportedError as exc:
            return str(exc)

    return tool(bash)


def _status_text(h: HarnessContext, read: JobOutput, limit: int, head: int) -> str:
    return cap_output(_text(read.data).rstrip(), limit, head)


def make_bash_output(h: HarnessContext) -> Tool:
    """The ``bash_output`` tool over ``h``."""

    def bash_output(handle: str, since: int | None = None) -> str:
        """Read what a background command has printed, and whether it is still running.

        Args:
            handle: The handle ``bash`` returned, like ``sh3``.
            since: Byte offset to read from. Leave it out to get everything new
                since your last read; ``0`` reads from the start (what is still
                kept of it).
        """
        key = handle.strip()
        start = h.cursors.get(key, 0) if since is None else max(0, int(since))
        try:
            read = h.backend.read_background(key, start)
        except (JobError, ExecUnsupportedError) as exc:
            return str(exc)
        h.cursors[key] = read.offset
        out = _status_text(h, read, h.config.output_chars, h.config.output_head_chars)
        return f"{read.status.describe()}\n{out or '(no new output)'}\n[next since={read.offset}]"

    return tool(bash_output)


def make_kill_shell(h: HarnessContext) -> Tool:
    """The ``kill_shell`` tool over ``h``."""

    def kill_shell(handle: str) -> str:
        """Stop a background command and everything it started.

        Args:
            handle: The handle ``bash`` returned, like ``sh3``.
        """
        key = handle.strip()
        try:
            before = h.backend.read_background(key, h.cursors.get(key, 0))
            if not before.status.running:
                return f"{before.status.describe()} — nothing to stop"
            status = h.backend.kill_background(key)
            read = h.backend.read_background(key, h.cursors.get(key, 0))
        except (JobError, ExecUnsupportedError) as exc:
            return str(exc)
        h.cursors[key] = read.offset
        out = _status_text(h, read, 4_000, 1_000)
        return f"killed {status.handle}: {status.command}" + (
            f"\nlast output:\n{out}" if out else ""
        )

    return tool(kill_shell)


def make_write_stdin(h: HarnessContext) -> Tool:
    """The ``write_stdin`` tool over ``h``."""

    def write_stdin(handle: str, text: str, close: bool = False) -> str:
        """Send input to a background command — an answer to a prompt, a REPL line.

        Input to a shell or an interpreter is a command, and is gated as one.

        Args:
            handle: The handle ``bash`` returned for a command started with
                ``background=true``.
            text: What to write. Add a trailing newline to submit a line.
            close: Close its stdin afterwards, for a command that reads to the end.
        """
        try:
            h.backend.write_background(handle.strip(), text.encode("utf-8"), close=bool(close))
        except (JobError, ExecUnsupportedError) as exc:
            return str(exc)
        return f"wrote {len(text):,} characters to {handle.strip()}" + (
            " and closed its stdin" if close else ""
        )

    return tool(write_stdin)
