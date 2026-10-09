# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Keep what the runner holds away from the commands it runs.

The runner holds three things a command in the box must not use: its workload
token (it answers the gateway as this run), and the OpenShell placeholders of
the model's key and of each connector's credential (the sandbox's proxy turns
a placeholder into the real credential on its host, for whoever sends it).
A ``bash`` call is the model's, admitted by the gateway as a command — not as
a model call or a connector call — so it must not be able to make those.

Two layers, both applied by the runner before any command can run:

1. **No inheritance.** Commands get :func:`command_env`: the runner's
   environment without those variables, so a child process never receives
   them.
2. **No reading them back.** :func:`protect` wipes the values out of the
   runner's initial environment block, the memory ``/proc/<pid>/environ``
   shows, once the runner has read them, and drops them from ``os.environ``.
   A command running as the same uid then finds only empty values there.
   Reading the runner's memory itself needs ptrace access, which Linux's Yama
   (``kernel.yama.ptrace_scope`` of 1 or more) refuses to a child towards its
   parent.

The runner stays **dumpable** by default. NVIDIA OpenShell identifies the
process behind every DNS lookup and connection through ``/proc/<pid>``
(``require_binary_identity``); a non-dumpable process cannot be identified,
so OpenShell refuses its lookups and the runner could never reach its gateway.
``TULIP_RUNNER_HARDEN=non-dumpable`` restores :func:`make_non_dumpable`
(``prctl(PR_SET_DUMPABLE, 0)``) for a sandbox that does not identify processes
that way; it adds to the wipe, it does not replace it.

What this does **not** cover: a process with ``CAP_SYS_PTRACE`` or root in the
box, or a kernel with Yama's ptrace scope at 0. The runner runs as a non-root
user with no capabilities, and OpenShell's process policy drops them; that is
the boundary. The box guard still meters every model call per run, whoever
sends it.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
import sys
from collections.abc import Iterable, Mapping


__all__ = [
    "HARDEN_VAR",
    "command_env",
    "make_non_dumpable",
    "protect",
    "scrub_environ",
    "wipe_initial_environ",
    "withheld_names",
]

logger = logging.getLogger(__name__)

#: ``prctl`` option: set the process's dumpable flag.
_PR_SET_DUMPABLE = 4

#: How the runner keeps its values from its commands: ``wipe`` (the default) or
#: ``non-dumpable`` (the wipe, then ``prctl(PR_SET_DUMPABLE, 0)``).
HARDEN_VAR = "TULIP_RUNNER_HARDEN"


def command_env(environ: Mapping[str, str], withheld: Iterable[str]) -> dict[str, str]:
    """``environ`` without the ``withheld`` variables: what a command may inherit."""
    drop = {name for name in withheld if name}
    return {key: value for key, value in environ.items() if key not in drop}


def make_non_dumpable() -> bool:
    """Mark this process non-dumpable. True when it took effect (Linux only)."""
    if not sys.platform.startswith("linux"):
        return False
    name = ctypes.util.find_library("c")
    try:
        libc = ctypes.CDLL(name or "libc.so.6", use_errno=True)
        result = libc.prctl(_PR_SET_DUMPABLE, 0, 0, 0, 0)
    except (OSError, AttributeError):  # pragma: no cover - no libc / no prctl
        logger.warning("could not make the runner non-dumpable: libc has no prctl")
        return False
    if result != 0:  # pragma: no cover - prctl refused (never for this option)
        logger.warning("prctl(PR_SET_DUMPABLE) failed: errno %s", ctypes.get_errno())
        return False
    return True


def withheld_names(names: Iterable[str | None]) -> tuple[str, ...]:
    """The non-empty names, deduplicated, in order."""
    return tuple(dict.fromkeys(name for name in names if name))


def scrub_environ(names: Iterable[str]) -> None:
    """Drop ``names`` from this process's ``os.environ`` (after they were read)."""
    for name in names:
        os.environ.pop(name, None)


def wipe_initial_environ(names: Iterable[str]) -> int:
    """Zero the values of ``names`` in this process's initial environment block.

    That block is what ``/proc/<pid>/environ`` shows another process; changing
    ``os.environ`` does not touch it. Its bounds are fields 50 and 51 of
    ``/proc/self/stat`` (``env_start``, ``env_end``). Each ``NAME=value`` entry
    keeps its name and length; the value bytes become NUL. Returns how many
    entries were wiped (0 off Linux, or when the block cannot be read).
    """
    if not sys.platform.startswith("linux"):
        return 0
    keys = [f"{name}=".encode() for name in names if name]
    if not keys:
        return 0
    try:
        with open("/proc/self/stat", "rb") as stat:
            fields = stat.read().rsplit(b")", 1)[1].split()
        start, end = int(fields[47]), int(fields[48])
    except (OSError, IndexError, ValueError):  # pragma: no cover - no /proc
        logger.warning("could not find the runner's initial environment block")
        return 0
    if end <= start:  # pragma: no cover - an empty block
        return 0
    block = (ctypes.c_char * (end - start)).from_address(start)
    raw = bytes(block)
    wiped = 0
    offset = 0
    while offset < len(raw):
        stop = raw.find(b"\0", offset)
        stop = len(raw) if stop == -1 else stop
        entry = raw[offset:stop]
        for key in keys:
            if entry.startswith(key) and len(entry) > len(key):
                ctypes.memset(start + offset + len(key), 0, len(entry) - len(key))
                wiped += 1
                break
        offset = stop + 1
    return wiped


def protect(names: Iterable[str], environ: Mapping[str, str] | None = None) -> None:
    """Keep ``names`` from the commands this runner will start; call once they are read.

    Wipes them from the initial environment block and ``os.environ``; with
    ``TULIP_RUNNER_HARDEN=non-dumpable`` also marks the process non-dumpable
    (which NVIDIA OpenShell's process identity does not allow, see the module).
    """
    chosen = list(names)
    mode = (
        str((os.environ if environ is None else environ).get(HARDEN_VAR) or "wipe").strip().lower()
    )
    wipe_initial_environ(chosen)
    scrub_environ(chosen)
    if mode == "non-dumpable":
        make_non_dumpable()
    elif mode != "wipe":
        logger.warning("%s=%r is not a mode this runner knows; using wipe", HARDEN_VAR, mode)
