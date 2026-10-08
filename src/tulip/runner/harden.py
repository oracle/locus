# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Keep what the runner holds away from the commands it runs.

The runner holds three things a command in the box must not use: its workload
token (it answers the gateway as this run), and the OpenShell placeholders of
the model's key and of each connector's credential (the sandbox's proxy turns
a placeholder into the real credential on its host, for whoever sends it).
A ``bash`` call is the model's, admitted by the gateway as a command — not as
a model call or a connector call — so it must not be able to make those.

Two layers, both applied by the runner at start:

1. **No inheritance.** Commands get :func:`command_env`: the runner's
   environment without those variables, so a child process never receives
   them.
2. **No reading them back.** :func:`make_non_dumpable` sets the runner
   process non-dumpable (``prctl(PR_SET_DUMPABLE, 0)``). Linux then makes the
   runner's ``/proc/<pid>/environ`` and memory owned by root and refuses
   ``ptrace`` from the same user, so a command running as the same uid cannot
   read the variables out of its parent. Children are dumpable again after
   ``execve``; that is fine, they never had the values.

What this does **not** cover: a process with ``CAP_SYS_PTRACE`` or root in the
box. The runner image runs as a non-root user with no capabilities, and
OpenShell's process policy drops them; that is the boundary. The box guard
still meters every model call per run, whoever sends it.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
import sys
from collections.abc import Iterable, Mapping


__all__ = ["command_env", "make_non_dumpable", "scrub_environ", "withheld_names"]

logger = logging.getLogger(__name__)

#: ``prctl`` option: set the process's dumpable flag.
_PR_SET_DUMPABLE = 4


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
