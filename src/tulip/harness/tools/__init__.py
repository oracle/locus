# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""The harness tool bodies, written once over any workspace backend.

Each ``make_<name>`` builds one :class:`~tulip.tools.decorator.Tool` bound
to a :class:`~tulip.harness.tools.common.HarnessContext`. No body calls a
gate: governance is applied around the built tool, in one place, by
:func:`tulip.harness.toolset.build_harness`'s ``wrap``.
"""

from __future__ import annotations

from collections.abc import Callable

from tulip.harness.tools.common import HarnessConfig, HarnessContext, Plan
from tulip.harness.tools.files import (
    make_edit,
    make_ls,
    make_multi_edit,
    make_read,
    make_write,
    plan_edit,
    plan_multi_edit,
    plan_write,
)
from tulip.harness.tools.notebook import make_notebook_edit, plan_notebook_edit
from tulip.harness.tools.patch import make_apply_patch, plan_patch
from tulip.harness.tools.search import make_glob, make_grep
from tulip.harness.tools.shell import (
    make_bash,
    make_bash_output,
    make_kill_shell,
    make_write_stdin,
)
from tulip.harness.tools.todos import make_todo_read, make_todo_write
from tulip.tools.decorator import Tool


__all__ = ["FACTORIES", "PLANNERS", "SHELL_TOOLS", "HarnessConfig", "HarnessContext", "Plan"]

#: Every harness tool, by name, in the order a model tends to reach for them.
FACTORIES: dict[str, Callable[[HarnessContext], Tool]] = {
    "read": make_read,
    "ls": make_ls,
    "glob": make_glob,
    "grep": make_grep,
    "edit": make_edit,
    "multi_edit": make_multi_edit,
    "write": make_write,
    "apply_patch": make_apply_patch,
    "notebook_edit": make_notebook_edit,
    "bash": make_bash,
    "bash_output": make_bash_output,
    "kill_shell": make_kill_shell,
    "write_stdin": make_write_stdin,
    "todo_write": make_todo_write,
    "todo_read": make_todo_read,
}

#: The tools that need a shell.
SHELL_TOOLS = frozenset({"bash", "bash_output", "kill_shell", "write_stdin"})

#: The file-changing tools' planners: the change a call would make, without
#: making it, or why it cannot be made.
PLANNERS: dict[str, Callable[..., Plan | str]] = {
    "write": plan_write,
    "edit": plan_edit,
    "multi_edit": plan_multi_edit,
    "apply_patch": plan_patch,
    "notebook_edit": plan_notebook_edit,
}
