# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""The harness tool bodies, written once over any workspace backend.

Each ``make_<name>`` builds one :class:`~locus.tools.decorator.Tool` bound
to a :class:`~locus.harness.tools.common.HarnessContext`. No body calls a
gate: governance is applied around the built tool, in one place, by
:func:`locus.harness.toolset.build_harness`'s ``wrap``.
"""

from __future__ import annotations

#: No harness tool is declared ``idempotent``. The agent loop answers a repeated call to an
#: idempotent tool from the run's earlier result without running it, and a workspace read is
#: not repeatable: after an edit, ``read`` of the same path must show the edit. Seen live on
#: dev (functional F23): a subagent's read after its own edit returned the file as it was.
from collections.abc import Callable

from locus.harness.tools.common import HarnessConfig, HarnessContext, Plan
from locus.harness.tools.files import (
    make_edit,
    make_ls,
    make_multi_edit,
    make_read,
    make_write,
    plan_edit,
    plan_multi_edit,
    plan_write,
)
from locus.harness.tools.notebook import make_notebook_edit, plan_notebook_edit
from locus.harness.tools.patch import make_apply_patch, plan_patch
from locus.harness.tools.results import make_fetch_tool_result
from locus.harness.tools.search import make_glob, make_grep
from locus.harness.tools.shell import (
    make_bash,
    make_bash_output,
    make_kill_shell,
    make_write_stdin,
)
from locus.harness.tools.todos import make_todo_read, make_todo_write
from locus.tools.decorator import Tool


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
    "fetch_tool_result": make_fetch_tool_result,
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
