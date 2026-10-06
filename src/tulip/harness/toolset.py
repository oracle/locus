# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Build a coding harness over a workspace: the tools, their labels, a prompt.

:func:`build_harness` is the one entry point. It binds the tool bodies to a
:class:`~tulip.harness.backend.WorkspaceBackend`, pairs each tool with the
:data:`~tulip.control.action.ActionSpec` that labels its calls
(:mod:`tulip.harness.labels`), and passes every pair through ``wrap``.

**The single-gate rule.** Tool bodies never call a gate. ``wrap`` is the one
place governance is applied, and it is applied to every tool the harness
returns: the local CLI passes the :mod:`tulip.control` gate, the gateway
passes its own, and both see the same action for the same call because the
labels come from here. One gate per call means one decision per call, one
audit record per call, and no path to a side effect that the gate did not
see — a check inside a body, beside one around it, is a second policy that
drifts from the first.

**Without ``wrap`` the tools are ungated.** ``wrap=None`` gives you tools that
read, write and run commands with nothing between the model and the
workspace. That is right for a test, or for a workspace that is itself the
boundary (a throwaway sandbox nobody else uses); it is not right for a host
shell. :func:`build_harness` logs a warning when it builds ungated shell
tools over an unisolated backend.

    from tulip.control import ControlPolicy, gate_tool
    from tulip.harness import LocalBackend, build_harness

    policy = ControlPolicy(require_human_for={"workspace.exec"})
    harness = build_harness(
        LocalBackend("."),
        wrap=lambda t, spec: gate_tool(t, policy=policy, action=spec),
    )
    agent = Agent(model=model, tools=harness.tools, system_prompt=harness.prompt_fragment)
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from tulip.control.action import ActionSpec
from tulip.harness.backend import WorkspaceBackend
from tulip.harness.labels import action_spec
from tulip.harness.ledger import ReadLedger
from tulip.harness.tools import FACTORIES, PLANNERS, SHELL_TOOLS
from tulip.harness.tools.common import HarnessConfig, HarnessContext
from tulip.tools.decorator import Tool


__all__ = ["ALL_TOOLS", "DEFAULT_TOOLS", "PATCH_TOOLS", "Harness", "build_harness"]

logger = logging.getLogger(__name__)

#: Every tool the harness can build.
ALL_TOOLS: tuple[str, ...] = tuple(FACTORIES)

#: The default set: everything but ``apply_patch``, which is the edit tool
#: for models trained on the patch format and replaces the three below.
DEFAULT_TOOLS: tuple[str, ...] = tuple(n for n in FACTORIES if n != "apply_patch")

#: The set for a model that edits with patches (the GPT family): the
#: search-and-replace tools swapped for ``apply_patch``.
PATCH_TOOLS: tuple[str, ...] = tuple(
    n for n in FACTORIES if n not in {"edit", "multi_edit", "write"}
)

Wrap = Callable[[Tool, ActionSpec], Tool]


@dataclass
class Harness:
    """A built harness.

    Attributes:
        tools: The tools, wrapped, ready for ``Agent(tools=...)``.
        specs: Each tool's :data:`~tulip.control.action.ActionSpec`, by name.
        prompt_fragment: What the model should know about the workspace and
            the tools, for the system prompt.
        context: The session state the tools share: the backend, the read
            ledger, the plan.
        gated: Whether a ``wrap`` was applied.
    """

    tools: list[Tool]
    specs: dict[str, ActionSpec]
    prompt_fragment: str
    context: HarnessContext
    gated: bool

    @property
    def backend(self) -> WorkspaceBackend:
        return self.context.backend

    def tool(self, name: str) -> Tool:
        """The built tool called ``name``."""
        for built in self.tools:
            if built.name == name:
                return built
        raise KeyError(name)

    def preview(self, name: str, arguments: Mapping[str, Any]) -> str | None:
        """What a file-changing call would do, as a diff, without doing it.

        For a gate that wants to show a person the change before approving
        it. Returns the reason the call would be refused when it would be,
        and ``None`` for a tool that changes no file.
        """
        planner = PLANNERS.get(name)
        if planner is None:
            return None
        plan = planner(self.context, **dict(arguments))
        return plan if isinstance(plan, str) else plan.preview


def _prompt(backend: WorkspaceBackend, names: list[str], config: HarnessConfig) -> str:
    caps = backend.capabilities
    lines = [
        "## Workspace",
        f"The workspace is {caps.label}, rooted at {caps.root}. Paths are relative to that root.",
    ]
    if caps.can_exec and not caps.isolated:
        lines.append(
            "Commands run directly on the host machine, not in a sandbox: "
            "anything a command does is real."
        )
    if config.require_read and {"edit", "write", "multi_edit"} & set(names):
        lines.append(
            "Read a file before you change it. A change to a file you have not read, "
            "or that changed since you read it, is refused."
        )
    if "bash" in names:
        lines.append(
            f"bash waits {config.bash_timeout}s by default (at most "
            f"{config.bash_max_timeout}s). Run servers and watchers with background=true "
            "and read them with bash_output."
        )
    if "apply_patch" in names:
        lines.append("Change files with apply_patch.")
    return "\n".join(lines)


def build_harness(
    backend: WorkspaceBackend,
    *,
    tools: Iterable[str] | None = None,
    wrap: Wrap | None = None,
    config: HarnessConfig | None = None,
) -> Harness:
    """Build the harness tools over ``backend``.

    Args:
        backend: The workspace.
        tools: Which tools, by name (see :data:`ALL_TOOLS`). Default:
            :data:`DEFAULT_TOOLS`, without the shell tools when the backend
            has no shell.
        wrap: Applied to every tool with its action spec — the single place
            governance goes. ``None`` leaves the tools **ungated**.
        config: Behaviour; tulip-code's defaults when omitted.

    Raises:
        ValueError: A tool name is unknown, or names a shell tool on a
            backend with no shell.
    """
    config = config or HarnessConfig()
    caps = backend.capabilities
    if tools is None:
        names = [n for n in DEFAULT_TOOLS if caps.can_exec or n not in SHELL_TOOLS]
    else:
        names = list(dict.fromkeys(tools))
        unknown = [n for n in names if n not in FACTORIES]
        if unknown:
            raise ValueError(f"no harness tool named {', '.join(map(repr, unknown))}")
        shell = [n for n in names if n in SHELL_TOOLS]
        if shell and not caps.can_exec:
            raise ValueError(f"the {caps.label} workspace has no shell for {', '.join(shell)}")
    context = HarnessContext(
        backend=backend, config=config, ledger=ReadLedger(require_read=config.require_read)
    )
    specs = {n: action_spec(n, environment=config.environment) for n in names}
    built: list[Tool] = []
    for name in names:
        made = FACTORIES[name](context)
        built.append(wrap(made, specs[name]) if wrap is not None else made)
    if wrap is None and caps.can_exec and not caps.isolated and SHELL_TOOLS & set(names):
        logger.warning(
            "harness built with ungated shell tools on %s: commands run on this host "
            "with nothing in between; pass wrap= to gate them",
            caps.label,
        )
    return Harness(
        tools=built,
        specs=specs,
        prompt_fragment=_prompt(backend, names, config),
        context=context,
        gated=wrap is not None,
    )
