# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Codex on Locus: the skill that teaches OpenAI Codex the SDK, and how to start Codex with it.

Codex discovers skills in ``.agents/skills/<name>/SKILL.md`` at the root of
the project it works in and in ``~/.agents/skills``. The Locus skill is the
repository's ``.agents/skills/locus`` directory, shipped inside the wheel as
``locus/codex/skill``. The ``locus codex`` command puts it where Codex looks
and starts Codex with the sandbox settings the skill's commands need::

    locus codex init            # this project: .agents/skills/locus + a Locus section in AGENTS.md
    locus codex init --home     # every project: ~/.agents/skills/locus
    locus codex run "Build an agent that triages tickets with a gated refund tool"
    locus codex path            # where the bundled skill is, for a copy or a symlink by hand
"""

from __future__ import annotations

import os
import shutil
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "AGENTS_SECTION",
    "CODEX_FLAGS",
    "SKILL_DIR",
    "SKILL_NAME",
    "CodexError",
    "add_agents_section",
    "codex_argv",
    "install",
    "launch",
    "skill_source",
]

#: The skill's name, which is also the directory Codex looks for.
SKILL_NAME = "locus"

#: Where the skill goes, relative to a project root or to the home directory.
SKILL_DIR = Path(".agents") / "skills" / SKILL_NAME

#: ``codex`` settings every command in the skill's recipes needs: writes inside
#: the project (``.venv``, output files) and network for ``pip`` and the model.
CODEX_FLAGS: tuple[str, ...] = (
    "-s",
    "workspace-write",
    "-c",
    "sandbox_workspace_write.network_access=true",
)

_MARKER = "<!-- locus-skill -->"

#: The section ``init`` adds to a project's ``AGENTS.md``, once.
AGENTS_SECTION = f"""
## Locus

{_MARKER}
Code that imports `locus` follows the `locus` skill in
`.agents/skills/locus/SKILL.md`; read it before writing or changing such
code. Every tool with a side effect runs through `gate_tool()` or `admit()`
from `locus.control` with a `ControlPolicy` and an `AuditTrail`. Never print,
log or store a credential; providers read theirs from the environment or
from `~/.oci/config`.
"""

_INSTALL_HINT = (
    "codex is not on PATH; install it with `npm install -g @openai/codex` "
    "(https://developers.openai.com/codex) and sign in"
)


class CodexError(Exception):
    """A problem the person has to fix, reported without a traceback."""


def skill_source() -> Path:
    """The bundled skill directory.

    Inside an installed wheel that is ``locus/codex/skill``; in an editable
    install of the repository it is the repository's ``.agents/skills/locus``.
    """
    packaged = Path(str(resources.files(__name__) / "skill"))
    if (packaged / "SKILL.md").is_file():
        return packaged
    checkout = Path(__file__).resolve().parents[3] / SKILL_DIR
    if (checkout / "SKILL.md").is_file():
        return checkout
    msg = "the Locus skill is missing from this installation; reinstall locus-sdk"
    raise CodexError(msg)


def _files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts)


def install(dest: Path) -> list[Path]:
    """Copy the skill to ``dest`` and return the files written.

    A file that is already there with the same content is left alone, so a
    second run writes nothing; files ``dest`` has that the skill does not are
    kept.
    """
    source = skill_source()
    written: list[Path] = []
    for src in _files(source):
        target = dest / src.relative_to(source)
        data = src.read_bytes()
        if target.is_file() and target.read_bytes() == data:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        written.append(target)
    return written


def add_agents_section(project: Path) -> bool:
    """Add the Locus section to ``project/AGENTS.md``; ``True`` when the file changed.

    The section carries a marker, so it is added once however many times this
    runs. A missing ``AGENTS.md`` is created with the section as its body.
    """
    path = project / "AGENTS.md"
    if path.is_file():
        text = path.read_text()
        if _MARKER in text:
            return False
        path.write_text(text.rstrip("\n") + "\n" + AGENTS_SECTION)
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# Agent instructions\n" + AGENTS_SECTION)
    return True


def codex_argv(
    prompt: str | None,
    *,
    exec_mode: bool = False,
    codex_profile: str | None = None,
    extra: Sequence[str] = (),
) -> list[str]:
    """The ``codex`` command line: the sandbox settings, then the caller's own arguments, then the prompt."""
    argv = ["codex"]
    if exec_mode:
        argv.append("exec")
    argv.extend(CODEX_FLAGS)
    if codex_profile:
        argv.extend(["-p", codex_profile])
    rest = list(extra)
    if rest and rest[0] == "--":
        rest = rest[1:]
    argv.extend(rest)
    if prompt is not None:
        argv.append(prompt)
    return argv


def launch(argv: Sequence[str]) -> None:
    """Replace this process with ``codex``.

    Codex takes over the terminal, so there is nothing to return to.
    """
    executable = shutil.which(argv[0])
    if executable is None:
        raise CodexError(_INSTALL_HINT)
    os.execv(executable, list(argv))  # noqa: S606 - the point of the command is to hand over to codex
