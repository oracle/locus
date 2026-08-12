# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""`tulip.__version__` must match the version the package ships as.

Not hypothetical: `tulip_agents-2.5.0-py3-none-any.whl` went to PyPI with
`METADATA Version: 2.5.0` and `tulip.__version__ == "2.4.0"` inside it,
because the literal in `src/tulip/__init__.py` and the one in
`pyproject.toml` are maintained by hand and drifted. Anything reading
`__version__` -- telemetry, a bug report, a compatibility check in the
gateway -- was told the wrong release for the whole of 2.5.0.

Two duplicated literals will drift again; this is the check that makes
CI notice rather than PyPI.
"""

from __future__ import annotations

import pathlib
import re

import tulip


def _pyproject_version() -> str | None:
    """The version in pyproject.toml, or None when running from an
    installed wheel where the source tree is not present."""
    root = pathlib.Path(__file__).resolve().parents[2]
    pyproject = root / "pyproject.toml"
    if not pyproject.is_file():
        return None
    match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject.read_text(), re.MULTILINE)
    return match.group(1) if match else None


def test_dunder_version_matches_pyproject() -> None:
    declared = _pyproject_version()
    if declared is None:
        return  # installed-wheel run; the metadata test below still applies
    assert tulip.__version__ == declared, (
        f"tulip.__version__ is {tulip.__version__!r} but pyproject.toml declares "
        f"{declared!r}. Both are hand-maintained -- update src/tulip/__init__.py."
    )


def test_dunder_version_matches_installed_metadata() -> None:
    """Guards the built artifact, which is what a user actually gets."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        installed = version("tulip-agents")
    except PackageNotFoundError:
        return  # not installed in this env (source-only run)
    assert tulip.__version__ == installed, (
        f"tulip.__version__ is {tulip.__version__!r} but the installed "
        f"distribution is {installed!r}."
    )


def test_dunder_version_is_a_release_number() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+([abrc.\-+].*)?", tulip.__version__), (
        f"{tulip.__version__!r} is not a recognisable version"
    )
