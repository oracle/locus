# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Say which extra to install when a model provider's package is missing.

Model providers are optional extras, so ``pip install locus-sdk`` alone has
no provider client. Python's own error for that names a module
(``No module named 'requests'``), which does not say what to do about it.
The helpers here raise the same :class:`ModuleNotFoundError`, with the
install command in the message, so ``except ImportError`` callers keep
working.
"""

from __future__ import annotations

import importlib.util


__all__ = ["missing_extra", "require_packages"]


def missing_extra(what: str, extra: str, package: str | None) -> ModuleNotFoundError:
    """The error for ``what`` being used without the package its ``extra`` installs."""
    needs = f"the {package} package" if package else "packages that are not installed"
    return ModuleNotFoundError(
        f'{what} needs {needs}: pip install "locus-sdk[{extra}]"',
        name=package,
    )


def require_packages(what: str, extra: str, packages: tuple[str, ...]) -> None:
    """Raise :func:`missing_extra` for the first of ``packages`` that is not installed."""
    for package in packages:
        if importlib.util.find_spec(package) is None:
            raise missing_extra(what, extra, package)
