# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""``locus.security`` is a compatibility shim over two new homes.

The grounding / verification names moved to ``locus.control`` and still import
from the old path with a deprecation warning. The security-domain tooling is no
longer part of Locus (it last shipped as ``locus-sdk-security`` at v5.5.0):
the old path resolves it when that package is installed and raises an
ImportError saying so when it is not. Submodule paths alias to the
same module objects, so nothing is loaded twice.
"""

from __future__ import annotations

import importlib
import sys
import warnings

import pytest

import locus.security as shim
from locus.core.warnings import LocusDeprecationWarning


_CORE = sorted(shim._CORE_NAMES)
_DOMAIN = sorted(shim._DOMAIN_NAMES)


@pytest.fixture
def fresh_shim(monkeypatch: pytest.MonkeyPatch) -> object:
    """The shim with no names resolved yet (it caches each one on first use)."""
    for name in (*_CORE, *_DOMAIN):
        monkeypatch.delitem(vars(shim), name, raising=False)
    return shim


@pytest.fixture
def without_security_package(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``locus_security`` unimportable, as on a core-only install."""
    for name in list(sys.modules):
        if name == "locus_security" or name.startswith(("locus_security.", "locus.security.")):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "locus_security", None)


def test_the_two_name_sets_are_disjoint_and_cover_the_old_surface() -> None:
    assert not set(_CORE) & set(_DOMAIN)
    assert sorted(shim.__all__) == sorted([*_CORE, *_DOMAIN])
    assert dir(shim) == sorted([*_CORE, *_DOMAIN])


@pytest.mark.parametrize("name", _CORE)
def test_core_names_resolve_from_control_with_a_warning(fresh_shim: object, name: str) -> None:
    from locus import control

    with pytest.warns(LocusDeprecationWarning, match=rf"locus\.control\.{name}"):
        value = getattr(fresh_shim, name)
    assert value is getattr(control, name)


def test_a_core_name_warns_once(fresh_shim: object) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from locus.security import Evidence  # noqa: F401
        from locus.security import Evidence as Again  # noqa: F401
    assert len([w for w in caught if issubclass(w.category, LocusDeprecationWarning)]) == 1


def test_core_names_resolve_without_the_security_package(
    fresh_shim: object, without_security_package: None
) -> None:
    with pytest.warns(LocusDeprecationWarning):
        from locus.security import Severity, ground_finding

    from locus import control

    assert Severity is control.Severity
    assert ground_finding is control.ground_finding


def test_domain_names_without_the_package_say_what_to_install(
    fresh_shim: object, without_security_package: None
) -> None:
    with pytest.raises(ImportError, match="no longer part of Locus") as exc:
        from locus.security import red_team  # noqa: F401
    assert exc.value.name == "locus_security"


def test_unknown_names_are_attribute_errors(fresh_shim: object) -> None:
    assert not hasattr(fresh_shim, "ControlPolicy")
    with pytest.raises(ImportError):
        from locus.security import ControlPolicy  # noqa: F401


@pytest.mark.parametrize(("old", "new"), sorted(shim._CORE_MODULES.items()))
def test_core_submodules_alias_the_moved_module(old: str, new: str) -> None:
    sys.modules.pop(f"locus.security.{old}", None)
    with pytest.warns(LocusDeprecationWarning, match=new.replace(".", r"\.")):
        module = importlib.import_module(f"locus.security.{old}")
    assert module is importlib.import_module(new)
    # The alias keeps the real module's identity for reload / pickling.
    assert module.__spec__ is not None
    assert module.__spec__.name == new


def test_domain_submodules_without_the_package_say_what_to_install(
    without_security_package: None,
) -> None:
    with pytest.raises(ImportError, match="no longer part of Locus"):
        importlib.import_module("locus.security.redteam.probes")


def test_unknown_submodules_are_not_found() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("locus.security.no_such_module")


def test_the_finder_is_installed_once() -> None:
    importlib.reload(shim)
    finders = [f for f in sys.meta_path if type(f).__name__ == "_AliasFinder"]
    assert len(finders) == 1
