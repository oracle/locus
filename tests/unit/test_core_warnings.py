# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for the Locus warning hierarchy.

The warning classes are tiny, but coverage was sitting at 0% because
nothing in the test suite imports them. The hierarchy is part of the
documented deprecation contract (consumers do
``simplefilter("error", LocusDeprecationWarning)``), so it deserves a
regression test that pins the inheritance shape.
"""

from __future__ import annotations

import warnings

import pytest

from locus.core.warnings import LocusDeprecationWarning, LocusWarning


class TestLocusWarningHierarchy:
    """Pin the inheritance shape — it's part of the public contract."""

    def test_locus_warning_is_user_warning(self) -> None:
        assert issubclass(LocusWarning, UserWarning)

    def test_locus_deprecation_warning_inherits_locus_warning(self) -> None:
        assert issubclass(LocusDeprecationWarning, LocusWarning)

    def test_locus_deprecation_warning_inherits_stdlib_deprecation(self) -> None:
        # Inheriting DeprecationWarning lets ``-W error::DeprecationWarning``
        # catch Locus deprecations the same way stdlib ones get caught.
        assert issubclass(LocusDeprecationWarning, DeprecationWarning)


class TestLocusWarningFiltering:
    """Smoke tests for the documented filter usage."""

    def test_locus_warning_can_be_elevated_to_error(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", LocusWarning)
            with pytest.raises(LocusWarning):
                warnings.warn("test", LocusWarning, stacklevel=1)

    def test_locus_deprecation_caught_by_locus_warning_filter(self) -> None:
        # Critical: the documented usage in ``locus.core.warnings``
        # promises that filtering on ``LocusWarning`` catches every
        # Locus-originated subclass, including the deprecation one.
        with warnings.catch_warnings():
            warnings.simplefilter("error", LocusWarning)
            with pytest.raises(LocusDeprecationWarning):
                warnings.warn("deprecated", LocusDeprecationWarning, stacklevel=1)

    def test_locus_deprecation_caught_by_stdlib_filter(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            with pytest.raises(LocusDeprecationWarning):
                warnings.warn("deprecated", LocusDeprecationWarning, stacklevel=1)
