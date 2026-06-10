# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Tulip warning hierarchy.

The warning classes are tiny, but coverage was sitting at 0% because
nothing in the test suite imports them. The hierarchy is part of the
documented deprecation contract (consumers do
``simplefilter("error", TulipDeprecationWarning)``), so it deserves a
regression test that pins the inheritance shape.
"""

from __future__ import annotations

import warnings

import pytest

from tulip.core.warnings import TulipDeprecationWarning, TulipWarning


class TestTulipWarningHierarchy:
    """Pin the inheritance shape — it's part of the public contract."""

    def test_tulip_warning_is_user_warning(self) -> None:
        assert issubclass(TulipWarning, UserWarning)

    def test_tulip_deprecation_warning_inherits_tulip_warning(self) -> None:
        assert issubclass(TulipDeprecationWarning, TulipWarning)

    def test_tulip_deprecation_warning_inherits_stdlib_deprecation(self) -> None:
        # Inheriting DeprecationWarning lets ``-W error::DeprecationWarning``
        # catch Tulip deprecations the same way stdlib ones get caught.
        assert issubclass(TulipDeprecationWarning, DeprecationWarning)


class TestTulipWarningFiltering:
    """Smoke tests for the documented filter usage."""

    def test_tulip_warning_can_be_elevated_to_error(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", TulipWarning)
            with pytest.raises(TulipWarning):
                warnings.warn("test", TulipWarning, stacklevel=1)

    def test_tulip_deprecation_caught_by_tulip_warning_filter(self) -> None:
        # Critical: the documented usage in ``tulip.core.warnings``
        # promises that filtering on ``TulipWarning`` catches every
        # Tulip-originated subclass, including the deprecation one.
        with warnings.catch_warnings():
            warnings.simplefilter("error", TulipWarning)
            with pytest.raises(TulipDeprecationWarning):
                warnings.warn("deprecated", TulipDeprecationWarning, stacklevel=1)

    def test_tulip_deprecation_caught_by_stdlib_filter(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            with pytest.raises(TulipDeprecationWarning):
                warnings.warn("deprecated", TulipDeprecationWarning, stacklevel=1)
