"""Retired event names: served with a deprecation warning until 7.0.0, absent from the registry."""

from __future__ import annotations

import importlib
import sys

import pytest

from locus.core.warnings import LocusDeprecationWarning


def _emit_module():
    importlib.import_module("locus.observability.emit")
    return sys.modules["locus.observability.emit"]


class TestRetiredEventNames:
    def test_the_retired_name_still_resolves_with_a_warning(self) -> None:
        with pytest.warns(LocusDeprecationWarning, match="removed in Locus 7.0.0"):
            value = _emit_module().EV_MEMORY_CONVERSATION_ADDED
        assert value == "memory.conversation.added"

    def test_a_from_import_warns_too(self) -> None:
        with pytest.warns(LocusDeprecationWarning):
            from locus.observability.emit import EV_MEMORY_CONVERSATION_ADDED  # noqa: PLC0415
        assert EV_MEMORY_CONVERSATION_ADDED == "memory.conversation.added"

    def test_the_registry_no_longer_lists_it(self) -> None:
        assert "EV_MEMORY_CONVERSATION_ADDED" not in dir(_emit_module())
        assert "EV_MEMORY_CONVERSATION_ADDED" not in vars(_emit_module())

    def test_other_unknown_names_raise(self) -> None:
        with pytest.raises(AttributeError):
            _emit_module().EV_NOT_A_REAL_EVENT  # noqa: B018
