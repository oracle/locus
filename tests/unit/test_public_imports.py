# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL

"""Guard the documented public import surface.

Past regressions:

- ``from tulip import Reflexion`` raised ``ImportError`` because the
  package's lazy-import map pointed at a class name (``Reflexion``) that
  doesn't exist in source — the real class is ``Reflector``. Both names are
  now valid imports (``Reflexion`` is an alias).
- ``from tulip.hooks.builtin import ModelRetryHook`` and
  ``SteeringHook`` failed because the submodule's ``__init__`` only
  re-exported logging / telemetry / guardrails, even though the README
  hook table named both.

Each test below imports a name the README documents. Failing here means a
user following the docs gets an ImportError — the loudest possible kind of
"feature exists in source but is unreachable" bug.
"""

from __future__ import annotations

import importlib


class TestTopLevelTulip:
    def test_reflexion_alias(self):
        import tulip

        # ``Reflexion`` should resolve via the lazy importer to the real
        # ``Reflector`` class — not raise ImportError.
        cls = tulip.Reflexion
        assert cls.__name__ == "Reflector"

    def test_reflector_direct(self):
        import tulip

        cls = tulip.Reflector
        assert cls.__name__ == "Reflector"

    def test_reflexion_and_reflector_are_same_class(self):
        import tulip

        assert tulip.Reflexion is tulip.Reflector

    def test_reflexion_in_dunder_all(self):
        import tulip

        # ``__all__`` must list both spellings so ``from tulip import *``
        # exposes them and tooling (pyright, sphinx) doesn't drop them.
        assert "Reflexion" in tulip.__all__
        assert "Reflector" in tulip.__all__


class TestHookBuiltinExports:
    """Every hook the README's hook-table names must be importable from
    ``tulip.hooks.builtin`` directly.
    """

    def test_logging_hooks(self):
        from tulip.hooks.builtin import LoggingHook, StructuredLoggingHook

        assert LoggingHook is not None
        assert StructuredLoggingHook is not None

    def test_telemetry_hooks(self):
        from tulip.hooks.builtin import NoOpTelemetryHook, TelemetryHook

        assert TelemetryHook is not None
        assert NoOpTelemetryHook is not None

    def test_guardrails_hooks(self):
        from tulip.hooks.builtin import ContentFilterHook, GuardrailsHook

        assert GuardrailsHook is not None
        assert ContentFilterHook is not None

    def test_model_retry_hook_exported(self):
        # The README hook table names ``ModelRetryHook`` but earlier versions
        # only re-exported logging / telemetry / guardrails.
        from tulip.hooks.builtin import ModelRetryHook

        assert ModelRetryHook is not None

    def test_steering_hook_exported(self):
        from tulip.hooks.builtin import (
            SteeringAction,
            SteeringContext,
            SteeringDecision,
            SteeringHook,
        )

        assert SteeringHook is not None
        # Bonus: the supporting types should ship together.
        assert SteeringAction is not None
        assert SteeringContext is not None
        assert SteeringDecision is not None

    def test_dunder_all_lists_new_exports(self):
        mod = importlib.import_module("tulip.hooks.builtin")
        for name in (
            "ModelRetryHook",
            "SteeringHook",
            "SteeringAction",
            "SteeringContext",
            "SteeringDecision",
        ):
            assert name in mod.__all__, f"{name} missing from tulip.hooks.builtin.__all__"
