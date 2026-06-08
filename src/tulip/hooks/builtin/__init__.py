# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL

"""Built-in hook providers for Tulip.

This module provides ready-to-use hook providers for common use cases:

- LoggingHook / StructuredLoggingHook: Log all lifecycle events
- TelemetryHook: OpenTelemetry integration
- GuardrailsHook / ContentFilterHook: Security guardrails and content filtering
- ModelRetryHook: Backoff retries for empty / rate-limited model responses
- SteeringHook: LLM-as-judge tool gating

Example:
    from tulip.hooks.builtin import LoggingHook, GuardrailsHook, ModelRetryHook

    agent = Agent(..., hooks=[LoggingHook(), GuardrailsHook(), ModelRetryHook()])
"""

from tulip.hooks.builtin.guardrails import (
    ContentFilterHook,
    GuardrailAction,
    GuardrailConfig,
    GuardrailsHook,
    GuardrailViolation,
)
from tulip.hooks.builtin.logging import LoggingHook, StructuredLoggingHook
from tulip.hooks.builtin.retry import ModelRetryHook
from tulip.hooks.builtin.steering import (
    SteeringAction,
    SteeringContext,
    SteeringDecision,
    SteeringHook,
)
from tulip.hooks.builtin.telemetry import (
    NoOpTelemetryHook,
    TelemetryHook,
    create_telemetry_hook,
)


__all__ = [
    # Logging
    "LoggingHook",
    "StructuredLoggingHook",
    # Telemetry
    "TelemetryHook",
    "NoOpTelemetryHook",
    "create_telemetry_hook",
    # Guardrails
    "GuardrailsHook",
    "GuardrailConfig",
    "GuardrailAction",
    "GuardrailViolation",
    "ContentFilterHook",
    # Retry
    "ModelRetryHook",
    # Steering
    "SteeringHook",
    "SteeringAction",
    "SteeringContext",
    "SteeringDecision",
]
