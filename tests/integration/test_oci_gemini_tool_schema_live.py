# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Live integration test for #281 — OCI Gemini tool-schema sanitization.

Proves the fix end-to-end against a real OCI GenAI Gemini endpoint.

The pre-fix bug (V1 OpenAI-compat transport, ``google.gemini-2.5-flash``):
Pydantic emits ``type: ["string", "null"]`` for ``Optional[str]`` and
``type: "any"`` for ``Any``. Gemini's Proto schema validator rejected
the list form with a hard 400::

    400 Bad Request: Unknown name 'type' at
    'tools[0].function_declarations[N].parameters.properties[M].value...':
    Proto field is not repeating, cannot start list.

The fix (``locus.core.structured.sanitize_oci_tool_schema``, wired into
``OCIChatCompletionsModel._sanitize_tools_for_oci``) sanitizes every
tool's ``function.parameters`` before the V1 call leaves Locus, mirroring
the ``langchain_oci.common.utils.OCIUtils.sanitize_schema`` reference
pattern.

This test calls the OCI provider directly (no agent runtime loop) so
the assertion is precisely "the OCI endpoint accepted the schema." A
deep-agent run would add the orthogonal empty-content concern from #280,
which is not what this test is verifying.

Cost: single ``complete()`` call per test, ``max_output_tokens=128`` —
roughly a cent on Gemini 2.5 Flash.

Usage::

    OCI_PROFILE=API_FREE_TIER OCI_REGION=us-chicago-1 RUN_LIVE_OCI=1 \\
      .venv/bin/python -m pytest \\
      tests/integration/test_oci_gemini_tool_schema_live.py -v
"""

from __future__ import annotations

import asyncio
import os

import pytest

from locus.core.messages import Message


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_OCI") != "1",
    reason="Live OCI Gemini tests gated by RUN_LIVE_OCI=1",
)


def _build_gemini_model():
    """Build an OCI Gemini chat model via the V1 transport.

    Mirrors the auth pattern used by ``test_deepagent_token_budget_live``
    so a single ``OCI_PROFILE`` env var covers both suites."""
    from locus.models import get_model

    profile = os.environ.get("OCI_PROFILE", "API_FREE_TIER")
    region = os.environ.get("OCI_REGION", "us-chicago-1")
    compartment = os.environ.get("OCI_COMPARTMENT") or os.environ.get(
        "OCI_TENANCY",
        # API_FREE_TIER tenancy default; harmless for any GenAI-enabled tenancy.
        "ocid1.tenancy.oc1..aaaaaaaaqlhpnytg33ztkwrdpq62p5yxx5gn5ltmkah23m7qebwjzc7x3lcq",
    )
    return get_model(
        "oci:google.gemini-2.5-flash",
        profile=profile,
        region=region,
        compartment_id=compartment,
    )


# Tool schema that triggered the pre-fix 400. Hand-built (rather than
# Pydantic-derived) so the test pinpoints the exact failing shapes and
# stays stable across Pydantic versions that occasionally change how
# they emit Optional fields.
_BUG_TRIGGERING_TOOL = {
    "type": "function",
    "function": {
        "name": "search_traces",
        "description": "Search distributed traces by free-text query.",
        "parameters": {
            "type": "object",
            "title": "SearchTracesParams",  # stripped by sanitizer
            "properties": {
                "query": {"type": "string"},
                # Optional[str] — list type → Proto rejected pre-fix
                "cluster": {"type": ["string", "null"]},
                # Optional[int] — list type → Proto rejected pre-fix
                "limit": {"type": ["null", "integer"], "default": None},
                # Any annotation — "any" not in Gemini's vocabulary
                "filters": {"type": "any"},
                # Nested array-of-objects with Optional fields inside —
                # the exact deepest-nesting shape from the issue's
                # ``items.properties[N].value`` error path.
                "tags": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "title": "Tag",  # stripped
                        "properties": {
                            "key": {"type": "string"},
                            "value": {"type": ["string", "null"]},
                        },
                    },
                },
            },
            "required": ["query"],
        },
    },
}


_CLEAN_TOOL = {
    "type": "function",
    "function": {
        "name": "list_services",
        "description": "Return the static service inventory.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}


def test_gemini_accepts_optional_field_tool_schema() -> None:
    """The smoking-gun: a tool schema that previously raised 400 from
    OCI's Gemini Proto validator now completes without error.

    The assertion is deliberately weak ("no exception, response object
    returned") — we're verifying the SCHEMA path. Whether Gemini
    decides to invoke the tool or answer free-form is irrelevant; both
    outcomes prove the validator cleared the schema."""
    model = _build_gemini_model()
    response = asyncio.run(
        model.complete(
            messages=[
                Message.system("You are a tracing assistant. Be terse."),
                Message.user("If you need traces, call search_traces."),
            ],
            tools=[_BUG_TRIGGERING_TOOL],
            max_tokens=128,
        )
    )
    # Pre-fix: this line was unreachable — model.complete() raised
    # `openai.BadRequestError: 400 ... Unknown name 'type' at
    # 'tools[0].function_declarations[0].parameters...'`.
    assert response is not None


def test_gemini_accepts_clean_schema_unchanged() -> None:
    """Control: a schema with NO bug shapes still completes. Proves
    the sanitizer is idempotent on already-clean schemas — it doesn't
    accidentally over-strip and break the no-Optional tool case."""
    model = _build_gemini_model()
    response = asyncio.run(
        model.complete(
            messages=[
                Message.system("You are a service catalog. Be terse."),
                Message.user("If asked about services, call list_services."),
            ],
            tools=[_CLEAN_TOOL],
            max_tokens=128,
        )
    )
    assert response is not None


# A ``tools=None`` smoke test would belong here, but the OCI Gemini
# endpoint occasionally returns ``usage.completion_tokens=None`` on
# ultra-short completions and Locus's OpenAIModel response parser
# doesn't tolerate that — a separate, orthogonal issue. The two tests
# above cover the #281 schema-sanitization surface; tools=None isn't
# touched by this fix (``_sanitize_tools_for_oci`` short-circuits at
# the top), so the unit test ``test_sanitize_tools_for_oci_returns_none_for_none``
# is the canonical guard for that path.
