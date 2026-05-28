# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for #281 — OCI GenAI tool-schema sanitization.

Covers :func:`locus.core.structured.sanitize_oci_tool_schema` and the
wiring in :class:`OCIChatCompletionsModel` that applies it to every
``tools[*].function.parameters`` before the V1 (OpenAI-compat) call
leaves Locus.

The bug shape (observed against ``google.gemini-2.5-flash`` via OCI's
``/openai/v1/chat/completions`` endpoint):

    400 Bad Request: Unknown name 'type' at
    'tools[0].function_declarations[10].parameters.properties[7]
     .value.items.properties[0].value':
    Proto field is not repeating, cannot start list.

Pydantic emits ``type: ["string", "null"]`` for ``Optional[str]`` and
``type: "any"`` for ``Any``. Gemini's Proto schema validator requires
``type`` to be a single string. OCI's tool API also strips a handful
of metadata keys (``title``, ``const``, ``x-*``, ``default: None``).

The fix mirrors ``langchain_oci.common.utils.OCIUtils.sanitize_schema``
— the validated reference pattern used by Oracle's own langchain
package against the same endpoint. It applies UNCONDITIONALLY to OCI
tool params (no vendor gate) because the keys it strips are pure
metadata that non-Gemini vendors ignore.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, Field

from locus.core.structured import sanitize_oci_tool_schema


# ---------------------------------------------------------------------------
# Helper behaviour — direct invocation on crafted + Pydantic-generated schemas
# ---------------------------------------------------------------------------


class _SearchQuery(BaseModel):
    """Tool param with an Optional[str] — exact shape of the observed bug."""

    query: str = Field(description="Search string")
    cluster: str | None = Field(default=None, description="Cluster scope")
    limit: int = 10


def test_optional_field_collapses_type_list_to_single_string() -> None:
    """The core bug: ``Optional[str]`` emits ``type: ["string", "null"]``;
    Gemini's Proto rejects the list. Sanitizer must keep the first non-null."""
    schema = _SearchQuery.model_json_schema()
    cleaned = sanitize_oci_tool_schema(schema)

    # Pydantic v2 may emit Optional[str] as either ``type: ["string", "null"]``
    # or as ``anyOf: [{type: string}, {type: null}]``. We care about the
    # invariant: no ``type`` field anywhere in the cleaned schema is a list.

    def _walk_no_list_types(node: Any) -> None:
        if isinstance(node, dict):
            if "type" in node:
                assert not isinstance(node["type"], list), (
                    f"type leaked as list: {node['type']} at {node}"
                )
            for v in node.values():
                _walk_no_list_types(v)
        elif isinstance(node, list):
            for item in node:
                _walk_no_list_types(item)

    _walk_no_list_types(cleaned)


def test_explicit_type_list_collapses_to_first_non_null() -> None:
    """Crafted schema with the exact Proto-rejecting shape."""
    schema = {
        "type": "object",
        "properties": {
            "x": {"type": ["string", "null"]},
            "y": {"type": ["null", "number"]},
            "z": {"type": ["null"]},  # degenerate — sanitizer falls back to "string"
        },
    }
    cleaned = sanitize_oci_tool_schema(schema)
    assert cleaned["properties"]["x"]["type"] == "string"
    assert cleaned["properties"]["y"]["type"] == "number"
    assert cleaned["properties"]["z"]["type"] == "string"


def test_type_any_normalises_to_object() -> None:
    """Pydantic emits ``type: "any"`` for ``Any`` annotations. Gemini
    doesn't know ``any`` — sanitizer remaps to ``object``."""
    schema = {"type": "object", "properties": {"payload": {"type": "any"}}}
    cleaned = sanitize_oci_tool_schema(schema)
    assert cleaned["properties"]["payload"]["type"] == "object"


def test_metadata_keys_stripped() -> None:
    """``title``, ``const``, ``x-*``, ``default: None`` — OCI rejects them
    on tool params. Sanitizer drops them."""
    schema = {
        "type": "object",
        "title": "MySchema",
        "x-internal-note": "ignore me",
        "properties": {
            "name": {
                "type": "string",
                "title": "Name",
                "default": None,
                "x-display": "inline",
            },
            "kind": {"const": "search"},
        },
    }
    cleaned = sanitize_oci_tool_schema(schema)
    assert "title" not in cleaned
    assert "x-internal-note" not in cleaned
    name = cleaned["properties"]["name"]
    assert "title" not in name
    assert "default" not in name
    assert "x-display" not in name
    # ``const: "search"`` is promoted to ``enum: ["search"]``
    assert cleaned["properties"]["kind"]["enum"] == ["search"]
    assert "const" not in cleaned["properties"]["kind"]


def test_user_field_named_title_survives_recursion() -> None:
    """The metadata-strip is applied to schema-level keys ONLY. A user
    field literally called ``title`` (under ``properties``) must survive
    — otherwise we'd silently drop legitimate fields. This is why the
    helper recurses into ``properties`` differently than into arbitrary
    sub-schemas."""
    schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "User's title"},
            "body": {"type": "string"},
        },
        "required": ["title", "body"],
    }
    cleaned = sanitize_oci_tool_schema(schema)
    assert "title" in cleaned["properties"]
    assert cleaned["properties"]["title"]["type"] == "string"
    assert cleaned["required"] == ["title", "body"]


def test_array_without_items_gets_default_items() -> None:
    """An array without ``items`` is invalid for some validators; OCI
    tool API needs a default. Sanitizer fills in ``{"type": "object"}``."""
    schema = {"type": "object", "properties": {"tags": {"type": "array"}}}
    cleaned = sanitize_oci_tool_schema(schema)
    assert cleaned["properties"]["tags"]["items"] == {"type": "object"}


def test_does_not_mutate_input() -> None:
    """Caller may need the original schema for a second non-OCI call;
    sanitizer must return a fresh structure."""
    schema = {
        "type": "object",
        "title": "Foo",
        "properties": {"x": {"type": ["string", "null"]}},
    }
    import copy

    snapshot = copy.deepcopy(schema)
    _ = sanitize_oci_tool_schema(schema)
    assert schema == snapshot, "input schema was mutated"


def test_nested_recursion_through_items_properties() -> None:
    """The deepest path in the observed bug: ``items.properties[].type``
    as a list. Confirms recursion reaches arbitrary depth."""
    schema = {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "title": "Result",  # should be stripped
                    "properties": {
                        "name": {"type": ["string", "null"]},
                        "score": {"type": ["null", "number"]},
                    },
                },
            }
        },
    }
    cleaned = sanitize_oci_tool_schema(schema)
    items = cleaned["properties"]["results"]["items"]
    assert "title" not in items
    assert items["properties"]["name"]["type"] == "string"
    assert items["properties"]["score"]["type"] == "number"


# ---------------------------------------------------------------------------
# Wiring — OCIChatCompletionsModel.complete / .stream apply the sanitizer
# ---------------------------------------------------------------------------


@pytest.fixture
def gemini_oci_model(monkeypatch: pytest.MonkeyPatch):
    """Build an OCIChatCompletionsModel without touching the network.

    The constructor reads ``OCI_PROFILE`` / ``OCI_COMPARTMENT`` from env
    when no explicit auth is passed — we satisfy both so __init__ runs
    cleanly. The model's request methods are wrapped by the test, so
    no real OCI client is needed."""
    monkeypatch.setenv("OCI_PROFILE", "DEFAULT")
    monkeypatch.setenv("OCI_COMPARTMENT", "ocid1.tenancy.oc1..test")
    monkeypatch.setenv(
        "OCI_GENAI_ENDPOINT",
        "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    )
    from locus.models.providers.oci import OCIChatCompletionsModel

    return OCIChatCompletionsModel(model="google.gemini-2.5-flash", profile="DEFAULT")


def test_sanitize_tools_for_oci_applies_to_each_tool(gemini_oci_model) -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search",
                "parameters": _SearchQuery.model_json_schema(),
            },
        },
        {
            "type": "function",
            "function": {
                "name": "noop",
                "description": "Noop",
                "parameters": {
                    "type": "object",
                    "title": "Noop",
                    "properties": {"x": {"type": ["string", "null"]}},
                },
            },
        },
    ]
    cleaned = gemini_oci_model._sanitize_tools_for_oci(tools)
    assert cleaned is not None
    # First tool: Optional[str] / Optional[int] collapsed
    params = cleaned[0]["function"]["parameters"]

    def _walk_no_list_types(node: Any) -> None:
        if isinstance(node, dict):
            if "type" in node:
                assert not isinstance(node["type"], list)
            for v in node.values():
                _walk_no_list_types(v)
        elif isinstance(node, list):
            for item in node:
                _walk_no_list_types(item)

    _walk_no_list_types(params)
    # Second tool: title stripped, type collapsed
    second = cleaned[1]["function"]["parameters"]
    assert "title" not in second
    assert second["properties"]["x"]["type"] == "string"


def test_sanitize_tools_for_oci_returns_none_for_none(gemini_oci_model) -> None:
    """Tool-free calls (e.g., plain chat) pass ``tools=None`` — sanitizer
    must not crash and must not synthesise a list."""
    assert gemini_oci_model._sanitize_tools_for_oci(None) is None
    assert gemini_oci_model._sanitize_tools_for_oci([]) == []


def test_sanitize_tools_for_oci_passes_through_non_function_tools(
    gemini_oci_model,
) -> None:
    """A tool dict without a ``function`` block (e.g., a future custom
    tool shape) is left alone — the sanitizer only knows the OpenAI
    function-calling shape and shouldn't corrupt anything else."""
    tools = [{"type": "retrieval"}, {"type": "function", "function": {"name": "x"}}]
    cleaned = gemini_oci_model._sanitize_tools_for_oci(tools)
    assert cleaned[0] == {"type": "retrieval"}
    # No "parameters" key on the function block → passes through untouched
    assert cleaned[1] == {"type": "function", "function": {"name": "x"}}


def test_sanitize_does_not_mutate_caller_tools(gemini_oci_model) -> None:
    """Caller may want to send the same tools to a non-OCI provider next.
    The sanitizer must return a fresh structure."""
    import copy

    tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "parameters": _SearchQuery.model_json_schema(),
            },
        }
    ]
    snapshot = copy.deepcopy(tools)
    _ = gemini_oci_model._sanitize_tools_for_oci(tools)
    assert tools == snapshot
