# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Structured output support - 100% Pydantic."""

from __future__ import annotations

import json
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError


T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(Exception):
    """Error parsing structured output."""

    def __init__(self, message: str, raw_content: str, errors: list[Any] | None = None):
        super().__init__(message)
        self.raw_content = raw_content
        self.errors = errors or []


class StructuredOutput(BaseModel):
    """Wrapper for structured output with validation."""

    raw: str
    parsed: BaseModel | None = None
    error: str | None = None
    validation_errors: list[dict[str, Any]] = []

    model_config = {"arbitrary_types_allowed": True}

    @property
    def success(self) -> bool:
        """Whether parsing succeeded."""
        return self.parsed is not None and self.error is None

    def unwrap(self) -> BaseModel:
        """Get parsed value or raise error."""
        if self.parsed is None:
            raise StructuredOutputError(
                self.error or "No parsed output",
                self.raw,
                self.validation_errors or None,
            )
        return self.parsed


def extract_json(content: str) -> str:
    """Extract JSON from content (handles markdown code blocks)."""
    content = content.strip()

    # Try to find JSON in code blocks
    if "```json" in content:
        start = content.find("```json") + 7
        end = content.find("```", start)
        if end > start:
            return content[start:end].strip()

    if "```" in content:
        start = content.find("```") + 3
        end = content.find("```", start)
        if end > start:
            extracted = content[start:end].strip()
            # Skip language identifier if present
            if extracted and not extracted.startswith("{") and not extracted.startswith("["):
                lines = extracted.split("\n", 1)
                if len(lines) > 1:
                    extracted = lines[1].strip()
            return extracted

    # Try to find raw JSON object or array
    obj_start = content.find("{")
    arr_start = content.find("[")
    if obj_start == -1 and arr_start == -1:
        return content

    if obj_start == -1:
        start, opener, closer = arr_start, "[", "]"
    elif arr_start == -1 or obj_start < arr_start:
        start, opener, closer = obj_start, "{", "}"
    else:
        start, opener, closer = arr_start, "[", "]"

    depth = 0
    for i, char in enumerate(content[start:], start):
        if char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return content[start : i + 1]

    return content


def parse_structured(
    content: str,
    schema: type[T],
    strict: bool = True,
) -> StructuredOutput:
    """
    Parse content into a structured Pydantic model.

    Args:
        content: Raw content from model
        schema: Pydantic model class to parse into
        strict: Whether to raise on parse failure

    Returns:
        StructuredOutput with parsed model or error
    """
    try:
        # Extract JSON from content
        json_str = extract_json(content)

        # Parse JSON
        data = json.loads(json_str)

        # Validate with Pydantic
        parsed = schema.model_validate(data)

        return StructuredOutput(raw=content, parsed=parsed)

    except json.JSONDecodeError as e:
        error = f"JSON parse error: {e}"
        if strict:
            raise StructuredOutputError(error, content) from e
        return StructuredOutput(raw=content, error=error)

    except ValidationError as e:
        error = f"Validation error: {e}"
        # Pydantic returns ``list[ErrorDetails]`` (TypedDict). Coerce to plain
        # dicts for downstream feedback rendering and serialization.
        errors: list[dict[str, Any]] = [dict(err) for err in e.errors()]
        if strict:
            raise StructuredOutputError(error, content, errors) from e
        return StructuredOutput(raw=content, error=error, validation_errors=errors)


def create_schema_prompt(schema: type[BaseModel]) -> str:
    """Create a prompt fragment describing the expected schema."""
    json_schema = schema.model_json_schema()

    # Clean up schema for prompt
    if "title" in json_schema:
        del json_schema["title"]

    return f"""Respond with a JSON object matching this schema:

```json
{json.dumps(json_schema, indent=2)}
```

Return ONLY the JSON object, no additional text."""


def create_output_instructions(schema: type[BaseModel]) -> str:
    """Create detailed instructions for structured output."""
    json_schema = schema.model_json_schema()
    properties = json_schema.get("properties", {})
    required = json_schema.get("required", [])

    lines = ["Your response must be a valid JSON object with these fields:", ""]

    for name, prop in properties.items():
        prop_type = prop.get("type", "any")
        description = prop.get("description", "")
        is_required = name in required
        req_marker = "(required)" if is_required else "(optional)"

        lines.append(f"- `{name}` ({prop_type}) {req_marker}: {description}")

    lines.extend(
        [
            "",
            "Return ONLY the JSON object. Do not include markdown code blocks or explanations.",
        ]
    )

    return "\n".join(lines)


def _strip_keywords_alongside_ref(node: Any) -> None:
    """Strict ``json_schema`` mode rejects ``$ref`` nodes that carry sibling
    keywords (``description``, ``default``, ``title``, …).

    Pydantic emits these when a field uses ``Field(description=…)`` on a
    referenced sub-schema (an enum or nested model). The OpenAI API
    rejects the schema with::

        $ref cannot have keywords {'description'}.

    We strip every sibling key from any node carrying ``$ref`` so the
    schema satisfies the OpenAI strict-mode contract.
    """
    if isinstance(node, dict):
        if "$ref" in node:
            ref_value = node["$ref"]
            for key in list(node.keys()):
                if key != "$ref":
                    del node[key]
            node["$ref"] = ref_value
        for v in node.values():
            _strip_keywords_alongside_ref(v)
    elif isinstance(node, list):
        for item in node:
            _strip_keywords_alongside_ref(item)


def _enforce_all_properties_required(node: Any) -> None:
    """Walk a JSON Schema and set ``required`` to every property on object
    nodes that have ``properties``.

    OpenAI's strict ``json_schema`` response format additionally requires
    ``required`` to list *every* key in ``properties`` — fields that are
    optional in Pydantic (default values, ``Optional[...]``) still need
    to appear, but their schema can include ``null`` in their type union
    so the model can emit ``null``. Pydantic doesn't add defaulted fields
    to ``required`` by default; we add them here so strict mode accepts
    schemas that have any optional field.
    """
    if isinstance(node, dict):
        if node.get("type") == "object" and "properties" in node:
            node["required"] = list(node["properties"].keys())
        for v in node.values():
            _enforce_all_properties_required(v)
    elif isinstance(node, list):
        for item in node:
            _enforce_all_properties_required(item)


def _enforce_additional_properties_false(node: Any) -> None:
    """Walk a JSON Schema and set ``additionalProperties: false`` on every
    object node.

    OpenAI's strict ``json_schema`` response format requires every object
    in the schema (including nested ones inside ``$defs``) to declare
    ``additionalProperties: false`` — otherwise the API rejects the
    request with ``Invalid schema for response_format``. Pydantic's
    ``model_json_schema()`` doesn't emit this by default.
    """
    if isinstance(node, dict):
        if node.get("type") == "object" and "additionalProperties" not in node:
            node["additionalProperties"] = False
        for v in node.values():
            _enforce_additional_properties_false(v)
    elif isinstance(node, list):
        for item in node:
            _enforce_additional_properties_false(item)


def inline_schema_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve all ``$ref`` pointers in a JSON schema by inlining the definitions.

    Gemini (and some other providers) reject schemas that contain ``$ref``
    — they require every type to be spelled out inline rather than referenced
    from ``$defs``.  This function performs a deep substitution: every
    ``{"$ref": "#/$defs/Foo"}`` is replaced with the corresponding definition
    from ``schema["$defs"]``, recursively, until no ``$ref`` remains.

    Args:
        schema: A JSON schema dict that may contain ``$defs`` and ``$ref``.

    Returns:
        A new schema dict with all ``$ref`` inlined and ``$defs`` removed.
    """
    import copy

    defs = schema.get("$defs", {})
    if not defs:
        return schema

    def _resolve(node: Any, seen: frozenset[str] = frozenset()) -> Any:  # noqa: B008
        if isinstance(node, dict):
            if "$ref" in node:
                ref = node["$ref"]
                name = ref.split("/")[-1]
                if name in seen:
                    return node
                definition = copy.deepcopy(defs.get(name, node))
                return _resolve(definition, seen | {name})
            return {k: _resolve(v, seen) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [_resolve(item, seen) for item in node]
        return node

    resolved: dict[str, Any] = _resolve(copy.deepcopy(schema))
    resolved.pop("$defs", None)
    return resolved


def build_response_format(schema: type[BaseModel], *, strict: bool = True) -> dict[str, Any]:
    """Build an OpenAI ``response_format`` for a Pydantic schema.

    The OpenAI ``json_schema`` response format requires the schema's root to be
    an object with no extra keys. Pydantic's ``model_json_schema()`` already
    emits an object schema, but we need to ensure ``additionalProperties`` is
    set to ``false`` (recursively, for nested ``$defs`` too) so the schema is
    self-contained.

    Args:
        schema: Pydantic model class describing the expected output.
        strict: When True, request strict mode (provider-enforced constrained
            decoding). Some OpenAI-compatible providers ignore this; falsy
            providers fall back to best-effort JSON mode.

    Returns:
        A ``response_format`` dict suitable for ``chat.completions.create``.
    """
    json_schema = schema.model_json_schema()
    json_schema.pop("title", None)
    if strict:
        _enforce_additional_properties_false(json_schema)
        _enforce_all_properties_required(json_schema)
        _strip_keywords_alongside_ref(json_schema)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema.__name__,
            "schema": json_schema,
            "strict": strict,
        },
    }


_PARTIAL_PAIRS = {"}": "{", "]": "["}


def _scan_brackets(text: str) -> list[str]:
    """Compute the bracket stack of a JSON prefix that contains no strings."""
    stack: list[str] = []
    for ch in text:
        if ch in "{[":
            stack.append(ch)
        elif ch in "}]" and stack and stack[-1] == _PARTIAL_PAIRS[ch]:
            stack.pop()
    return stack


def _close_partial_json(content: str) -> str:
    """Close any unbalanced braces / brackets / strings in partial JSON.

    Used by :func:`parse_partial` to make a best-effort completion of a
    truncated JSON stream so it can be validated against a Pydantic schema.
    The tactic is intentionally simple: track string state (with backslash
    escape) and a bracket stack, and append the necessary closers in
    reverse order. We do not attempt to repair missing commas or trailing
    keys; partial values that cannot be coerced will simply fail validation
    and the caller will retry on the next chunk.
    """
    extracted = extract_json(content)
    stack: list[str] = []
    in_string = False
    escape = False
    for ch in extracted:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "{[":
            stack.append(ch)
        elif ch in "}]" and stack and stack[-1] == _PARTIAL_PAIRS[ch]:
            stack.pop()

    # If we ended mid-key/value with an unterminated string, drop the dangling
    # ``"key": "partial`` so the JSON is at least syntactically valid. The
    # alternative — closing the string — would surface a half-token to the
    # validator, which is rarely useful.
    if in_string:
        last_safe = max(extracted.rfind(","), extracted.rfind("{"), extracted.rfind("["))
        if last_safe >= 0:
            extracted = extracted[: last_safe + 1].rstrip(", ")
            stack = _scan_brackets(extracted)

    suffix = "".join("}" if opener == "{" else "]" for opener in reversed(stack))
    return extracted + suffix


def parse_partial(content: str, schema: type[T]) -> T | None:
    """Best-effort parse of a *partial* JSON string into a Pydantic schema.

    Returns the parsed model if the (auto-completed) JSON is valid, or
    ``None`` if the buffer cannot yet be coerced. Designed for streaming:
    call this on accumulated ``ModelChunkEvent.content`` and surface
    incremental snapshots to the user.

    Validation is strict — only schemas where every required field has been
    emitted return non-``None``. Use ``model_construct`` directly if you
    need to surface partial-but-invalid drafts.
    """
    completed = _close_partial_json(content)
    if not completed.strip():
        return None
    try:
        data = json.loads(completed)
    except json.JSONDecodeError:
        return None
    try:
        return schema.model_validate(data)
    except ValidationError:
        return None


def format_validation_errors(errors: list[dict[str, Any]]) -> str:
    """Render Pydantic validation errors into a compact, model-readable bullet list.

    Used when re-prompting the model after a parse failure — the model gets a
    direct, structured account of which fields failed and why so it can repair
    the next attempt.
    """
    if not errors:
        return "(no error details)"
    lines = []
    for err in errors:
        loc = ".".join(str(x) for x in err.get("loc", ())) or "<root>"
        msg = err.get("msg", "invalid")
        type_ = err.get("type", "")
        suffix = f" [{type_}]" if type_ else ""
        lines.append(f"- {loc}: {msg}{suffix}")
    return "\n".join(lines)
