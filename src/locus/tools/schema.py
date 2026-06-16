# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""JSON Schema generation from Python types - 100% Pydantic."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Union, get_args, get_origin, get_type_hints

from pydantic import BaseModel


if TYPE_CHECKING:
    from collections.abc import Callable


def python_type_to_json_type(py_type: type) -> dict[str, Any]:  # noqa: PLR0911
    """Convert a Python type to JSON Schema type."""
    origin = get_origin(py_type)
    args = get_args(py_type)

    # Handle None / NoneType
    if py_type is type(None):
        return {"type": "null"}

    # Handle Optional[X] -> Union[X, None]
    if origin is Union:
        non_none_args = [a for a in args if a is not type(None)]
        if len(non_none_args) == 1:
            return python_type_to_json_type(non_none_args[0])
        # Union of multiple types
        return {"anyOf": [python_type_to_json_type(a) for a in args]}

    # Handle list[X]
    if origin is list:
        if args:
            return {"type": "array", "items": python_type_to_json_type(args[0])}
        return {"type": "array"}

    # Handle dict[K, V]
    if origin is dict:
        if len(args) == 2:
            return {
                "type": "object",
                "additionalProperties": python_type_to_json_type(args[1]),
            }
        return {"type": "object"}

    # Handle tuple
    if origin is tuple:
        if args:
            return {
                "type": "array",
                "prefixItems": [python_type_to_json_type(a) for a in args],
                "items": False,
            }
        return {"type": "array"}

    # Handle Pydantic models
    if isinstance(py_type, type) and issubclass(py_type, BaseModel):
        return pydantic_to_json_schema(py_type)

    # Basic types
    type_map = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        bytes: {"type": "string", "contentEncoding": "base64"},
    }

    if py_type in type_map:
        return type_map[py_type]

    # Default to string
    return {"type": "string"}


def pydantic_to_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Convert a Pydantic model to JSON Schema."""
    return model.model_json_schema()


def generate_schema(fn: Callable[..., Any], description: str | None = None) -> dict[str, Any]:
    """
    Generate OpenAI-compatible tool schema from a function.

    Args:
        fn: The function to generate schema for
        description: Override description (uses docstring if not provided)

    Returns:
        Tool schema in OpenAI function format
    """
    sig = inspect.signature(fn)
    hints = get_type_hints(fn)

    # Get description from docstring if not provided
    if description is None:
        description = inspect.getdoc(fn) or f"Call the {fn.__name__} function"

    # Parse docstring for parameter descriptions
    param_descriptions = _parse_docstring_params(fn)

    # Build parameters schema
    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        # Skip self, cls, and context parameters
        if name in ("self", "cls", "ctx", "context"):
            continue

        # Get type hint
        hint = hints.get(name, str)

        # Skip ToolContext type
        if _is_tool_context(hint):
            continue

        # Convert to JSON schema
        prop = python_type_to_json_type(hint)

        # Add description from docstring
        if name in param_descriptions:
            prop["description"] = param_descriptions[name]

        # Handle default values
        if param.default is not inspect.Parameter.empty:
            prop["default"] = param.default
        else:
            required.append(name)

        properties[name] = prop

    return {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def _is_tool_context(hint: type) -> bool:
    """Check if a type hint is ToolContext."""
    from locus.tools.context import ToolContext  # noqa: PLC0415

    origin = get_origin(hint)
    if origin is Union:
        args = get_args(hint)
        return any(_is_tool_context(a) for a in args)

    if isinstance(hint, type):
        return issubclass(hint, ToolContext)

    return False


def _parse_docstring_params(fn: Callable[..., Any]) -> dict[str, str]:
    """Parse parameter descriptions from docstring."""
    doc = inspect.getdoc(fn)
    if not doc:
        return {}

    params: dict[str, str] = {}
    in_args = False

    for line in doc.split("\n"):
        stripped = line.strip()

        # Detect Args section
        if stripped.lower() in ("args:", "arguments:", "parameters:"):
            in_args = True
            continue

        # Detect end of Args section
        if (
            in_args
            and stripped.endswith(":")
            and not stripped.startswith(" ")
            and stripped.lower() not in ("args:", "arguments:", "parameters:")
        ):
            in_args = False
            continue

        # Parse parameter lines
        if in_args and ":" in stripped:
            parts = stripped.split(":", 1)
            if len(parts) == 2:
                param_name = parts[0].strip()
                # Remove type annotation in parentheses
                if "(" in param_name:
                    param_name = param_name.split("(")[0].strip()
                description = parts[1].strip()
                if param_name and description:
                    params[param_name] = description

    return params
