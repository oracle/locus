# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for tools schema generation."""

from typing import Optional, Union

from pydantic import BaseModel

from locus.tools.context import ToolContext
from locus.tools.schema import (
    _is_tool_context,
    _parse_docstring_params,
    generate_schema,
    pydantic_to_json_schema,
    python_type_to_json_type,
)


class TestPythonTypeToJsonType:
    """Tests for python_type_to_json_type function."""

    def test_none_type(self):
        """Test NoneType conversion."""
        result = python_type_to_json_type(type(None))
        assert result == {"type": "null"}

    def test_optional_type(self):
        """Test Optional[X] conversion."""
        # Exercising the typing-module spelling deliberately — user
        # tools that still use `Optional[X]` must resolve the same way
        # as `X | None`. Parallel X|None coverage is in test_modern_union_*.
        result = python_type_to_json_type(Optional[str])  # noqa: UP045
        assert result == {"type": "string"}

    def test_union_multiple_types(self):
        """Test Union of multiple non-None types."""
        result = python_type_to_json_type(Union[str, int])  # noqa: UP007
        assert "anyOf" in result

    def test_list_without_args(self):
        """Test list without type argument falls back to string."""
        # Bare `list` has no origin, so falls through to default
        result = python_type_to_json_type(list)
        assert result == {"type": "string"}

    def test_list_with_args(self):
        """Test list with type argument."""
        result = python_type_to_json_type(list[str])
        assert result == {"type": "array", "items": {"type": "string"}}

    def test_dict_without_args(self):
        """Test dict without type arguments falls back to string."""
        # Bare `dict` has no origin, so falls through to default
        result = python_type_to_json_type(dict)
        assert result == {"type": "string"}

    def test_dict_with_args(self):
        """Test dict with type arguments."""
        result = python_type_to_json_type(dict[str, int])
        assert result == {"type": "object", "additionalProperties": {"type": "integer"}}

    def test_tuple_without_args(self):
        """Test tuple without type arguments falls back to string."""
        # Bare `tuple` has no origin, so falls through to default
        result = python_type_to_json_type(tuple)
        assert result == {"type": "string"}

    def test_tuple_with_args(self):
        """Test tuple with type arguments."""
        result = python_type_to_json_type(tuple[str, int])
        assert result["type"] == "array"
        assert "prefixItems" in result
        assert len(result["prefixItems"]) == 2

    def test_pydantic_model(self):
        """Test Pydantic model conversion."""

        class MyModel(BaseModel):
            name: str
            age: int

        result = python_type_to_json_type(MyModel)
        assert "properties" in result

    def test_str_type(self):
        """Test string type conversion."""
        result = python_type_to_json_type(str)
        assert result == {"type": "string"}

    def test_int_type(self):
        """Test integer type conversion."""
        result = python_type_to_json_type(int)
        assert result == {"type": "integer"}

    def test_float_type(self):
        """Test float type conversion."""
        result = python_type_to_json_type(float)
        assert result == {"type": "number"}

    def test_bool_type(self):
        """Test boolean type conversion."""
        result = python_type_to_json_type(bool)
        assert result == {"type": "boolean"}

    def test_bytes_type(self):
        """Test bytes type conversion."""
        result = python_type_to_json_type(bytes)
        assert result == {"type": "string", "contentEncoding": "base64"}

    def test_unknown_type(self):
        """Test unknown type defaults to string."""

        class CustomClass:
            pass

        result = python_type_to_json_type(CustomClass)
        assert result == {"type": "string"}


class TestPydanticToJsonSchema:
    """Tests for pydantic_to_json_schema function."""

    def test_simple_model(self):
        """Test converting simple Pydantic model."""

        class Person(BaseModel):
            name: str
            age: int

        result = pydantic_to_json_schema(Person)
        assert "properties" in result
        assert "name" in result["properties"]
        assert "age" in result["properties"]


class TestGenerateSchema:
    """Tests for generate_schema function."""

    def test_simple_function(self):
        """Test generating schema for simple function."""

        def greet(name: str) -> str:
            """Greet a person."""
            return f"Hello, {name}!"

        result = generate_schema(greet)
        assert result["type"] == "function"
        assert result["function"]["name"] == "greet"
        assert result["function"]["description"] == "Greet a person."
        assert "name" in result["function"]["parameters"]["properties"]

    def test_function_with_default(self):
        """Test generating schema for function with default."""

        def greet(name: str, greeting: str = "Hello") -> str:
            """Greet someone."""
            return f"{greeting}, {name}!"

        result = generate_schema(greet)
        params = result["function"]["parameters"]
        assert "name" in params["required"]
        assert "greeting" not in params["required"]
        assert params["properties"]["greeting"]["default"] == "Hello"

    def test_function_with_custom_description(self):
        """Test generating schema with custom description."""

        def greet(name: str) -> str:
            """Original description."""
            return f"Hello, {name}!"

        result = generate_schema(greet, description="Custom description")
        assert result["function"]["description"] == "Custom description"

    def test_function_without_docstring(self):
        """Test generating schema for function without docstring."""

        def my_func(x: int) -> int:
            return x * 2

        result = generate_schema(my_func)
        assert "my_func" in result["function"]["description"]

    def test_function_skips_self_cls(self):
        """Test that self and cls parameters are skipped."""

        def method(self, name: str) -> str:
            """Method."""
            return name

        result = generate_schema(method)
        assert "self" not in result["function"]["parameters"]["properties"]

    def test_function_skips_context(self):
        """Test that context parameters are skipped."""

        def tool_fn(name: str, ctx: ToolContext) -> str:
            """Tool function."""
            return name

        result = generate_schema(tool_fn)
        assert "ctx" not in result["function"]["parameters"]["properties"]

    def test_function_with_docstring_params(self):
        """Test parsing parameter descriptions from docstring."""

        def greet(name: str, age: int) -> str:
            """Greet a person.

            Args:
                name: The person's name
                age: The person's age
            """
            return f"Hello {name}, you are {age}"

        result = generate_schema(greet)
        props = result["function"]["parameters"]["properties"]
        assert props["name"]["description"] == "The person's name"
        assert props["age"]["description"] == "The person's age"


class TestIsToolContext:
    """Tests for _is_tool_context function."""

    def test_tool_context_type(self):
        """Test with ToolContext type."""
        assert _is_tool_context(ToolContext) is True

    def test_optional_tool_context(self):
        """Test with Optional[ToolContext]."""
        # Typing-module spelling — user tools using `Optional[ToolContext]`
        # must be recognised identically to `ToolContext | None`.
        assert _is_tool_context(Optional[ToolContext]) is True  # noqa: UP045

    def test_non_context_type(self):
        """Test with non-context type."""
        assert _is_tool_context(str) is False

    def test_non_type_hint(self):
        """Test with non-type hint."""
        assert _is_tool_context("str") is False


class TestParseDocstringParams:
    """Tests for _parse_docstring_params function."""

    def test_no_docstring(self):
        """Test function without docstring."""

        def no_doc():
            pass

        result = _parse_docstring_params(no_doc)
        assert result == {}

    def test_docstring_with_args_section(self):
        """Test docstring with Args section."""

        def func():
            """Description.

            Args:
                name: The name
                value: The value
            """

        result = _parse_docstring_params(func)
        assert result["name"] == "The name"
        assert result["value"] == "The value"

    def test_docstring_with_typed_params(self):
        """Test docstring with type annotations in params."""

        def func():
            """Description.

            Args:
                name (str): The name parameter
                count (int): The count parameter
            """

        result = _parse_docstring_params(func)
        assert result["name"] == "The name parameter"
        assert result["count"] == "The count parameter"

    def test_docstring_ends_args_section(self):
        """Test docstring that ends Args section."""

        def func():
            """Description.

            Args:
                name: The name

            Returns:
                Something
            """

        result = _parse_docstring_params(func)
        assert "name" in result
        # Should not parse Returns as a param
        assert "Returns" not in result
