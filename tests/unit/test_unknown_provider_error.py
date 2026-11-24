# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""``get_model`` and ``get_checkpointer`` raise a typed ``UnknownProviderError``.

The error sits under ``ConfigError`` with a stable ``kind``, carries the
provider that was asked for and the ones that were registered, and is still
a ``ValueError``, which is what both registries raised before.
"""

from __future__ import annotations

import copy
import pickle

import pytest

import locus.core
from locus.core import errors
from locus.core.errors import ConfigError, LocusError, UnknownProviderError
from locus.memory.registry import get_checkpointer, list_checkpointers
from locus.models.registry import get_model, list_providers


def test_it_is_a_config_error_and_a_value_error_with_its_own_kind() -> None:
    assert issubclass(UnknownProviderError, ConfigError)
    assert issubclass(UnknownProviderError, LocusError)
    assert issubclass(UnknownProviderError, ValueError)
    assert UnknownProviderError.kind == "unknown_provider"
    assert "UnknownProviderError" in errors.__all__
    assert locus.core.UnknownProviderError is UnknownProviderError
    assert "UnknownProviderError" in locus.core.__all__


def test_the_attributes_default_to_no_provider_and_nothing_available() -> None:
    error = UnknownProviderError("no such provider")
    assert str(error) == "no such provider"
    assert error.provider is None
    assert error.available == []
    cause = KeyError("x")
    chained = UnknownProviderError("m", provider="p", available=("a", "b"), cause=cause)
    assert chained.__cause__ is cause
    assert chained.available == ["a", "b"]


def test_get_model_raises_it_for_an_unregistered_prefix() -> None:
    with pytest.raises(UnknownProviderError) as caught:
        get_model("nope:some-model")
    error = caught.value
    assert error.kind == "unknown_provider"
    assert error.provider == "nope"
    assert error.available == list_providers()
    assert "openai" in error.available
    assert str(error) == f"Unknown provider: nope. Available: {list_providers()}"


def test_get_model_raises_it_for_a_string_without_a_prefix() -> None:
    with pytest.raises(UnknownProviderError) as caught:
        get_model("gpt-4o")
    error = caught.value
    assert error.provider is None
    assert error.available == list_providers()
    assert str(error) == (
        f"Model string must be 'provider:model', got: gpt-4o. "
        f"Available providers: {list_providers()}"
    )


@pytest.mark.parametrize("model_string", [":gpt-4o", ":"])
def test_an_empty_prefix_names_no_provider_in_get_model(model_string: str) -> None:
    with pytest.raises(UnknownProviderError) as caught:
        get_model(model_string)
    assert caught.value.provider is None
    assert str(caught.value).startswith("Unknown provider: .")


def test_an_unknown_prefix_with_an_empty_or_odd_model_part_still_names_the_provider() -> None:
    for model_string in ("nope:", "nope::x", "nope:a:b"):
        with pytest.raises(UnknownProviderError) as caught:
            get_model(model_string)
        assert caught.value.provider == "nope", model_string


def test_get_checkpointer_raises_it_with_the_registered_names() -> None:
    with pytest.raises(UnknownProviderError) as caught:
        get_checkpointer("nope:hint")
    error = caught.value
    assert error.kind == "unknown_provider"
    assert error.provider == "nope"
    assert error.available == list_checkpointers()
    assert "memory" in error.available
    assert str(error) == (
        f"Unknown checkpointer provider: 'nope'. "
        f"Available providers: {list_checkpointers()}. "
        f"Install optional dependencies or register a custom provider."
    )


@pytest.mark.parametrize("spec", ["", ":x"])
def test_an_empty_checkpointer_name_names_no_provider(spec: str) -> None:
    with pytest.raises(UnknownProviderError) as caught:
        get_checkpointer(spec)
    assert caught.value.provider is None


def test_available_is_a_copy_the_caller_can_change() -> None:
    with pytest.raises(UnknownProviderError) as caught:
        get_model("nope:x")
    caught.value.available.clear()
    assert list_providers(), "the registry is untouched"


def test_existing_handlers_keep_working() -> None:
    for call, arg in ((get_model, "nope:x"), (get_checkpointer, "nope")):
        with pytest.raises(ValueError, match="Unknown"):
            call(arg)
        with pytest.raises(ConfigError):
            call(arg)
        with pytest.raises(LocusError):
            call(arg)


def test_it_survives_pickle_and_copy_with_its_attributes() -> None:
    with pytest.raises(UnknownProviderError) as caught:
        get_model("nope:x")
    original = caught.value
    for clone in (
        pickle.loads(pickle.dumps(original)),  # noqa: S301 - our own object
        copy.copy(original),
        copy.deepcopy(original),
    ):
        assert type(clone) is UnknownProviderError
        assert str(clone) == str(original)
        assert clone.provider == "nope"
        assert clone.available == original.available
        assert clone.kind == "unknown_provider"
