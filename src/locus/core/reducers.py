# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""State reducers for composable state updates.

Reducers define how state fields are merged when updates occur.
Use with typing.Annotated to declare reducer behavior on state fields.

Example:
    from typing import Annotated
    from locus.core.reducers import add_messages, merge_dict

    class MyState(BaseModel):
        messages: Annotated[list[Message], add_messages]
        context: Annotated[dict, merge_dict]
        count: int  # Default: last-write-wins
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Hashable
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, get_args, get_origin

from pydantic import BaseModel


if TYPE_CHECKING:
    from locus.core.messages import Message


T = TypeVar("T")
K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


# =============================================================================
# Reducer Protocol
# =============================================================================


class Reducer(Protocol[T]):
    """
    Protocol for state reducers.

    A reducer takes the current value and an update value,
    returning the new merged value.
    """

    def __call__(self, current: T, update: T) -> T:
        """Merge current value with update."""
        ...


# =============================================================================
# Built-in Reducers
# =============================================================================


class AddMessages:
    """
    Reducer that appends messages with ID-based deduplication.

    If a message has an ID that already exists, it replaces the existing message.
    Messages without IDs are always appended.

    Special handling:
    - REMOVE_ALL marker clears the list
    - Messages with matching IDs are replaced in-place
    """

    REMOVE_ALL = "__REMOVE_ALL_MESSAGES__"

    def __call__(
        self,
        current: list[Message],
        update: list[Message] | Message | str,
    ) -> list[Message]:
        """Merge message lists with deduplication.

        ``update`` accepts:

        - a ``list[Message]`` — appended (with ID dedup);
        - a single ``Message`` — wrapped in a one-element list;
        - the ``REMOVE_ALL`` sentinel string — clears ``current``.

        Any other string raises ``ValueError``.
        """
        # Handle the string-only sentinel up front so mypy can narrow
        # ``update`` to a Message-shaped value for the rest of the function.
        if isinstance(update, str):
            if update == self.REMOVE_ALL:
                return []
            err = f"AddMessages received unsupported string sentinel: {update!r}"
            raise ValueError(err)

        # Single Message → one-element list; preserves the historical
        # ergonomic of `add_messages(current, Message.user("hi"))`.
        if not isinstance(update, list):
            update = [update]

        if not current:
            return list(update)

        if not update:
            return list(current)

        # Build ID index for existing messages
        result = list(current)
        existing_ids: dict[str, int] = {}
        for i, msg in enumerate(result):
            if hasattr(msg, "id") and msg.id:
                existing_ids[msg.id] = i

        # Process updates
        for msg in update:
            msg_id = getattr(msg, "id", None) if hasattr(msg, "id") else None
            if msg_id and msg_id in existing_ids:
                # Replace existing message
                result[existing_ids[msg_id]] = msg
            else:
                # Append new message
                result.append(msg)
                if msg_id:
                    existing_ids[msg_id] = len(result) - 1

        return result


class MergeDict:
    """
    Reducer that merges dictionaries.

    Uses dict.update() semantics - later values override earlier ones.
    Nested dicts are NOT deep-merged (use DeepMergeDict for that).
    """

    def __call__(
        self,
        current: dict[K, V],
        update: dict[K, V],
    ) -> dict[K, V]:
        """Merge dictionaries."""
        if not current:
            return dict(update) if update else {}
        if not update:
            return dict(current)

        result = dict(current)
        result.update(update)
        return result


class DeepMergeDict:
    """
    Reducer that deep-merges nested dictionaries.

    Recursively merges nested dicts. Non-dict values are overwritten.
    """

    def __call__(
        self,
        current: dict[str, Any],
        update: dict[str, Any],
    ) -> dict[str, Any]:
        """Deep merge dictionaries."""
        if not current:
            return dict(update) if update else {}
        if not update:
            return dict(current)

        result = dict(current)
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self(result[key], value)
            else:
                result[key] = value
        return result


class AppendList:
    """
    Reducer that appends lists without deduplication.

    Simple concatenation of lists.
    """

    def __call__(self, current: list[T], update: list[T]) -> list[T]:
        """Append lists."""
        if not current:
            return list(update) if update else []
        if not update:
            return list(current)
        return [*current, *update]


class UniqueAppendList:
    """
    Reducer that appends lists with deduplication.

    Only adds items that aren't already in the list.
    Preserves order of first occurrence.
    """

    def __call__(self, current: list[T], update: list[T]) -> list[T]:
        """Append unique items only."""
        if not current:
            return list(update) if update else []
        if not update:
            return list(current)

        seen = set(current)
        result = list(current)
        for item in update:
            if item not in seen:
                result.append(item)
                seen.add(item)
        return result


class AddNumbers:
    """Reducer that adds numeric values."""

    def __call__(self, current: float, update: float) -> float | int:
        """Add numbers."""
        return current + update


class MaxValue:
    """Reducer that keeps the maximum value."""

    def __call__(self, current: T, update: T) -> T:
        """Return maximum."""
        # T is unbounded so mypy can't prove SupportsRichComparison; the
        # Reducer Protocol contract is "current and update are the same
        # type", so if one is comparable, the other is too.
        return max(current, update)  # type: ignore[call-overload,no-any-return]


class MinValue:
    """Reducer that keeps the minimum value."""

    def __call__(self, current: T, update: T) -> T:
        """Return minimum."""
        # See MaxValue.__call__ — same TypeVar-bound limitation.
        return min(current, update)  # type: ignore[call-overload,no-any-return]


class LastValue:
    """
    Reducer that keeps the last (most recent) value.

    This is the default behavior when no reducer is specified.
    """

    def __call__(self, current: T, update: T) -> T:
        """Return update (last value wins)."""
        return update


class FirstValue:
    """Reducer that keeps the first (original) value."""

    def __call__(self, current: T, update: T) -> T:
        """Return current (first value wins)."""
        return current


class SetUnion:
    """Reducer that unions sets."""

    def __call__(self, current: set[T], update: set[T]) -> set[T]:
        """Union sets."""
        if not current:
            return set(update) if update else set()
        if not update:
            return set(current)
        return current | update


class SetIntersection:
    """Reducer that intersects sets."""

    def __call__(self, current: set[T], update: set[T]) -> set[T]:
        """Intersect sets."""
        if not current or not update:
            return set()
        return current & update


# =============================================================================
# Reducer Instances (for use with Annotated)
# =============================================================================

# Message handling
add_messages = AddMessages()

# Dictionary merging
merge_dict = MergeDict()
deep_merge_dict = DeepMergeDict()

# List operations
append_list = AppendList()
unique_append_list = UniqueAppendList()

# Numeric operations
add_numbers = AddNumbers()
max_value = MaxValue()
min_value = MinValue()

# Value selection
last_value = LastValue()
first_value = FirstValue()

# Set operations
set_union = SetUnion()
set_intersection = SetIntersection()

# Operator aliases for common cases
add = operator.add  # For numbers
or_ = operator.or_  # For dicts (same as merge_dict for dicts)


# =============================================================================
# Reducer Extraction
# =============================================================================


def get_reducer(annotation: Any) -> Reducer[Any] | None:
    """
    Extract reducer from an Annotated type hint.

    Args:
        annotation: Type annotation, possibly Annotated[T, reducer]

    Returns:
        Reducer if found in annotation, None otherwise

    Example:
        >>> from typing import Annotated
        >>> hint = Annotated[list, add_messages]
        >>> reducer = get_reducer(hint)
        >>> reducer([msg1], [msg2])  # Returns merged list
    """
    # Check if it's an Annotated type
    if get_origin(annotation) is not None:
        # For Python 3.9+ Annotated types
        try:
            from typing import Annotated
            from typing import get_origin as get_origin_typing

            if get_origin_typing(annotation) is Annotated:
                args = get_args(annotation)
                if len(args) >= 2:
                    # Second arg should be the reducer. ``get_args`` returns
                    # ``tuple[Any, ...]`` so the callable narrowing isn't
                    # visible to the type checker.
                    potential_reducer = args[1]
                    if callable(potential_reducer):
                        return potential_reducer  # type: ignore[no-any-return]
        except ImportError:
            pass

    return None


def extract_reducers_from_model(
    model_class: type[BaseModel],
) -> dict[str, Reducer[Any]]:
    """
    Extract all reducers from a Pydantic model's field annotations.

    Args:
        model_class: Pydantic model class

    Returns:
        Dict mapping field names to their reducers

    Example:
        >>> class State(BaseModel):
        ...     messages: Annotated[list, add_messages]
        ...     data: dict
        >>> reducers = extract_reducers_from_model(State)
        >>> reducers  # {'messages': <AddMessages>}
    """
    reducers: dict[str, Reducer[Any]] = {}

    for field_name, field_info in model_class.model_fields.items():
        # In Pydantic v2, Annotated metadata is stored in field_info.metadata
        if field_info.metadata:
            for meta in field_info.metadata:
                if callable(meta):
                    reducers[field_name] = meta
                    break

    return reducers


def apply_reducers(
    current: dict[str, Any],
    update: dict[str, Any],
    reducers: dict[str, Reducer[Any]],
) -> dict[str, Any]:
    """
    Apply reducers to merge current state with update.

    Args:
        current: Current state dict
        update: Update to apply
        reducers: Map of field names to reducers

    Returns:
        New merged state dict

    Example:
        >>> reducers = {"messages": add_messages}
        >>> current = {"messages": [msg1], "count": 1}
        >>> update = {"messages": [msg2], "count": 5}
        >>> result = apply_reducers(current, update, reducers)
        >>> # result["messages"] is [msg1, msg2] (reduced)
        >>> # result["count"] is 5 (last-write-wins)
    """
    result = dict(current)

    for key, value in update.items():
        if key in reducers and key in current:
            # Apply reducer
            result[key] = reducers[key](current[key], value)
        else:
            # Last-write-wins (default)
            result[key] = value

    return result


# =============================================================================
# Convenience Functions
# =============================================================================


def create_reducer(fn: Callable[[T, T], T]) -> Reducer[T]:
    """
    Create a reducer from a simple function.

    Args:
        fn: Binary function (current, update) -> merged

    Returns:
        Reducer wrapping the function

    Example:
        >>> concat = create_reducer(lambda a, b: a + b)
        >>> Annotated[str, concat]
    """

    class FunctionReducer:
        def __call__(self, current: T, update: T) -> T:
            return fn(current, update)

    return FunctionReducer()


def reducer(fn: Callable[[T, T], T]) -> Reducer[T]:
    """Decorator to create a reducer from a function."""
    return create_reducer(fn)
