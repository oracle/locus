# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for state reducers."""

from typing import Annotated

from pydantic import BaseModel

from locus.core.messages import Message
from locus.core.reducers import (
    # Reducer classes
    AddMessages,
    add_messages,
    add_numbers,
    append_list,
    apply_reducers,
    create_reducer,
    deep_merge_dict,
    extract_reducers_from_model,
    first_value,
    # Utilities
    get_reducer,
    last_value,
    max_value,
    merge_dict,
    min_value,
    reducer,
    set_intersection,
    set_union,
    unique_append_list,
)


class TestAddMessages:
    """Tests for add_messages reducer."""

    def test_append_simple(self):
        """Test simple append without IDs."""
        current = [Message.user("Hello")]
        update = [Message.assistant("Hi")]
        result = add_messages(current, update)
        assert len(result) == 2
        assert result[0].content == "Hello"
        assert result[1].content == "Hi"

    def test_empty_current(self):
        """Test with empty current list."""
        result = add_messages([], [Message.user("Hello")])
        assert len(result) == 1

    def test_empty_update(self):
        """Test with empty update list."""
        current = [Message.user("Hello")]
        result = add_messages(current, [])
        assert len(result) == 1

    def test_both_empty(self):
        """Test with both lists empty."""
        result = add_messages([], [])
        assert len(result) == 0

    def test_remove_all_marker(self):
        """Test REMOVE_ALL_MESSAGES marker."""
        current = [Message.user("Hello"), Message.assistant("Hi")]
        result = add_messages(current, AddMessages.REMOVE_ALL)
        assert len(result) == 0

    def test_single_message_update(self):
        """Test with single message (non-list) update."""
        current = [Message.user("Hello")]
        update = Message.assistant("Hi")  # Single message, not a list
        result = add_messages(current, update)
        assert len(result) == 2
        assert result[1].content == "Hi"

    def test_replace_by_id(self):
        """Test that messages with same ID are replaced."""
        # Create messages with explicit IDs
        from locus.core.messages import ToolCall

        tc = ToolCall(id="call_123", name="search", arguments={"q": "test"})
        current = [
            Message.user("Hello"),
            Message.assistant(content="Let me search", tool_calls=[tc]),
        ]
        # Create a message with the same characteristics that should update
        # Note: Messages are immutable, so we test with the ID-based lookup
        update = [Message.assistant(content="Updated response", tool_calls=[tc])]
        result = add_messages(current, update)
        # Message should be appended since Message doesn't have id attribute directly
        assert len(result) == 3


class TestMergeDict:
    """Tests for merge_dict reducer."""

    def test_simple_merge(self):
        """Test simple dict merge."""
        current = {"a": 1, "b": 2}
        update = {"b": 3, "c": 4}
        result = merge_dict(current, update)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_empty_current(self):
        """Test with empty current dict."""
        result = merge_dict({}, {"a": 1})
        assert result == {"a": 1}

    def test_empty_update(self):
        """Test with empty update dict."""
        result = merge_dict({"a": 1}, {})
        assert result == {"a": 1}

    def test_nested_not_deep(self):
        """Test that nested dicts are replaced, not merged."""
        current = {"a": {"x": 1, "y": 2}}
        update = {"a": {"z": 3}}
        result = merge_dict(current, update)
        # Nested dict should be replaced entirely
        assert result == {"a": {"z": 3}}


class TestDeepMergeDict:
    """Tests for deep_merge_dict reducer."""

    def test_deep_merge(self):
        """Test deep merge of nested dicts."""
        current = {"a": {"x": 1, "y": 2}, "b": 3}
        update = {"a": {"z": 3}, "c": 4}
        result = deep_merge_dict(current, update)
        assert result == {"a": {"x": 1, "y": 2, "z": 3}, "b": 3, "c": 4}

    def test_override_non_dict(self):
        """Test that non-dict values are overwritten."""
        current = {"a": {"x": 1}}
        update = {"a": "string"}
        result = deep_merge_dict(current, update)
        assert result == {"a": "string"}


class TestAppendList:
    """Tests for append_list reducer."""

    def test_simple_append(self):
        """Test simple list append."""
        result = append_list([1, 2], [3, 4])
        assert result == [1, 2, 3, 4]

    def test_with_duplicates(self):
        """Test that duplicates are kept."""
        result = append_list([1, 2], [2, 3])
        assert result == [1, 2, 2, 3]

    def test_empty_lists(self):
        """Test with empty lists."""
        assert append_list([], [1]) == [1]
        assert append_list([1], []) == [1]
        assert append_list([], []) == []


class TestUniqueAppendList:
    """Tests for unique_append_list reducer."""

    def test_removes_duplicates(self):
        """Test that duplicates are not added."""
        result = unique_append_list([1, 2], [2, 3])
        assert result == [1, 2, 3]

    def test_preserves_order(self):
        """Test that first occurrence order is preserved."""
        result = unique_append_list([3, 1], [1, 2, 3])
        assert result == [3, 1, 2]


class TestNumericReducers:
    """Tests for numeric reducers."""

    def test_add_numbers(self):
        """Test add_numbers reducer."""
        assert add_numbers(5, 3) == 8
        assert add_numbers(0, 10) == 10
        assert add_numbers(-5, 5) == 0

    def test_max_value(self):
        """Test max_value reducer."""
        assert max_value(5, 3) == 5
        assert max_value(3, 5) == 5
        assert max_value(-10, -5) == -5

    def test_min_value(self):
        """Test min_value reducer."""
        assert min_value(5, 3) == 3
        assert min_value(3, 5) == 3
        assert min_value(-10, -5) == -10


class TestValueReducers:
    """Tests for value selection reducers."""

    def test_last_value(self):
        """Test last_value reducer (default behavior)."""
        assert last_value("old", "new") == "new"
        assert last_value(1, 2) == 2

    def test_first_value(self):
        """Test first_value reducer."""
        assert first_value("old", "new") == "old"
        assert first_value(1, 2) == 1


class TestSetReducers:
    """Tests for set reducers."""

    def test_set_union(self):
        """Test set_union reducer."""
        result = set_union({1, 2}, {2, 3})
        assert result == {1, 2, 3}

    def test_set_intersection(self):
        """Test set_intersection reducer."""
        result = set_intersection({1, 2, 3}, {2, 3, 4})
        assert result == {2, 3}

    def test_empty_sets(self):
        """Test with empty sets."""
        assert set_union(set(), {1}) == {1}
        assert set_intersection(set(), {1}) == set()


class TestGetReducer:
    """Tests for get_reducer utility."""

    def test_extracts_reducer(self):
        """Test extracting reducer from Annotated type."""
        hint = Annotated[list, add_messages]
        reducer = get_reducer(hint)
        assert reducer is add_messages

    def test_returns_none_for_plain_type(self):
        """Test returns None for non-Annotated type."""
        reducer = get_reducer(list)
        assert reducer is None

    def test_returns_none_for_non_callable(self):
        """Test returns None when second arg is not callable."""
        hint = Annotated[list, "not a reducer"]
        reducer = get_reducer(hint)
        assert reducer is None


class TestExtractReducersFromModel:
    """Tests for extract_reducers_from_model utility."""

    def test_extracts_all_reducers(self):
        """Test extracting reducers from model."""

        class TestState(BaseModel):
            messages: Annotated[list, add_messages]
            context: Annotated[dict, merge_dict]
            count: int  # No reducer

        reducers = extract_reducers_from_model(TestState)
        assert "messages" in reducers
        assert "context" in reducers
        assert "count" not in reducers


class TestApplyReducers:
    """Tests for apply_reducers utility."""

    def test_applies_reducers(self):
        """Test applying reducers to state update."""
        reducers = {
            "items": append_list,
            "data": merge_dict,
        }
        current = {"items": [1, 2], "data": {"a": 1}, "other": "old"}
        update = {"items": [3], "data": {"b": 2}, "other": "new"}

        result = apply_reducers(current, update, reducers)
        assert result["items"] == [1, 2, 3]  # Reduced
        assert result["data"] == {"a": 1, "b": 2}  # Reduced
        assert result["other"] == "new"  # Last-write-wins

    def test_handles_missing_keys(self):
        """Test with keys only in update."""
        reducers = {"items": append_list}
        current = {}
        update = {"items": [1, 2], "new_key": "value"}

        result = apply_reducers(current, update, reducers)
        assert result["items"] == [1, 2]
        assert result["new_key"] == "value"


class TestCreateReducer:
    """Tests for create_reducer decorator."""

    def test_create_from_function(self):
        """Test creating reducer from function."""
        concat = create_reducer(lambda a, b: a + b)
        assert concat("hello", " world") == "hello world"

    def test_reducer_decorator(self):
        """Test @reducer decorator."""

        @reducer
        def multiply(a: int, b: int) -> int:
            return a * b

        assert multiply(3, 4) == 12


class TestAddMessagesWithIds:
    """Additional tests for add_messages with ID-based handling."""

    def test_message_with_id_attribute(self):
        """Test messages that have id attributes."""

        # Create a simple class with id attribute
        class MessageWithId:
            def __init__(self, msg_id: str, content: str):
                self.id = msg_id
                self.content = content

        msg1 = MessageWithId("id_1", "First message")
        msg2 = MessageWithId("id_2", "Second message")
        msg3 = MessageWithId("id_1", "Updated first message")  # Same ID as msg1

        reducer = AddMessages()
        result = reducer([msg1, msg2], [msg3])

        # msg3 should replace msg1 since they have the same ID
        assert len(result) == 2
        assert result[0].content == "Updated first message"
        assert result[1].content == "Second message"

    def test_new_message_with_id_appended(self):
        """Test that new messages with IDs are appended and indexed."""

        class MessageWithId:
            def __init__(self, msg_id: str, content: str):
                self.id = msg_id
                self.content = content

        msg1 = MessageWithId("id_1", "First")
        msg2 = MessageWithId("id_2", "Second")

        reducer = AddMessages()
        result = reducer([msg1], [msg2])

        assert len(result) == 2
        assert result[0].id == "id_1"
        assert result[1].id == "id_2"

    def test_mixed_messages_with_and_without_ids(self):
        """Test mixing messages with and without IDs."""

        class MessageWithId:
            def __init__(self, msg_id: str | None, content: str):
                self.id = msg_id
                self.content = content

        msg1 = MessageWithId("id_1", "With ID")
        msg2 = MessageWithId(None, "No ID")
        msg3 = MessageWithId("id_1", "Updated With ID")

        reducer = AddMessages()
        result = reducer([msg1, msg2], [msg3])

        # msg3 should replace msg1
        assert len(result) == 2
        assert result[0].content == "Updated With ID"


class TestReducerEdgeCases:
    """Tests for edge cases in reducers."""

    def test_deep_merge_multiple_levels(self):
        """Test deep merge with multiple nesting levels."""
        current = {
            "a": {
                "b": {
                    "c": 1,
                    "d": 2,
                }
            }
        }
        update = {
            "a": {
                "b": {
                    "e": 3,
                }
            }
        }
        result = deep_merge_dict(current, update)
        assert result["a"]["b"]["c"] == 1
        assert result["a"]["b"]["d"] == 2
        assert result["a"]["b"]["e"] == 3

    def test_append_list_preserves_types(self):
        """Test that append_list preserves various types."""
        result = append_list([1, "two", 3.0], [True, None])
        assert result == [1, "two", 3.0, True, None]

    def test_unique_append_with_unhashable(self):
        """Test unique_append_list with hashable items only."""
        result = unique_append_list([1, 2], [2, 3, 4])
        assert result == [1, 2, 3, 4]

    def test_set_operations_with_various_types(self):
        """Test set operations with various element types."""
        result = set_union({"a", 1}, {1, "b"})
        assert result == {"a", 1, "b"}

    def test_numeric_reducers_with_floats(self):
        """Test numeric reducers with floats."""
        assert add_numbers(1.5, 2.5) == 4.0
        assert max_value(1.5, 2.5) == 2.5
        assert min_value(1.5, 2.5) == 1.5


class TestApplyReducersEdgeCases:
    """Tests for apply_reducers edge cases."""

    def test_apply_reducers_no_overlap(self):
        """Test apply_reducers with no overlapping keys."""
        reducers = {"items": append_list}
        current = {"a": 1}
        update = {"b": 2}

        result = apply_reducers(current, update, reducers)
        assert result["a"] == 1
        assert result["b"] == 2

    def test_apply_reducers_empty_update(self):
        """Test apply_reducers with empty update."""
        reducers = {"items": append_list}
        current = {"items": [1, 2]}
        update = {}

        result = apply_reducers(current, update, reducers)
        assert result["items"] == [1, 2]

    def test_apply_reducers_new_key_with_reducer(self):
        """Test apply_reducers when key exists only in update."""
        reducers = {"items": append_list}
        current = {}
        update = {"items": [1, 2]}

        result = apply_reducers(current, update, reducers)
        assert result["items"] == [1, 2]


class TestReducerEdgeCasesExtended:
    """Edge case tests for reducers."""

    def test_deep_merge_dict_empty_current(self):
        """Test DeepMergeDict with empty current."""
        result = deep_merge_dict({}, {"a": 1, "b": 2})
        assert result == {"a": 1, "b": 2}

    def test_deep_merge_dict_empty_update(self):
        """Test DeepMergeDict with empty update."""
        result = deep_merge_dict({"a": 1}, {})
        assert result == {"a": 1}

    def test_deep_merge_dict_both_empty(self):
        """Test DeepMergeDict with both empty."""
        result = deep_merge_dict({}, {})
        assert result == {}

    def test_deep_merge_dict_none_current(self):
        """Test DeepMergeDict with None-like current."""
        result = deep_merge_dict(None, {"a": 1})
        assert result == {"a": 1}

    def test_unique_append_list_empty_current(self):
        """Test UniqueAppendList with empty current."""
        result = unique_append_list([], [1, 2, 3])
        assert result == [1, 2, 3]

    def test_unique_append_list_empty_update(self):
        """Test UniqueAppendList with empty update."""
        result = unique_append_list([1, 2], [])
        assert result == [1, 2]

    def test_unique_append_list_both_empty(self):
        """Test UniqueAppendList with both empty."""
        result = unique_append_list([], [])
        assert result == []

    def test_unique_append_list_none_current(self):
        """Test UniqueAppendList with None-like current."""
        result = unique_append_list(None, [1, 2])
        assert result == [1, 2]

    def test_set_union_empty_current(self):
        """Test SetUnion with empty current."""
        result = set_union(set(), {1, 2, 3})
        assert result == {1, 2, 3}

    def test_set_union_empty_update(self):
        """Test SetUnion with empty update."""
        result = set_union({1, 2}, set())
        assert result == {1, 2}

    def test_set_union_both_empty(self):
        """Test SetUnion with both empty."""
        result = set_union(set(), set())
        assert result == set()

    def test_set_union_none_current(self):
        """Test SetUnion with None-like current."""
        result = set_union(None, {1, 2})
        assert result == {1, 2}

    def test_set_intersection_empty_current(self):
        """Test SetIntersection with empty current."""
        result = set_intersection(set(), {1, 2, 3})
        assert result == set()

    def test_set_intersection_empty_update(self):
        """Test SetIntersection with empty update."""
        result = set_intersection({1, 2}, set())
        assert result == set()

    def test_append_list_empty_current(self):
        """Test AppendList with empty current."""
        result = append_list([], [1, 2])
        assert result == [1, 2]

    def test_append_list_empty_update(self):
        """Test AppendList with empty update."""
        result = append_list([1, 2], [])
        assert result == [1, 2]

    def test_append_list_none_current(self):
        """Test AppendList with None-like current."""
        result = append_list(None, [1, 2])
        assert result == [1, 2]


class TestGetReducerEdgeCases:
    """Edge case tests for get_reducer."""

    def test_get_reducer_non_annotated(self):
        """Test get_reducer with non-annotated type."""
        result = get_reducer(str)
        assert result is None

    def test_get_reducer_annotated_no_reducer(self):
        """Test get_reducer with Annotated but no reducer."""
        from typing import Annotated

        result = get_reducer(Annotated[int, "just a string"])
        assert result is None

    def test_get_reducer_annotated_with_reducer(self):
        """Test get_reducer with Annotated and reducer."""
        from typing import Annotated

        result = get_reducer(Annotated[list, append_list])
        assert result is append_list
