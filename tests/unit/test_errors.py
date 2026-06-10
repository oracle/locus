# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the Tulip exception hierarchy.

A single ``except TulipError:`` handler must catch any exception
raised from inside Tulip. Every subclass ships a stable ``kind``
attribute for structured logging.
"""

from __future__ import annotations

import inspect

import pytest

from tulip.core import errors


class TestTulipErrorHierarchy:
    def test_tulip_error_is_exception(self) -> None:
        assert issubclass(errors.TulipError, Exception)

    @pytest.mark.parametrize(
        "cls",
        [
            errors.ToolError,
            errors.ToolNotFoundError,
            errors.ToolValidationError,
            errors.ToolExecutionError,
            errors.ModelError,
            errors.ModelAuthError,
            errors.ModelThrottledError,
            errors.ModelResponseError,
            errors.CheckpointError,
            errors.CheckpointNotFoundError,
            errors.CheckpointSerializationError,
            errors.RAGError,
            errors.EmbeddingError,
            errors.VectorStoreError,
            errors.ValidationError,
            errors.ConfigError,
        ],
    )
    def test_every_error_subclasses_tulip_error(self, cls: type[Exception]) -> None:
        """One handler catches them all."""
        assert issubclass(cls, errors.TulipError)

    def test_sub_hierarchies(self) -> None:
        """Within-subsystem catches work too."""
        assert issubclass(errors.ToolExecutionError, errors.ToolError)
        assert issubclass(errors.ToolNotFoundError, errors.ToolError)
        assert issubclass(errors.ModelAuthError, errors.ModelError)
        assert issubclass(errors.ModelThrottledError, errors.ModelError)
        assert issubclass(errors.CheckpointNotFoundError, errors.CheckpointError)
        assert issubclass(errors.EmbeddingError, errors.RAGError)
        assert issubclass(errors.VectorStoreError, errors.RAGError)

    def test_kind_is_snake_case_and_unique(self) -> None:
        """Every leaf class has a distinct snake_case ``kind``."""
        leaves = [
            c
            for _, c in inspect.getmembers(errors, inspect.isclass)
            if issubclass(c, errors.TulipError) and c is not errors.TulipError
        ]
        kinds = [c.kind for c in leaves]
        # All lower-case + underscores
        for k in kinds:
            assert k == k.lower()
            assert " " not in k
        # Every subclass overrode the default
        assert all(k != "tulip_error" for k in kinds)
        # No duplicates
        assert len(kinds) == len(set(kinds))

    def test_message_and_cause(self) -> None:
        """Constructor passes message through and chains cause."""
        root = ValueError("original")
        err = errors.CheckpointError("save failed", cause=root)
        assert str(err) == "save failed"
        assert err.__cause__ is root

    def test_can_be_caught_as_tulip_error(self) -> None:
        """The headline ergonomic: one handler for everything."""
        with pytest.raises(errors.TulipError):
            raise errors.ToolExecutionError("boom")
        with pytest.raises(errors.TulipError):
            raise errors.ModelThrottledError("slow down")
        with pytest.raises(errors.TulipError):
            raise errors.CheckpointNotFoundError("missing thread")
