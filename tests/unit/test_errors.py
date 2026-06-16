# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Contract tests for the Locus exception hierarchy.

A single ``except LocusError:`` handler must catch any exception
raised from inside Locus. Every subclass ships a stable ``kind``
attribute for structured logging.
"""

from __future__ import annotations

import inspect

import pytest

from locus.core import errors


class TestLocusErrorHierarchy:
    def test_locus_error_is_exception(self) -> None:
        assert issubclass(errors.LocusError, Exception)

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
    def test_every_error_subclasses_locus_error(self, cls: type[Exception]) -> None:
        """One handler catches them all."""
        assert issubclass(cls, errors.LocusError)

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
            if issubclass(c, errors.LocusError) and c is not errors.LocusError
        ]
        kinds = [c.kind for c in leaves]
        # All lower-case + underscores
        for k in kinds:
            assert k == k.lower()
            assert " " not in k
        # Every subclass overrode the default
        assert all(k != "locus_error" for k in kinds)
        # No duplicates
        assert len(kinds) == len(set(kinds))

    def test_message_and_cause(self) -> None:
        """Constructor passes message through and chains cause."""
        root = ValueError("original")
        err = errors.CheckpointError("save failed", cause=root)
        assert str(err) == "save failed"
        assert err.__cause__ is root

    def test_can_be_caught_as_locus_error(self) -> None:
        """The headline ergonomic: one handler for everything."""
        with pytest.raises(errors.LocusError):
            raise errors.ToolExecutionError("boom")
        with pytest.raises(errors.LocusError):
            raise errors.ModelThrottledError("slow down")
        with pytest.raises(errors.LocusError):
            raise errors.CheckpointNotFoundError("missing thread")
