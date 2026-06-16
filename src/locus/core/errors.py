# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Locus exception hierarchy.

All exceptions raised from within Locus subclass :class:`LocusError`,
so consumers can catch "any Locus-originated failure" with a single
handler::

    try:
        await agent.run(prompt, thread_id=thread_id)
    except LocusError as exc:
        logger.exception("agent run failed", extra={"kind": exc.kind})
        raise

Sub-hierarchies correspond to subsystems:

- :class:`ToolError` — tool registration, schema, or execution failure
- :class:`ModelError` — LLM provider call, authentication, throttling
- :class:`CheckpointError` — state save / load / list / delete
- :class:`RAGError` — embeddings, vector store, retrieval
- :class:`ValidationError` — bad input at a public API boundary
- :class:`ConfigError` — invalid or missing configuration

Each subclass of :class:`LocusError` exposes a ``kind`` attribute
(a short snake-case string) that can be used as a stable key in
structured logs and metrics, independent of the class name.
"""

from __future__ import annotations

from typing import Any


class LocusError(Exception):
    """Base class for every exception raised by Locus.

    The ``kind`` class attribute is a short stable identifier suitable
    for structured logging and metrics. Subclasses override it.
    """

    kind: str = "locus_error"

    def __init__(self, message: str, *, cause: BaseException | None = None) -> None:
        super().__init__(message)
        if cause is not None:
            self.__cause__ = cause


# =============================================================================
# Tooling
# =============================================================================


class ToolError(LocusError):
    """Base for tool-related failures (registration, schema, execution)."""

    kind = "tool_error"


class ToolNotFoundError(ToolError):
    """Requested tool is not registered with the agent."""

    kind = "tool_not_found"


class ToolValidationError(ToolError):
    """Tool arguments failed schema validation."""

    kind = "tool_validation"


class ToolExecutionError(ToolError):
    """Tool raised an exception during execution."""

    kind = "tool_execution"


# =============================================================================
# Models
# =============================================================================


class ModelError(LocusError):
    """Base for LLM provider failures.

    Instances may optionally carry a :class:`~locus.models.failover.FailoverReason`
    that tells retry / credential-rotation / compaction layers *what to do*,
    independent of the exception class. See
    :func:`locus.models.failover.classify` for the classifier that produces
    the reason from an arbitrary provider SDK exception.
    """

    kind = "model_error"

    def __init__(
        self,
        message: str,
        *,
        cause: BaseException | None = None,
        reason: Any = None,
    ) -> None:
        super().__init__(message, cause=cause)
        self.reason = reason


class ModelAuthError(ModelError):
    """Authentication / authorization against the model provider failed."""

    kind = "model_auth"


class ModelThrottledError(ModelError):
    """Provider is rate-limiting or refusing capacity."""

    kind = "model_throttled"


class ModelResponseError(ModelError):
    """Provider returned an unusable response (malformed JSON, empty, refused)."""

    kind = "model_response"


# =============================================================================
# Memory / checkpointing
# =============================================================================


class CheckpointError(LocusError):
    """Base for checkpointer / storage-backend failures."""

    kind = "checkpoint_error"


class CheckpointNotFoundError(CheckpointError):
    """Requested thread / checkpoint does not exist."""

    kind = "checkpoint_not_found"


class CheckpointSerializationError(CheckpointError):
    """Saving or loading failed during (de)serialization."""

    kind = "checkpoint_serialization"


# =============================================================================
# RAG
# =============================================================================


class RAGError(LocusError):
    """Base for RAG-subsystem failures (embeddings, vector stores, retrieval)."""

    kind = "rag_error"


class EmbeddingError(RAGError):
    """Embedding-provider call failed."""

    kind = "embedding_error"


class VectorStoreError(RAGError):
    """Vector-store read/write failed."""

    kind = "vector_store_error"


# =============================================================================
# Public-API boundaries
# =============================================================================


class ValidationError(LocusError):
    """Caller passed invalid or inconsistent input at a public API boundary."""

    kind = "validation_error"


class ConfigError(LocusError):
    """Configuration is invalid or missing a required value."""

    kind = "config_error"


__all__ = [
    "CheckpointError",
    "CheckpointNotFoundError",
    "CheckpointSerializationError",
    "ConfigError",
    "EmbeddingError",
    "LocusError",
    "ModelAuthError",
    "ModelError",
    "ModelResponseError",
    "ModelThrottledError",
    "RAGError",
    "ToolError",
    "ToolExecutionError",
    "ToolNotFoundError",
    "ToolValidationError",
    "ValidationError",
    "VectorStoreError",
]
