# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""The request-to-frame seam at the front of the router.

Everything after the :class:`~locus.router.goal_frame.GoalFrame` is
deterministic; the frame itself has to come from somewhere. By default an
:class:`~locus.Agent` with ``output_schema=GoalFrame`` fills it, one
structured-output call per request. :class:`RequestClassifier` is the
contract that step satisfies, so a deployment can put something else
behind it: a trained classifier, a lookup table for a closed set of
requests, a cache in front of the agent. The :class:`Router` never knows
which it is talking to.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from locus.router.goal_frame import GoalFrame


if TYPE_CHECKING:
    from locus.agent.agent import Agent


class FrameExtractionError(RuntimeError):
    """Raised when a classifier failed to produce a valid GoalFrame."""


@runtime_checkable
class RequestClassifier(Protocol):
    """Turn a natural-language request into a :class:`GoalFrame`.

    The only requirement is the one method. A classifier that cannot frame
    a request raises :class:`FrameExtractionError`; the router reports it
    on the event bus and re-raises, because the compiler cannot recover
    from a missing frame.
    """

    async def classify(self, text: str) -> GoalFrame:
        """Return the frame for ``text``."""
        ...


class AgentFrameExtractor:
    """The default :class:`RequestClassifier`: an agent fills the frame.

    Wraps an :class:`~locus.Agent` configured with
    ``output_schema=GoalFrame`` and runs its blocking ``invoke`` off the
    event loop. This is the extraction :class:`Router` always did; it is
    now one implementation of the seam rather than the seam itself.
    """

    def __init__(self, extractor: Agent) -> None:
        self.extractor = extractor

    async def classify(self, text: str) -> GoalFrame:
        result: Any = await asyncio.to_thread(self.extractor.invoke, text)
        parsed = result.parsed
        if not isinstance(parsed, GoalFrame):
            err_msg = (
                "Extractor did not produce a GoalFrame. "
                f"parse_error={result.parse_error!r}, message={result.message!r}"
            )
            raise FrameExtractionError(err_msg)
        return parsed


__all__ = ["AgentFrameExtractor", "FrameExtractionError", "RequestClassifier"]
