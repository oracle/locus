# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""ReAct loop primitives. **Deprecated — scheduled for removal in 3.0.0.**

This is a second ReAct implementation, parallel to the one the supported
``Agent`` actually runs. ``Agent`` has never used it: the only reference from
the production runtime was one private helper, and that has moved to
:mod:`locus.tools.executor`.

Two implementations of the same idea is worse than either alone. They drift,
a bug fixed in one stays live in the other, and a reader deciding whether this
project is disciplined greps for exactly this. Nothing here is a capability
:class:`~locus.agent.Agent` lacks.

What to use instead:

===========================  ==================================================
``ReActLoop`` / ``create_react_loop``  :class:`locus.agent.Agent`
``LoopRunner``               ``await agent.arun(prompt)``
``BatchRunner``              :class:`locus.evaluation.EvalRunner`, or
                             ``asyncio.gather`` over ``arun``
``StreamingCollector``       ``async for event in agent.run(prompt)``
``ThinkNode`` / ``ExecuteNode`` / ``ReflectNode``
                             the phases are internal to ``Agent``; hook them
                             with :mod:`locus.hooks`
``ConditionalRouter``        :class:`locus.multiagent.StateGraph` conditional
                             edges, or :mod:`locus.router`
===========================  ==================================================

Importing a name from here still works and will keep working until 3.0.0, per
:doc:`/DEPRECATION`. Each access emits
:class:`~locus.core.warnings.LocusDeprecationWarning`; to find them in your own
code, run with ``-W error::DeprecationWarning``.
"""

from __future__ import annotations

import warnings
from typing import Any

from locus.core.warnings import LocusDeprecationWarning
from locus.loop.nodes import (
    ExecuteNode,
    Node,
    NodeResult,
    ReflectNode,
    ThinkNode,
)
from locus.loop.react import (
    ReActLoop,
    ReActLoopConfig,
    create_react_loop,
)
from locus.loop.router import (
    ConditionalRouter,
    NodeType,
    RouteDecision,
    Router,
)
from locus.loop.runner import (
    BatchRunner,
    LoopRunner,
    StreamingCollector,
    create_runner,
)


__all__ = [
    # Nodes
    "Node",
    "NodeResult",
    "ThinkNode",
    "ExecuteNode",
    "ReflectNode",
    # React
    "ReActLoop",
    "ReActLoopConfig",
    "create_react_loop",
    # Router
    "Router",
    "ConditionalRouter",
    "NodeType",
    "RouteDecision",
    # Runner
    "LoopRunner",
    "BatchRunner",
    "StreamingCollector",
    "create_runner",
]


#: What to reach for instead, per name. Only the ones with a direct answer —
#: a vague pointer is worse than none, because it sends the reader off to
#: check something that was never going to fit.
_REPLACEMENTS = {
    "ReActLoop": "locus.agent.Agent",
    "ReActLoopConfig": "locus.agent.AgentConfig",
    "create_react_loop": "locus.agent.Agent",
    "LoopRunner": "Agent.arun()",
    "BatchRunner": "locus.evaluation.EvalRunner",
    "StreamingCollector": "Agent.run()",
    "ConditionalRouter": "locus.multiagent.StateGraph conditional edges",
}

# Bind the eager imports aside, then re-serve them through __getattr__ so every
# access warns. Module-level __getattr__ only fires for names NOT already in the
# module namespace, so leaving them bound would make the warning unreachable —
# which is how a deprecation ships without deprecating anything.
_DEPRECATED = {name: globals().pop(name) for name in __all__ if name in globals()}


def __getattr__(name: str) -> Any:
    """Serve a deprecated name, and say so."""
    if name not in _DEPRECATED:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)

    instead = _REPLACEMENTS.get(name)
    warnings.warn(
        f"locus.loop.{name} is deprecated and will be removed in 6.0.0. "
        f"locus.loop is a second ReAct implementation that Agent never used"
        + (f"; use {instead} instead." if instead else "."),
        LocusDeprecationWarning,
        stacklevel=2,
    )
    return _DEPRECATED[name]


def __dir__() -> list[str]:
    """Keep tab-completion and ``dir()`` working through ``__getattr__``."""
    return sorted(__all__)
