# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Agent server — expose agents (and graphs) as HTTP endpoints.

Wraps a Tulip Agent as a FastAPI application with invoke and stream
endpoints. Requires ``fastapi`` and ``uvicorn`` optional dependencies.

Example — publish an Agent::

    from tulip.agent import Agent, AgentConfig
    from tulip.server import AgentServer

    agent = Agent(
        config=AgentConfig(
            system_prompt="You are a helpful assistant.",
            model=my_model,
        )
    )

    server = AgentServer(agent=agent)
    server.run(port=8000)

Example — publish a Graph via :class:`GraphRunnable` (closes #213)::

    from tulip.multiagent.graph import StateGraph
    from tulip.server import AgentServer, GraphRunnable

    graph = StateGraph(...).compile()
    server = AgentServer(
        agent=GraphRunnable(graph, input_key="prompt", output_key="answer"),
    )
    server.run(port=8000)

The same :class:`GraphRunnable` slots into ``tulip.a2a.A2AServer`` —
publish a graph as a spec-compliant A2A endpoint with no extra wiring.
"""

from tulip.server.adapters import GraphRunnable
from tulip.server.app import AgentServer


__all__ = ["AgentServer", "GraphRunnable"]
