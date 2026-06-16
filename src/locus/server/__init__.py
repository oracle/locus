# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Agent server — expose agents (and graphs) as HTTP endpoints.

Wraps a Locus Agent as a FastAPI application with invoke and stream
endpoints. Requires ``fastapi`` and ``uvicorn`` optional dependencies.

Example — publish an Agent::

    from locus.agent import Agent, AgentConfig
    from locus.server import AgentServer

    agent = Agent(
        config=AgentConfig(
            system_prompt="You are a helpful assistant.",
            model=my_model,
        )
    )

    server = AgentServer(agent=agent)
    server.run(port=8000)

Example — publish a Graph via :class:`GraphRunnable` (closes #213)::

    from locus.multiagent.graph import StateGraph
    from locus.server import AgentServer, GraphRunnable

    graph = StateGraph(...).compile()
    server = AgentServer(
        agent=GraphRunnable(graph, input_key="prompt", output_key="answer"),
    )
    server.run(port=8000)

The same :class:`GraphRunnable` slots into ``locus.a2a.A2AServer`` —
publish a graph as a spec-compliant A2A endpoint with no extra wiring.
"""

from locus.server.adapters import GraphRunnable
from locus.server.app import AgentServer


__all__ = ["AgentServer", "GraphRunnable"]
