"""The agent server this deployment ships. Edit freely; ``app`` is what the image serves.

Environment the deployment provides (see locus-deploy.yaml and the Terraform
variables): OCI_COMPARTMENT_ID and OCI_REGION always; LOCUS_OCI_AUTH_TYPE names
the principal the platform gives the container; LOCUS_SERVER_API_KEY arrives
through OCI Vault when server_api_key_secret_id is set. Set LOCUS_MODEL=mock in
``environment:`` for a first deployment that proves the path before a model is
wired, then remove it.
"""

from __future__ import annotations

import os
from typing import Any

from locus.agent import Agent
from locus.server import AgentServer


MODEL = os.environ.get("LOCUS_MODEL", "oci:openai.gpt-4.1")


def _model_kwargs() -> dict[str, str]:
    if not MODEL.startswith("oci:"):
        return {}
    return {
        "auth_type": os.environ.get("LOCUS_OCI_AUTH_TYPE", "resource_principal"),
        "compartment_id": os.environ["OCI_COMPARTMENT_ID"],
    }


def _build_agent() -> Agent:
    if MODEL == "mock":
        from locus.testing import FunctionModel

        def reply(messages: list[Any], tools: list[Any]) -> str:
            del tools
            prompt = messages[-1].content if messages else ""
            return f"mock model received: {prompt}"

        return Agent(model=FunctionModel(reply), tools=[], reflexion=False, grounding=False)
    return Agent(
        model=MODEL,
        model_kwargs=_model_kwargs(),
        tools=[],
        system_prompt="Answer concisely.",
    )


server = AgentServer(
    agent=_build_agent(),
    title=os.environ.get("LOCUS_SERVER_TITLE", "Locus agent"),
    api_key=os.environ.get("LOCUS_SERVER_API_KEY"),
)

# The ASGI callable uvicorn serves; locus deploy names it as server:app.
app = server.app
