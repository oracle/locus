"""The agent server this deployment ships. Edit freely; ``app`` is what the image serves.

Environment the deployment provides (see locus-deploy.yaml and the Terraform
variables): OCI_COMPARTMENT_ID and OCI_REGION always; LOCUS_OCI_AUTH_TYPE names
the principal the platform gives the container (``api_key`` with OCI_PROFILE and
a mounted ``~/.oci`` runs the image locally); LOCUS_SERVER_API_KEY arrives
through OCI Vault when server_api_key_secret_id is set. The server refuses to
start without that key unless LOCUS_SERVER_ALLOW_UNAUTHENTICATED=1 says an
unauthenticated agent is intended (allow_unauthenticated in the Terraform
variables). Set LOCUS_MODEL=mock in ``environment:`` for a first deployment that
proves the path before a model is wired, then remove it.
"""

from __future__ import annotations

import os
import sys
from typing import Any

from locus.agent import Agent
from locus.server import AgentServer


MODEL = os.environ.get("LOCUS_MODEL", "oci:openai.gpt-4.1")
API_KEY = os.environ.get("LOCUS_SERVER_API_KEY") or None
ALLOW_UNAUTHENTICATED = os.environ.get("LOCUS_SERVER_ALLOW_UNAUTHENTICATED") == "1"

if API_KEY is None and not ALLOW_UNAUTHENTICATED:
    sys.stderr.write(
        "server.py: LOCUS_SERVER_API_KEY is not set, so every caller could use this agent. "
        "Set server_api_key_secret_id (a Vault secret holding the bearer key) in "
        "locus-deploy.yaml, or allow_unauthenticated: true "
        "(LOCUS_SERVER_ALLOW_UNAUTHENTICATED=1) if that is intended.\n"
    )
    sys.exit(1)


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
    api_key=API_KEY,
    allow_unauthenticated=API_KEY is None,
)

# The ASGI callable uvicorn serves; locus deploy names it as server:app.
app = server.app
