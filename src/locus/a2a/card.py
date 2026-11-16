# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""An A2A v1.0 Agent Card from an agent definition.

:func:`card_from_agent` reads what an :class:`~locus.agent.Agent` already
declares, its tools, its skills, its name and system prompt, and writes the
card a peer needs to discover it: one ``AgentSkill`` per tool and per skill,
``capabilities.streaming`` (every Locus run streams), ``pushNotifications``
when the server will deliver them, and one ``AgentInterface`` per transport
the server is configured with. ``locus a2a card`` is the command-line face
of it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from locus.a2a.spec import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentProvider,
    AgentSkill,
)
from locus.a2a.spec_v1 import (
    A2A_V1_PROTOCOL_VERSION,
    BINDING_GRPC,
    BINDING_HTTP_JSON,
    BINDING_JSONRPC,
)
from locus.a2a.transports import normalise_binding


DEFAULT_TRANSPORTS: tuple[str, ...] = (BINDING_JSONRPC, BINDING_HTTP_JSON)


def _slug(text: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "-" for ch in text).strip("-")
    while "--" in out:
        out = out.replace("--", "-")
    return out or "skill"


def _first_line(text: str | None, fallback: str) -> str:
    if not text:
        return fallback
    line = text.strip().splitlines()[0].strip() if text.strip() else ""
    return line or fallback


def grpc_interface_url(url: str, grpc_port: int | None, grpc_url: str | None = None) -> str:
    """The gRPC interface address: ``grpc_url`` as given, else the HTTP host with ``grpc_port``."""
    if grpc_url:
        return grpc_url
    parts = urlsplit(url or "http://localhost")
    host = parts.hostname or "localhost"
    if grpc_port is None:
        raise ValueError("a GRPC interface needs grpc_port or grpc_url")
    return f"{host}:{grpc_port}"


def interfaces_for(
    url: str,
    transports: Sequence[str],
    *,
    grpc_port: int | None = None,
    grpc_url: str | None = None,
) -> list[AgentInterface]:
    """One ``AgentInterface`` per transport, in the order given."""
    base = (url or "http://localhost").rstrip("/")
    interfaces: list[AgentInterface] = []
    for binding in dict.fromkeys(normalise_binding(t) for t in transports):
        if binding == BINDING_GRPC:
            address = grpc_interface_url(base, grpc_port, grpc_url)
        else:
            address = base + "/"
        interfaces.append(
            AgentInterface(
                protocolBinding=binding, url=address, protocolVersion=A2A_V1_PROTOCOL_VERSION
            )
        )
    return interfaces


def skills_from_agent(agent: Any) -> list[AgentSkill]:
    """``AgentSkill`` entries for an agent's skills and tools.

    Skills (``agent.config.skills``: :class:`~locus.skills.Skill` objects or
    paths to skill directories) come first, tagged ``skill``; tools
    (``agent.config.tools``, or the compiled registry when the agent has
    one) follow, tagged ``tool`` and ``read-only`` when they declare it. The
    built-in ``ask_user``, ``task_complete``, ``skills`` and ``tool_search``
    tools are the agent's own plumbing and are left out.
    """
    out: list[AgentSkill] = []
    seen: set[str] = set()
    config = getattr(agent, "config", None)
    plumbing = {"ask_user", "task_complete", "skills", "tool_search"}

    for item in list(getattr(config, "skills", None) or []):
        skill = item
        if isinstance(item, str | Path):
            try:
                from locus.skills.models import Skill

                skill = Skill.from_file(item)
            except Exception:  # noqa: BLE001 - a path that does not load is not a skill to advertise
                continue
        name = getattr(skill, "name", None)
        description = getattr(skill, "description", None)
        if not name:
            continue
        skill_id = _slug(str(name))
        if skill_id in seen:
            continue
        seen.add(skill_id)
        out.append(
            AgentSkill(
                id=skill_id,
                name=str(name),
                description=_first_line(description, str(name)),
                tags=["skill"],
            )
        )

    tools: list[Any] = list(getattr(config, "tools", None) or [])
    registry = getattr(agent, "_tool_registry", None)
    registry_tools = getattr(registry, "tools", None)
    if isinstance(registry_tools, dict) and registry_tools:
        tools = list(registry_tools.values())
    for tool in tools:
        name = getattr(tool, "name", None) or getattr(tool, "__name__", None)
        if not name or name in plumbing:
            continue
        skill_id = _slug(str(name))
        if skill_id in seen:
            continue
        seen.add(skill_id)
        tags = ["tool"]
        if getattr(tool, "read_only", False):
            tags.append("read-only")
        description = getattr(tool, "description", None) or getattr(tool, "__doc__", None)
        out.append(
            AgentSkill(
                id=skill_id,
                name=str(name),
                description=_first_line(description, str(name)),
                tags=tags,
            )
        )
    return out


def card_from_agent(
    agent: Any,
    *,
    url: str,
    name: str | None = None,
    description: str | None = None,
    version: str = "0.1.0",
    transports: Sequence[str] = DEFAULT_TRANSPORTS,
    grpc_port: int | None = None,
    grpc_url: str | None = None,
    push_notifications: bool = False,
    provider: AgentProvider | None = None,
    skills: Sequence[AgentSkill] | None = None,
    documentation_url: str | None = None,
    bearer_auth: bool = False,
) -> AgentCard:
    """A v1.0 Agent Card describing ``agent`` as served at ``url``.

    Args:
        agent: The agent (or anything with ``config.tools`` / ``config.skills``).
        url: The public base URL of the A2A server.
        name: Card name; defaults to the agent's ``name`` or ``"Locus Agent"``.
        description: Card description; defaults to the first line of the
            agent's system prompt.
        version: The agent's version string.
        transports: Protocol bindings to advertise (``JSONRPC``, ``HTTP+JSON``,
            ``GRPC``; case-insensitive).
        grpc_port: Port of the gRPC interface, when advertised.
        grpc_url: Address of the gRPC interface instead of the HTTP host plus
            ``grpc_port``.
        push_notifications: Whether the server delivers push notifications.
        provider: The publishing organisation.
        skills: Skills to advertise instead of the ones derived from the agent.
        documentation_url: A link for the card's ``documentationUrl``.
        bearer_auth: Declare the server's bearer-token requirement in
            ``securitySchemes`` / ``securityRequirements``.
    """
    config = getattr(agent, "config", None)
    card_name = (
        name or getattr(agent, "name", None) or getattr(config, "name", None) or "Locus Agent"
    )
    card_description = description or _first_line(
        getattr(config, "system_prompt", None), f"A2A-compatible {card_name}"
    )
    security_schemes: dict[str, Any] = {}
    requirements: list[dict[str, list[str]]] = []
    if bearer_auth:
        security_schemes = {
            "bearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "description": "Bearer token required on every route.",
            }
        }
        requirements = [{"bearerAuth": []}]
    resolved_skills = list(skills) if skills is not None else skills_from_agent(agent)
    return AgentCard(
        name=str(card_name),
        description=card_description,
        url=(url or "http://localhost").rstrip("/") or "http://localhost",
        provider=provider,
        version=version,
        protocolVersion=A2A_V1_PROTOCOL_VERSION,
        documentationUrl=documentation_url,
        capabilities=AgentCapabilities(
            streaming=True,
            pushNotifications=push_notifications,
            stateTransitionHistory=False,
        ),
        securitySchemes=security_schemes,
        security=requirements or None,
        securityRequirements=requirements,
        supportedInterfaces=interfaces_for(url, transports, grpc_port=grpc_port, grpc_url=grpc_url),
        defaultInputModes=["text/plain"],
        defaultOutputModes=["text/plain"],
        skills=[
            skill.model_copy(
                update={
                    "inputModes": skill.inputModes or ["text/plain"],
                    "outputModes": skill.outputModes or ["text/plain"],
                }
            )
            for skill in resolved_skills
        ],
    )


def card_to_json(card: AgentCard) -> str:
    """The card as the JSON document served at ``/.well-known/agent-card.json``."""
    return json.dumps(card.model_dump(exclude_none=True), indent=2, ensure_ascii=False) + "\n"


def write_card(card: AgentCard, path: str | Path) -> Path:
    """Write ``card`` to ``path`` and return it."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(card_to_json(card), encoding="utf-8")
    return target


__all__ = [
    "DEFAULT_TRANSPORTS",
    "card_from_agent",
    "card_to_json",
    "grpc_interface_url",
    "interfaces_for",
    "skills_from_agent",
    "write_card",
]
