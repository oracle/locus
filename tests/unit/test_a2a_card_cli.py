# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""``card_from_agent`` and ``locus a2a card``."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

import pytest

from locus.a2a import A2AServer, card_from_agent
from locus.a2a.card import card_to_json, grpc_interface_url, interfaces_for, skills_from_agent
from locus.agent import Agent
from locus.cli import main
from locus.skills.models import Skill
from locus.testing import FunctionModel, text
from locus.tools.decorator import tool


@tool(read_only=True)
def lookup_order(order_id: str) -> str:
    """Look up an order by id.

    Second paragraph that must not appear in the card.
    """
    return order_id


@tool
def issue_refund(order_id: str, amount_usd: float) -> str:
    """Refund an order."""
    return f"{order_id} {amount_usd}"


def _agent() -> Agent:
    return Agent(
        name="refunds",
        model=FunctionModel(lambda messages, tools: text("ok")),
        tools=[lookup_order, issue_refund],
        skills=[Skill(name="Refund policy", description="When to refund.\nNot this line.")],
        system_prompt="Handle refund requests for the store.\nBe terse.",
        reflexion=False,
        grounding=False,
    )


EXPECTED = {
    "name": "refunds",
    "description": "Handle refund requests for the store.",
    "url": "https://agent.example.com",
    "version": "1.2.3",
    "protocolVersion": "1.0",
    "capabilities": {
        "streaming": True,
        "pushNotifications": True,
        "stateTransitionHistory": False,
        "extensions": [],
    },
    "securitySchemes": {},
    "securityRequirements": [],
    "supportedInterfaces": [
        {
            "protocolBinding": "JSONRPC",
            "url": "https://agent.example.com/",
            "protocolVersion": "1.0",
        },
        {
            "protocolBinding": "HTTP+JSON",
            "url": "https://agent.example.com/",
            "protocolVersion": "1.0",
        },
        {"protocolBinding": "GRPC", "url": "agent.example.com:7422", "protocolVersion": "1.0"},
    ],
    "defaultInputModes": ["text/plain"],
    "defaultOutputModes": ["text/plain"],
    "skills": [
        {
            "id": "refund-policy",
            "name": "Refund policy",
            "description": "When to refund.",
            "tags": ["skill"],
            "examples": [],
            "inputModes": ["text/plain"],
            "outputModes": ["text/plain"],
        },
        {
            "id": "lookup-order",
            "name": "lookup_order",
            "description": "Look up an order by id.",
            "tags": ["tool", "read-only"],
            "examples": [],
            "inputModes": ["text/plain"],
            "outputModes": ["text/plain"],
        },
        {
            "id": "issue-refund",
            "name": "issue_refund",
            "description": "Refund an order.",
            "tags": ["tool"],
            "examples": [],
            "inputModes": ["text/plain"],
            "outputModes": ["text/plain"],
        },
    ],
    "supportsAuthenticatedExtendedCard": False,
}


class TestCardFromAgent:
    def test_snapshot(self) -> None:
        card = card_from_agent(
            _agent(),
            url="https://agent.example.com/",
            version="1.2.3",
            transports=["jsonrpc", "http+json", "grpc"],
            grpc_port=7422,
            push_notifications=True,
        )
        assert card.model_dump(exclude_none=True) == EXPECTED
        assert json.loads(card_to_json(card)) == EXPECTED

    def test_defaults_and_overrides(self) -> None:
        agent = Agent(
            model=FunctionModel(lambda m, t: text("ok")), tools=[], reflexion=False, grounding=False
        )
        card = card_from_agent(agent, url="")
        assert card.name == "Locus Agent"
        # The agent's own default system prompt, first line.
        assert card.description == "You are a helpful AI assistant."
        assert card.url == "http://localhost"
        assert [i.protocolBinding for i in card.supportedInterfaces] == ["JSONRPC", "HTTP+JSON"]
        assert card.skills == []
        assert card.capabilities.pushNotifications is False

        from locus.a2a import AgentProvider, AgentSkill

        custom = card_from_agent(
            agent,
            url="https://x",
            name="custom",
            description="desc",
            provider=AgentProvider(organization="Acme"),
            skills=[AgentSkill(id="s", name="S", description="d")],
            documentation_url="https://docs",
            bearer_auth=True,
        )
        assert custom.provider is not None
        assert custom.provider.organization == "Acme"
        assert [s.id for s in custom.skills] == ["s"]
        assert custom.skills[0].inputModes == ["text/plain"]
        assert custom.securityRequirements == [{"bearerAuth": []}]
        assert custom.documentationUrl == "https://docs"

    def test_skills_from_registry_paths_and_duplicates(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "triage"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: triage\ndescription: Triage alerts.\n---\nSteps.\n"
        )
        agent = Agent(
            model=FunctionModel(lambda m, t: text("ok")),
            tools=[lookup_order],
            skills=[
                str(skill_dir),
                str(tmp_path / "missing"),
                Skill(name="triage", description="dup"),
            ],
            reflexion=False,
            grounding=False,
        )
        _ = agent.tools  # compiles config.tools and config.skills into the registry
        skills = skills_from_agent(agent)
        assert [s.id for s in skills] == ["triage", "lookup-order"]
        assert skills[0].description == "Triage alerts."
        # The built-in plumbing tools are never advertised.
        assert "ask-user" not in {s.id for s in skills}

    def test_interfaces_and_grpc_url(self) -> None:
        assert (
            grpc_interface_url("https://h.example.com:8443", None, "grpc.example.com:443")
            == "grpc.example.com:443"
        )
        assert grpc_interface_url("http://h", 50051) == "h:50051"
        with pytest.raises(ValueError, match="grpc_port"):
            grpc_interface_url("http://h", None)
        with pytest.raises(ValueError, match="grpc_port"):
            interfaces_for("http://h", ["grpc"])
        ifaces = interfaces_for("http://h/", ["jsonrpc", "JSONRPC", "grpc"], grpc_url="h:1")
        assert [(i.protocolBinding, i.url) for i in ifaces] == [
            ("JSONRPC", "http://h/"),
            ("GRPC", "h:1"),
        ]


_APP = """
from locus.agent import Agent
from locus.testing import FunctionModel, text
from locus.tools.decorator import tool


@tool
def lookup_order(order_id: str) -> str:
    \"\"\"Look up an order.\"\"\"
    return order_id


agent = Agent(
    name="refunds",
    model=FunctionModel(lambda messages, tools: text("ok")),
    tools=[lookup_order],
    system_prompt="Handle refunds.",
    reflexion=False,
    grounding=False,
)
"""


def _write_app(tmp_path: Path) -> Path:
    path = tmp_path / "app.py"
    path.write_text(textwrap.dedent(_APP))
    return path


class TestCli:
    def test_card_to_stdout_and_file(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        app = _write_app(tmp_path)
        assert main(["a2a", "card", f"{app}:agent", "--url", "https://agent.example.com"]) == 0
        card = json.loads(capsys.readouterr().out)
        assert card["name"] == "refunds"
        assert card["description"] == "Handle refunds."
        assert [s["id"] for s in card["skills"]] == ["lookup-order"]
        assert [i["protocolBinding"] for i in card["supportedInterfaces"]] == [
            "JSONRPC",
            "HTTP+JSON",
        ]
        assert card["capabilities"]["pushNotifications"] is False

        out = tmp_path / "cards" / "agent-card.json"
        code = main(
            [
                "a2a",
                "card",
                f"{app}:agent",
                "--url",
                "https://agent.example.com",
                "--transport",
                "grpc",
                "--transport",
                "jsonrpc",
                "--grpc-port",
                "7422",
                "--push",
                "--provider",
                "Acme",
                "--docs-url",
                "https://docs",
                "--version",
                "2.0.0",
                "--api-key",
                "k",
                "-o",
                str(out),
            ]
        )
        assert code == 0
        assert "agent card written to" in capsys.readouterr().out
        written = json.loads(out.read_text())
        assert [i["protocolBinding"] for i in written["supportedInterfaces"]] == ["GRPC", "JSONRPC"]
        assert written["supportedInterfaces"][0]["url"] == "agent.example.com:7422"
        assert written["capabilities"]["pushNotifications"] is True
        assert written["provider"] == {"organization": "Acme"}
        assert written["version"] == "2.0.0"
        assert written["securityRequirements"] == [{"bearerAuth": []}]

    def test_grpc_needs_a_port(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        app = _write_app(tmp_path)
        assert main(["a2a", "card", f"{app}:agent", "--transport", "grpc"]) == 2
        assert "--grpc-port" in capsys.readouterr().err

    def test_serve_hands_the_card_to_the_server(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app = _write_app(tmp_path)
        runs: list[dict[str, Any]] = []

        def fake_run(self: A2AServer, host: str, port: int, **kwargs: Any) -> None:
            runs.append(
                {
                    "host": host,
                    "port": port,
                    "card": self._build_card().model_dump(exclude_none=True),
                    "push": self.push_notifications is not None,
                    "api_key": self.api_key,
                }
            )

        monkeypatch.setattr(A2AServer, "run", fake_run)
        monkeypatch.delenv("LOCUS_A2A_API_KEY", raising=False)
        code = main(
            [
                "a2a",
                "card",
                f"{app}:agent",
                "--serve",
                "--port",
                "7431",
                "--url",
                "http://127.0.0.1:7431",
                "--push",
                "--allow-private-push",
                "--api-key",
                "k",
                "-o",
                str(tmp_path / "card.json"),
            ]
        )
        assert code == 0
        assert runs[0]["host"] == "127.0.0.1"
        assert runs[0]["port"] == 7431
        assert runs[0]["push"] is True
        assert runs[0]["api_key"] == "k"
        assert runs[0]["card"]["name"] == "refunds"
        assert runs[0]["card"]["capabilities"]["pushNotifications"] is True
        assert runs[0]["card"]["securityRequirements"] == [{"bearerAuth": []}]
        assert runs[0]["push"] is True

    def test_serve_refusal_is_reported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        app = _write_app(tmp_path)

        def refusing_run(self: A2AServer, host: str, port: int, **kwargs: Any) -> None:
            raise RuntimeError("Refusing to bind")

        monkeypatch.setattr(A2AServer, "run", refusing_run)
        monkeypatch.delenv("LOCUS_A2A_API_KEY", raising=False)
        assert (
            main(
                [
                    "a2a",
                    "card",
                    f"{app}:agent",
                    "--serve",
                    "--host",
                    "0.0.0.0",  # noqa: S104
                    "-o",
                    str(tmp_path / "c.json"),
                ]
            )
            == 1
        )  # noqa: S104
        assert "Refusing to bind" in capsys.readouterr().err
