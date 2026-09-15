# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""The ``tulip`` command, end to end through ``main(argv)``.

The flow that matters is the one a person runs by hand: an agent pauses on a
held refund, someone approves it from the command line, the run resumes and the
refund happens once; then the exported audit trail verifies from a public key.
"""

from __future__ import annotations

import json
import textwrap
from typing import TYPE_CHECKING, Any

import pytest

from tulip.cli import EXIT_PAUSED, load_agent, main


if TYPE_CHECKING:
    from pathlib import Path


_SIMPLE_APP = """
from tulip.agent import Agent
from tulip.testing import FunctionModel, text


def make():
    return Agent(
        model=FunctionModel(lambda messages, tools: text("hello from the agent")),
        tools=[],
        reflexion=False,
        grounding=False,
    )


agent = make()
not_an_agent = 42
"""

_HELD_APP = '''
import os
from pathlib import Path

from tulip.agent import Agent
from tulip.control import Action, ApprovalAuthority, ApproverRule, ControlPolicy, FileApprovals, gate_tool
from tulip.core.messages import Role
from tulip.memory.backends.file import FileCheckpointer
from tulip.testing import FunctionModel, text, tool_call
from tulip.tools.decorator import tool

ROOT = Path(os.environ["TULIP_CLI_TEST_ROOT"])


@tool
def issue_refund(order_id: str, amount_usd: float) -> str:
    """Refund an order."""
    with open(ROOT / "refunds.log", "a") as log:
        log.write(f"{order_id} {amount_usd}\\n")
    return f"refunded {amount_usd} on {order_id}"


def _model(messages, tools):
    if any(m.role == Role.TOOL for m in messages):
        return text("refund issued")
    return tool_call("issue_refund", order_id="o1", amount_usd=250.0)


def make():
    refund = gate_tool(
        issue_refund,
        policy=ControlPolicy(require_verification_score=0.0, require_human_for=frozenset({"payment"})),
        action=lambda name, kwargs: Action(name=name, asset=kwargs["order_id"], kind="payment"),
        approval=FileApprovals(ROOT / "approvals.json"),
        on_refusal="interrupt",
        principal="svc-billing",
    )
    return Agent(
        model=FunctionModel(_model),
        tools=[refund],
        checkpointer=FileCheckpointer(ROOT / "checkpoints"),
        reflexion=False,
        grounding=False,
    )


authority = ApprovalAuthority(rules=(ApproverRule(approvers=frozenset({"alice"})),))
'''


def _write(tmp_path: Path, name: str, source: str) -> Path:
    path = tmp_path / name
    path.write_text(textwrap.dedent(source))
    return path


def test_run_streams_the_answer(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    app = _write(tmp_path, "simple_app.py", _SIMPLE_APP)

    assert main(["run", f"{app}:agent", "hi"]) == 0
    assert "hello from the agent" in capsys.readouterr().out


def test_a_factory_is_called(tmp_path: Path) -> None:
    app = _write(tmp_path, "simple_app.py", _SIMPLE_APP)

    from tulip.agent import Agent

    assert isinstance(load_agent(f"{app}:make"), Agent)


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ("no-colon", "expected MODULE:ATTR"),
        ("missing.py:agent", "no such file"),
        ("no_such_module_xyz:agent", "cannot import"),
        ("{app}:missing", "has no attribute"),
        ("{app}:not_an_agent", "is not an Agent"),
    ],
)
def test_a_bad_agent_spec_is_a_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], spec: str, message: str
) -> None:
    app = _write(tmp_path, "simple_app.py", _SIMPLE_APP)

    assert main(["run", spec.format(app=app), "hi"]) == 2
    assert message in capsys.readouterr().err


def test_hold_approve_resume_from_the_command_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("TULIP_CLI_TEST_ROOT", str(tmp_path))
    app = _write(tmp_path, "held_app.py", _HELD_APP)
    store = str(tmp_path / "approvals.json")

    assert main(["run", f"{app}:make", "refund o1", "--thread", "t1"]) == EXIT_PAUSED
    paused = capsys.readouterr().out
    approval_id = next(line.split()[-1] for line in paused.splitlines() if "approval:" in line)
    assert "tulip resume" in paused
    assert not (tmp_path / "refunds.log").exists()

    assert main(["approvals", "--store", store, "list"]) == 0
    assert approval_id in capsys.readouterr().out
    assert main(["approvals", "--store", store, "list", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["tool"] == "issue_refund"

    assert main(["approvals", "--store", store, "approve", approval_id, "--by", "alice"]) == 0
    assert f"{approval_id}: approved" in capsys.readouterr().out

    code = main(["resume", f"{app}:make", "--thread", "t1", "--answer", "approved", "--perform"])
    out = capsys.readouterr().out
    assert code == 0
    assert "refund issued" in out
    assert (tmp_path / "refunds.log").read_text() == "o1 250.0\n"

    assert main(["approvals", "--store", store, "show", approval_id]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "consumed"
    assert main(["approvals", "--store", store, "list"]) == 0
    assert "no pending approvals" in capsys.readouterr().out


def test_decisions_are_checked_against_an_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("TULIP_CLI_TEST_ROOT", str(tmp_path))
    app = _write(tmp_path, "held_app.py", _HELD_APP)
    store = str(tmp_path / "approvals.json")
    main(["run", f"{app}:make", "refund o1", "--thread", "t1"])
    approval_id = next(
        line.split()[-1] for line in capsys.readouterr().out.splitlines() if "approval:" in line
    )
    guarded = ["approvals", "--store", store, "--authority", f"{app}:authority"]

    assert main([*guarded, "approve", approval_id, "--by", "mallory"]) == 1
    assert "refused" in capsys.readouterr().err
    assert main([*guarded, "reject", approval_id, "--by", "alice"]) == 0
    assert f"{approval_id}: denied" in capsys.readouterr().out
    assert main(["approvals", "--store", store, "approve", approval_id, "--by", "alice"]) == 1
    assert "already denied" in capsys.readouterr().err
    assert main(["approvals", "--store", store, "show", "appr-nope"]) == 1
    assert main([*guarded[:4], f"{app}:make", "list"]) == 2


def test_audit_verify_from_public_keys(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("cryptography")
    from tulip.control import AuditTrail, Ed25519Signer

    signer = Ed25519Signer.generate(key_id="audit-1")
    trail = AuditTrail(signer=signer)
    for i in range(3):
        trail.record("action-admission", {"action": f"refund-{i}"})
    export = tmp_path / "trail.jsonl"
    export.write_text(trail.export_jsonl())
    pem = tmp_path / "audit-1.pub.pem"
    pem.write_bytes(signer.public_key_pem())

    assert (
        main(["audit", "verify", str(export), "--key", f"audit-1={pem}", "--head", trail.head]) == 0
    )
    assert "verified: 3 record(s), signatures checked against 1 key(s), head matches" in (
        capsys.readouterr().out
    )
    assert main(["audit", "verify", str(export)]) == 0
    assert "chain only" in capsys.readouterr().out

    lines = export.read_text().splitlines()
    tampered = json.loads(lines[1])
    tampered["payload"]["action"] = "refund-forged"
    lines[1] = json.dumps(tampered)
    export.write_text("\n".join(lines))
    assert main(["audit", "verify", str(export), "--key", f"audit-1={pem}"]) == 1
    assert "FAILED" in capsys.readouterr().err

    assert main(["audit", "verify", str(export), "--key", "no-equals-sign"]) == 2
    assert main(["audit", "verify", str(export), "--key", f"audit-1={tmp_path / 'nope.pem'}"]) == 2
    assert main(["audit", "verify", str(tmp_path / "missing.jsonl")]) == 2


def test_serve_hands_the_agent_to_the_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    app = _write(tmp_path, "simple_app.py", _SIMPLE_APP)
    calls: list[dict[str, Any]] = []

    def fake_run(self: Any, host: str, port: int) -> None:
        calls.append({"host": host, "port": port, "api_key": self._api_key})

    monkeypatch.setattr("tulip.server.app.AgentServer.run", fake_run)
    assert main(["serve", f"{app}:agent", "--port", "9123", "--api-key", "k"]) == 0
    assert calls == [{"host": "127.0.0.1", "port": 9123, "api_key": "k"}]

    def refusing_run(self: Any, host: str, port: int) -> None:
        raise RuntimeError("Refusing to bind")

    monkeypatch.setattr("tulip.server.app.AgentServer.run", refusing_run)
    assert main(["serve", f"{app}:agent", "--host", "0.0.0.0"]) == 1  # noqa: S104
    assert "Refusing to bind" in capsys.readouterr().err


def test_version_is_printed(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.startswith("tulip-agents ")
