# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""``locus approvals`` against any store, for an operator without a UI.

``--store`` takes a spec (``memory://``, ``file://``, ``postgresql://``,
``redis://``, ``oracle://``, or a bare path as before) or comes from
``LOCUS_APPROVALS_STORE``; ``approve`` and ``reject`` record who decided
(``--by``, or ``LOCUS_APPROVER``) and why (``--reason``). The database path is
exercised through the asyncpg fake, so the whole command runs with no service.
"""

from __future__ import annotations

import json
import textwrap
from typing import TYPE_CHECKING

import pytest

from locus.cli import EXIT_PAUSED, main
from locus.control import PostgresApprovals
from tests.unit.approval_store_fakes import FakeServer, install_asyncpg


if TYPE_CHECKING:
    from pathlib import Path


_HELD_APP = '''
import os
from pathlib import Path

from locus.agent import Agent
from locus.control import Action, ControlPolicy, approval_store, gate_tool
from locus.core.messages import Role
from locus.memory.backends.file import FileCheckpointer
from locus.testing import FunctionModel, text, tool_call
from locus.tools.decorator import tool

ROOT = Path(os.environ["LOCUS_CLI_TEST_ROOT"])


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
        approval=approval_store(os.environ["LOCUS_APPROVALS_STORE"]),
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
'''


def _app(tmp_path: Path) -> Path:
    path = tmp_path / "held_app.py"
    path.write_text(textwrap.dedent(_HELD_APP))
    return path


def _approval_id(out: str) -> str:
    return next(line.split()[-1] for line in out.splitlines() if "approval:" in line)


def test_the_store_spec_and_the_approver_can_come_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("LOCUS_CLI_TEST_ROOT", str(tmp_path))
    monkeypatch.setenv("LOCUS_APPROVALS_STORE", f"file://{tmp_path / 'approvals.json'}")
    monkeypatch.setenv("LOCUS_APPROVER", "alice")
    app = _app(tmp_path)

    assert main(["run", f"{app}:make", "refund o1", "--thread", "t1"]) == EXIT_PAUSED
    approval_id = _approval_id(capsys.readouterr().out)

    assert main(["approvals", "list"]) == 0
    assert approval_id in capsys.readouterr().out
    assert (
        main(["approvals", "approve", approval_id, "--reason", "within the retention budget"]) == 0
    )
    assert f"{approval_id}: approved by alice" in capsys.readouterr().out
    assert main(["approvals", "show", approval_id]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["decided_by"] == "alice"
    assert record["approvals"][-1]["reason"] == "within the retention budget"

    code = main(["resume", f"{app}:make", "--thread", "t1", "--answer", "approved", "--perform"])
    assert code == 0
    assert (tmp_path / "refunds.log").read_text() == "o1 250.0\n"


def test_reject_records_who_and_why(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("LOCUS_CLI_TEST_ROOT", str(tmp_path))
    store = f"file://{tmp_path / 'approvals.json'}"
    monkeypatch.setenv("LOCUS_APPROVALS_STORE", store)
    app = _app(tmp_path)
    main(["run", f"{app}:make", "refund o1", "--thread", "t1"])
    approval_id = _approval_id(capsys.readouterr().out)

    decision = [
        "approvals",
        "--store",
        store,
        "reject",
        approval_id,
        "--by",
        "bob",
        "--reason",
        "no",
    ]
    assert main(decision) == 0
    assert f"{approval_id}: denied by bob" in capsys.readouterr().out
    assert main(["approvals", "--store", store, "show", approval_id]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["approvals"] == [
        {**record["approvals"][0], "by": "bob", "verdict": "denied", "reason": "no"}
    ]


def test_usage_errors_are_reported_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("LOCUS_APPROVALS_STORE", raising=False)
    monkeypatch.delenv("LOCUS_APPROVER", raising=False)

    assert main(["approvals", "list"]) == 2
    assert "--store is required (or set LOCUS_APPROVALS_STORE)" in capsys.readouterr().err
    assert main(["approvals", "--store", "ftp://x", "list"]) == 2
    assert "unknown approval store scheme 'ftp'" in capsys.readouterr().err
    assert main(["approvals", "--store", "memory://", "approve", "appr-1"]) == 2
    assert "--by is required (or set LOCUS_APPROVER)" in capsys.readouterr().err
    assert main(["approvals", "--store", "memory://", "list"]) == 0
    assert "no pending approvals" in capsys.readouterr().out


def test_a_database_store_is_opened_from_its_spec(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    install_asyncpg(monkeypatch, FakeServer())
    spec = "postgresql://locus:pw@db.internal:5432/locus?table=ops.approvals"
    agent_side = PostgresApprovals(
        dsn="postgresql://locus:pw@db.internal:5432/locus",
        table_name="approvals",
        schema_name="ops",
    )
    approval_id = agent_side.submit("svc", "issue_refund", {"order_id": "o1"}, reason="payment")

    assert main(["approvals", "--store", spec, "list"]) == 0
    assert approval_id in capsys.readouterr().out
    assert main(["approvals", "--store", spec, "approve", approval_id, "--by", "alice"]) == 0
    assert f"{approval_id}: approved by alice" in capsys.readouterr().out

    record = agent_side.get(approval_id)
    assert record is not None
    assert (record.status, record.decided_by) == ("approved", "alice")
    agent_side.close()
