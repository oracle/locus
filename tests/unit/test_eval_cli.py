# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""``locus eval``: run a suite from a file, save the report, diff two, gate a build.

The command exists so a suite can run from CI without a Python harness
around it: cases in a file, an agent named by ``module:attr``, the gate's
verdict as the exit code.
"""

from __future__ import annotations

import json
import textwrap
from typing import TYPE_CHECKING

import pytest

from locus.cli import main


if TYPE_CHECKING:
    from pathlib import Path


_APP = """
from types import SimpleNamespace

from locus.agent import Agent
from locus.core.messages import Message, ToolCall
from locus.models.base import ModelResponse
from locus.testing import FunctionModel, text


def make():
    return Agent(model=FunctionModel(lambda m, t: text("yes")), tools=[])


def make_no():
    return Agent(model=FunctionModel(lambda m, t: text("no")), tools=[])


class _JudgeModel:
    config = SimpleNamespace(model="judge-model")

    async def complete(self, messages, tools=None, **kwargs):
        call = ToolCall(
            id="v",
            name="submit_verdict",
            arguments={
                "dimensions": [{"key": "quality", "score": 85, "reason": "fine"}],
                "issues": [],
                "passed": True,
            },
        )
        return ModelResponse(message=Message.assistant(tool_calls=[call]), usage={})

    async def stream(self, *a, **k):
        raise NotImplementedError


model = _JudgeModel()
not_a_judge = 42
"""

_CASES = [
    {
        "name": "says_yes",
        "prompt": "say yes",
        "expected_output_contains": ["yes"],
        "tags": ["smoke"],
    },
    {"name": "says_no", "prompt": "say no", "expected_output_contains": ["no"], "tags": ["slow"]},
]


@pytest.fixture
def app(tmp_path: Path) -> Path:
    path = tmp_path / "app.py"
    path.write_text(textwrap.dedent(_APP), encoding="utf-8")
    return path


@pytest.fixture
def cases(tmp_path: Path) -> Path:
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(_CASES), encoding="utf-8")
    return path


def test_run_prints_the_report_and_the_gate_decides_the_exit_code(
    app: Path, cases: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["eval", "run", str(cases), "--agent", f"{app}:make"])
    out = capsys.readouterr().out
    assert code == 1
    assert "Eval Report: 1/2 passed" in out
    assert "[FAIL] says_no" in out
    assert "Gate: FAILED" in out
    assert "pass rate 0.50 below 1.00" in out


def test_a_relaxed_gate_passes_and_the_report_is_saved(
    app: Path, cases: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_file = tmp_path / "report.json"
    code = main(
        [
            "eval",
            "run",
            str(cases),
            "--agent",
            f"{app}:make",
            "--min-pass-rate",
            "0.5",
            "--report",
            str(out_file),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "Gate: passed" in out
    assert out_file.is_file()
    assert json.loads(out_file.read_text())["passed"] == 1


def test_tags_select_the_cases(app: Path, cases: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["eval", "run", str(cases), "--agent", f"{app}:make", "--tags", "smoke"])
    assert code == 0
    assert "1/1 passed" in capsys.readouterr().out
    assert main(["eval", "run", str(cases), "--agent", f"{app}:make", "--tags", "nothing"]) == 2
    assert "no cases to run" in capsys.readouterr().err


def test_summary_and_markdown_read_a_saved_report(
    app: Path, cases: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    saved = tmp_path / "report.json"
    main(["eval", "run", str(cases), "--agent", f"{app}:make", "--report", str(saved)])
    capsys.readouterr()

    assert main(["eval", "summary", str(saved)]) == 0
    assert "Eval Report: 1/2 passed" in capsys.readouterr().out
    assert main(["eval", "summary", str(saved), "--markdown"]) == 0
    assert "| says_yes | pass |" in capsys.readouterr().out


def test_diff_names_the_regression_and_fails(
    app: Path, cases: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    before, after = tmp_path / "before.json", tmp_path / "after.json"
    main(["eval", "run", str(cases), "--agent", f"{app}:make", "--report", str(before)])
    main(["eval", "run", str(cases), "--agent", f"{app}:make_no", "--report", str(after)])
    capsys.readouterr()

    code = main(["eval", "diff", str(before), str(after)])
    out = capsys.readouterr().out
    assert code == 1
    assert "1 regressed, 1 improved" in out
    assert "[REGRESSED] says_yes" in out
    assert main(["eval", "diff", str(before), str(after), "--markdown"]) == 1
    assert "| says_yes | regressed |" in capsys.readouterr().out


def test_gate_applies_to_a_saved_report_against_a_baseline(
    app: Path, cases: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    before, after = tmp_path / "before.json", tmp_path / "after.json"
    main(["eval", "run", str(cases), "--agent", f"{app}:make", "--report", str(before)])
    main(["eval", "run", str(cases), "--agent", f"{app}:make_no", "--report", str(after)])
    capsys.readouterr()

    code = main(["eval", "gate", str(after), "--baseline", str(before), "--min-pass-rate", "0.5"])
    out = capsys.readouterr().out
    assert code == 1
    assert "regressed against the baseline: says_yes" in out
    assert (
        main(
            [
                "eval",
                "gate",
                str(after),
                "--baseline",
                str(before),
                "--min-pass-rate",
                "0.5",
                "--allow-regressions",
            ]
        )
        == 0
    )


def test_a_judge_spec_may_name_a_model_to_wrap(
    app: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cases = tmp_path / "graded.json"
    cases.write_text(
        json.dumps([{"name": "graded", "prompt": "p", "rubric": "Passes if it answers."}]),
        encoding="utf-8",
    )
    rubric = tmp_path / "rubric.json"
    rubric.write_text(
        json.dumps(
            {
                "dimensions": [{"key": "quality"}],
                "grade_thresholds": {"A": 90, "B": 75, "C": 60, "D": 45, "F": 0},
            }
        ),
        encoding="utf-8",
    )
    plain = tmp_path / "plain.json"
    plain.write_text(json.dumps([{"name": "graded", "prompt": "p"}]), encoding="utf-8")

    code = main(
        [
            "eval",
            "run",
            str(plain),
            "--agent",
            f"{app}:make",
            "--judge",
            f"{app}:model",
            "--rubric",
            str(rubric),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "grade B" in out  # 85 under the default bands

    code = main(
        ["eval", "run", str(cases), "--agent", f"{app}:make", "--judge", f"{app}:not_a_judge"]
    )
    assert code == 2
    assert "is not a judge" in capsys.readouterr().err


def test_usage_errors_are_reported_without_a_traceback(
    app: Path, cases: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["eval", "run", str(cases), "--agent", f"{app}:missing"]) == 2
    assert "has no attribute 'missing'" in capsys.readouterr().err
    assert main(["eval", "run", str(tmp_path / "nope.json"), "--agent", f"{app}:make"]) == 2
    assert "cannot read cases" in capsys.readouterr().err
    assert main(["eval", "summary", str(tmp_path / "nope.json")]) == 2
    assert "cannot read report" in capsys.readouterr().err
