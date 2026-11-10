# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Cases as data: load a suite from a file, write one back.

A suite that lives in code is reviewed as code and run from Python. One
that lives in a file beside the rubric can be reviewed by the people who
own the behaviour, run from ``locus eval``, and fingerprinted so two reports
are known to be of the same thing.

Three shapes are read, chosen by the file's suffix:

- ``.json``: a list of case objects, or an object ``{"rubric": ...,
  "cases": [...]}`` whose rubric applies to every case that names none;
- ``.jsonl``: one case object per line;
- ``.yaml`` / ``.yml``: the same two shapes as JSON, with ``pyyaml``.

A case's ``rubric`` may be a string, a rubric object, or absent.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from locus.evaluation.framework import EvalCase
from locus.evaluation.rubric import Rubric


__all__ = ["dump_cases", "load_cases"]


def _read(path: Path) -> Any:
    raw = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return [json.loads(line) for line in raw.splitlines() if line.strip()]
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore[import-untyped]  # noqa: PLC0415 — optional dependency
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError("loading a YAML suite needs pyyaml: pip install pyyaml") from exc
        return yaml.safe_load(raw)
    return json.loads(raw)


def _rubric(value: Any, fallback: Rubric | None) -> Rubric | str | None:
    if value is None:
        return fallback
    if isinstance(value, str | Rubric):
        return value
    if isinstance(value, Mapping):
        return Rubric.model_validate(dict(value))
    raise TypeError(f"a case rubric must be a string or an object, got {type(value).__name__}")


def load_cases(path: str | Path, *, rubric: Rubric | None = None) -> list[EvalCase]:
    """Read the cases in ``path``.

    Args:
        path: A ``.json``, ``.jsonl``, ``.yaml`` or ``.yml`` file.
        rubric: Applied to every case that names no rubric of its own, after
            the file's own top-level rubric when it has one.
    """
    file = Path(path)
    data = _read(file)
    file_rubric: Rubric | None = rubric
    if isinstance(data, Mapping):
        top = data.get("rubric")
        if top is not None:
            resolved = _rubric(top, rubric)
            file_rubric = Rubric.from_text(resolved) if isinstance(resolved, str) else resolved
        items = data.get("cases")
    else:
        items = data
    if not isinstance(items, Sequence) or isinstance(items, str | bytes):
        raise TypeError(f"{file}: expected a list of cases or an object with a 'cases' list")

    cases: list[EvalCase] = []
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise TypeError(f"{file}: case {index} is {type(item).__name__}, not an object")
        fields = dict(item)
        fields["rubric"] = _rubric(fields.get("rubric"), file_rubric)
        cases.append(EvalCase.model_validate(fields))
    return cases


def dump_cases(cases: Sequence[EvalCase], path: str | Path) -> None:
    """Write ``cases`` as a JSON list that :func:`load_cases` reads back.

    A :class:`~locus.evaluation.rubric.Rubric` on a case is written as its
    object form; a string rubric as the string.
    """
    file = Path(path)
    payload = [case.model_dump(mode="json", exclude_defaults=True) for case in cases]
    file.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
