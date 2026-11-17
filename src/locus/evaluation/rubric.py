# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""A rubric as data: dimensions, weights, grade thresholds, a revision.

A rubric written as one sentence ("passes if it names the evidence") asks the
judge for one boolean and one number, and the judge's number becomes the
score. That is the probabilistic part deciding what the deterministic part
could have decided. A :class:`Rubric` keeps the judge to the part only a
reader can do, scoring each named dimension, and leaves the arithmetic to
code: the weighted composite, the letter, the band a cap moves a verdict to,
and the revision string that says which rubric produced a number.

    rubric = Rubric(
        dimensions=[
            Dimension(key="root_cause", weight=0.5,
                      guidance="Names the failing component and the mechanism."),
            Dimension(key="evidence", weight=0.3,
                      guidance="Every figure it states appears in a tool result."),
            Dimension(key="actionable", weight=0.2,
                      guidance="The recommendation can be executed as written."),
        ],
        grade_thresholds=Rubric.default_grades(),
    )
    rubric.composite({"root_cause": 80, "evidence": 100, "actionable": 60})  # 82
    rubric.grade_for(82)                                                     # "B"

Two readings are deliberately absent. Missing dimensions score zero rather
than being skipped: a judge that forgot a dimension has not graded it, and
renormalising over the rest would hide that. And the composite rounds half
up, so the same scores give the same letter on every platform.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


__all__ = ["DEFAULT_GRADE_THRESHOLDS", "Cap", "Dimension", "Rubric"]


#: The letter bands most rubrics reach for. Offered, not imposed: a rubric
#: without ``grade_thresholds`` reports no letter at all.
DEFAULT_GRADE_THRESHOLDS: dict[str, int] = {"A": 90, "B": 75, "C": 60, "D": 45, "F": 0}

#: Conditions a :class:`Cap` may name. Each is decided by code from the
#: evidence the runner collected, never by the judge.
CAP_TRIGGERS: frozenset[str] = frozenset({"contradicted_claim", "receipts_failed", "ungrounded"})

_REVISION_CHARS = 12


class Dimension(BaseModel):
    """One axis the judge scores 0-100.

    Attributes:
        key: Identifier the judge returns the score under. Stable across
            rubric edits so reports stay comparable.
        label: Human-readable name for reports. Defaults to the key.
        weight: Relative weight in the composite. Weights are normalised, so
            ``3, 2, 1`` and ``0.5, 0.33, 0.17`` grade the same.
        guidance: What the judge reads when scoring this axis. Say what a
            90 looks like and what a 50 looks like; a judge handed a bare
            label invents its own scale.
    """

    key: str = Field(min_length=1, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    label: str = ""
    weight: float = Field(default=1.0, gt=0)
    guidance: str = ""

    @property
    def title(self) -> str:
        """The label, or the key when no label was given."""
        return self.label or self.key


class Cap(BaseModel):
    """A ceiling the verdict's grade drops to when a code-decided condition holds.

    The judge never applies a cap; the runner does, from what it measured:
    ``contradicted_claim`` when a claim check came back contradicted,
    ``receipts_failed`` when a figure in the answer appears in no tool result,
    ``ungrounded`` when the grounding score fell below its threshold.
    """

    when: str
    grade: str = Field(min_length=1)

    @field_validator("when")
    @classmethod
    def _known_trigger(cls, value: str) -> str:
        if value not in CAP_TRIGGERS:
            raise ValueError(
                f"unknown cap trigger {value!r}; expected one of {sorted(CAP_TRIGGERS)}"
            )
        return value


class Rubric(BaseModel):
    """Dimensions, weights and grade bands, with the arithmetic that reads them.

    Attributes:
        dimensions: At least one :class:`Dimension`; keys must be unique.
        grade_thresholds: ``{letter: minimum composite}``. The letter with the
            highest threshold at or below the composite wins; a composite
            below every threshold takes the lowest letter. ``None`` means the
            rubric reports no letter.
        pass_score: Composite at or above which a verdict passes. 60 by
            default, the floor of a C in :data:`DEFAULT_GRADE_THRESHOLDS`.
        caps: Ceilings applied by the runner on code-decided conditions.
        version: Free-form tag carried into :meth:`revision`.
    """

    dimensions: list[Dimension] = Field(min_length=1)
    grade_thresholds: dict[str, int] | None = None
    pass_score: int = Field(default=60, ge=0, le=100)
    caps: list[Cap] = Field(default_factory=list)
    version: str = "1"

    @field_validator("dimensions")
    @classmethod
    def _unique_keys(cls, value: list[Dimension]) -> list[Dimension]:
        seen: set[str] = set()
        for dimension in value:
            if dimension.key in seen:
                raise ValueError(f"duplicate dimension key {dimension.key!r}")
            seen.add(dimension.key)
        return value

    @model_validator(mode="after")
    def _thresholds_in_range(self) -> Rubric:
        if self.grade_thresholds is not None:
            if not self.grade_thresholds:
                raise ValueError("grade_thresholds must name at least one letter")
            for letter, floor in self.grade_thresholds.items():
                if not 0 <= floor <= 100:
                    raise ValueError(f"threshold for {letter!r} must be within 0-100, got {floor}")
            for cap in self.caps:
                if cap.grade not in self.grade_thresholds:
                    raise ValueError(
                        f"cap grade {cap.grade!r} is not one of the rubric's letters "
                        f"{sorted(self.grade_thresholds)}"
                    )
        elif self.caps:
            raise ValueError("caps need grade_thresholds to name a letter to drop to")
        return self

    # -- construction helpers -------------------------------------------------

    @classmethod
    def default_grades(cls) -> dict[str, int]:
        """A copy of :data:`DEFAULT_GRADE_THRESHOLDS`."""
        return dict(DEFAULT_GRADE_THRESHOLDS)

    @classmethod
    def from_text(cls, text: str, *, key: str = "quality") -> Rubric:
        """One dimension carrying ``text`` as its guidance, and no letters.

        This is how a plain-string rubric on an :class:`~locus.evaluation.EvalCase`
        is read when a :class:`Rubric` is needed for its revision or prompt
        block. It does not change how the judge treats a string rubric.
        """
        return cls(dimensions=[Dimension(key=key, guidance=text.strip())])

    @classmethod
    def load(cls, path: str | Path) -> Rubric:
        """Read a rubric from a ``.json``, ``.yaml`` or ``.yml`` file.

        YAML needs the optional ``pyyaml`` package; the error names it.
        """
        file = Path(path)
        raw = file.read_text(encoding="utf-8")
        if file.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore[import-untyped]  # noqa: PLC0415 — optional dependency
            except ImportError as exc:  # pragma: no cover - depends on the environment
                raise ImportError("loading a YAML rubric needs pyyaml: pip install pyyaml") from exc
            data = yaml.safe_load(raw)
        else:
            data = json.loads(raw)
        if not isinstance(data, dict):
            raise TypeError(
                f"{file}: a rubric file must hold one object, got {type(data).__name__}"
            )
        return cls.model_validate(data)

    # -- arithmetic -----------------------------------------------------------

    @property
    def keys(self) -> list[str]:
        """Dimension keys in rubric order."""
        return [d.key for d in self.dimensions]

    @property
    def weights(self) -> dict[str, float]:
        """``{key: normalised weight}``; the values sum to 1."""
        total = sum(d.weight for d in self.dimensions)
        return {d.key: d.weight / total for d in self.dimensions}

    def composite(self, scores: Mapping[str, int | float]) -> int:
        """Weighted composite of per-dimension scores, 0-100, rounded half up.

        A dimension absent from ``scores`` counts as 0: it was not graded,
        and that must lower the composite rather than vanish from it. Keys
        the rubric does not know are ignored.
        """
        total = Decimal(0)
        for key, weight in self.weights.items():
            raw = scores.get(key, 0) or 0
            clamped = min(100, max(0, float(raw)))
            total += Decimal(str(weight)) * Decimal(str(clamped))
        rounded = int(total.quantize(Decimal(1), rounding=ROUND_HALF_UP))
        return min(100, max(0, rounded))

    def grade_for(self, score: float) -> str | None:
        """The letter for a composite, or ``None`` when the rubric has no letters."""
        if self.grade_thresholds is None:
            return None
        ordered = sorted(self.grade_thresholds.items(), key=lambda item: -item[1])
        for letter, floor in ordered:
            if score >= floor:
                return letter
        return ordered[-1][0]

    def band_top(self, grade: str) -> int:
        """Highest composite that still maps to ``grade``.

        One below the next letter's threshold; 100 for the top letter, and
        100 for a letter the rubric does not know (no cap).
        """
        if self.grade_thresholds is None or grade not in self.grade_thresholds:
            return 100
        own = self.grade_thresholds[grade]
        higher = [floor for floor in self.grade_thresholds.values() if floor > own]
        return min(higher) - 1 if higher else 100

    def rank(self, grade: str) -> int | None:
        """Position of a letter, best first (``A`` is 0), or ``None`` if unknown."""
        if self.grade_thresholds is None or grade not in self.grade_thresholds:
            return None
        ordered = sorted(self.grade_thresholds, key=lambda letter: -self.grade_thresholds[letter])  # type: ignore[index]
        return ordered.index(grade)

    # -- identity -------------------------------------------------------------

    def revision(self, prompt_text: str | None = None) -> str:
        """A short hash of the rubric and, when given, the judge prompt.

        Two verdicts are comparable only when their revisions match: the same
        dimensions, the same weights, the same thresholds, the same prompt.
        Editing any of them moves the revision, which is the point.
        """
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256()
        digest.update(canonical.encode("utf-8"))
        if prompt_text:
            digest.update(b"\n")
            digest.update(prompt_text.encode("utf-8"))
        return digest.hexdigest()[:_REVISION_CHARS]

    def prompt_block(self) -> str:
        """The rubric rendered for the judge: one line per dimension, then the bands."""
        lines = ["Score each dimension from 0 to 100."]
        for dimension, weight in zip(self.dimensions, self.weights.values(), strict=True):
            guidance = f" {dimension.guidance}" if dimension.guidance else ""
            lines.append(f"- {dimension.key} (weight {weight:.0%}): {dimension.title}.{guidance}")
        if self.grade_thresholds:
            bands = ", ".join(
                f"{letter} >= {floor}"
                for letter, floor in sorted(self.grade_thresholds.items(), key=lambda i: -i[1])
            )
            lines.append(f"Letter bands on the weighted composite: {bands}.")
        lines.append(f"A composite of {self.pass_score} or more passes.")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form, the shape :meth:`load` reads."""
        return self.model_dump(mode="json", exclude_defaults=False)
