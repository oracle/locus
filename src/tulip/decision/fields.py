# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""The questions a decision model answers, the answers it gives, and the wire format.

A *field* is one question with a fixed list of answers: a :class:`Choice` among
named options, a :class:`YesNo`, or a :class:`Score` on ordered levels. A
provider answers each field of one input with a probability for every listed
answer — an :class:`Answer` — and the answers to one input form a
:class:`Decision`.

The wire format is :func:`render` and :data:`SYSTEM_PROMPT`. It is part of the
contract, not an implementation detail: a head trained on it is served by any
:class:`~tulip.decision.LogprobDecider` with no glue, and a change of one
character makes every trained head answer a question it was not trained on.
``tests/unit/test_decision.py`` pins it verbatim.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


__all__ = [
    "LETTERS",
    "SYSTEM_PROMPT",
    "YES_NO",
    "Answer",
    "Choice",
    "Decision",
    "DecisionError",
    "Field",
    "Score",
    "YesNo",
    "answer_from_logprobs",
    "check_fields",
    "render",
]

#: The answer codes, in order. One uppercase letter is one token in every
#: tokenizer in use, so an answer is read from the first generated position.
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

#: The system message every request carries.
SYSTEM_PROMPT = (
    "You answer one question about the input. "
    "Reply with the letter of one listed answer and nothing else."
)

#: A :class:`YesNo` field's labels, in this order: ``A) yes``, ``B) no``.
YES_NO = ("yes", "no")


class DecisionError(RuntimeError):
    """A decision could not be made: the server failed, or answered off the list.

    Raised rather than guessed. A provider that cannot read an answer must not
    return one, because a caller thresholding a probability would act on it.
    """


def render(question: str, labels: Sequence[str], text: str) -> str:
    """The user message for one field asked of one input.

    The input comes first, so every field asked of the same input shares a
    prefix and a server with prefix caching computes it once.
    """
    lines = ["[input]", text.strip(), "", "[question]", question.strip(), "", "[answers]"]
    lines += [f"{letter}) {label}" for letter, label in zip(LETTERS, labels, strict=False)]
    lines += ["", "Answer with one letter."]
    return "\n".join(lines)


def _check(name: str, question: str, labels: Sequence[str]) -> None:
    if not name or not name.strip():
        raise ValueError("a field needs a name")
    if not question or not question.strip():
        raise ValueError(f"field {name!r} needs a question")
    if len(labels) < 2:
        raise ValueError(f"field {name!r} needs at least two answers")
    if len(labels) > len(LETTERS):
        raise ValueError(f"field {name!r} has {len(labels)} answers; at most {len(LETTERS)}")
    for label in labels:
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"field {name!r} has an empty answer")
        if "\n" in label:
            raise ValueError(f"field {name!r}: an answer may not span lines ({label!r})")
    if len(set(labels)) != len(labels):
        raise ValueError(f"field {name!r} lists an answer twice")


@dataclass(frozen=True)
class Choice:
    """Pick one of ``options``."""

    name: str
    question: str
    options: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "options", tuple(self.options))
        _check(self.name, self.question, self.options)

    @property
    def labels(self) -> tuple[str, ...]:
        """The answers, in the order their letters are assigned."""
        return self.options

    def render(self, text: str) -> str:
        """This field's user message for ``text``."""
        return render(self.question, self.labels, text)


@dataclass(frozen=True)
class YesNo:
    """Answer yes or no. Rendered as ``A) yes`` and ``B) no``."""

    name: str
    question: str

    def __post_init__(self) -> None:
        _check(self.name, self.question, YES_NO)

    @property
    def labels(self) -> tuple[str, ...]:
        """Always :data:`YES_NO`."""
        return YES_NO

    def render(self, text: str) -> str:
        """This field's user message for ``text``."""
        return render(self.question, self.labels, text)


@dataclass(frozen=True)
class Score:
    """Place the input on ordered ``levels``, lowest first.

    ``levels`` is the level labels, or an int ``n`` for the levels ``"1"`` to
    ``"n"``. The answer's :attr:`Answer.expected` is the probability-weighted
    level, normalised to [0, 1].
    """

    name: str
    question: str
    levels: tuple[str, ...] | int

    def __post_init__(self) -> None:
        levels: tuple[str, ...]
        if isinstance(self.levels, int):
            levels = tuple(str(i) for i in range(1, self.levels + 1))
        else:
            levels = tuple(self.levels)
        object.__setattr__(self, "levels", levels)
        _check(self.name, self.question, self.labels)

    @property
    def labels(self) -> tuple[str, ...]:
        """The levels, lowest first."""
        assert isinstance(self.levels, tuple)  # noqa: S101 - normalised in __post_init__
        return self.levels

    def render(self, text: str) -> str:
        """This field's user message for ``text``."""
        return render(self.question, self.labels, text)


#: Any one question a decision model answers.
Field = Choice | YesNo | Score


def _kind(question: Field) -> str:
    if isinstance(question, YesNo):
        return "yesno"
    if isinstance(question, Score):
        return "score"
    return "choice"


@dataclass(frozen=True)
class Answer:
    """One field's answer: a probability for every listed label.

    ``distribution`` is renormalised over the listed labels. ``coverage`` is the
    probability mass the listed letters had *before* renormalising: near 1 when
    the model answered the question, low when it wanted to say something else.
    A low-coverage answer is a model out of its depth, whatever its argmax.
    """

    field: str
    kind: str
    labels: tuple[str, ...]
    distribution: Mapping[str, float]
    coverage: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "distribution", MappingProxyType(dict(self.distribution)))

    @property
    def label(self) -> str:
        """The most probable label (the first listed, on a tie)."""
        return max(self.labels, key=lambda lbl: self.distribution[lbl])

    @property
    def probability(self) -> float:
        """The probability of :attr:`label`."""
        return self.distribution[self.label]

    @property
    def margin(self) -> float:
        """How far the winner led the runner-up. 0 is a coin toss."""
        ranked = sorted(self.distribution.values(), reverse=True)
        return ranked[0] - ranked[1]

    def p(self, label: str) -> float:
        """The probability of ``label``. A label off the list is an error."""
        if label not in self.distribution:
            raise KeyError(f"{self.field!r} has no answer {label!r}; its answers are {self.labels}")
        return self.distribution[label]

    @property
    def p_yes(self) -> float:
        """P(yes), for a :class:`YesNo` field."""
        if self.kind != "yesno":
            raise ValueError(f"{self.field!r} is not a yes/no question")
        return self.distribution["yes"]

    @property
    def expected(self) -> float:
        """The probability-weighted level of a :class:`Score`, in [0, 1]."""
        if self.kind != "score":
            raise ValueError(f"{self.field!r} is not a score")
        top = len(self.labels) - 1
        return sum(i / top * self.distribution[lbl] for i, lbl in enumerate(self.labels))

    def flagged(self, threshold: float, label: str | None = None) -> bool:
        """Whether P(``label``) is at least ``threshold``. ``label`` defaults to yes.

        A threshold belongs to the weights it was calibrated on: carry it with
        the model, and recalibrate when the model changes.
        """
        if label is None:
            if self.kind != "yesno":
                raise ValueError(f"{self.field!r}: name the label to threshold")
            label = "yes"
        return self.p(label) >= threshold

    def as_record(self) -> dict[str, Any]:
        """The answer as plain data, for an audit payload."""
        return {
            "label": self.label,
            "probability": round(self.probability, 6),
            "distribution": {k: round(v, 6) for k, v in self.distribution.items()},
            "coverage": round(self.coverage, 6),
        }


@dataclass(frozen=True)
class Decision:
    """The answers to every field asked of one input."""

    answers: Mapping[str, Answer]
    model: str
    provider: str
    latency_ms: float
    #: Extra facts a provider or router attaches (e.g. the tenant). Plain data.
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "answers", MappingProxyType(dict(self.answers)))
        object.__setattr__(self, "meta", MappingProxyType(dict(self.meta)))

    def __getitem__(self, name: str) -> Answer:
        try:
            return self.answers[name]
        except KeyError:
            raise KeyError(f"no field {name!r} was asked; asked: {sorted(self.answers)}") from None

    def __contains__(self, name: object) -> bool:
        return name in self.answers

    def __iter__(self) -> Iterator[str]:
        return iter(self.answers)

    def __len__(self) -> int:
        return len(self.answers)

    def as_record(self) -> dict[str, Any]:
        """The decision as plain data: labels and probabilities, never the input."""
        return {
            "model": self.model,
            "provider": self.provider,
            "latency_ms": round(self.latency_ms, 3),
            "answers": {name: answer.as_record() for name, answer in self.answers.items()},
        }


def check_fields(fields: Sequence[Field]) -> None:
    """Refuse an empty request or two fields with one name."""
    if not fields:
        raise ValueError("ask at least one field")
    names = [f.name for f in fields]
    if len(set(names)) != len(names):
        raise ValueError(f"field names must be unique within a decision: {names}")


def answer_from_logprobs(
    question: Field, top_logprobs: Mapping[str, float] | Iterable[tuple[str, float]]
) -> Answer:
    """Read one field's answer from the first position's top log-probabilities.

    ``top_logprobs`` is ``(token text, log-probability)`` pairs, or a mapping of
    them. Whitespace around a token is ignored, so ``" A"`` and ``"A"`` both
    count for ``A`` (and both are summed when a server lists both: they are
    different tokens, each a way of giving that answer). Tokens that are not one of this
    field's letters are mass the model put elsewhere: they lower
    :attr:`Answer.coverage` and are otherwise ignored.

    Raises :class:`DecisionError` when none of the field's letters is present.
    """
    letters = {LETTERS[i]: label for i, label in enumerate(question.labels)}
    mass = dict.fromkeys(question.labels, 0.0)
    pairs = top_logprobs.items() if isinstance(top_logprobs, Mapping) else top_logprobs
    for token, logprob in pairs:
        label = letters.get(token.strip())
        if label is not None and math.isfinite(logprob):
            mass[label] += math.exp(logprob)
    coverage = sum(mass.values())
    if coverage <= 0.0:
        raise DecisionError(
            f"field {question.name!r}: none of the letters "
            f"{''.join(letters)} was among the model's top answers"
        )
    return Answer(
        field=question.name,
        kind=_kind(question),
        labels=question.labels,
        distribution={label: m / coverage for label, m in mass.items()},
        coverage=min(1.0, coverage),
    )
