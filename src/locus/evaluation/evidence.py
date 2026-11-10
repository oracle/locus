# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""The run's own evidence, with ids a claim can cite, and receipts settled by code.

Structural checks look at the answer and at which tools ran. Neither notices
the most common quality failure of an agent that reads data: a figure in the
answer that no tool returned. The answer names the right tool, mentions the
right words, and states a number it made up.

An :class:`EvidenceLedger` is the run's tool executions laid out as rows with
stable ids — ``E0``, ``E1``, ... in execution order, plus ``P`` for the
prompt the agent was given — so a claim can cite the row that backs it and a
check can say which row it looked in. Two things are then decided by code,
with no model call:

- :func:`settle_receipts` extracts every figure and identifier from the
  answer and looks each one up in the rows. A value found nowhere is a
  missing receipt. Variants a tool output legitimately takes (``1,200`` and
  ``1200``, ``5`` and ``5.0``, ``71.85`` against a row that holds
  ``71.8506``) count as the same value; a different number does not.
- :func:`settle_claims` confirms a claim outright when every figure it states
  sits in one row, and hands the rest to the judge as open claims. The judge
  then cites rows by id and quotes what it read, and code checks that the
  quote really is in the row it cites.

The gate's refusals are evidence too. When :func:`~locus.control.gate_tool`
refuses a call it hands the model a JSON object whose ``status`` is
``denied`` or ``held_for_approval`` (a public contract). A row whose result is
such an object is a refusal, not an execution, which is what lets an eval
case assert that a forbidden tool never ran and that the gate answered.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from typing import Any, Literal

from pydantic import BaseModel, Field


__all__ = [
    "REFUSAL_STATUSES",
    "ClaimCheck",
    "EvidenceLedger",
    "EvidenceRow",
    "ReceiptCheck",
    "ReceiptRow",
    "ToolCallExpectation",
    "extract_values",
    "refusal_status",
    "settle_claims",
    "settle_receipts",
    "value_variants",
]


#: The ``status`` values the gate puts on a refused call's result.
REFUSAL_STATUSES: frozenset[str] = frozenset({"denied", "held_for_approval"})

#: The prompt's row id. Figures the agent was *given* are receipted by it.
PROMPT_ROW_ID = "P"

_MAX_ROW_CHARS = 2000
#: Bare integers up to this value are counts in prose ("3 retries"), not
#: readings, and are not asked to carry a receipt.
_SMALL_COUNT_MAX = 10

# A number: optional sign, digits with optional thousands commas, optional
# decimals, optional percent sign. Not glued to a word, and not one segment
# of a dotted name or version (``1.2.3`` is none of ``1.2``, ``2`` or ``3``);
# a sentence-ending period after it is fine.
_NUMBER = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?%?(?!\w)(?!\.\d)")
# An ISO date or timestamp, taken whole so its parts are not read as numbers.
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?Z?)?\b")
# An identifier such as ``ord-4821``, ``SOC-4821`` or ``job_ln889e``: a
# letter prefix, a separator, and at least one digit somewhere after it.
_IDENT = re.compile(r"\b[A-Za-z]{2,}[-_][A-Za-z0-9-]*\d[A-Za-z0-9-]*\b")
# A quoted string of reasonable length, in any common quote style.
_QUOTED = re.compile(r"[\"'`“”‘’]([^\"'`“”‘’\n]{2,80})[\"'`“”‘’]")
_WS = re.compile(r"\s+")


def refusal_status(result: str | None) -> str | None:
    """The gate's refusal status carried by a tool result, or ``None``.

    Reads the public refusal contract of :mod:`locus.control.gate`: a JSON
    object with ``"status"`` set to ``"denied"`` or ``"held_for_approval"``.
    Anything else, including ordinary JSON a tool returned, is not a refusal.
    """
    if not result:
        return None
    text = result.strip()
    if not text.startswith("{"):
        return None
    try:
        data = json.loads(text)
    except ValueError:
        return None
    status = data.get("status") if isinstance(data, dict) else None
    return status if isinstance(status, str) and status in REFUSAL_STATUSES else None


def _normalise(text: str) -> str:
    return _WS.sub(" ", text).strip().lower()


def _as_number(token: str) -> float | None:
    cleaned = token.replace(",", "").rstrip("%")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _decimals(token: str) -> int:
    cleaned = token.replace(",", "").rstrip("%")
    return len(cleaned.split(".", 1)[1]) if "." in cleaned else 0


def extract_values(text: str) -> list[str]:
    """Every figure and identifier in ``text`` that should have a receipt.

    Returns, without duplicates: identifiers that carry a digit (``ord-4821``,
    ``db-07``), ISO dates and timestamps, quoted strings, then numbers (with
    a percent sign or decimals, or above a small-count floor). Small bare
    integers are left alone: "3 retries" is a count in prose, not a reading.
    The digits inside an identifier or a date are not read again as numbers.
    """
    seen: set[str] = set()
    values: list[str] = []

    def add(value: str) -> None:
        if value not in seen:
            seen.add(value)
            values.append(value)

    masked = text
    for pattern in (_IDENT, _DATE):
        for match in pattern.finditer(text):
            add(match.group(0))
        masked = pattern.sub(lambda m: " " * len(m.group(0)), masked)
    for match in _QUOTED.finditer(text):
        add(match.group(1).strip())
    for match in _NUMBER.finditer(masked):
        token = match.group(0)
        number = _as_number(token)
        if number is None:
            continue
        bare_small = "%" not in token and "." not in token and abs(number) <= _SMALL_COUNT_MAX
        if bare_small:
            continue
        add(token)
    return values


def value_variants(value: str) -> list[str]:
    """The spellings a tool output may use for ``value``.

    ``1,200`` and ``1200``; ``5`` and ``5.0``; ``71.85%`` and ``71.85``; a
    signed number without its sign. The first entry is the value itself.
    """
    variants: list[str] = [value]

    def add(candidate: str) -> None:
        if candidate and candidate not in variants:
            variants.append(candidate)

    stripped = value.strip().rstrip("%")
    add(stripped)
    no_commas = stripped.replace(",", "")
    add(no_commas)
    unsigned = no_commas.lstrip("+-")
    add(unsigned)
    number = _as_number(unsigned)
    if number is not None:
        if number.is_integer():
            add(str(int(number)))
            add(f"{int(number)}.0")
        else:
            add(repr(number))
    return variants


class ToolCallExpectation(BaseModel):
    """One tool call the run must have made, with the arguments that matter.

    ``arguments`` is a subset: every key given must be present on the call
    with an equal value, and keys not given are not checked.
    """

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)

    def matches(self, tool_name: str, arguments: Any) -> bool:
        """Whether an execution of ``tool_name`` with ``arguments`` satisfies this."""
        if tool_name != self.name:
            return False
        if not self.arguments:
            return True
        if not isinstance(arguments, dict):
            return False
        return all(
            key in arguments and arguments[key] == value for key, value in self.arguments.items()
        )


class EvidenceRow(BaseModel):
    """One piece of evidence the run had: a tool execution, or the prompt."""

    id: str
    source: Literal["tool", "prompt"] = "tool"
    tool_name: str = ""
    arguments: dict[str, Any] = Field(default_factory=dict)
    text: str = Field(default="", description="The tool's result, or the prompt itself")
    error: str | None = None
    refusal: str | None = Field(
        default=None, description="The gate's status when the call was refused, else None"
    )

    @property
    def executed(self) -> bool:
        """The tool ran and returned: no error, and no refusal from the gate."""
        return self.source == "tool" and self.error is None and self.refusal is None

    def contains(self, value: str) -> bool:
        """Whether ``value``, in any of its spellings, appears in this row's text.

        Numbers also match a figure in the row that rounds to the value at
        the value's own precision, so ``71.9`` finds ``71.8506``.
        """
        haystack = _normalise(self.text)
        if any(_normalise(variant) in haystack for variant in value_variants(value)):
            return True
        wanted = _as_number(value)
        if wanted is None:
            return False
        places = _decimals(value)
        return any(
            (number := _as_number(token)) is not None and round(number, places) == wanted
            for token in _NUMBER.findall(self.text)
        )

    def quotes(self, observed: str) -> bool:
        """Whether ``observed`` is a quote from this row (whitespace and case folded)."""
        needle = _normalise(observed)
        return bool(needle) and needle in _normalise(self.text)


class EvidenceLedger(BaseModel):
    """The run's evidence as rows with ids: ``E<i>`` per tool execution, ``P`` for the prompt."""

    rows: list[EvidenceRow] = Field(default_factory=list)

    @classmethod
    def from_executions(
        cls, executions: Iterable[Any], *, prompt: str | None = None
    ) -> EvidenceLedger:
        """Build the ledger from ``ToolExecution``-shaped objects, in order."""
        rows: list[EvidenceRow] = []
        if prompt:
            rows.append(EvidenceRow(id=PROMPT_ROW_ID, source="prompt", text=prompt))
        for index, execution in enumerate(executions):
            result = getattr(execution, "result", None)
            text = "" if result is None else str(result)
            arguments = getattr(execution, "arguments", None)
            rows.append(
                EvidenceRow(
                    id=f"E{index}",
                    tool_name=str(getattr(execution, "tool_name", "") or ""),
                    arguments=dict(arguments) if isinstance(arguments, dict) else {},
                    text=text,
                    error=getattr(execution, "error", None),
                    refusal=refusal_status(text),
                )
            )
        return cls(rows=rows)

    @classmethod
    def from_result(cls, result: Any, *, prompt: str | None = None) -> EvidenceLedger:
        """Build the ledger from an ``AgentResult`` (anything with ``tool_executions``)."""
        return cls.from_executions(getattr(result, "tool_executions", ()) or (), prompt=prompt)

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def ids(self) -> list[str]:
        return [row.id for row in self.rows]

    def get(self, row_id: str) -> EvidenceRow | None:
        return next((row for row in self.rows if row.id == row_id), None)

    @property
    def tool_rows(self) -> list[EvidenceRow]:
        return [row for row in self.rows if row.source == "tool"]

    def executions_of(self, tool_name: str) -> list[EvidenceRow]:
        """Rows where ``tool_name`` actually ran: no error, no refusal."""
        return [row for row in self.tool_rows if row.tool_name == tool_name and row.executed]

    def refusals_of(self, tool_name: str) -> list[EvidenceRow]:
        """Rows where the gate refused ``tool_name``."""
        return [row for row in self.tool_rows if row.tool_name == tool_name and row.refusal]

    def find(self, value: str) -> list[str]:
        """Ids of every row that carries ``value``."""
        return [row.id for row in self.rows if row.contains(value)]

    def corpus(self, *, max_chars_per_row: int = _MAX_ROW_CHARS) -> str:
        """The rows rendered for a judge, one per line, id first."""
        lines: list[str] = []
        for row in self.rows:
            if row.source == "prompt":
                head = f"{row.id} [prompt]"
            else:
                args = (
                    json.dumps(row.arguments, sort_keys=True, default=str) if row.arguments else ""
                )
                head = f"{row.id} [{row.tool_name}({args})]"
                if row.refusal:
                    head += f" refused: {row.refusal}"
                elif row.error:
                    head += f" error: {row.error}"
            body = row.text
            if len(body) > max_chars_per_row:
                body = (
                    body[:max_chars_per_row] + f" [cut: {len(row.text) - max_chars_per_row} chars]"
                )
            lines.append(f"{head}: {body}")
        return "\n".join(lines)


class ReceiptRow(BaseModel):
    """One value from the answer and the rows that carry it."""

    value: str
    evidence_ids: list[str] = Field(default_factory=list)


class ReceiptCheck(BaseModel):
    """Every figure and identifier in the answer, and where each was found."""

    values: list[ReceiptRow] = Field(default_factory=list)

    @property
    def missing(self) -> list[str]:
        return [row.value for row in self.values if not row.evidence_ids]

    @property
    def passed(self) -> bool:
        return not self.missing

    @property
    def summary(self) -> str:
        """``found/total``, the label the runner records the check under."""
        found = sum(1 for row in self.values if row.evidence_ids)
        return f"{found}/{len(self.values)}"


def settle_receipts(output: str, ledger: EvidenceLedger) -> ReceiptCheck:
    """Look every figure and identifier in ``output`` up in the ledger.

    No model call. A value found in no row is a missing receipt: the answer
    states something the run never read and was never given.
    """
    return ReceiptCheck(
        values=[
            ReceiptRow(value=value, evidence_ids=ledger.find(value))
            for value in extract_values(output)
        ]
    )


class ClaimCheck(BaseModel):
    """One claim about the run, settled by code or by the judge."""

    claim: str
    verdict: Literal["confirmed", "contradicted", "unverifiable"]
    settled_by: Literal["code", "judge"]
    evidence_ids: list[str] = Field(default_factory=list)
    observed: str = Field(default="", description="What the cited row says, quoted")
    reason: str = ""


def settle_claims(
    claims: Sequence[str], ledger: EvidenceLedger
) -> tuple[list[ClaimCheck], list[str]]:
    """Confirm by code every claim whose figures the ledger carries; return the rest.

    Returns ``(settled, open_claims)``. A claim is confirmed when each figure
    and identifier it states is in some row, and the check cites those rows.
    A claim with no figure to look up, or with a figure the ledger does not
    carry, is open: only a reader can settle it, and the judge gets it with
    the ledger as its evidence.
    """
    settled: list[ClaimCheck] = []
    open_claims: list[str] = []
    for claim in claims:
        values = extract_values(claim)
        if not values:
            open_claims.append(claim)
            continue
        cited: set[str] = set()
        for value in values:
            found = ledger.find(value)
            if not found:
                break
            cited.update(found)
        else:
            settled.append(
                ClaimCheck(
                    claim=claim,
                    verdict="confirmed",
                    settled_by="code",
                    evidence_ids=[row_id for row_id in ledger.ids if row_id in cited][:5],
                    reason=f"every figure ({', '.join(values)}) is in the evidence",
                )
            )
            continue
        open_claims.append(claim)
    return settled, open_claims
