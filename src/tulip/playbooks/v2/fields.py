# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed process data: the one field model a v2 playbook's data is declared with.

Ported from the registry (``tulip_registry.definitions.fields``), which checks every field
strictly at publish. A playbook's inputs, each step's outputs and the answers a step asks
a person for are all :class:`Field` values: a name, a plain label, a type, and whether the
value is **sensitive**. The type is what a value must look like (:func:`validate_value`,
whose sentences are the registry's word for word); the sensitivity is where it may be
seen. A sensitive value (the default) stays digested in the control plane's mirror and is
read in clear only on the runtime; a field marked ``sensitive: false`` may be shown in
Studio and used in notices (:meth:`~tulip.playbooks.v2.engine.PlaybookRuntime.public_outputs`).

The types, and what a value of each is:

* ``text``, ``choice`` (one of ``choices``), ``email``, ``url`` (``http`` or ``https``):
  a string;
* ``number``: an integer or a decimal number (never ``true`` / ``false``);
* ``money``: ``{"amount": <number>, "currency": "<ISO 4217 code>"}``;
* ``date``: an ISO-8601 calendar date, ``YYYY-MM-DD``;
* ``boolean``: ``true`` or ``false``;
* ``file``: a reference to a file held in the data plane,
  ``{"ref", "name", "size"?, "media_type"?}`` -- never the file's bytes.

Reading a definition is tolerant (:func:`parse_field`): the registry refused anything
malformed at publish, so here an unknown type reads as ``text`` and a stray key is ignored
rather than refusing a run of a playbook that was published.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any
from urllib.parse import urlsplit

from tulip.playbooks.v2.when import Day, Money


#: Every field type, in the order the schema lists them.
FIELD_TYPES: tuple[str, ...] = (
    "text",
    "number",
    "money",
    "date",
    "boolean",
    "choice",
    "email",
    "url",
    "file",
)

#: The parts of a structured value a condition may read (``inputs.amount.currency``),
#: and the type each part has.
FIELD_PARTS: Mapping[str, Mapping[str, str]] = {
    "money": {"amount": "number", "currency": "text"},
    "file": {"ref": "text", "name": "text", "size": "number", "media_type": "text"},
}

_CURRENCY = re.compile(r"^[A-Z]{3}$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_FILE_KEYS = frozenset({"ref", "name", "size", "media_type"})


def label_from_name(name: str) -> str:
    """The label a field shows when it declares none: ``vendor_id`` -> ``Vendor id``."""
    words = re.sub(r"[_\-:]+", " ", name).strip()
    return words[:1].upper() + words[1:] if words else name


@dataclass(frozen=True)
class Field:
    """One piece of process data: an input, a step's output, or a person's answer.

    The registry's ``PlaybookField``. ``label`` is never empty once read
    (:func:`parse_field` makes it from the name); ``choices`` only for ``choice``.
    """

    name: str
    label: str = ""
    type: str = "text"
    choices: tuple[str, ...] = ()
    required: bool = True
    sensitive: bool = True
    description: str = ""

    @property
    def shown_as(self) -> str:
        """What people see: the label, or one made from the name."""
        return self.label or label_from_name(self.name)


def text_field(name: str) -> Field:
    """The field an untyped name stands for (an old ``expected_outputs`` entry, say)."""
    return Field(name=name, label=label_from_name(name))


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def parse_field(raw: Any, *, name: str = "") -> Field | None:
    """Read one field of a definition tolerantly; ``None`` when it names nothing.

    An old input, ``{name, description, required}``, reads as a sensitive text field. A
    ``name`` given here is used when the field carries none (a question's ``field``
    takes the question's name). An unknown type reads as ``text``, and so does a
    ``choice`` without choices -- the registry refuses both at publish, so a stored
    playbook carrying one predates that check and is not refused here. Anything but an
    explicit ``false`` keeps ``required`` and ``sensitive`` on: a field fails toward
    being asked for, and toward staying digested.
    """
    if not isinstance(raw, Mapping):
        return None
    field_name = _text(raw.get("name")) or name
    if not field_name:
        return None
    kind = _text(raw.get("type")) or "text"
    if kind not in FIELD_TYPES:
        kind = "text"
    choices: tuple[str, ...] = ()
    if kind == "choice":
        listed = raw.get("choices")
        if isinstance(listed, list | tuple):
            choices = tuple(c for c in listed if isinstance(c, str) and c.strip())
        if not choices:
            kind = "text"
    return Field(
        name=field_name,
        label=_text(raw.get("label")) or label_from_name(field_name),
        type=kind,
        choices=choices,
        required=raw.get("required", True) is not False,
        sensitive=raw.get("sensitive", True) is not False,
        description=_text(raw.get("description")),
    )


def parse_fields(raw: Any) -> tuple[Field, ...]:
    """Read a list of fields tolerantly: the ones that name something, each name once."""
    if not isinstance(raw, list | tuple):
        return ()
    fields: list[Field] = []
    seen: set[str] = set()
    for item in raw:
        parsed = parse_field(item)
        if parsed is not None and parsed.name not in seen:
            seen.add(parsed.name)
            fields.append(parsed)
    return tuple(fields)


# ── values ───────────────────────────────────────────────────────────────────


def _is_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    return not (isinstance(value, float) and not math.isfinite(value))


def is_iso_date(value: Any) -> bool:
    """Whether ``value`` is an ISO-8601 calendar date string, ``YYYY-MM-DD``."""
    if not isinstance(value, str) or not _DATE.match(value):
        return False
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def _money_problem(label: str, value: Any) -> str | None:
    shape = f"{label} must be an amount of money: an amount and a three-letter currency code."
    if not isinstance(value, Mapping) or set(value) != {"amount", "currency"}:
        return shape
    if not _is_number(value["amount"]):
        return f"{label} must have a number as its amount."
    currency = value["currency"]
    if not isinstance(currency, str) or not _CURRENCY.match(currency):
        return f"{label} must have an ISO 4217 currency code, such as USD or EUR."
    return None


def _file_problem(label: str, value: Any) -> str | None:
    if isinstance(value, bytes | bytearray | memoryview):
        return f"{label} must be a reference to a stored file, never the file itself."
    if not isinstance(value, Mapping):
        return f"{label} must be a reference to a stored file, with its ref and name."
    extra = sorted(str(k) for k in set(value) - _FILE_KEYS)
    if extra:
        return (
            f"{label} must be a reference to a stored file (ref, name, size, media_type); "
            f"it cannot carry {', '.join(extra)}."
        )
    for key in ("ref", "name"):
        part = value.get(key)
        if not isinstance(part, str) or not part.strip():
            return f"{label} must name the stored file's {key}."
    size = value.get("size")
    if size is not None and (isinstance(size, bool) or not isinstance(size, int) or size < 0):
        return f"{label} must give its size as a whole number of bytes."
    media_type = value.get("media_type")
    if media_type is not None and (not isinstance(media_type, str) or "/" not in media_type):
        return f"{label} must give its media type as type/subtype, such as application/pdf."
    return None


def _url_ok(value: str) -> bool:
    try:
        parts = urlsplit(value)
    except ValueError:
        return False
    return parts.scheme in ("http", "https") and bool(parts.netloc) and " " not in value


def validate_value(field: Field, value: Any) -> str | None:
    """Whether ``value`` is a valid value of ``field``: ``None`` when it is, else why not.

    The reason is one plain sentence naming the field by its label, fit to show the
    person or the model that gave the value -- the registry's sentence, word for word.
    A missing value (``None``) is fine only for a field that is not required.
    """
    label = field.shown_as
    if value is None:
        return f"{label} is required." if field.required else None
    kind = field.type
    problem: str | None = None
    if kind in ("text", "choice", "email", "url") and not isinstance(value, str):
        problem = f"{label} must be text."
    elif kind == "choice":
        if value not in field.choices:
            problem = f"{label} must be one of: {', '.join(field.choices)}."
    elif kind == "email":
        if not _EMAIL.match(value):
            problem = f"{label} must be an email address."
    elif kind == "url":
        if not _url_ok(value):
            problem = f"{label} must be a web address starting with http:// or https://."
    elif kind == "number":
        if not _is_number(value):
            problem = f"{label} must be a number."
    elif kind == "money":
        problem = _money_problem(label, value)
    elif kind == "date":
        if not is_iso_date(value):
            problem = f"{label} must be a date written as YYYY-MM-DD."
    elif kind == "boolean":
        if not isinstance(value, bool):
            problem = f"{label} must be yes or no (true or false)."
    elif kind == "file":
        problem = _file_problem(label, value)
    return problem


def typed_value(field: Field, value: Any) -> Any:
    """``value`` as a condition reads it: a valid money value as :class:`Money`, a date as
    :class:`Day`, anything else unchanged.

    So ``inputs.amount > 10000`` compares the amount, two amounts compare only in one
    currency, and dates compare as dates (:mod:`tulip.playbooks.v2.when`). A value that is
    not valid for its field is left as it is.
    """
    if validate_value(field, value) is not None or value is None:
        return value
    if field.type == "money":
        return Money(value["amount"], value["currency"])
    if field.type == "date":
        return Day(value)
    return value


def describe(field: Field) -> str:
    """How a value of ``field`` is written, for the model: empty for plain text."""
    if field.type == "text":
        return ""
    if field.type == "money":
        return 'money: {"amount": <number>, "currency": "<ISO 4217 code>"}'
    if field.type == "date":
        return "date: YYYY-MM-DD"
    if field.type == "boolean":
        return "true or false"
    if field.type == "choice":
        return "one of: " + ", ".join(field.choices)
    if field.type == "file":
        return 'file: {"ref", "name"} of a stored file, never its bytes'
    return field.type


__all__ = [
    "FIELD_PARTS",
    "FIELD_TYPES",
    "Field",
    "describe",
    "is_iso_date",
    "label_from_name",
    "parse_field",
    "parse_fields",
    "text_field",
    "typed_value",
    "validate_value",
]
