# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""The policy as a document: load, dump, hash, and reload a :class:`ControlPolicy`.

A :class:`~locus.control.policy.ControlPolicy` is a handful of rules. The
people who own those rules review documents, not Python, so the same policy
has a file form that a security or platform team can version, diff and
reload without a deploy::

    # refund-policy.yaml
    schema: "locus.control.policy/v1"
    version: "refunds-2026-11"
    require_verification_score: 0.0
    max_blast_radius: 1
    require_human_for:
      - "high_value"
    deny_for: []
    min_severity: "low"
    require_sandbox_for: []
    require_human_over_usd: null
    spend_limit_usd: null

Every key is a :class:`ControlPolicy` field with the same name, type and
default, so the two forms round-trip: ``ControlPolicy.load(path)`` and
``policy.dump(path)`` are inverses, and a file never says something the
Python API cannot. A key the policy does not have is an error that names
it, so a typo cannot silently widen authority. JSON works everywhere; YAML
reading needs the optional ``pyyaml`` package (writing does not).

:func:`policy_hash` is the SHA-256 of the rules. Equal policies hash the
same whether they came from YAML, JSON or a constructor call, and
:func:`~locus.control.admission.admit` records it on every trail entry, so
an auditor can tell exactly which rules a decision was made under.

:class:`PolicyWatcher` re-reads the file when it changes and hands
:func:`~locus.control.admission.admit` or :func:`~locus.control.gate.gate_tool`
the current policy, one per decision. A file that stops parsing keeps the
last good policy in force rather than leaving the gate without one.

A document may also say what the policy is meant to cover, for
``locus policy lint``::

    governs:
      environments: ["production", "staging"]
      kinds: ["payment"]

That block is read by :class:`PolicyDocument` and the linter; the policy
itself does not carry it.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import math
import os
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, fields
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from locus.control.policy import ControlPolicy
from locus.control.taxonomy import Severity


#: The document format this module reads and writes.
SCHEMA = "locus.control.policy/v1"

Format = Literal["yaml", "json"]

#: The policy's fields, in declaration order: the order a document is written in.
_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(ControlPolicy))
_LABEL_FIELDS = frozenset({"require_human_for", "deny_for", "require_sandbox_for"})
_OPTIONAL_USD_FIELDS = frozenset({"require_human_over_usd", "spend_limit_usd"})
_DOCUMENT_KEYS = ("schema", "governs")
_GOVERNS_KEYS = ("environments", "kinds")


class PolicyFileError(ValueError):
    """A policy document that cannot become a :class:`ControlPolicy`.

    ``key`` is the path to the offending key (``require_human_for[2]``,
    ``governs.kinds``), ``source`` the file it was read from.
    """

    def __init__(self, message: str, *, key: str = "", source: str = "") -> None:
        self.key = key
        self.source = source
        where = f"{source}: " if source else ""
        at = f"at {key}: " if key else ""
        super().__init__(f"{where}{at}{message}")


@dataclass(frozen=True)
class PolicyDocument:
    """A policy file as a whole: the policy, plus what it declares it governs.

    ``environments`` and ``kinds`` are the labels the author expects the
    policy to have a rule for; ``locus policy lint`` reports the ones it does
    not. They are document metadata, not rules, so a :class:`ControlPolicy`
    does not carry them.
    """

    policy: ControlPolicy
    environments: frozenset[str] = frozenset()
    kinds: frozenset[str] = frozenset()
    source: str = "<string>"

    @classmethod
    def load(cls, source: str | os.PathLike[str]) -> PolicyDocument:
        """Read a document from a path, or from text when ``source`` is one."""
        if isinstance(source, str) and _looks_like_text(source):
            return cls.loads(source)
        path = Path(source)
        return cls.loads(
            path.read_text(encoding="utf-8"), format=_format_for(path), source=str(path)
        )

    @classmethod
    def loads(
        cls,
        text: str,
        *,
        format: Format | None = None,  # noqa: A002
        source: str = "<string>",
    ) -> PolicyDocument:
        """Read a document from ``text``; the format is guessed when not given."""
        data = _parse(text, format or _format_for_text(text), source)
        return _document_from(data, source)

    def dump(
        self,
        path: str | os.PathLike[str] | None = None,
        *,
        format: Format | None = None,  # noqa: A002
    ) -> str:
        """The document as text, written to ``path`` when one is given."""
        data = to_dict(self.policy)
        governs: dict[str, list[str]] = {}
        if self.environments:
            governs["environments"] = sorted(self.environments)
        if self.kinds:
            governs["kinds"] = sorted(self.kinds)
        if governs:
            data["governs"] = governs
        return _write({"schema": SCHEMA, **data}, path, format)


def load_policy(source: str | os.PathLike[str]) -> ControlPolicy:
    """The :class:`ControlPolicy` in a file, or in a document passed as text."""
    return PolicyDocument.load(source).policy


def loads_policy(text: str, *, format: Format | None = None) -> ControlPolicy:  # noqa: A002
    """The :class:`ControlPolicy` in ``text``."""
    return PolicyDocument.loads(text, format=format).policy


def dump_policy(
    policy: ControlPolicy,
    path: str | os.PathLike[str] | None = None,
    *,
    format: Format | None = None,  # noqa: A002
) -> str:
    """``policy`` as a document, written to ``path`` when one is given."""
    return _write({"schema": SCHEMA, **to_dict(policy)}, path, format)


def to_dict(policy: ControlPolicy) -> dict[str, Any]:
    """The policy's rules as plain JSON-ready values, in field order.

    Label sets become sorted lists and the severity its name, so two equal
    policies produce identical dictionaries.
    """
    data: dict[str, Any] = {}
    for name in _FIELDS:
        value = getattr(policy, name)
        if isinstance(value, frozenset):
            data[name] = sorted(value)
        elif isinstance(value, Severity):
            data[name] = value.value
        else:
            data[name] = value
    return data


def policy_hash(policy: ControlPolicy) -> str:
    """SHA-256 (hex) over the canonical form of the rules.

    The same for equal policies however they were built, and different for
    any change to any rule, the ``version`` label included.
    """
    canonical = json.dumps(to_dict(policy), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- reading


def _looks_like_text(source: str) -> bool:
    stripped = source.lstrip()
    return "\n" in source or stripped.startswith(("{", "["))


def _format_for(path: Path) -> Format | None:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return "json"
    if suffix in (".yaml", ".yml"):
        return "yaml"
    return None


def _format_for_text(text: str) -> Format:
    return "json" if text.lstrip().startswith(("{", "[")) else "yaml"


def _parse(text: str, fmt: Format, source: str) -> Any:
    if fmt == "json":
        try:
            return json.loads(text)
        except ValueError as exc:
            raise PolicyFileError(f"not valid JSON: {exc}", source=source) from exc
    try:
        import yaml  # type: ignore[import-untyped]  # noqa: PLC0415 — optional dependency
    except ImportError as exc:
        raise ImportError(
            "reading a YAML policy needs pyyaml: pip install pyyaml "
            "(a JSON policy needs nothing extra)"
        ) from exc
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise PolicyFileError(f"not valid YAML: {exc}", source=source) from exc


def _document_from(data: Any, source: str) -> PolicyDocument:
    mapping = _mapping(data, key="", source=source)
    known = (*_DOCUMENT_KEYS, *_FIELDS)
    for key in mapping:
        if not isinstance(key, str) or key not in known:
            raise PolicyFileError(
                f"unknown key {key!r}{_suggestion(str(key), known)}",
                key=str(key),
                source=source,
            )
    schema = mapping.get("schema", SCHEMA)
    if schema != SCHEMA:
        raise PolicyFileError(f"expected {SCHEMA!r}, got {schema!r}", key="schema", source=source)
    kwargs = {name: _coerce(name, mapping[name], source) for name in _FIELDS if name in mapping}
    environments, kinds = _governs(mapping.get("governs"), source)
    return PolicyDocument(
        policy=ControlPolicy(**kwargs),
        environments=environments,
        kinds=kinds,
        source=source,
    )


def _governs(data: Any, source: str) -> tuple[frozenset[str], frozenset[str]]:
    if data is None:
        return frozenset(), frozenset()
    mapping = _mapping(data, key="governs", source=source)
    for key in mapping:
        if key not in _GOVERNS_KEYS:
            raise PolicyFileError(
                f"unknown key {key!r}{_suggestion(str(key), _GOVERNS_KEYS)}",
                key=f"governs.{key}",
                source=source,
            )
    return (
        _labels(mapping.get("environments", []), key="governs.environments", source=source),
        _labels(mapping.get("kinds", []), key="governs.kinds", source=source),
    )


def _mapping(data: Any, *, key: str, source: str) -> Mapping[Any, Any]:
    if not isinstance(data, Mapping):
        raise PolicyFileError(
            f"expected a mapping of rules, got {_kind(data)}", key=key, source=source
        )
    return data


def _coerce(name: str, value: Any, source: str) -> Any:
    """``value`` as the type :class:`ControlPolicy` field ``name`` takes."""
    if name in _LABEL_FIELDS:
        return _labels(value, key=name, source=source)
    if name == "min_severity":
        if isinstance(value, Severity):
            return value
        values = [member.value for member in Severity]
        if not isinstance(value, str) or value not in values:
            raise PolicyFileError(
                f"expected one of {values}, got {value!r}", key=name, source=source
            )
        return Severity(value)
    if name == "version":
        if not isinstance(value, str):
            raise PolicyFileError(f"expected a string, got {_kind(value)}", key=name, source=source)
        return value
    if name == "max_blast_radius":
        if isinstance(value, bool) or not isinstance(value, int):
            raise PolicyFileError(
                f"expected an integer, got {_kind(value)}", key=name, source=source
            )
        if value < 0:
            raise PolicyFileError(f"expected 0 or more, got {value}", key=name, source=source)
        return value
    if name in _OPTIONAL_USD_FIELDS and value is None:
        return None
    return _number(value, key=name, source=source)


def _number(value: Any, *, key: str, source: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise PolicyFileError(f"expected a number, got {_kind(value)}", key=key, source=source)
    number = float(value)
    if not math.isfinite(number):
        raise PolicyFileError(f"expected a finite number, got {value!r}", key=key, source=source)
    if number < 0:
        raise PolicyFileError(f"expected 0 or more, got {value!r}", key=key, source=source)
    return number


def _labels(value: Any, *, key: str, source: str) -> frozenset[str]:
    if isinstance(value, str | bytes | Mapping) or not isinstance(value, Iterable):
        raise PolicyFileError(
            f"expected a list of labels, got {_kind(value)}", key=key, source=source
        )
    labels: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item:
            raise PolicyFileError(
                f"expected a non-empty string, got {_kind(item)}",
                key=f"{key}[{index}]",
                source=source,
            )
        labels.add(item)
    return frozenset(labels)


def _kind(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "a boolean"
    if isinstance(value, int | float):
        return "a number"
    if isinstance(value, str):
        return f"the string {value!r}" if value else "an empty string"
    if isinstance(value, Mapping):
        return "a mapping"
    if isinstance(value, list | tuple):
        return "a list"
    return type(value).__name__


def _suggestion(key: str, known: Iterable[str]) -> str:
    close = difflib.get_close_matches(key, list(known), n=1, cutoff=0.6)
    return f" (did you mean {close[0]!r}?)" if close else ""


# --------------------------------------------------------------------------- writing


def _write(data: dict[str, Any], path: str | os.PathLike[str] | None, fmt: Format | None) -> str:
    if fmt is None:
        fmt = (_format_for(Path(path)) if path is not None else None) or "yaml"
    text = json.dumps(data, indent=2) + "\n" if fmt == "json" else _emit_yaml(data)
    if path is not None:
        Path(path).write_text(text, encoding="utf-8")
    return text


def _emit_yaml(data: Mapping[str, Any]) -> str:
    """The document as YAML, with no YAML library.

    The schema is closed (scalars, lists of strings, one nested mapping), so
    a small emitter covers it: strings are written as JSON strings, which
    YAML reads as double-quoted scalars, so a label such as ``yes`` or
    ``1.0`` stays a string. Writing needs nothing installed; only reading
    YAML does.
    """
    lines: list[str] = []
    for key, value in data.items():
        if isinstance(value, Mapping):
            lines.append(f"{key}:")
            for inner_key, inner in value.items():
                lines.extend(_yaml_entry(inner_key, inner, indent="  "))
        else:
            lines.extend(_yaml_entry(key, value, indent=""))
    return "\n".join(lines) + "\n"


def _yaml_entry(key: str, value: Any, *, indent: str) -> list[str]:
    if isinstance(value, list):
        if not value:
            return [f"{indent}{key}: []"]
        return [f"{indent}{key}:", *(f"{indent}  - {_yaml_scalar(item)}" for item in value)]
    return [f"{indent}{key}: {_yaml_scalar(value)}"]


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"cannot write {value!r} to a policy document")
        text = repr(value)
        # YAML 1.1 reads ``1e-05`` as a string; spell exponents out.
        return format(Decimal(text), "f") if "e" in text else text
    if isinstance(value, int):
        return str(value)
    return json.dumps(value, ensure_ascii=False)


# --------------------------------------------------------------------------- reloading


class PolicyWatcher:
    """A :class:`~locus.control.policy.PolicySource` that follows a file.

    Reads the file once on construction, so a broken file fails at startup,
    and after that re-reads it whenever its modification time or size has
    changed, checking at most once per ``interval`` seconds. The swap is one
    assignment: an admission in flight finishes under the policy it started
    with, and the next one sees the new policy whole.

    A rewrite that no longer parses does not change the policy in force. The
    last good one stays, the problem is kept on :attr:`last_error` for the
    operator, and the next change to the file is tried again. Passing
    ``on_reload`` is how a service logs a swap or refuses to run under rules
    it did not expect.

    Hand it anywhere a policy goes::

        watcher = PolicyWatcher("policy/refunds.yaml", interval=2.0)
        refund = gate_tool(issue_refund, policy=watcher, trail=trail)
        await admit(action, perform, policy=watcher, trail=trail)

    Each decision records the hash of the policy it was made under.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        interval: float = 1.0,
        on_reload: Callable[[ControlPolicy, ControlPolicy], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._path = Path(path)
        self._interval = interval
        self._on_reload = on_reload
        self._clock = clock
        self._lock = threading.Lock()
        self._document = PolicyDocument.load(self._path)
        self._stamp = self._stat()
        self._checked = clock()
        self._last_error: Exception | None = None
        self._reloads = 0

    @property
    def path(self) -> Path:
        """The file being followed."""
        return self._path

    @property
    def document(self) -> PolicyDocument:
        """The document as last read, without checking the file."""
        return self._document

    @property
    def policy(self) -> ControlPolicy:
        """The policy as last read, without checking the file."""
        return self._document.policy

    @property
    def hash(self) -> str:
        """The :func:`policy_hash` of the policy in force."""
        return policy_hash(self._document.policy)

    @property
    def reloads(self) -> int:
        """How many times the file was re-read successfully."""
        return self._reloads

    @property
    def last_error(self) -> Exception | None:
        """Why the latest change to the file was not taken, or ``None``."""
        return self._last_error

    def current(self) -> ControlPolicy:
        """The policy in force, after checking the file if it is time to."""
        if self._clock() - self._checked >= self._interval:
            self.reload()
        return self._document.policy

    def reload(self) -> bool:
        """Re-read the file now if it changed; ``True`` when the rules changed."""
        with self._lock:
            self._checked = self._clock()
            stamp = self._stat()
            if stamp == self._stamp:
                return False
            self._stamp = stamp
            if stamp is None:
                self._last_error = FileNotFoundError(str(self._path))
                return False
            try:
                document = PolicyDocument.load(self._path)
            except (PolicyFileError, OSError, ImportError) as exc:
                self._last_error = exc
                return False
            previous = self._document
            self._document = document
            self._last_error = None
            self._reloads += 1
        changed = previous.policy != document.policy
        if changed and self._on_reload is not None:
            self._on_reload(previous.policy, document.policy)
        return changed

    def _stat(self) -> tuple[int, int, int] | None:
        try:
            stat = self._path.stat()
        except FileNotFoundError:
            return None
        return (stat.st_mtime_ns, stat.st_size, stat.st_ino)


__all__ = [
    "SCHEMA",
    "Format",
    "PolicyDocument",
    "PolicyFileError",
    "PolicyWatcher",
    "dump_policy",
    "load_policy",
    "loads_policy",
    "policy_hash",
    "to_dict",
]
