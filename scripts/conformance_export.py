#!/usr/bin/env python3
# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Regenerate the control-layer conformance fixtures under ``conformance/v1``.

The fixtures pin what a port of the governance core has to reproduce byte for
byte: the canonical JSON the hashes are taken over, hash-chained and signed
audit trails with every tamper class and the report each one yields, policy
documents with their hash, lint findings and decisions, approval digests and
decision sequences, and the exact ``action-admission`` line an admission
writes. Everything is derived from a fixed clock, fixed inputs and signing
keys derived from a fixed seed, so running this twice gives the same bytes.

``tests/unit/test_conformance_fixtures.py`` regenerates into a temporary
directory and fails on any difference from the committed tree: a change to a
format is a deliberate ``conformance/v2``, never a silent edit of ``v1``.

Usage::

    python scripts/conformance_export.py            # rewrite conformance/v1
    python scripts/conformance_export.py --check    # exit 1 if it would change
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime, timedelta, tzinfo
from itertools import count
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self
from unittest import mock

from locus.control import (
    Action,
    AdmissionError,
    AuditTrail,
    ControlPolicy,
    Ed25519Signer,
    InMemoryApprovals,
    InMemorySpendLedger,
    JudgeVerdict,
    PolicyDocument,
    PolicyFileError,
    admit_sync,
    approve,
    call_digest,
    check_jsonl,
    evaluate_rules,
    policy_hash,
)
from locus.control.approvals import (
    ApprovalAuthority,
    ApprovalAuthorityError,
    ApproverRule,
    Delegation,
)
from locus.control.findings import Evidence
from locus.control.policy_file import to_dict
from locus.control.policy_lint import action_from_mapping, lint_policy
from locus.control.taxonomy import Severity
from locus.control.verification import VerificationResult


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "conformance" / "v1"

#: The directories this script owns. ``README.md`` and ``tools/`` are written
#: by hand and left alone.
GENERATED = ("canonical-json", "audit", "policies", "approvals", "admission")

#: Every timestamp in the fixtures starts here and advances one second per tick.
START = datetime(2026, 1, 1, tzinfo=UTC)

GENESIS = "0" * 64

#: The seeds the two signing keys are derived from. Published on purpose: the
#: keys sign nothing but these fixtures.
KEY_SEEDS = {
    "primary": b"locus conformance v1: audit signing key, primary",
    "rotated": b"locus conformance v1: audit signing key, rotated",
}
ROTATED_KEY_ID = "conformance-v1-rotated"


# --------------------------------------------------------------------------- helpers


def canonical(value: Any, *, ensure_ascii: bool = True) -> str:
    """The canonical form the SDK hashes: sorted keys, compact, ``default=str``."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=ensure_ascii, default=str
    )


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def iso_clock(start: datetime = START) -> Callable[[], str]:
    """A trail clock: ISO-8601 UTC, one second later on every call."""
    ticks = count()

    def clock() -> str:
        return (start + timedelta(seconds=next(ticks))).isoformat()

    return clock


@contextmanager
def fixed_clock(start: datetime = START) -> Iterator[None]:
    """Make the approval store's clock tick one second per reading, from ``start``."""
    import locus.control.approvals as approvals_module

    ticks = count()

    class Clock(datetime):
        @classmethod
        def now(cls, tz: tzinfo | None = None) -> Self:  # noqa: ARG003 — always UTC here
            moment = start + timedelta(seconds=next(ticks))
            return cls.fromtimestamp(moment.timestamp(), tz=UTC)

    with mock.patch.object(approvals_module, "datetime", Clock):
        yield


def _expect(condition: bool, message: str) -> None:  # noqa: FBT001 — a check, not a flag
    if not condition:
        raise RuntimeError(f"conformance export self-check failed: {message}")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")


def _write_json(path: Path, data: Any) -> None:
    _write_text(path, json.dumps(data, indent=2))


def _finding(severity: str) -> Evidence:
    return Evidence(
        title="conformance finding",
        description="a finding the conformance fixtures weigh against min_severity",
        severity=Severity(severity),
        asset="svc-1",
        remediation="none",
        gsar_score=1.0,
        evidence_refs=["conformance"],
    )


def _verdict(spec: Mapping[str, Any] | None) -> VerificationResult | None:
    if spec is None:
        return None
    return VerificationResult(
        survives=spec["survives"],
        confidence=spec["confidence"],
        evidence_quality=spec["evidence_quality"],
    )


# --------------------------------------------------------------------------- keys


def signers() -> tuple[Ed25519Signer, Ed25519Signer]:
    """The two fixture signers: the primary with the derived key id, the rotated one named."""
    from cryptography.hazmat.primitives.asymmetric import ed25519

    def key(seed: bytes) -> Any:
        return ed25519.Ed25519PrivateKey.from_private_bytes(hashlib.sha256(seed).digest())

    primary = Ed25519Signer(key(KEY_SEEDS["primary"]))
    rotated = Ed25519Signer(key(KEY_SEEDS["rotated"]), key_id=ROTATED_KEY_ID)
    return primary, rotated


# --------------------------------------------------------------------------- audit


EVENTS: tuple[tuple[str, dict[str, Any]], ...] = (
    (
        "session-start",
        {"agent": "conformance", "schema": "locus.control.policy/v1", "ratio": 1.0, "zero": 0.0},
    ),
    (
        "action-admission",
        {
            "action": "issue_refund",
            "asset": "order-4821",
            "outcome": "allow",
            "reason": "all policy checks passed",
            "policy_hash": policy_hash(ControlPolicy()),
            "cost_usd": 12.5,
            "note": "café ☕ 😀",
        },
    ),
    (
        "tool-call",
        {
            "tool": "send_email",
            "arguments": {"to": "ops@example.com", "subject": "Résumé", "attachments": []},
            "nested": {"b": [1, 2.0, "3"], "a": {"y": None, "x": True}},
        },
    ),
    (
        "session-end",
        {"empty": {}, "list": [], "text": 'line1\nline2\t"quoted" \\ / \x7f', "count": 4},
    ),
)


def build_trail(signer: Ed25519Signer | None, rotate_to: Ed25519Signer | None = None) -> AuditTrail:
    """The fixture trail: four records; ``rotate_to`` signs from the third on."""
    trail = AuditTrail(clock=iso_clock(), signer=signer)
    for index, (event_type, payload) in enumerate(EVENTS):
        if rotate_to is not None and index == 2:
            trail.use_signer(rotate_to)
        trail.record(event_type, payload)
    for record in trail.records():
        body = {
            "seq": record.seq,
            "ts": record.ts,
            "event_type": record.event_type,
            "payload": record.payload,
            "prev_hash": record.prev_hash,
        }
        _expect(sha256_hex(canonical(body)) == record.hash, f"record {record.seq} hash")
    return trail


def report_dict(
    text: str, *, keys: Mapping[str, bytes] | None, expected_head: str | None
) -> dict[str, Any]:
    return asdict(check_jsonl(text, keys=keys, expected_head=expected_head))


def _check(
    text: str,
    key_ids: list[str] | None,
    expected_head: str | None,
    public: Mapping[str, bytes],
) -> dict[str, Any]:
    keys = {key_id: public[key_id] for key_id in key_ids} if key_ids is not None else None
    return {
        "keys": key_ids,
        "expected_head": expected_head,
        "report": report_dict(text, keys=keys, expected_head=expected_head),
    }


def _lines(trail: AuditTrail) -> list[str]:
    return trail.export_jsonl().split("\n")


def _edit(lines: list[str], index: int, change: Callable[[dict[str, Any]], None]) -> list[str]:
    edited = list(lines)
    record = json.loads(edited[index])
    change(record)
    edited[index] = json.dumps(record)
    return edited


def _rebuild(lines: list[str], start: int) -> list[str]:
    """Recompute every hash from ``start`` on, keeping the old signatures."""
    records = [json.loads(line) for line in lines]
    prev = records[start - 1]["hash"] if start else GENESIS
    for record in records[start:]:
        record["prev_hash"] = prev
        body = {key: record[key] for key in ("seq", "ts", "event_type", "payload", "prev_hash")}
        record["hash"] = prev = sha256_hex(canonical(body))
    return [json.dumps(record) for record in records]


def tamper_cases(signed: list[str]) -> list[tuple[str, str, list[str]]]:
    """``(name, description, lines)`` for every tamper class, from the signed trail."""

    def set_payload(record: dict[str, Any]) -> None:
        record["payload"]["cost_usd"] = 1250.0

    def set_key_id(record: dict[str, Any]) -> None:
        record["key_id"] = "conformance-v1-unknown"

    def set_bad_base64(record: dict[str, Any]) -> None:
        record["signature"] = "not*base64!"

    def set_other_signature(record: dict[str, Any]) -> None:
        record["signature"] = json.loads(signed[0])["signature"]

    def add_field(record: dict[str, Any]) -> None:
        record["note"] = "added after the fact"

    def set_prev_hash(record: dict[str, Any]) -> None:
        record["prev_hash"] = "f" * 64

    reordered = list(signed)
    reordered[1], reordered[2] = reordered[2], reordered[1]
    return [
        (
            "edited-payload",
            "record 1's payload changed; its hash no longer matches",
            _edit(signed, 1, set_payload),
        ),
        ("reordered-records", "records 1 and 2 swapped", reordered),
        ("deleted-record", "record 2 removed from the middle", signed[:2] + signed[3:]),
        (
            "truncated-file",
            "the last record dropped; only the expected head catches it",
            signed[:-1],
        ),
        (
            "unknown-key-id",
            "record 2 names a key the verifier does not hold",
            _edit(signed, 2, set_key_id),
        ),
        (
            "bad-base64-signature",
            "record 3's signature is not base64",
            _edit(signed, 3, set_bad_base64),
        ),
        (
            "wrong-signature",
            "record 1 carries record 0's (valid base64) signature",
            _edit(signed, 1, set_other_signature),
        ),
        ("extra-field", "record 0 has a field a record does not have", _edit(signed, 0, add_field)),
        (
            "edited-prev-hash",
            "record 2 points at a parent that is not record 1",
            _edit(signed, 2, set_prev_hash),
        ),
        (
            "rebuilt-chain",
            "record 1 edited and every later hash recomputed; chain consistent, signatures and head not",
            _rebuild(_edit(signed, 1, set_payload), 1),
        ),
        (
            "not-a-record-line",
            "a line that is not JSON between records 1 and 2",
            signed[:2] + ["not a record"] + signed[2:],
        ),
    ]


def write_audit(out: Path, primary: Ed25519Signer, rotated: Ed25519Signer) -> dict[str, AuditTrail]:
    public = {
        primary.key_id: primary.public_key_pem(),
        rotated.key_id: rotated.public_key_pem(),
    }
    for signer in (primary, rotated):
        _write_text(
            out / "keys" / f"{signer.key_id}.priv.pem", signer.private_key_pem().decode("ascii")
        )
        _write_text(
            out / "keys" / f"{signer.key_id}.pub.pem", signer.public_key_pem().decode("ascii")
        )

    trails = {
        "unsigned": build_trail(None),
        "signed": build_trail(primary),
        "rotated": build_trail(primary, rotate_to=rotated),
    }
    both = [primary.key_id, rotated.key_id]
    verifier_runs: dict[str, list[tuple[list[str] | None, bool]]] = {
        "unsigned": [(None, False), (None, True), ([primary.key_id], True)],
        "signed": [(None, True), ([primary.key_id], True), (both, True), ([rotated.key_id], True)],
        "rotated": [([primary.key_id], True), (both, True)],
    }
    for name, trail in trails.items():
        text = trail.export_jsonl()
        _write_text(out / name / "trail.jsonl", text)
        _write_text(out / name / "expected_head", trail.head)
        _write_json(
            out / name / "check.json",
            {
                "trail": "trail.jsonl",
                "expected_head": trail.head,
                "checks": [
                    _check(text, key_ids, trail.head if with_head else None, public)
                    for key_ids, with_head in verifier_runs[name]
                ],
            },
        )

    signed = trails["signed"]
    for name, description, lines in tamper_cases(_lines(signed)):
        text = "\n".join(lines)
        runs: list[tuple[list[str] | None, str | None]] = [([primary.key_id], signed.head)]
        if name in ("truncated-file", "rebuilt-chain"):
            runs.append(([primary.key_id], None))
            runs.append((None, None))
        _write_text(out / "tamper" / f"{name}.jsonl", text)
        _write_json(
            out / "tamper" / f"{name}.check.json",
            {
                "base": "signed",
                "description": description,
                "checks": [_check(text, key_ids, head, public) for key_ids, head in runs],
            },
        )
    return trails


# --------------------------------------------------------------------------- canonical JSON


_ESCAPES = 'quote " backslash \\ slash / tab \t newline \n nul \x00 del \x7f ls  '


def build_vectors(unsigned: AuditTrail, approval_body: dict[str, Any]) -> list[dict[str, Any]]:
    first = unsigned.records()[0]
    entry_body = {
        "seq": first.seq,
        "ts": first.ts,
        "event_type": first.event_type,
        "payload": first.payload,
        "prev_hash": first.prev_hash,
    }
    cases: list[tuple[str, str, Any, str]] = [
        ("float-zero", "ascii", 0.0, "a float keeps its point: 0.0, never 0"),
        ("float-one", "ascii", 1.0, "1.0 and 1 are different bytes and different hashes"),
        ("int-one", "ascii", 1, ""),
        ("negative-zero", "ascii", -0.0, "the sign of zero is kept"),
        ("float-small-exponent", "ascii", 1e-05, "repr() switches to the exponent form below 1e-4"),
        ("float-large-exponent", "ascii", 1e16, "and above 1e16: 1e+16 with an explicit plus"),
        ("float-shortest-repr", "ascii", 0.1 + 0.2, "the shortest round-tripping repr"),
        ("float-whole", "ascii", 100.0, ""),
        (
            "int-large",
            "ascii",
            12345678901234567890123,
            "integers are arbitrary precision and print in full",
        ),
        (
            "int-vs-float-in-object",
            "ascii",
            {"int": 1, "float": 1.0, "zero": 0, "float_zero": 0.0},
            "a parser that reads 1.0 as 1 cannot re-hash this",
        ),
        ("bool-null", "ascii", {"t": True, "f": False, "n": None}, ""),
        ("empty-containers", "ascii", {"object": {}, "array": [], "string": ""}, ""),
        (
            "key-order",
            "ascii",
            {"b": 1, "a": 2, "B": 3, "_": 4, "10": 5, "9": 6, "aa": 7, "a0": 8},
            "keys sort by code point: digits, upper case, underscore, lower case; '10' before '9'",
        ),
        (
            "nested",
            "ascii",
            {"z": {"y": [3, {"x": 1, "w": 2}], "a": None}, "m": [[], {}]},
            "objects sort at every depth; arrays keep their order",
        ),
        (
            "escaping-ascii",
            "ascii",
            {"s": _ESCAPES},
            "every byte outside 0x20-0x7e is escaped; '/' is not",
        ),
        (
            "escaping-utf8",
            "utf8",
            {"s": _ESCAPES},
            "only control characters, the quote and the backslash are escaped",
        ),
        (
            "non-ascii-key-ascii",
            "ascii",
            {"é": "ü", "e": 1, "z": 2},
            "e (U+0065) sorts before é (U+00E9) before z",
        ),
        (
            "non-ascii-key-utf8",
            "utf8",
            {"é": "ü", "e": 1, "z": 2},
            "the same order, written as UTF-8",
        ),
        (
            "astral-key-order-ascii",
            "ascii",
            {"😀": 1, "！": 2, "a": 3},
            "U+FF01 sorts before U+1F600 by code point; UTF-16 code-unit order would put the emoji first",
        ),
        ("astral-key-order-utf8", "utf8", {"😀": 1, "！": 2, "a": 3}, ""),
        ("astral-value-ascii", "ascii", {"s": "𝄞"}, "a surrogate pair in ascii mode"),
        (
            "no-unicode-normalisation",
            "ascii",
            {"é": 1, "é": 2},
            "NFC and NFD are different keys; nothing is normalised",
        ),
        ("numeric-string", "ascii", {"a": "1.0", "b": 1.0}, "a string stays a string"),
        (
            "audit-entry-body",
            "ascii",
            entry_body,
            "the hash of record 0 in audit/unsigned/trail.jsonl",
        ),
        ("policy-default", "ascii", to_dict(ControlPolicy()), "policies/default.hash"),
        ("approval-call", "ascii", approval_body, "approvals/digests.json[0].digest"),
        (
            "integrity-call-utf8",
            "utf8",
            {
                "id": "call_1",
                "name": "send_email",
                "arguments": {"to": "ops@example.com", "subject": "Résumé", "n": 1.0},
            },
            "the tool-call integrity canonical form uses ensure_ascii=False",
        ),
    ]
    vectors: list[dict[str, Any]] = []
    for name, mode, value, note in cases:
        text = canonical(value, ensure_ascii=mode == "ascii")
        vectors.append(
            {
                "name": name,
                "mode": mode,
                "input": value,
                "input_json": json.dumps(value, ensure_ascii=False),
                "canonical": text,
                "sha256": sha256_hex(text),
                "note": note,
            }
        )
    by_name = {vector["name"]: vector for vector in vectors}
    _expect(by_name["audit-entry-body"]["sha256"] == first.hash, "audit-entry-body vector")
    _expect(
        by_name["policy-default"]["sha256"] == policy_hash(ControlPolicy()), "policy-default vector"
    )
    return vectors


# --------------------------------------------------------------------------- policies


def _doc(
    policy: ControlPolicy, environments: tuple[str, ...] = (), kinds: tuple[str, ...] = ()
) -> PolicyDocument:
    return PolicyDocument(
        policy=policy, environments=frozenset(environments), kinds=frozenset(kinds)
    )


def _case(
    action: dict[str, Any],
    *,
    verdict: dict[str, Any] | None = None,
    finding: str | None = None,
    spent_usd: float = 0.0,
) -> dict[str, Any]:
    return {
        "action": action,
        "verdict": verdict,
        "finding": {"severity": finding} if finding else None,
        "spent_usd": spent_usd,
    }


_GOOD = {"survives": True, "confidence": 0.95, "evidence_quality": 0.9}
_WEAK = {"survives": True, "confidence": 0.5, "evidence_quality": 0.5}
_REFUTED = {"survives": False, "confidence": 0.2, "evidence_quality": 0.1}


def _action(name: str, **fields: Any) -> dict[str, Any]:
    return {"name": name, **fields}


POLICIES: dict[str, tuple[str, PolicyDocument, list[dict[str, Any]]]] = {
    "default": (
        "json",
        _doc(ControlPolicy()),
        [
            _case(_action("rotate_key", asset="svc-1", environment="staging"), verdict=_GOOD),
            _case(_action("rotate_key", asset="svc-1", environment="production"), verdict=_GOOD),
            _case(_action("rotate_key", environment="staging")),
            _case(_action("rotate_key", environment="staging"), verdict=_WEAK),
            _case(_action("rotate_key", environment="staging"), verdict=_REFUTED),
            _case(_action("rotate_key", environment="staging", blast_radius=3), verdict=_GOOD),
            _case(_action("rotate_key", environment="staging"), verdict=_GOOD, finding="info"),
            _case(_action("rotate_key", environment="staging"), verdict=_GOOD, finding="high"),
            _case(_action("rotate_key", environment="production"), verdict=_REFUTED),
        ],
    ),
    "refunds": (
        "yaml",
        _doc(
            ControlPolicy(
                require_verification_score=0.0,
                max_blast_radius=1,
                require_human_for=frozenset({"high_value", "production"}),
                version="refunds-2026-11",
                require_human_over_usd=500.0,
                spend_limit_usd=10000.0,
            ),
            environments=("production", "staging"),
            kinds=("payment",),
        ),
        [
            _case(_action("issue_refund", kind="payment", environment="staging", cost_usd=120.0)),
            _case(_action("issue_refund", kind="payment", environment="staging", cost_usd=750.0)),
            _case(
                _action("issue_refund", kind="payment", environment="staging", cost_usd=150.0),
                spent_usd=9900.0,
            ),
            _case(
                _action(
                    "issue_refund",
                    kind="payment",
                    environment="staging",
                    tags=["high_value"],
                    cost_usd=120.0,
                )
            ),
            _case(
                _action("issue_refund", kind="payment", environment="production", cost_usd=750.0)
            ),
            _case(
                _action("issue_refund", kind="payment", environment="staging", cost_usd=1234.5),
                spent_usd=8900.25,
            ),
        ],
    ),
    "locked-down": (
        "json",
        _doc(
            ControlPolicy(
                require_verification_score=1.5,
                max_blast_radius=0,
                require_human_for=frozenset({"production"}),
                deny_for=frozenset({"production", "delete"}),
                min_severity=Severity.HIGH,
                require_sandbox_for=frozenset({"shell"}),
                version="locked-down",
                require_human_over_usd=100.0,
                spend_limit_usd=50.0,
            ),
            environments=("dev", "production"),
            kinds=("payment", "delete", "shell"),
        ),
        [
            _case(_action("run", kind="shell", environment="dev")),
            _case(
                _action("run", kind="shell", environment="dev", tags=["sandboxed"], blast_radius=0),
                verdict={"survives": True, "confidence": 1.0, "evidence_quality": 1.0},
            ),
            _case(_action("drop_table", kind="delete", environment="production")),
            _case(_action("buy", kind="payment", environment="dev", blast_radius=0, cost_usd=60.0)),
            _case(
                _action("scan", environment="dev", blast_radius=0),
                verdict={"survives": True, "confidence": 1.0, "evidence_quality": 1.0},
                finding="critical",
            ),
        ],
    ),
    "floats": (
        "yaml",
        _doc(
            ControlPolicy(
                require_verification_score=1e-05,
                max_blast_radius=1,
                require_human_for=frozenset(),
                version="floats",
                require_human_over_usd=0.0,
                spend_limit_usd=1e16,
            )
        ),
        [
            _case(_action("ping", environment="staging"), verdict=_WEAK),
            _case(_action("ping", environment="staging")),
            _case(_action("ping", environment="staging", cost_usd=0.5), verdict=_WEAK),
        ],
    ),
    "sandbox": (
        "yaml",
        _doc(
            ControlPolicy(
                require_verification_score=0.0,
                require_human_for=frozenset(),
                min_severity=Severity.INFO,
                require_sandbox_for=frozenset({"shell", "network"}),
            ),
            kinds=("shell",),
        ),
        [
            _case(_action("fetch", tags=["network"])),
            _case(_action("fetch", tags=["network", "sandboxed"])),
            _case(_action("fetch")),
        ],
    ),
    "quoted-labels": (
        "yaml",
        _doc(
            ControlPolicy(
                require_verification_score=0.0,
                require_human_for=frozenset({"on", "1.0"}),
                deny_for=frozenset({"yes", "café"}),
                version="1.0",
            )
        ),
        [
            _case(_action("act", tags=["yes"])),
            _case(_action("act", tags=["café"])),
            _case(_action("act", kind="on")),
            _case(_action("act", environment="1.0")),
        ],
    ),
}

_SCHEMA_LINE = '"schema": "locus.control.policy/v1"'


def _json_doc(body: str) -> str:
    """A JSON policy document with the schema line and ``body`` as further keys."""
    return "{" + _SCHEMA_LINE + ", " + body + "}\n"


MALFORMED: dict[str, tuple[str, str]] = {
    "unknown-key": ("json", _json_doc('"requier_human_for": []')),
    "bad-schema": ("json", '{"schema": "locus.control.policy/v2"}\n'),
    "not-a-mapping": ("json", "[]\n"),
    "invalid-json": ("json", '{"schema":\n'),
    "label-not-string": ("json", _json_doc('"require_human_for": ["prod", 3]')),
    "labels-not-list": ("json", _json_doc('"deny_for": "production"')),
    "empty-label": ("json", _json_doc('"deny_for": [""]')),
    "negative-blast-radius": ("json", _json_doc('"max_blast_radius": -1')),
    "boolean-blast-radius": ("json", _json_doc('"max_blast_radius": true')),
    "float-blast-radius": ("json", _json_doc('"max_blast_radius": 1.0')),
    "bad-severity": ("json", _json_doc('"min_severity": "severe"')),
    "negative-usd": ("json", _json_doc('"spend_limit_usd": -5')),
    "score-not-number": ("json", _json_doc('"require_verification_score": "high"')),
    "version-not-string": ("json", _json_doc('"version": 3')),
    "governs-unknown-key": ("json", _json_doc('"governs": {"regions": []}')),
    "governs-not-mapping": ("json", _json_doc('"governs": []')),
    "governs-label-not-string": ("json", _json_doc('"governs": {"kinds": ["payment", null]}')),
    "yaml-unquoted-boolean": (
        "yaml",
        'schema: "locus.control.policy/v1"\nrequire_human_for:\n  - yes\n',
    ),
    "yaml-unquoted-float": ("yaml", 'schema: "locus.control.policy/v1"\ndeny_for:\n  - 1.0\n'),
    "yaml-infinite": ("yaml", 'schema: "locus.control.policy/v1"\nspend_limit_usd: .inf\n'),
    "invalid-yaml": ("yaml", 'schema: "locus.control.policy/v1"\ndeny_for: [unclosed\n'),
}


def decide(policy: ControlPolicy, case: Mapping[str, Any]) -> dict[str, Any]:
    """What the policy decides for one fixture case, and every rule it weighed."""
    action = action_from_mapping(case["action"])
    verdict = _verdict(case.get("verdict"))
    finding = _finding(case["finding"]["severity"]) if case.get("finding") else None
    spent = float(case.get("spent_usd", 0.0))
    decision = approve(action, policy=policy, finding=finding, verdict=verdict, spent_usd=spent)
    rules = evaluate_rules(action, policy=policy, finding=finding, verdict=verdict, spent_usd=spent)
    return {
        "outcome": decision.outcome,
        "checks": list(decision.checks),
        "reason": decision.reason,
        "rules": [{"rule": r.rule, "outcome": r.outcome, "detail": r.detail} for r in rules],
    }


def lint_dict(document: PolicyDocument) -> dict[str, Any]:
    """The lint report of a document without its path, so it does not depend on where it was read."""
    return lint_policy(
        document.policy, environments=document.environments, kinds=document.kinds
    ).to_dict()


def malformed_error(text: str, fmt: str) -> dict[str, Any]:
    try:
        PolicyDocument.loads(text, format="json" if fmt == "json" else "yaml", source="")
    except PolicyFileError as exc:
        message = str(exc)
        if message.startswith(("not valid JSON:", "not valid YAML:")):
            # The parser's own detail varies between interpreter and library
            # versions; the SDK's prefix is the stable part.
            message = message.split(":", 1)[0]
        return {"key": exc.key, "message": message}
    raise RuntimeError("a malformed document loaded")


def load_policies(policies: Path) -> dict[str, ControlPolicy]:
    """The committed policy documents by name, read back through the SDK."""
    return {
        path.name.split(".")[0]: PolicyDocument.load(path).policy
        for path in sorted(policies.glob("*.policy.*"))
    }


def write_policies(out: Path) -> dict[str, ControlPolicy]:
    policies: dict[str, ControlPolicy] = {}
    for name, (fmt, document, cases) in POLICIES.items():
        path = out / f"{name}.policy.{fmt}"
        path.parent.mkdir(parents=True, exist_ok=True)
        document.dump(path)
        reloaded = PolicyDocument.load(path)
        _expect(reloaded.policy == document.policy, f"{name} round-trips")
        _expect(reloaded.environments == document.environments, f"{name} governs.environments")
        _expect(reloaded.kinds == document.kinds, f"{name} governs.kinds")
        _write_text(out / f"{name}.hash", policy_hash(document.policy))
        _write_json(out / f"{name}.lint.json", lint_dict(document))
        _write_json(
            out / f"{name}.decisions.json",
            [{**case, "expected": decide(document.policy, case)} for case in cases],
        )
        policies[name] = document.policy
    for name, (fmt, text) in MALFORMED.items():
        _write_text(out / "malformed" / f"{name}.policy.{fmt}", text)
        _write_json(out / "malformed" / f"{name}.error.json", malformed_error(text, fmt))
    return policies


# --------------------------------------------------------------------------- approvals


_REFUND_ARGS: dict[str, Any] = {"order_id": "4821", "amount_usd": 120.0}

DIGEST_CASES: list[dict[str, Any]] = [
    {
        "principal": "agent-7",
        "tool": "issue_refund",
        "arguments": _REFUND_ARGS,
        "context": None,
        "note": "",
    },
    {
        "principal": "agent-7",
        "tool": "issue_refund",
        "arguments": {"amount_usd": 120.0, "order_id": "4821"},
        "context": None,
        "note": "key order does not matter: the same digest as the first case",
    },
    {
        "principal": "agent-7",
        "tool": "issue_refund",
        "arguments": _REFUND_ARGS,
        "context": {},
        "note": "an empty context hashes like no context",
    },
    {
        "principal": "agent-7",
        "tool": "issue_refund",
        "arguments": _REFUND_ARGS,
        "context": {"policy_version": "refunds-2026-11", "thread_id": "t1"},
        "note": "a context is part of the call",
    },
    {
        "principal": "agent-7",
        "tool": "issue_refund",
        "arguments": {"order_id": "4821", "amount_usd": 120},
        "context": None,
        "note": "120 is not 120.0",
    },
    {
        "principal": "agent-8",
        "tool": "issue_refund",
        "arguments": _REFUND_ARGS,
        "context": None,
        "note": "the principal is part of the call",
    },
    {
        "principal": "agent-7",
        "tool": "send_email",
        "arguments": {
            "to": "ops@example.com",
            "subject": "Résumé 😀",
            "body": "line1\nline2",
            "cc": [],
        },
        "context": None,
        "note": "",
    },
    {
        "principal": "agent-7",
        "tool": "update_record",
        "arguments": {"id": 7, "patch": {"b": [1, 2.0], "a": {"nested": None}}},
        "context": None,
        "note": "",
    },
]


def digest_case(case: Mapping[str, Any]) -> dict[str, Any]:
    """The digest and first approval id the SDK gives one call."""
    digest = call_digest(case["principal"], case["tool"], case["arguments"], case["context"])
    approval_id = InMemoryApprovals().submit(
        case["principal"], case["tool"], case["arguments"], context=case["context"]
    )
    return {"digest": digest, "approval_id": approval_id}


def _submit(**fields: Any) -> dict[str, Any]:
    return {
        "op": "submit",
        "principal": "agent-7",
        "tool": "issue_refund",
        "arguments": _REFUND_ARGS,
        "reason": "labels ['production'] require human approval",
        "labels": ["payment", "production"],
        "context": None,
        **fields,
    }


def _decide(ref: int, verdict: str, by: str, **fields: Any) -> dict[str, Any]:
    return {"op": "decide", "ref": ref, "verdict": verdict, "by": by, **fields}


def _rule(**fields: Any) -> dict[str, Any]:
    return {
        "labels": [],
        "approvers": [],
        "roles": [],
        "quorum": 1,
        "allow_self_approval": False,
        "name": "",
        **fields,
    }


SEQUENCES: dict[str, dict[str, Any]] = {
    "approve-then-consume": {
        "authority": None,
        "steps": [
            _submit(),
            {"op": "state", "ref": 0},
            {"op": "pending"},
            _decide(0, "approved", "olga", reason="verified with the customer"),
            {"op": "state", "ref": 0},
            {"op": "consume", "ref": 0},
            {"op": "pending"},
            _submit(),
        ],
    },
    "deny-and-resubmit": {
        "authority": None,
        "steps": [
            _submit(),
            _decide(0, "denied", "olga", reason="duplicate of an earlier refund"),
            _submit(),
            {"op": "consume", "ref": 0},
            _submit(),
        ],
    },
    "edited-arguments": {
        "authority": None,
        "steps": [
            _submit(),
            _decide(0, "approved", "olga", arguments={"amount": 100.0}),
            _decide(0, "denied", "olga", arguments={"order_id": "4821", "amount_usd": 100.0}),
            _decide(0, "approved", "olga", arguments={"order_id": "4821", "amount_usd": 100.0}),
        ],
    },
    "errors": {
        "authority": None,
        "steps": [
            _submit(),
            {"op": "consume", "ref": 0},
            {
                "op": "decide",
                "approval_id": "appr-0000000000000000-1",
                "verdict": "approved",
                "by": "olga",
            },
            {"op": "state", "approval_id": "appr-0000000000000000-1"},
            _decide(0, "approved", "olga"),
            _decide(0, "approved", "dan"),
            {"op": "consume", "ref": 0},
            {"op": "consume", "ref": 0},
        ],
    },
    "quorum-of-two": {
        "authority": {
            "rules": [
                _rule(labels=["production"], approvers=["olga", "dan", "eve"], quorum=2, name="SRE")
            ],
        },
        "steps": [
            _submit(),
            _decide(0, "approved", "olga"),
            {"op": "state", "ref": 0},
            _decide(0, "approved", "olga"),
            _decide(0, "approved", "dan"),
        ],
    },
    "authority-refusals": {
        "authority": {
            "rules": [_rule(labels=["payment"], roles=["finance"], name="Finance")],
            "roles": {"pat": ["finance"]},
        },
        "steps": [
            _submit(),
            _decide(0, "approved", "mallory"),
            _decide(0, "approved", "agent-7"),
            _decide(0, "approved", "pat"),
        ],
    },
    "delegation": {
        "authority": {
            "rules": [_rule(labels=["production"], approvers=["olga"])],
            "delegations": [
                {
                    "grantor": "olga",
                    "grantee": "dan",
                    "expires_at": "2026-06-01T00:00:00+00:00",
                    "labels": [],
                },
                {
                    "grantor": "olga",
                    "grantee": "eve",
                    "expires_at": "2025-06-01T00:00:00+00:00",
                    "labels": [],
                },
            ],
        },
        "steps": [
            _submit(),
            _decide(0, "approved", "eve"),
            _decide(0, "approved", "dan"),
        ],
    },
    "count-once": {
        "authority": {
            "rules": [
                _rule(labels=["payment"], roles=["finance"], name="Finance"),
                _rule(labels=["payment"], approvers=["cfo"], name="CFO"),
            ],
            "roles": {"pat": ["finance"], "pat2": ["finance"], "cfo": ["finance"]},
            "count_once": True,
        },
        "steps": [
            _submit(),
            _decide(0, "approved", "pat"),
            _decide(0, "approved", "pat2"),
            _decide(0, "approved", "cfo"),
        ],
    },
    "authority-denial": {
        "authority": {
            "rules": [_rule(labels=["production"], approvers=["olga", "dan"], quorum=2)],
        },
        "steps": [
            _submit(),
            _decide(0, "approved", "olga"),
            _decide(0, "denied", "dan", reason="the customer withdrew the request"),
            {"op": "pending"},
        ],
    },
    "uncovered-action": {
        "authority": {"rules": [_rule(labels=["payment"], approvers=["pat"])]},
        "steps": [
            _submit(labels=["production"]),
            _decide(0, "approved", "pat"),
        ],
    },
}


def build_authority(spec: Mapping[str, Any] | None) -> ApprovalAuthority | None:
    if spec is None:
        return None
    roles = {principal: frozenset(names) for principal, names in spec.get("roles", {}).items()}
    return ApprovalAuthority(
        rules=tuple(
            ApproverRule(
                labels=frozenset(rule["labels"]),
                approvers=frozenset(rule["approvers"]),
                roles=frozenset(rule["roles"]),
                quorum=rule["quorum"],
                allow_self_approval=rule["allow_self_approval"],
                name=rule["name"],
            )
            for rule in spec["rules"]
        ),
        delegations=tuple(
            Delegation(
                grantor=grant["grantor"],
                grantee=grant["grantee"],
                expires_at=grant["expires_at"],
                labels=frozenset(grant["labels"]),
            )
            for grant in spec.get("delegations", [])
        ),
        roles_of=(lambda principal: roles.get(principal, frozenset())) if roles else None,
        count_once=spec.get("count_once", False),
    )


def replay_sequence(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Run one decision sequence against an in-memory store under the fixed clock."""
    with fixed_clock():
        store = InMemoryApprovals(build_authority(spec["authority"]))
        ids: list[str] = []
        results: list[dict[str, Any]] = []
        for step in spec["steps"]:
            op = step["op"]
            try:
                if op == "submit":
                    approval_id = store.submit(
                        step["principal"],
                        step["tool"],
                        step["arguments"],
                        reason=step["reason"],
                        labels=step["labels"],
                        context=step["context"],
                    )
                    ids.append(approval_id)
                    results.append({"approval_id": approval_id})
                    continue
                if op == "pending":
                    results.append({"pending": [record.approval_id for record in store.pending()]})
                    continue
                approval_id = step["approval_id"] if "approval_id" in step else ids[step["ref"]]
                if op == "state":
                    results.append({"state": store.state(approval_id)})
                elif op == "decide":
                    record = store.decide(
                        approval_id,
                        step["verdict"],
                        by=step["by"],
                        arguments=step.get("arguments"),
                        reason=step.get("reason", ""),
                    )
                    results.append({"status": record.status})
                elif op == "consume":
                    results.append({"status": store.consume(approval_id).status})
                else:
                    raise RuntimeError(f"unknown step {op!r}")
            except (ApprovalAuthorityError, ValueError, KeyError) as exc:
                results.append({"error": {"type": type(exc).__name__, "message": str(exc)}})
        final: dict[str, Any] = {}
        for approval_id in dict.fromkeys(ids):
            stored = store.get(approval_id)
            if stored is None:
                raise RuntimeError(f"record {approval_id} vanished from the store")
            final[approval_id] = asdict(stored)
    return {"results": results, "final": final}


def write_approvals(out: Path) -> list[dict[str, Any]]:
    digests = [{**case, **digest_case(case)} for case in DIGEST_CASES]
    _expect(
        digests[0]["digest"] == digests[1]["digest"] == digests[2]["digest"], "digest equalities"
    )
    _expect(digests[0]["digest"] != digests[4]["digest"], "120 differs from 120.0")
    _write_json(out / "digests.json", digests)
    for name, spec in SEQUENCES.items():
        _write_json(out / "sequences" / f"{name}.json", {**spec, **replay_sequence(spec)})
    return digests


# --------------------------------------------------------------------------- admission


class _Advisor:
    """A control model that answers from the fixture, or fails."""

    def __init__(self, spec: Mapping[str, Any]) -> None:
        self._spec = spec

    def advise(self, action: Action) -> str | None:  # noqa: ARG002 — the protocol's signature
        if "raises" in self._spec:
            raise RuntimeError(self._spec["raises"])
        advice: str | None = self._spec.get("advice")
        return advice


def _admission(
    name: str,
    policy: str,
    action: dict[str, Any],
    *,
    verdict: dict[str, Any] | None = None,
    finding: str | None = None,
    approved_by: str | None = None,
    ledger: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
    advisor: dict[str, Any] | None = None,
    judgement: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "policy": policy,
        "action": action,
        "verdict": verdict,
        "finding": {"severity": finding} if finding else None,
        "approved_by": approved_by,
        "ledger": ledger,
        "context": context,
        "advisor": advisor,
        "judgement": judgement,
        "clock": START.isoformat(),
    }


_STAGING = _action("rotate_key", asset="svc-1", environment="staging", kind="maintenance")
_PRODUCTION = _action("rotate_key", asset="svc-1", environment="production", kind="maintenance")
_REFUND = _action(
    "issue_refund", asset="order-4821", environment="staging", kind="payment", cost_usd=150.0
)


def _judge(
    verdict: str, source: str = "model", reason: str = "", error: str | None = None
) -> dict[str, Any]:
    return {
        "tool": "rotate_key",
        "verdict": verdict,
        "reason": reason,
        "model": "judge-model-v1" if source in ("model", "cache") else "",
        "source": source,
        "error": error,
    }


ADMISSION_CASES: list[dict[str, Any]] = [
    _admission("allow", "default", _STAGING, verdict=_GOOD),
    _admission("require-human-production", "default", _PRODUCTION, verdict=_GOOD),
    _admission("approved-by", "default", _PRODUCTION, verdict=_GOOD, approved_by="olga"),
    _admission(
        "approved-by-does-not-override-deny",
        "default",
        _PRODUCTION,
        verdict=_REFUTED,
        approved_by="olga",
    ),
    _admission("no-verdict", "default", _STAGING),
    _admission(
        "deny-labels",
        "locked-down",
        _action("drop_table", asset="db-1", environment="production", kind="delete"),
    ),
    _admission("finding-below-minimum", "default", _STAGING, verdict=_GOOD, finding="info"),
    _admission("finding-meets-minimum", "default", _STAGING, verdict=_GOOD, finding="high"),
    _admission("ledger-deny", "refunds", _REFUND, ledger={"scope": "acme", "spent": 9900.0}),
    _admission("ledger-allow", "refunds", _REFUND, ledger={"scope": "acme", "spent": 100.0}),
    _admission(
        "ledger-hold-spends-nothing",
        "refunds",
        {**_REFUND, "cost_usd": 750.0},
        ledger={"scope": "acme", "spent": 100.0},
    ),
    _admission(
        "context",
        "default",
        _STAGING,
        verdict=_GOOD,
        context={
            "rule": "maintenance-window",
            "principal": "agent-7",
            "arguments": {"key": "k-1", "ttl": 3600},
        },
    ),
    _admission("advisor-escalates", "default", _STAGING, verdict=_GOOD, advisor={"advice": "deny"}),
    _admission(
        "advisor-holds", "default", _STAGING, verdict=_GOOD, advisor={"advice": "require_human"}
    ),
    _admission("advisor-agrees", "default", _STAGING, verdict=_GOOD, advisor={"advice": "allow"}),
    _admission(
        "advisor-unrecognised",
        "default",
        _STAGING,
        verdict=_GOOD,
        advisor={"advice": "probably fine"},
    ),
    _admission("advisor-silent", "default", _STAGING, verdict=_GOOD, advisor={"advice": None}),
    _admission(
        "advisor-fails",
        "default",
        _STAGING,
        verdict=_GOOD,
        advisor={"raises": "model endpoint unavailable"},
    ),
    _admission(
        "advisor-not-asked-on-hold",
        "default",
        _PRODUCTION,
        verdict=_GOOD,
        advisor={"advice": "deny"},
    ),
    _admission(
        "judge-holds",
        "default",
        _STAGING,
        verdict=_GOOD,
        judgement=_judge("hold", reason="rotates a production secret"),
    ),
    _admission(
        "judge-denies",
        "default",
        _STAGING,
        verdict=_GOOD,
        judgement=_judge("deny", reason="the key is shared with billing"),
    ),
    _admission(
        "judge-allows",
        "default",
        _STAGING,
        verdict=_GOOD,
        judgement=_judge("allow", reason="routine rotation"),
    ),
    _admission(
        "judge-allows-cached",
        "default",
        _STAGING,
        verdict=_GOOD,
        judgement=_judge("allow", source="cache", reason="routine rotation"),
    ),
    _admission(
        "judge-read-only",
        "default",
        _STAGING,
        verdict=_GOOD,
        judgement=_judge("allow", source="read_only", reason="read-only tool"),
    ),
    _admission(
        "judge-error",
        "default",
        _STAGING,
        verdict=_GOOD,
        judgement=_judge(
            "hold", source="error", reason="the judge failed closed", error="timeout after 10s"
        ),
    ),
    _admission(
        "judge-budget",
        "default",
        _STAGING,
        verdict=_GOOD,
        judgement=_judge("hold", source="budget", reason="judge call budget spent"),
    ),
    _admission(
        "judge-unknown-word",
        "default",
        _STAGING,
        verdict=_GOOD,
        judgement=_judge("maybe", reason="unclear"),
    ),
    _admission(
        "judge-cannot-widen",
        "default",
        _PRODUCTION,
        verdict=_GOOD,
        judgement=_judge("allow", reason="looks fine"),
    ),
    _admission(
        "advisor-and-judge",
        "default",
        _STAGING,
        verdict=_GOOD,
        advisor={"advice": "allow"},
        judgement=_judge("hold", reason="unusual hour"),
    ),
    _admission(
        "sandbox-missing",
        "sandbox",
        _action("fetch", asset="https://example.com", tags=["network"]),
    ),
    _admission(
        "sandbox-present",
        "sandbox",
        _action("fetch", asset="https://example.com", tags=["network", "sandboxed"]),
    ),
    _admission(
        "floats-policy",
        "floats",
        _action("ping", asset="svc-1", environment="staging"),
        verdict=_WEAK,
    ),
    _admission("quoted-labels", "quoted-labels", _action("act", asset="svc-1", tags=["café"])),
]


def run_admission_case(
    case: Mapping[str, Any], policies: Mapping[str, ControlPolicy]
) -> dict[str, Any]:
    """Admit one fixture case on a fresh trail and return what it wrote."""
    clock = case["clock"]
    trail = AuditTrail(clock=lambda: clock)
    ledger_spec = case["ledger"]
    ledger = InMemorySpendLedger() if ledger_spec is not None else None
    scope = ledger_spec["scope"] if ledger_spec is not None else "default"
    if ledger is not None and ledger_spec["spent"]:
        ledger.record(scope, ledger_spec["spent"], action="earlier")
    judgement = JudgeVerdict(**case["judgement"]) if case["judgement"] is not None else None
    admitted, error = True, None
    try:
        admit_sync(
            action_from_mapping(case["action"]),
            lambda: "performed",
            policy=policies[case["policy"]],
            finding=_finding(case["finding"]["severity"]) if case["finding"] else None,
            verdict=_verdict(case["verdict"]),
            trail=trail,
            approved_by=case["approved_by"],
            ledger=ledger,
            spend_scope=scope,
            context=case["context"],
            advisor=_Advisor(case["advisor"]) if case["advisor"] is not None else None,
            judgement=judgement,
        )
    except AdmissionError as exc:
        admitted, error = False, str(exc)
    return {
        "admitted": admitted,
        "error": error,
        "line": trail.export_jsonl(),
        "spent_after": ledger.spent(scope) if ledger is not None else None,
    }


def write_admission(out: Path, policies: Mapping[str, ControlPolicy]) -> None:
    cases = [{**case, "expected": run_admission_case(case, policies)} for case in ADMISSION_CASES]
    _expect(len({case["name"] for case in cases}) == len(cases), "admission case names are unique")
    _write_json(out / "cases.json", cases)


# --------------------------------------------------------------------------- entry points


def generate(out: Path) -> None:
    """Write every generated directory under ``out``, replacing what is there."""
    for name in GENERATED:
        shutil.rmtree(out / name, ignore_errors=True)
    primary, rotated = signers()
    trails = write_audit(out / "audit", primary, rotated)
    policies = write_policies(out / "policies")
    digests = write_approvals(out / "approvals")
    write_admission(out / "admission", policies)
    first = DIGEST_CASES[0]
    approval_body = {
        "principal": first["principal"],
        "tool": first["tool"],
        "arguments": first["arguments"],
    }
    vectors = build_vectors(trails["unsigned"], approval_body)
    _expect(
        next(v for v in vectors if v["name"] == "approval-call")["sha256"] == digests[0]["digest"],
        "approval-call vector",
    )
    _write_json(out / "canonical-json" / "vectors.json", vectors)


def tree_diff(expected: Path, actual: Path) -> list[str]:
    """Relative paths that differ between two fixture trees, over the generated directories."""

    def files(root: Path) -> dict[str, bytes]:
        found: dict[str, bytes] = {}
        for name in GENERATED:
            for path in sorted((root / name).rglob("*")):
                if path.is_file():
                    found[path.relative_to(root).as_posix()] = path.read_bytes()
        return found

    before, after = files(expected), files(actual)
    return sorted(
        {f"missing: {p}" for p in before.keys() - after.keys()}
        | {f"unexpected: {p}" for p in after.keys() - before.keys()}
        | {f"changed: {p}" for p in before.keys() & after.keys() if before[p] != after[p]}
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--out", type=Path, default=FIXTURES, help="where to write (default: conformance/v1)"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="regenerate into a temporary directory and report differences",
    )
    args = parser.parse_args(argv)
    if not args.check:
        generate(args.out)
        print(f"wrote {args.out}")
        return 0
    with tempfile.TemporaryDirectory() as scratch:
        generate(Path(scratch))
        differences = tree_diff(args.out, Path(scratch))
    for line in differences:
        print(line)
    if differences:
        print(
            f"{len(differences)} difference(s): a format change is a new conformance version",
            file=sys.stderr,
        )
        return 1
    print("conformance fixtures are current")
    return 0


if __name__ == "__main__":
    sys.exit(main())
