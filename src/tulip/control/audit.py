# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Tamper-evident audit trail — every agent action as replayable evidence.

A :class:`AuditTrail` records agent actions into a hash chain: each record
commits to the hash of the one before it, so a later edit, reorder, or
deletion from the middle breaks the chain and :meth:`AuditTrail.verify`
returns ``False``. The trail exports as JSONL for shipping to a SIEM.

Truncation is the exception, and it is worth stating plainly: lopping records
off the *end* of a chain leaves a shorter chain that still verifies, because
no link can attest to one that was never handed to it. Anchor
:attr:`AuditTrail.head` externally and pass it to
``verify(expected_head=...)`` to close that gap — see :meth:`AuditTrail.verify`.

**Signing.** A hash chain alone proves the records are consistent with each
other, not who wrote them: anyone who can write the log can rebuild the whole
chain around an edit and it verifies. Give the trail a signer and every record's
hash is signed with Ed25519, naming the key it was signed with::

    signer = Ed25519Signer.from_pem(private_pem, key_id="audit-2026-09")
    trail = AuditTrail(signer=signer)
    ...
    exported = trail.export_jsonl()

    # Anyone with only the public key, and no Tulip runtime state:
    verify_jsonl(exported, keys={"audit-2026-09": public_pem}, expected_head=anchor)

A rebuilt chain then fails, because it cannot be signed with a key the verifier
trusts. Keys rotate by naming a new signer (:meth:`AuditTrail.use_signer`) and
handing the verifier both public keys. Signing needs the ``cryptography``
package (``pip install "tulip-agents[audit]"``); an unsigned trail needs nothing
and exports exactly as before.

This is a *supporting property* of a trustworthy agent — the agent leaves a
record that holds up — not a governance/policy product. It does not block or
enforce; it makes the record auditable after the fact.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable


# The chain's anchor — `prev_hash` of the first record.
_GENESIS = "0" * 64

#: Fields a signed record adds. Left out of an unsigned record's export, so the
#: format a SIEM already parses does not change for trails that do not sign.
_SIGNATURE_FIELDS = ("key_id", "signature")


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _entry_hash(
    seq: int,
    ts: str,
    event_type: str,
    payload: Mapping[str, Any],
    prev_hash: str,
) -> str:
    """SHA-256 over the canonical (sorted, compact) record body + prev hash."""
    canonical = json.dumps(
        {
            "seq": seq,
            "ts": ts,
            "event_type": event_type,
            "payload": dict(payload),
            "prev_hash": prev_hash,
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _crypto() -> tuple[Any, Any, Any]:
    """``serialization``, ``ed25519`` and ``InvalidSignature``, imported lazily."""
    try:
        from cryptography.exceptions import InvalidSignature  # noqa: PLC0415
        from cryptography.hazmat.primitives import serialization  # noqa: PLC0415
        from cryptography.hazmat.primitives.asymmetric import ed25519  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise ImportError(
            "Signed audit trails need the cryptography package:\n"
            '    pip install "tulip-agents[audit]"'
        ) from exc
    return serialization, ed25519, InvalidSignature


@runtime_checkable
class AuditSigner(Protocol):
    """Signs a record's hash. ``key_id`` names the key a verifier should use."""

    key_id: str

    def sign(self, data: bytes) -> bytes:
        """A signature over ``data``."""
        ...


class Ed25519Signer:
    """An Ed25519 :class:`AuditSigner`.

    Args:
        private_key: A ``cryptography`` Ed25519 private key.
        key_id: The name verifiers look the public key up by. Defaults to the
            first 16 hex characters of the SHA-256 of the raw public key, so the
            same key always gets the same id.
    """

    def __init__(self, private_key: Any, *, key_id: str | None = None) -> None:
        serialization, _, _ = _crypto()
        self._key = private_key
        raw = private_key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
        self.key_id: str = key_id or hashlib.sha256(raw).hexdigest()[:16]

    @classmethod
    def generate(cls, *, key_id: str | None = None) -> Ed25519Signer:
        """A signer with a new random key."""
        _, ed25519, _ = _crypto()
        return cls(ed25519.Ed25519PrivateKey.generate(), key_id=key_id)

    @classmethod
    def from_pem(
        cls, pem: bytes | str, *, password: bytes | None = None, key_id: str | None = None
    ) -> Ed25519Signer:
        """A signer from a PEM-encoded (PKCS#8) Ed25519 private key."""
        serialization, ed25519, _ = _crypto()
        data = pem.encode("ascii") if isinstance(pem, str) else pem
        key = serialization.load_pem_private_key(data, password=password)
        if not isinstance(key, ed25519.Ed25519PrivateKey):
            raise TypeError("expected an Ed25519 private key")
        return cls(key, key_id=key_id)

    def private_key_pem(self, *, password: bytes | None = None) -> bytes:
        """The private key as PKCS#8 PEM, encrypted when ``password`` is given."""
        serialization, _, _ = _crypto()
        encryption = (
            serialization.BestAvailableEncryption(password)
            if password
            else serialization.NoEncryption()
        )
        pem: bytes = self._key.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, encryption
        )
        return pem

    def public_key_pem(self) -> bytes:
        """The public key as SubjectPublicKeyInfo PEM — what a verifier needs."""
        serialization, _, _ = _crypto()
        pem: bytes = self._key.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
        return pem

    def sign(self, data: bytes) -> bytes:
        signature: bytes = self._key.sign(data)
        return signature


@dataclass(frozen=True)
class AuditRecord:
    """One link in the audit chain. ``hash`` commits to ``prev_hash``."""

    seq: int
    ts: str
    event_type: str
    payload: dict[str, Any]
    prev_hash: str
    hash: str
    #: The signing key's id, when the trail signs.
    key_id: str | None = None
    #: Base64 signature over ``hash``, when the trail signs.
    signature: str | None = None


class AuditTrail:
    """An append-only, hash-chained log of agent actions.

    Append with :meth:`record` (or :meth:`record_event` for a Tulip event);
    check integrity with :meth:`verify`; ship with :meth:`export_jsonl`.
    Pass ``clock`` to make timestamps deterministic in tests, and ``signer``
    to sign every record.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], str] | None = None,
        signer: AuditSigner | None = None,
    ) -> None:
        self._records: list[AuditRecord] = []
        self._clock = clock or _utc_now_iso
        self._signer = signer

    @property
    def head(self) -> str:
        """Hash of the latest record, or the genesis anchor when empty."""
        return self._records[-1].hash if self._records else _GENESIS

    def __len__(self) -> int:
        return len(self._records)

    def use_signer(self, signer: AuditSigner | None) -> None:
        """Sign records from now on with ``signer`` — how a key is rotated.

        Records already written keep the key they were signed with; a verifier
        given both public keys accepts the whole trail.
        """
        self._signer = signer

    def record(self, event_type: str, payload: Mapping[str, Any] | None = None) -> AuditRecord:
        """Append a record committing to the current chain head."""
        seq = len(self._records)
        prev = self.head
        ts = self._clock()
        body = dict(payload or {})
        digest = _entry_hash(seq, ts, event_type, body, prev)
        key_id = signature = None
        if self._signer is not None:
            key_id = self._signer.key_id
            signature = base64.b64encode(self._signer.sign(digest.encode("ascii"))).decode("ascii")
        rec = AuditRecord(
            seq=seq,
            ts=ts,
            event_type=event_type,
            payload=body,
            prev_hash=prev,
            hash=digest,
            key_id=key_id,
            signature=signature,
        )
        self._records.append(rec)
        return rec

    def record_event(self, event: Any) -> AuditRecord:
        """Append a record for a Tulip event (duck-typed; safe scalar fields)."""
        payload: dict[str, Any] = {}
        for key in ("name", "tool", "final_message", "reason", "content", "asset"):
            val = getattr(event, key, None)
            if isinstance(val, str | int | float | bool):
                payload[key] = val
        return self.record(type(event).__name__, payload)

    def records(self) -> list[AuditRecord]:
        """A copy of the records, in order."""
        return list(self._records)

    def verify(
        self,
        *,
        expected_head: str | None = None,
        keys: Mapping[str, bytes | str] | None = None,
    ) -> bool:
        """Whether the chain is internally consistent, optionally un-truncated and signed.

        On its own this catches every edit, reorder, and deletion **from the
        middle** of the chain: each of those leaves a record whose stored hash
        no longer matches its contents, or whose ``prev_hash`` no longer points
        at the record before it.

        It cannot, on its own, catch a *truncation*. Dropping records from the
        end — or discarding the trail entirely — leaves a shorter chain that is
        perfectly valid on its own terms, so this returns ``True``. That is a
        property of hash chains in general, not of this implementation: nothing
        inside a chain can attest to a link that was never handed to it.

        Truncation is what ``expected_head`` is for. Persist :attr:`head`
        somewhere the agent cannot reach — a WORM bucket, a append-only log, a
        transparency log, a co-signer — and pass it back here. Every attack
        above, truncation included, changes the head:

        ```python
        anchor = trail.head  # written to durable, external storage
        ...
        trail.verify(expected_head=anchor)  # False if anything was removed
        ```

        Nor can a chain alone catch a *rebuild*: rewrite a record and recompute
        every hash after it, and the chain is consistent again. That is what
        ``keys`` is for. With it, every record must carry a signature that
        verifies under the public key its ``key_id`` names, so a rebuilt chain
        fails unless it was signed with a key the verifier trusts.

        Args:
            expected_head: The chain head recorded out-of-band. When given, the
                trail must also *end* on this hash. Omit it and truncation goes
                undetected — see above.
            keys: Trusted public keys as PEM, by key id. When given, an unsigned
                record, an unknown key id, or a bad signature fails verification.

        Returns:
            ``True`` if the chain is intact, ends at ``expected_head`` when one
            was supplied, and every record is validly signed when ``keys`` was.
        """
        prev = _GENESIS
        for i, rec in enumerate(self._records):
            if rec.seq != i or rec.prev_hash != prev:
                return False
            if _entry_hash(rec.seq, rec.ts, rec.event_type, rec.payload, rec.prev_hash) != rec.hash:
                return False
            prev = rec.hash
        if expected_head is not None and self.head != expected_head:
            return False
        if keys is not None:
            return _signatures_valid(self._records, keys)
        return True

    def export_jsonl(self) -> str:
        """The chain as newline-delimited JSON — one record per line, SIEM-ready."""
        return "\n".join(json.dumps(_exported(rec), default=str) for rec in self._records)

    @classmethod
    def from_records(cls, records: Iterable[AuditRecord]) -> AuditTrail:
        """Rebuild a trail from records (e.g. to :meth:`verify` an exported chain)."""
        trail = cls()
        trail._records = list(records)
        return trail


def _exported(rec: AuditRecord) -> dict[str, Any]:
    data = asdict(rec)
    for name in _SIGNATURE_FIELDS:
        if data[name] is None:
            del data[name]
    return data


def _signatures_valid(records: list[AuditRecord], keys: Mapping[str, bytes | str]) -> bool:
    serialization, ed25519, invalid_signature = _crypto()
    loaded: dict[str, Any] = {}
    for rec in records:
        if rec.key_id is None or rec.signature is None or rec.key_id not in keys:
            return False
        if rec.key_id not in loaded:
            pem = keys[rec.key_id]
            data = pem.encode("ascii") if isinstance(pem, str) else pem
            public = serialization.load_pem_public_key(data)
            if not isinstance(public, ed25519.Ed25519PublicKey):
                return False
            loaded[rec.key_id] = public
        try:
            loaded[rec.key_id].verify(
                base64.b64decode(rec.signature, validate=True), rec.hash.encode("ascii")
            )
        except (invalid_signature, binascii.Error, ValueError):
            return False
    return True


def verify_jsonl(
    text: str,
    *,
    keys: Mapping[str, bytes | str] | None = None,
    expected_head: str | None = None,
) -> bool:
    """Verify an exported trail from its JSONL alone.

    Needs no Tulip runtime state: an auditor with the export and the public
    keys runs this and nothing else. A line that is not a record fails.
    """
    records: list[AuditRecord] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            records.append(AuditRecord(**json.loads(line)))
        except (TypeError, ValueError):
            return False
    return AuditTrail.from_records(records).verify(expected_head=expected_head, keys=keys)


__all__ = ["AuditRecord", "AuditSigner", "AuditTrail", "Ed25519Signer", "verify_jsonl"]
