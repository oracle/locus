# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""A signed audit trail that someone outside the runtime can verify.

A hash chain proves records are consistent with each other; anyone who can write
the log can rebuild the chain around an edit and it still verifies. These tests
pin what signing adds: an export verifies from the public key alone, and a
modified, deleted, reordered, truncated, or rebuilt-and-re-signed trail does not.
"""

from __future__ import annotations

import json

import pytest


pytest.importorskip("cryptography")

from tulip.control import AuditTrail, Ed25519Signer, verify_jsonl  # noqa: E402


def _signed(
    count: int = 4, signer: Ed25519Signer | None = None
) -> tuple[AuditTrail, Ed25519Signer]:
    signer = signer or Ed25519Signer.generate(key_id="audit-1")
    trail = AuditTrail(signer=signer)
    for i in range(count):
        trail.record("action-admission", {"action": f"refund-{i}", "outcome": "allow"})
    return trail, signer


def _keys(*signers: Ed25519Signer) -> dict[str, bytes]:
    return {s.key_id: s.public_key_pem() for s in signers}


def test_an_export_verifies_from_the_public_key_alone() -> None:
    trail, signer = _signed()

    exported = trail.export_jsonl()

    assert all(json.loads(line)["signature"] for line in exported.splitlines())
    assert verify_jsonl(exported, keys=_keys(signer), expected_head=trail.head)


def test_a_modified_record_fails() -> None:
    trail, signer = _signed()
    lines = trail.export_jsonl().splitlines()
    record = json.loads(lines[1])
    record["payload"]["outcome"] = "deny"
    lines[1] = json.dumps(record)

    assert not verify_jsonl("\n".join(lines), keys=_keys(signer))


def test_a_rebuilt_chain_verifies_as_a_chain_but_not_under_the_trusted_key() -> None:
    """The attack a keyless chain cannot catch."""
    trail, signer = _signed()
    forger = Ed25519Signer.generate(key_id=signer.key_id)  # same name, different key
    forged, _ = _signed(signer=forger)

    assert verify_jsonl(forged.export_jsonl()), "a rebuilt chain is internally consistent"
    assert not verify_jsonl(forged.export_jsonl(), keys=_keys(signer))


@pytest.mark.parametrize("mutation", ["delete-middle", "reorder"])
def test_a_deleted_or_reordered_record_fails(mutation: str) -> None:
    trail, signer = _signed()
    lines = trail.export_jsonl().splitlines()
    if mutation == "delete-middle":
        del lines[1]
    else:
        lines[1], lines[2] = lines[2], lines[1]

    assert not verify_jsonl("\n".join(lines), keys=_keys(signer))


def test_truncation_needs_the_anchored_head() -> None:
    trail, signer = _signed()
    truncated = "\n".join(trail.export_jsonl().splitlines()[:2])

    assert verify_jsonl(truncated, keys=_keys(signer))
    assert not verify_jsonl(truncated, keys=_keys(signer), expected_head=trail.head)


def test_a_rotated_key_verifies_with_both_public_keys() -> None:
    first = Ed25519Signer.generate(key_id="audit-1")
    second = Ed25519Signer.generate(key_id="audit-2")
    trail, _ = _signed(count=2, signer=first)
    trail.use_signer(second)
    trail.record("action-admission", {"action": "refund-after-rotation"})

    assert trail.verify(keys=_keys(first, second))
    assert not trail.verify(keys=_keys(first))


def test_an_unsigned_record_fails_when_keys_are_required() -> None:
    signer = Ed25519Signer.generate(key_id="audit-1")
    trail = AuditTrail()
    trail.record("action-admission", {"action": "refund"})

    assert trail.verify()
    assert not trail.verify(keys=_keys(signer))


def test_an_unsigned_trail_exports_exactly_as_before() -> None:
    trail = AuditTrail(clock=lambda: "2026-09-15T00:00:00+00:00")
    trail.record("action-admission", {"action": "refund"})

    assert set(json.loads(trail.export_jsonl())) == {
        "seq",
        "ts",
        "event_type",
        "payload",
        "prev_hash",
        "hash",
    }


def test_a_signer_round_trips_through_pem_with_a_stable_default_id() -> None:
    signer = Ed25519Signer.generate()
    pem = signer.private_key_pem(password=b"correct horse")

    again = Ed25519Signer.from_pem(pem, password=b"correct horse")

    assert again.key_id == signer.key_id
    trail, _ = _signed(signer=again)
    assert trail.verify(keys=_keys(signer))


def test_a_line_that_is_not_a_record_fails() -> None:
    trail, signer = _signed()

    assert not verify_jsonl(trail.export_jsonl() + '\n{"not": "a record"}', keys=_keys(signer))


def _ec_private_pem() -> tuple[bytes, bytes]:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    key = ec.generate_private_key(ec.SECP256R1())
    private = key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    )
    public = key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    return private, public


def test_a_signer_refuses_a_key_that_is_not_ed25519() -> None:
    private, _ = _ec_private_pem()

    with pytest.raises(TypeError, match="Ed25519"):
        Ed25519Signer.from_pem(private)


def test_a_trusted_key_that_is_not_ed25519_fails_verification() -> None:
    trail, signer = _signed()
    _, ec_public = _ec_private_pem()

    assert not trail.verify(keys={signer.key_id: ec_public})


def test_blank_lines_in_an_export_are_ignored() -> None:
    trail, signer = _signed()
    spaced = "\n\n".join(trail.export_jsonl().splitlines()) + "\n\n"

    assert verify_jsonl(spaced, keys=_keys(signer), expected_head=trail.head)
