# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Encryption at rest for checkpoints, at the serialisation layer every backend shares.

A checkpoint is the agent's state as JSON: the conversation, tool results,
whatever the user said. :class:`EncryptedSerializer` seals that JSON with
AES-256-GCM before a backend stores it, so the store — a bucket, a table,
a Redis key, a file — holds ciphertext. Give it to any checkpointer as
``encryption=``; the backends call it through
:meth:`~locus.memory.checkpointer.BaseCheckpointer._encode_state` and
:meth:`~locus.memory.checkpointer.BaseCheckpointer._decode_state`.

What is stored is a versioned envelope::

    {
        "v": 1,
        "alg": "A256GCM",
        "kid": "3f9a1c0e",
        "nonce": "<b64>",
        "ct": "<b64>",
        "iteration": 3,
    }

``kid`` is a fingerprint of the key (the first eight hex digits of its
SHA-256), never the key; ``iteration`` is the state's iteration counter,
kept in the clear so a history listing can report it without the key.

**What stays plaintext.** Only the state is encrypted. Everything a backend
keeps beside it for ``list()`` and ``load()`` is not: the thread id, the
checkpoint id, the write time and the expiry time, the iteration, the
stored size, the parent checkpoint id and the fork origin, and the
``metadata`` passed to ``save()`` — so keep secrets out of that metadata.

**Plaintext checkpoints keep loading.** A checkpoint written before
encryption was turned on is not an envelope; the serializer hands it back
as it is. **Keys rotate** by moving the old key to ``previous_keys``: a
checkpoint is decrypted with the key its ``kid`` names, or, for an envelope
without one, with each key in turn. A ``key_provider`` callable fetches the
current key from a vault or a key-management service; it is called for every
encryption and decryption, so cache inside it when the fetch is slow.

Key material is never logged, never part of an exception message and never
in a ``repr``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any


__all__ = [
    "ENVELOPE_ALGORITHM",
    "ENVELOPE_VERSION",
    "CheckpointEncryptionError",
    "EncryptedSerializer",
    "is_envelope",
]

#: The envelope layout this module writes.
ENVELOPE_VERSION = 1
#: The cipher every envelope names.
ENVELOPE_ALGORITHM = "A256GCM"
_KEY_BYTES = 32
_NONCE_BYTES = 12


class CheckpointEncryptionError(RuntimeError):
    """A checkpoint could not be decrypted: no key given, or none of the keys fits."""


def is_envelope(data: Any) -> bool:
    """Whether ``data`` is an encrypted checkpoint envelope rather than a plaintext state."""
    return (
        isinstance(data, Mapping)
        and data.get("alg") == ENVELOPE_ALGORITHM
        and "v" in data
        and "nonce" in data
        and "ct" in data
    )


def _aesgcm() -> Any:
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "checkpoint encryption needs the 'cryptography' package. "
            'Install with: pip install "locus-sdk[encryption]"'
        ) from exc
    return AESGCM


def _check_key(key: Any, what: str) -> bytes:
    if not isinstance(key, bytes | bytearray) or len(key) != _KEY_BYTES:
        raise ValueError(f"{what} must be {_KEY_BYTES} bytes (AES-256); got {type(key).__name__}")
    return bytes(key)


def _kid(key: bytes) -> str:
    return hashlib.sha256(key).hexdigest()[:8]


class EncryptedSerializer:
    """Seals checkpoint state with AES-256-GCM; see the module docs for what stays plaintext.

    Args:
        key: The current 32-byte key. Give either this or ``key_provider``.
        key_provider: A callable returning the current 32-byte key, called on
            every encryption and decryption (cache inside it if the fetch is
            slow).
        previous_keys: Keys checkpoints may still be sealed with, tried after
            the current one — how a key is rotated without re-writing history.

    Example:
        >>> serializer = EncryptedSerializer(
        ...     key=os.environ["LOCUS_CHECKPOINT_KEY"].encode()
        ... )
        >>> checkpointer = FileCheckpointer("./threads", encryption=serializer)
    """

    def __init__(
        self,
        key: bytes | None = None,
        *,
        key_provider: Callable[[], bytes] | None = None,
        previous_keys: Sequence[bytes] = (),
    ) -> None:
        if (key is None) == (key_provider is None):
            raise ValueError("give exactly one of key= or key_provider=")
        self._key = _check_key(key, "key") if key is not None else None
        self._key_provider = key_provider
        self._previous = [_check_key(k, "each previous key") for k in previous_keys]

    def _current_key(self) -> bytes:
        if self._key is not None:
            return self._key
        assert self._key_provider is not None  # noqa: S101 - one of the two is set
        return _check_key(self._key_provider(), "the key the provider returned")

    @property
    def key_id(self) -> str:
        """The fingerprint (``kid``) envelopes written now carry."""
        return _kid(self._current_key())

    def encrypt(self, data: Mapping[str, Any]) -> dict[str, Any]:
        """The envelope for a plaintext checkpoint state."""
        from locus.memory.time_travel import checkpoint_iteration  # noqa: PLC0415

        key = self._current_key()
        nonce = os.urandom(_NONCE_BYTES)
        plaintext = json.dumps(data, separators=(",", ":"), default=str).encode("utf-8")
        ciphertext = _aesgcm()(key).encrypt(nonce, plaintext, None)
        return {
            "v": ENVELOPE_VERSION,
            "alg": ENVELOPE_ALGORITHM,
            "kid": _kid(key),
            "nonce": base64.b64encode(nonce).decode("ascii"),
            "ct": base64.b64encode(ciphertext).decode("ascii"),
            "iteration": checkpoint_iteration(data),
        }

    def decrypt(self, data: Mapping[str, Any]) -> dict[str, Any]:
        """The plaintext state of an envelope; a plaintext state is returned as it is.

        Raises:
            CheckpointEncryptionError: The envelope is of a layout this
                release does not read, or no key decrypts it.
        """
        if not is_envelope(data):
            return dict(data)
        if data.get("v") != ENVELOPE_VERSION:
            raise CheckpointEncryptionError(
                f"checkpoint envelope version {data.get('v')!r} is not readable by this release"
            )
        from cryptography.exceptions import InvalidTag  # noqa: PLC0415

        nonce = base64.b64decode(str(data["nonce"]))
        ciphertext = base64.b64decode(str(data["ct"]))
        kid = data.get("kid")
        aesgcm = _aesgcm()
        for key in self._candidate_keys(kid):
            try:
                plaintext = aesgcm(key).decrypt(nonce, ciphertext, None)
            except InvalidTag:
                continue
            loaded: dict[str, Any] = json.loads(plaintext.decode("utf-8"))
            return loaded
        raise CheckpointEncryptionError(
            "no configured key decrypts this checkpoint"
            + (f" (it names key {kid!r})" if kid else "")
            + "; add the key it was written with to previous_keys"
        )

    def _candidate_keys(self, kid: Any) -> list[bytes]:
        """The current key and the previous ones, the one ``kid`` names first."""
        keys = [self._current_key(), *self._previous]
        if isinstance(kid, str):
            keys.sort(key=lambda k: _kid(k) != kid)
        return keys

    def __repr__(self) -> str:
        return f"EncryptedSerializer(kid={self.key_id!r}, previous_keys={len(self._previous)})"
