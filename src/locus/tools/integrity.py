# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Tool-call integrity — signed tool calls between model output and execution.

A tool call travels some distance between the model that produced it and the
tool that runs it: through the after-model hooks, the before-tool hooks, the
idempotency cache, the executor. Any of those is a place where arguments can
change — a hook that rewrites them, a plugin with a bug, a consumer that
mutates the ``ToolCall`` it was handed on a ``ThinkEvent``. Without a record
of what the model actually said, the executor cannot tell a legitimate
rewrite from a modified call.

:class:`ToolCallSigner` is that record. The runtime creates one per run with a
random key that lives only in memory. Every :class:`~locus.core.messages.ToolCall`
the model produces is signed the moment the response is parsed — the
streaming path included — over a canonical form of its id, name and arguments
(sorted keys, no whitespace) with HMAC-SHA256. The executor verifies the call
it is about to run against that record and refuses one that no longer
matches: a typed :class:`ToolCallIntegrityError`, an entry on the agent's
:class:`~locus.control.audit.AuditTrail` when it has one, an event on the
bus, and a tool error the model reads so the loop goes on (or, configured so,
the run aborts).

**Chain of custody.** A hook that legitimately rewrites arguments — PII
redaction, an external ``PreToolUse`` hook — re-signs the call under its own
name through :meth:`ToolCallSigner.resign` (from a hook:
``event.resign(by="MyHook")``). The signature's ``by`` chain then reads
``("model", "GuardrailsHook")``: who last vouched for these arguments is
explicit, and the audit entry can name it. A hook that changes the arguments
*without* re-signing taints the record instead; the executor refuses the call
and the audit entry names that hook. Re-signing a tainted record does not
clear the taint — a later hook cannot launder an earlier silent change.

**What is and is not covered.** The signature is kept in a table keyed by
call id on the run's signer, not on the ``ToolCall`` itself: the message
types, their checkpoints and their wire formats are unchanged, and a copy of
a call (``model_copy``) cannot carry a stale signature along. The key is
per run and never persisted, so a call that reaches the executor without a
record in *this* run — one rehydrated from a checkpoint in a new process, or
issued by code that never passed the runtime's signing point — is *unsigned*:
it runs, unverified, exactly as every call did before this module existed.
The guarantee is therefore: **once the runtime has signed a call, nothing
may change it unnamed.**
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import logging
import secrets
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

from locus.core.errors import ToolError
from locus.core.messages import ToolCall, ToolResult


if TYPE_CHECKING:
    from locus.hooks.provider import BeforeToolCallEvent

logger = logging.getLogger(__name__)

#: The first link of every chain of custody: the model produced the call.
MODEL = "model"

#: The event type of the audit record a refusal appends.
AUDIT_EVENT_TYPE = "agent.tool.integrity.refused"

#: What a failed verification does to the run.
IntegrityViolation = Literal["refuse", "abort"]

#: Why a verification failed.
IntegrityFailure = Literal["mismatch", "tainted"]


# =============================================================================
# Canonical form
# =============================================================================


def _canonical(call_id: str, name: str, arguments: Mapping[str, Any]) -> bytes:
    return json.dumps(
        {"id": call_id, "name": name, "arguments": dict(arguments)},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")


def canonical_form(call: ToolCall) -> bytes:
    """The bytes a signature covers: ``{"arguments": …, "id": …, "name": …}``.

    Sorted keys at every level, no whitespace, UTF-8. Argument values that are
    not JSON types are rendered with ``str``.
    """
    return _canonical(call.id, call.name, call.arguments)


def call_digest(call: ToolCall) -> str:
    """SHA-256 of :func:`canonical_form`, hex. What an audit entry shows."""
    return hashlib.sha256(canonical_form(call)).hexdigest()


def _digest(call_id: str, name: str, arguments: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(call_id, name, arguments)).hexdigest()


# =============================================================================
# Records and errors
# =============================================================================


@dataclass(frozen=True)
class ToolCallSignature:
    """A signed tool call, as the run's signer remembers it.

    Attributes:
        call_id: The ``ToolCall.id`` this record is for.
        tool_name: The tool the call names.
        digest: SHA-256 of the canonical form that was signed.
        mac: HMAC-SHA256, under the run's key, over the canonical form and
            the ``by`` chain.
        by: The chain of custody, oldest first. ``("model",)`` for a call
            nobody rewrote; each :meth:`ToolCallSigner.resign` appends a name.
        key_id: Identifies the run's key, for diagnostics.
        tainted_by: The hook after which the arguments were found changed
            without a re-sign, or ``None``. A tainted record never verifies.
    """

    call_id: str
    tool_name: str
    digest: str
    mac: str
    by: tuple[str, ...]
    key_id: str
    tainted_by: str | None = None

    @property
    def signed_by(self) -> str:
        """Who last vouched for the arguments."""
        return self.by[-1]


class ToolCallIntegrityError(ToolError):
    """A tool call reached the executor changed from what was signed.

    Raised by :meth:`ToolCallSigner.verify`; in ``abort`` mode it ends the run.
    In ``refuse`` mode the executor returns its message to the model as the
    call's error instead, and the loop continues.

    Attributes:
        tool_name: The tool the call names.
        call_id: The call's id.
        reason: ``"mismatch"`` — the call differs from the signed form;
            ``"tainted"`` — a hook changed the arguments without re-signing.
        expected_digest: Digest of the canonical form last signed.
        actual_digest: Digest of the call as the executor received it.
        signed_by: The chain of custody at the time of the check.
        mutated_by: The hook the change is attributed to, when known.
    """

    kind = "tool_call_integrity"

    def __init__(
        self,
        *,
        tool_name: str,
        call_id: str,
        reason: IntegrityFailure,
        expected_digest: str,
        actual_digest: str,
        signed_by: tuple[str, ...],
        mutated_by: str | None = None,
    ) -> None:
        self.tool_name = tool_name
        self.call_id = call_id
        self.reason: IntegrityFailure = reason
        self.expected_digest = expected_digest
        self.actual_digest = actual_digest
        self.signed_by = signed_by
        self.mutated_by = mutated_by
        super().__init__(self._describe())

    def _describe(self) -> str:
        who = (
            f"changed by {self.mutated_by} outside the re-sign API"
            if self.mutated_by
            else f"changed after {self.signed_by[-1]} signed it"
        )
        return (
            f"tool call {self.call_id} ({self.tool_name}) was {who}: "
            f"signed digest {self.expected_digest[:12]}…, "
            f"received {self.actual_digest[:12]}…; the tool did not run"
        )

    def audit_payload(self, *, disposition: IntegrityViolation, key_id: str) -> dict[str, Any]:
        """The fields a refusal records on the audit trail and the event bus.

        The tool goes out as ``tool_name``, the key :class:`ToolCallIntegrityError`
        and the audit entries use.
        """
        return {
            "tool_name": self.tool_name,
            "call_id": self.call_id,
            "reason": self.reason,
            "expected_digest": self.expected_digest,
            "actual_digest": self.actual_digest,
            "signed_by": list(self.signed_by),
            "mutated_by": self.mutated_by,
            "disposition": disposition,
            "key_id": key_id,
        }


# =============================================================================
# The signer
# =============================================================================


class ToolCallSigner:
    """Signs the tool calls of one run and verifies them at the executor.

    Args:
        on_violation: ``"refuse"`` returns the failed check to the model as the
            call's error and the run goes on; ``"abort"`` additionally ends
            the run with the :class:`ToolCallIntegrityError` after the batch.
        trail: An :class:`~locus.control.audit.AuditTrail` (anything with a
            ``record(event_type, payload)`` method) a refusal is appended to.
        key: The HMAC key. A fresh random 32-byte key by default — pass one
            only to share a signer across runtimes you control.
    """

    def __init__(
        self,
        *,
        on_violation: IntegrityViolation = "refuse",
        trail: Any | None = None,
        key: bytes | None = None,
    ) -> None:
        if on_violation not in ("refuse", "abort"):
            raise ValueError(f"on_violation must be 'refuse' or 'abort', got {on_violation!r}")
        self._key = key if key is not None else secrets.token_bytes(32)
        self._key_id = hashlib.sha256(self._key).hexdigest()[:8]
        self._on_violation: IntegrityViolation = on_violation
        self._trail = trail
        self._records: dict[str, ToolCallSignature] = {}
        self._pending_abort: ToolCallIntegrityError | None = None
        #: Every refusal this signer issued, oldest first.
        self.refusals: list[ToolCallIntegrityError] = []

    @property
    def key_id(self) -> str:
        """Identifies this signer's key (not the key itself)."""
        return self._key_id

    @property
    def on_violation(self) -> IntegrityViolation:
        return self._on_violation

    @property
    def trail(self) -> Any | None:
        return self._trail

    def __len__(self) -> int:
        return len(self._records)

    # ----------------------------------------------------------- signing --

    def _mac(self, canonical: bytes, by: tuple[str, ...]) -> str:
        message = canonical + b"\x1f" + "\x1e".join(by).encode("utf-8")
        return hmac.new(self._key, message, hashlib.sha256).hexdigest()

    def _record(self, call: ToolCall, by: tuple[str, ...]) -> ToolCallSignature:
        canonical = canonical_form(call)
        return ToolCallSignature(
            call_id=call.id,
            tool_name=call.name,
            digest=hashlib.sha256(canonical).hexdigest(),
            mac=self._mac(canonical, by),
            by=by,
            key_id=self._key_id,
        )

    def sign(self, call: ToolCall, *, by: str = MODEL) -> ToolCallSignature:
        """Sign ``call`` as it is now, starting its chain of custody at ``by``.

        The runtime calls this for every call a model response carries. An
        earlier record for the same id is replaced.
        """
        record = self._record(call, (by,))
        self._records[call.id] = record
        return record

    def sign_all(self, calls: Iterable[ToolCall], *, by: str = MODEL) -> None:
        """:meth:`sign` each call."""
        for call in calls:
            self.sign(call, by=by)

    def resign(self, call: ToolCall, *, by: str) -> ToolCallSignature | None:
        """Vouch for ``call`` as it is now, appending ``by`` to its chain.

        This is the documented way for a hook that rewrites arguments to keep
        the chain of custody explicit; from a hook, ``event.resign(by=...)``
        calls it. Returns the new record, or ``None`` when nothing changed:
        a call this signer never signed stays unsigned, and a tainted record
        (changed earlier without a re-sign) is left as it is, so the executor
        still refuses it and names the hook that changed it.
        """
        if not by:
            raise ValueError("resign() needs the name of who is re-signing (by=...)")
        current = self._records.get(call.id)
        if current is None:
            return None
        if current.tainted_by is not None:
            logger.warning(
                "tool call %s (%s): %s re-signed a call %s changed without re-signing; "
                "the taint stands",
                call.id,
                call.name,
                by,
                current.tainted_by,
            )
            return None
        record = self._record(call, (*current.by, by))
        self._records[call.id] = record
        return record

    def taint(self, call_id: str, *, by: str) -> None:
        """Mark the call's arguments as changed by ``by`` without a re-sign.

        The record keeps the digest of the last signed form, so the refusal
        shows what was signed next to what arrived. The first taint wins.
        """
        current = self._records.get(call_id)
        if current is None or current.tainted_by is not None:
            return
        self._records[call_id] = replace(current, tainted_by=by)

    def signature(self, call_id: str) -> ToolCallSignature | None:
        """The record for ``call_id``, or ``None`` when the call is unsigned."""
        return self._records.get(call_id)

    # -------------------------------------------------------- verifying --

    def verify(self, call: ToolCall) -> ToolCallSignature | None:
        """Check ``call`` against its record.

        Returns the record when the call verifies, ``None`` when this signer
        holds no record for the call (unsigned — see the module docstring).

        Raises:
            ToolCallIntegrityError: The call differs from the signed form, or
                its record is tainted.
        """
        record = self._records.get(call.id)
        if record is None:
            return None
        actual = call_digest(call)
        if record.tainted_by is not None:
            raise ToolCallIntegrityError(
                tool_name=call.name,
                call_id=call.id,
                reason="tainted",
                expected_digest=record.digest,
                actual_digest=actual,
                signed_by=record.by,
                mutated_by=record.tainted_by,
            )
        expected_mac = self._mac(canonical_form(call), record.by)
        if not hmac.compare_digest(expected_mac, record.mac):
            raise ToolCallIntegrityError(
                tool_name=call.name,
                call_id=call.id,
                reason="mismatch",
                expected_digest=record.digest,
                actual_digest=actual,
                signed_by=record.by,
            )
        return record

    async def enforce(self, call: ToolCall) -> ToolResult | None:
        """Verify ``call`` for execution; the refusal to return when it fails.

        ``None`` means proceed (verified, or unsigned). Otherwise the call is
        refused: the failure is appended to the audit trail, published on the
        event bus, remembered for :meth:`raise_pending_abort`, and returned as
        the :class:`~locus.core.messages.ToolResult` the model reads.
        """
        try:
            self.verify(call)
        except ToolCallIntegrityError as exc:
            return await self._refuse(exc)
        return None

    async def _refuse(self, exc: ToolCallIntegrityError) -> ToolResult:
        self.refusals.append(exc)
        if self._on_violation == "abort" and self._pending_abort is None:
            self._pending_abort = exc
        payload = exc.audit_payload(disposition=self._on_violation, key_id=self._key_id)
        logger.warning("tool-call integrity: %s", exc)
        if self._trail is not None:
            try:
                self._trail.record(AUDIT_EVENT_TYPE, payload)
            except Exception:  # noqa: BLE001 — the refusal must stand even if the trail cannot take it
                logger.exception("tool-call integrity: audit trail refused the record")
        from locus.observability.emit import EV_TOOL_INTEGRITY_REFUSED, emit  # noqa: PLC0415

        await emit(EV_TOOL_INTEGRITY_REFUSED, **payload)
        return ToolResult(
            tool_call_id=exc.call_id,
            name=exc.tool_name,
            content="",
            error=(
                f"ToolCallIntegrityError: {exc}. The call was not executed because its "
                "arguments failed an integrity check. Do not repeat it; tell the user."
            ),
            duration_ms=0.0,
        )

    def raise_pending_abort(self) -> None:
        """In ``abort`` mode, raise the first refusal since the last call.

        The runtime calls this once a batch of tools has finished, so the
        siblings of a refused call complete and the run ends with the typed
        error rather than mid-batch.
        """
        exc, self._pending_abort = self._pending_abort, None
        if exc is not None:
            raise exc

    # ---------------------------------------------------- the hook seam --

    def watch(self, event: BeforeToolCallEvent) -> HookCustody:
        """Track what each before-tool hook does to ``event`` (runtime use)."""
        return HookCustody(self, event)


class HookCustody:
    """Attributes argument changes made during the before-tool hook seam.

    The hook orchestrator wraps each ``on_before_tool_call`` in
    :meth:`around`. A hook that changed ``event.arguments`` and did not
    re-sign taints the record under that hook's name. A hook that set
    ``event.secret_arguments`` is remembered so :meth:`close` can extend the
    signature over the merged invocation arguments — the executor receives
    them merged, and the chain names the hook that added them.
    """

    __slots__ = ("_event", "_secret_source", "_signer")

    def __init__(self, signer: ToolCallSigner, event: BeforeToolCallEvent) -> None:
        self._signer = signer
        self._event = event
        self._secret_source: str | None = None

    def _digest(self) -> str:
        event = self._event
        return _digest(event.tool_call_id, event.tool_name, event.arguments)

    def _secrets(self) -> dict[str, Any]:
        secret = getattr(self._event, "secret_arguments", None)
        return dict(secret) if secret else {}

    @contextlib.contextmanager
    def around(self, hook: Any) -> Iterator[None]:
        """Run one hook's ``on_before_tool_call`` inside this context."""
        if self._signer.signature(self._event.tool_call_id) is None:
            # Unsigned call: nothing to attribute changes against.
            yield
            return
        digest_before = self._digest()
        secrets_before = self._secrets()
        yield
        label = hook_label(hook)
        digest_after = self._digest()
        if digest_after != digest_before:
            record = self._signer.signature(self._event.tool_call_id)
            if record is None or record.digest != digest_after:
                self._signer.taint(self._event.tool_call_id, by=label)
        if self._secrets() != secrets_before:
            self._secret_source = label

    def close(self) -> None:
        """Extend the signature over ``secret_arguments``, once all hooks ran."""
        secret = self._secrets()
        if not secret:
            return
        event = self._event
        record = self._signer.signature(event.tool_call_id)
        if record is None or record.tainted_by is not None:
            return
        if self._digest() != record.digest:
            # Changed by something the seam did not see; the executor decides.
            return
        merged = ToolCall(
            id=event.tool_call_id,
            name=event.tool_name,
            arguments={**event.arguments, **secret},
        )
        self._signer.resign(merged, by=f"{self._secret_source or 'hook'}:secret_arguments")


# =============================================================================
# Wiring helpers
# =============================================================================


def hook_label(hook: Any) -> str:
    """The name a hook appears under in a chain of custody or an audit entry."""
    label = type(hook).__name__
    plugin = getattr(hook, "_plugin", None)
    name = getattr(plugin, "name", None) if plugin is not None else None
    if isinstance(name, str) and name:
        return f"{label}({name})"
    return label


def find_audit_trail(hooks: Iterable[Any]) -> Any | None:
    """The first ``AuditTrail`` an agent's hooks carry (an ``AuditHook``'s), or ``None``."""
    for hook in hooks:
        trail = getattr(hook, "trail", None)
        if trail is not None and callable(getattr(trail, "record", None)):
            return trail
    return None


def signer_for(mode: bool | IntegrityViolation, hooks: Iterable[Any]) -> ToolCallSigner | None:
    """The signer an :class:`~locus.agent.config.AgentConfig.tool_call_integrity` value asks for.

    ``False`` → ``None`` (off). ``True`` or ``"refuse"`` → refuse and continue.
    ``"abort"`` → refuse and end the run. The agent's audit trail, when it has
    one through an ``AuditHook``, receives every refusal.
    """
    if mode is False:
        return None
    on_violation: IntegrityViolation = "abort" if mode == "abort" else "refuse"
    return ToolCallSigner(on_violation=on_violation, trail=find_audit_trail(hooks))


__all__ = [
    "AUDIT_EVENT_TYPE",
    "MODEL",
    "HookCustody",
    "IntegrityFailure",
    "IntegrityViolation",
    "ToolCallIntegrityError",
    "ToolCallSignature",
    "ToolCallSigner",
    "call_digest",
    "canonical_form",
    "find_audit_trail",
    "hook_label",
    "signer_for",
]
