# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""A2A push notifications: the server side that posts, the client side that receives.

A task that outlives the request that created it needs a way back to the
caller. The specification's answer is a webhook: the client registers a
:class:`~locus.a2a.spec_v1.A2AV1TaskPushNotificationConfig` on the task and
the server POSTs a ``StreamResponse`` to its ``url`` on every status and
artifact change. This module holds:

- :class:`PushNotificationStore`: the per-task configs (in memory).
- :class:`PushNotificationDispatcher`: delivers notifications, with the
  headers the specification names (``X-A2A-Notification-Token`` and the
  ``Authorization`` the config asks for), a retry with back-off, and a URL
  policy that refuses private and link-local destinations unless told
  otherwise.
- :class:`PushNotificationSigner`: optional detached JWS over every body
  (Ed25519), with the public keys published as a JWKS so a receiver can
  check that a notification came from this agent and was not altered.
- :class:`PushNotificationReceiver`: the client-side helper. A small ASGI
  application that checks the token and, when keys are configured, the
  signature, then hands verified ``StreamResponse`` payloads to the caller.

Signing keys and credentials are never logged: log lines carry the task id,
the config id, the destination host and the outcome only.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import ipaddress
import json
import logging
import socket
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlsplit

from locus.a2a.spec_v1 import (
    A2A_JSON_MEDIA_TYPE,
    A2A_V1_PROTOCOL_VERSION,
    A2AV1TaskPushNotificationConfig,
)
from locus.core.loop_bound import loop_bound


if TYPE_CHECKING:
    import httpx

_logger = logging.getLogger(__name__)

#: Header carrying the config's ``token`` back to the receiver.
NOTIFICATION_TOKEN_HEADER = "X-A2A-Notification-Token"  # noqa: S105 — a header name
#: Header carrying the detached JWS over the body when the agent signs.
SIGNATURE_HEADER = "X-A2A-Signature"
#: Where :class:`PushNotificationSigner` publishes its public keys.
JWKS_PATH = "/.well-known/jwks.json"
#: Header carrying the protocol version on every notification.
VERSION_HEADER = "A2A-Version"


class PushNotificationError(Exception):
    """A push-notification config was refused or a notification failed."""


class PushVerificationError(Exception):
    """A received notification failed its token or signature check."""


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _host_of(url: str) -> str:
    return urlsplit(url).hostname or ""


# ---------------------------------------------------------------------------
# URL policy
# ---------------------------------------------------------------------------


def _address_is_private(host: str) -> bool:
    """Whether ``host`` is, or resolves to, a non-public address."""
    try:
        addresses = [ipaddress.ip_address(host)]
    except ValueError:
        try:
            infos = socket.getaddrinfo(host, None)
        except OSError:
            return True
        addresses = [ipaddress.ip_address(info[4][0]) for info in infos]
    return any(
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_multicast
        or addr.is_unspecified
        for addr in addresses
    )


def validate_push_url(url: str, *, allow_private: bool = False) -> None:
    """Refuse a webhook URL the server should not call.

    Only ``http`` and ``https`` schemes are accepted. Hosts that are, or
    resolve to, loopback, private, link-local, reserved or multicast
    addresses are refused unless ``allow_private`` is set (a local receiver
    in a test or a demo), which is the guard the specification asks of a
    server (SSRF).

    Raises:
        PushNotificationError: the URL is not acceptable.
    """
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise PushNotificationError(f"push URL scheme must be http or https, got {parts.scheme!r}")
    if not parts.hostname:
        raise PushNotificationError("push URL has no host")
    if not allow_private and _address_is_private(parts.hostname):
        raise PushNotificationError(
            f"push URL host {parts.hostname!r} is not a public address; "
            "pass allow_private_urls=True to deliver to private networks"
        )


# ---------------------------------------------------------------------------
# Signing
# ---------------------------------------------------------------------------


class PushNotificationSigner:
    """Sign push-notification bodies with an Ed25519 key, published as a JWKS.

    Every notification carries ``X-A2A-Signature: <compact JWS>`` whose
    payload is ``{"iat", "exp", "taskId", "request_body_sha256"}``; the
    receiver fetches ``/.well-known/jwks.json`` from the agent (or is given
    the keys) and verifies both the signature and the body hash. Needs the
    ``cryptography`` package (the ``audit`` extra). The private key is held
    in memory only and never appears in ``repr()`` or a log line.
    """

    ALGORITHM = "EdDSA"

    def __init__(self, private_key: Any, *, kid: str | None = None, ttl: float = 300.0) -> None:
        self._key = private_key
        self.kid = kid or uuid.uuid4().hex[:16]
        self.ttl = ttl

    @classmethod
    def generate(cls, *, kid: str | None = None, ttl: float = 300.0) -> PushNotificationSigner:
        """A signer with a fresh key; publish its ``jwks()`` for receivers."""
        try:
            from cryptography.hazmat.primitives.asymmetric import ed25519
        except ImportError as e:
            msg = (
                "push-notification signing needs the cryptography package: pip install cryptography"
            )
            raise ImportError(msg) from e
        return cls(ed25519.Ed25519PrivateKey.generate(), kid=kid, ttl=ttl)

    @classmethod
    def from_pem(
        cls, pem: bytes | str, *, password: bytes | None = None, kid: str | None = None
    ) -> PushNotificationSigner:
        """A signer from a PEM-encoded Ed25519 private key."""
        try:
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.primitives.asymmetric import ed25519
        except ImportError as e:
            msg = (
                "push-notification signing needs the cryptography package: pip install cryptography"
            )
            raise ImportError(msg) from e
        data = pem.encode() if isinstance(pem, str) else pem
        key = serialization.load_pem_private_key(data, password=password)
        if not isinstance(key, ed25519.Ed25519PrivateKey):
            raise TypeError("push-notification signing needs an Ed25519 private key")
        return cls(key, kid=kid)

    def __repr__(self) -> str:
        return f"PushNotificationSigner(kid={self.kid!r})"

    def public_jwk(self) -> dict[str, Any]:
        """This key's public half as a JWK."""
        from cryptography.hazmat.primitives import serialization

        raw = self._key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
        return {
            "kty": "OKP",
            "crv": "Ed25519",
            "kid": self.kid,
            "use": "sig",
            "alg": self.ALGORITHM,
            "x": _b64url(raw),
        }

    def jwks(self) -> dict[str, Any]:
        """The JWKS document served at :data:`JWKS_PATH`."""
        return {"keys": [self.public_jwk()]}

    def sign(self, body: bytes, *, task_id: str, now: float | None = None) -> str:
        """A compact JWS binding ``body`` to ``task_id`` for ``ttl`` seconds."""
        issued = int(now if now is not None else time.time())
        header = {"alg": self.ALGORITHM, "typ": "JWT", "kid": self.kid}
        claims = {
            "iat": issued,
            "exp": issued + int(self.ttl),
            "taskId": task_id,
            "request_body_sha256": hashlib.sha256(body).hexdigest(),
        }
        signing_input = (
            _b64url(json.dumps(header, separators=(",", ":")).encode())
            + "."
            + _b64url(json.dumps(claims, separators=(",", ":")).encode())
        )
        signature = self._key.sign(signing_input.encode("ascii"))
        return signing_input + "." + _b64url(signature)


def verify_signature(
    jws: str,
    body: bytes,
    *,
    jwks: Mapping[str, Any],
    task_id: str | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Check a detached JWS from :class:`PushNotificationSigner` against ``jwks``.

    Returns the verified claims. Raises :class:`PushVerificationError` when
    the key is unknown, the signature does not verify, the body hash differs,
    the token has expired, or ``task_id`` is given and does not match.
    """
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric import ed25519
    except ImportError as e:
        msg = "verifying push-notification signatures needs the cryptography package"
        raise ImportError(msg) from e
    try:
        header_b64, claims_b64, signature_b64 = jws.split(".")
        header = json.loads(_b64url_decode(header_b64))
        claims = json.loads(_b64url_decode(claims_b64))
        signature = _b64url_decode(signature_b64)
    except (ValueError, json.JSONDecodeError) as e:
        raise PushVerificationError("malformed signature") from e
    if header.get("alg") != PushNotificationSigner.ALGORITHM:
        raise PushVerificationError(f"unsupported signature algorithm {header.get('alg')!r}")
    key_entry = next(
        (k for k in jwks.get("keys", []) if k.get("kid") == header.get("kid")),
        None,
    )
    if key_entry is None:
        raise PushVerificationError(f"unknown signing key {header.get('kid')!r}")
    try:
        public_key = ed25519.Ed25519PublicKey.from_public_bytes(_b64url_decode(key_entry["x"]))
        public_key.verify(signature, f"{header_b64}.{claims_b64}".encode("ascii"))
    except (InvalidSignature, ValueError, KeyError) as e:
        raise PushVerificationError("signature does not verify") from e
    if claims.get("request_body_sha256") != hashlib.sha256(body).hexdigest():
        raise PushVerificationError("body hash does not match the signature")
    current = now if now is not None else time.time()
    if isinstance(claims.get("exp"), int | float) and claims["exp"] < current:
        raise PushVerificationError("signature has expired")
    if task_id is not None and claims.get("taskId") != task_id:
        raise PushVerificationError("signature was issued for another task")
    return dict(claims)


# ---------------------------------------------------------------------------
# Server side: store + dispatcher
# ---------------------------------------------------------------------------


class PushNotificationStore:
    """Push-notification configs per task, in this process."""

    def __init__(self) -> None:
        self._configs: dict[str, dict[str, A2AV1TaskPushNotificationConfig]] = {}

    def set(
        self, task_id: str, config: A2AV1TaskPushNotificationConfig
    ) -> A2AV1TaskPushNotificationConfig:
        """Store ``config`` for ``task_id``, assigning an ``id`` when it has none."""
        stored = config.model_copy(update={"taskId": task_id, "id": config.id or uuid.uuid4().hex})
        self._configs.setdefault(task_id, {})[str(stored.id)] = stored
        return stored

    def get(self, task_id: str, config_id: str) -> A2AV1TaskPushNotificationConfig | None:
        return self._configs.get(task_id, {}).get(config_id)

    def list(self, task_id: str) -> list[A2AV1TaskPushNotificationConfig]:
        return list(self._configs.get(task_id, {}).values())

    def delete(self, task_id: str, config_id: str) -> bool:
        """Remove one config; ``False`` when there was none (deletion is idempotent)."""
        return self._configs.get(task_id, {}).pop(config_id, None) is not None

    def clear(self, task_id: str) -> None:
        self._configs.pop(task_id, None)


@dataclass
class PushNotificationSettings:
    """How an :class:`~locus.a2a.A2AServer` delivers push notifications.

    Args:
        allow_private_urls: Deliver to loopback, private and link-local
            hosts. Off by default, as the specification advises; turn it on
            for a receiver on the same machine.
        signer: Sign every body and publish the JWKS at
            :data:`JWKS_PATH`. ``None`` sends unsigned notifications that
            still carry the token and the configured ``Authorization``.
        timeout: Seconds to wait for the receiver's 2xx.
        max_attempts: Delivery attempts per notification, with exponential
            back-off starting at ``backoff`` seconds.
        http_client: An ``httpx.AsyncClient`` to post with; one is created
            when ``None``. Tests pass one bound to an in-process receiver.
    """

    allow_private_urls: bool = False
    signer: PushNotificationSigner | None = None
    timeout: float = 10.0
    max_attempts: int = 3
    backoff: float = 0.5
    http_client: Any | None = None


class PushNotificationDispatcher:
    """Post ``StreamResponse`` payloads to every config registered on a task.

    Delivery is best effort and never raises into the task's run: a failure
    is logged (task id, config id, destination host, outcome) and retried up
    to ``settings.max_attempts`` times. :meth:`drain` waits for in-flight
    deliveries, which tests and graceful shutdown use.
    """

    def __init__(self, store: PushNotificationStore, settings: PushNotificationSettings) -> None:
        self._store = store
        self._settings = settings
        self._inflight: set[asyncio.Task[None]] = set()
        self._client: httpx.AsyncClient | None = settings.http_client

    @property
    def settings(self) -> PushNotificationSettings:
        return self._settings

    def register(
        self, task_id: str, config: A2AV1TaskPushNotificationConfig
    ) -> A2AV1TaskPushNotificationConfig:
        """Validate ``config.url`` under the URL policy and store it."""
        validate_push_url(config.url, allow_private=self._settings.allow_private_urls)
        return self._store.set(task_id, config)

    def get(self, task_id: str, config_id: str) -> A2AV1TaskPushNotificationConfig | None:
        return self._store.get(task_id, config_id)

    def list(self, task_id: str) -> list[A2AV1TaskPushNotificationConfig]:
        return self._store.list(task_id)

    def delete(self, task_id: str, config_id: str) -> bool:
        return self._store.delete(task_id, config_id)

    def notify(self, task_id: str, payload: Mapping[str, Any]) -> None:
        """Schedule delivery of one ``StreamResponse`` payload to the task's webhooks."""
        configs = self._store.list(task_id)
        if not configs:
            return
        body = json.dumps(dict(payload), ensure_ascii=False).encode("utf-8")
        for config in configs:
            task = asyncio.ensure_future(self._deliver(task_id, config, body))
            self._inflight.add(task)
            task.add_done_callback(self._inflight.discard)

    async def drain(self) -> None:
        """Wait for every scheduled delivery to finish."""
        # Finished tasks are dropped here as well as by their done callback:
        # ``gather`` over tasks that are all done completes without yielding
        # to the loop, so waiting on the callback alone could spin.
        while True:
            self._inflight.difference_update([t for t in self._inflight if t.done()])
            pending = list(self._inflight)
            if not pending:
                return
            await asyncio.gather(*pending, return_exceptions=True)

    def _headers(
        self, task_id: str, config: A2AV1TaskPushNotificationConfig, body: bytes
    ) -> dict[str, str]:
        headers = {
            "Content-Type": A2A_JSON_MEDIA_TYPE,
            VERSION_HEADER: A2A_V1_PROTOCOL_VERSION,
        }
        if config.token:
            headers[NOTIFICATION_TOKEN_HEADER] = config.token
        if config.authentication is not None and config.authentication.credentials:
            headers["Authorization"] = (
                f"{config.authentication.scheme} {config.authentication.credentials}"
            )
        if self._settings.signer is not None:
            headers[SIGNATURE_HEADER] = self._settings.signer.sign(body, task_id=task_id)
        return headers

    def _build_client(self) -> httpx.AsyncClient:
        import httpx

        return httpx.AsyncClient(timeout=self._settings.timeout)

    async def _client_for(self) -> httpx.AsyncClient:
        """The posting client: the injected one, or an owned one bound to the running loop."""
        if self._settings.http_client is not None:
            return cast("httpx.AsyncClient", self._settings.http_client)
        return loop_bound(self, "_client", self._build_client)

    async def _deliver(
        self, task_id: str, config: A2AV1TaskPushNotificationConfig, body: bytes
    ) -> None:
        import httpx

        client = await self._client_for()
        headers = self._headers(task_id, config, body)
        delay = self._settings.backoff
        attempts = max(1, self._settings.max_attempts)
        for attempt in range(1, attempts + 1):
            try:
                response = await client.post(config.url, content=body, headers=headers)
                if response.is_success:
                    _logger.info(
                        "A2A push delivered task=%s config=%s host=%s status=%s",
                        task_id,
                        config.id,
                        _host_of(config.url),
                        response.status_code,
                    )
                    return
                outcome = f"status {response.status_code}"
            except httpx.HTTPError as e:
                outcome = type(e).__name__
            _logger.warning(
                "A2A push attempt %d/%d failed task=%s config=%s host=%s outcome=%s",
                attempt,
                attempts,
                task_id,
                config.id,
                _host_of(config.url),
                outcome,
            )
            if attempt < attempts:
                await asyncio.sleep(delay)
                delay *= 2

    async def aclose(self) -> None:
        await self.drain()
        if self._client is not None and self._settings.http_client is None:
            await self._client.aclose()
            self._client = None


# ---------------------------------------------------------------------------
# Client side: receiver
# ---------------------------------------------------------------------------


@dataclass
class PushNotification:
    """One verified notification: the ``StreamResponse`` and what it was checked with."""

    task_id: str
    payload: dict[str, Any]
    claims: dict[str, Any] | None = None
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def kind(self) -> str:
        """``task``, ``message``, ``statusUpdate`` or ``artifactUpdate``."""
        return next(iter(self.payload), "")

    @property
    def state(self) -> str | None:
        """The task state the notification reports, when it carries one."""
        body = self.payload.get("statusUpdate") or self.payload.get("task") or {}
        status = body.get("status") or {}
        state = status.get("state")
        return str(state) if state is not None else None


def _payload_task_id(payload: Mapping[str, Any]) -> str:
    for key in ("statusUpdate", "artifactUpdate", "message"):
        body = payload.get(key)
        if isinstance(body, Mapping) and body.get("taskId"):
            return str(body["taskId"])
    task = payload.get("task")
    if isinstance(task, Mapping) and task.get("id"):
        return str(task["id"])
    return ""


class PushNotificationReceiver:
    """Receive and verify A2A push notifications.

    Mount :attr:`app` (an ASGI application) wherever the webhook URL points,
    or call :meth:`serve` to run it on its own port. Each accepted request
    becomes a :class:`PushNotification` on :meth:`events`.

    Args:
        token: The ``token`` registered in the push config. A notification
            with a different ``X-A2A-Notification-Token`` is refused with
            401. ``None`` skips the check.
        jwks: The agent's JWKS (from ``GET /.well-known/jwks.json``). When
            set, every notification must carry a valid ``X-A2A-Signature``.
        jwks_fetcher: An async callable returning the JWKS, used instead of
            ``jwks`` to fetch (and refresh once on an unknown key).
        path: The path the app answers on; other paths are 404.
    """

    def __init__(
        self,
        *,
        token: str | None = None,
        jwks: Mapping[str, Any] | None = None,
        jwks_fetcher: Callable[[], Awaitable[Mapping[str, Any]]] | None = None,
        path: str = "/a2a/push",
    ) -> None:
        self.token = token
        self._jwks: Mapping[str, Any] | None = jwks
        self._jwks_fetcher = jwks_fetcher
        self.path = path
        self._queue: asyncio.Queue[PushNotification] = asyncio.Queue()
        self.received: list[PushNotification] = []

    @property
    def requires_signature(self) -> bool:
        return self._jwks is not None or self._jwks_fetcher is not None

    async def _keys(self, *, refresh: bool = False) -> Mapping[str, Any]:
        if (self._jwks is None or refresh) and self._jwks_fetcher is not None:
            self._jwks = await self._jwks_fetcher()
        return self._jwks or {"keys": []}

    async def verify(self, headers: Mapping[str, str], body: bytes) -> PushNotification:
        """Check one request; return the notification or raise :class:`PushVerificationError`."""
        lower = {k.lower(): v for k, v in headers.items()}
        if self.token is not None and lower.get(NOTIFICATION_TOKEN_HEADER.lower()) != self.token:
            raise PushVerificationError("notification token does not match")
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as e:
            raise PushVerificationError("notification body is not JSON") from e
        if not isinstance(payload, dict):
            raise PushVerificationError("notification body is not a StreamResponse object")
        task_id = _payload_task_id(payload)
        if not task_id:
            raise PushVerificationError("notification names no task")
        claims: dict[str, Any] | None = None
        if self.requires_signature:
            jws = lower.get(SIGNATURE_HEADER.lower())
            if not jws:
                raise PushVerificationError("notification is not signed")
            try:
                claims = verify_signature(jws, body, jwks=await self._keys(), task_id=task_id)
            except PushVerificationError:
                if self._jwks_fetcher is None:
                    raise
                claims = verify_signature(
                    jws, body, jwks=await self._keys(refresh=True), task_id=task_id
                )
        return PushNotification(
            task_id=task_id,
            payload=payload,
            claims=claims,
            headers={k: v for k, v in lower.items() if k != SIGNATURE_HEADER.lower()},
        )

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        """The ASGI application."""
        if scope["type"] == "lifespan":
            while True:
                message = await receive()
                if message["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif message["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        if scope["type"] != "http":
            return
        if scope.get("path") != self.path or scope.get("method") != "POST":
            await self._respond(send, 404, {"error": "not found"})
            return
        body = b""
        while True:
            message = await receive()
            body += message.get("body", b"")
            if not message.get("more_body"):
                break
        headers = {k.decode("latin-1"): v.decode("latin-1") for k, v in scope.get("headers", [])}
        try:
            notification = await self.verify(headers, body)
        except PushVerificationError as e:
            _logger.warning("A2A push refused: %s", e)
            await self._respond(send, 401, {"error": str(e)})
            return
        self.received.append(notification)
        self._queue.put_nowait(notification)
        await self._respond(send, 202, {"accepted": True})

    @property
    def app(self) -> Callable[..., Awaitable[None]]:
        """The ASGI application: mount it or hand it to a server."""
        return self

    @staticmethod
    async def _respond(send: Any, status: int, body: Mapping[str, Any]) -> None:
        data = json.dumps(dict(body)).encode()
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(data)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": data})

    async def events(self) -> AsyncIterator[PushNotification]:
        """Verified notifications as they arrive, newest last."""
        while True:
            yield await self._queue.get()

    async def next(
        self,
        timeout: float | None = None,  # noqa: ASYNC109 — the wait is asyncio.wait_for below
    ) -> PushNotification:
        """The next verified notification, or ``TimeoutError`` after ``timeout`` seconds."""
        if timeout is None:
            return await self._queue.get()
        return await asyncio.wait_for(self._queue.get(), timeout)

    async def serve(self, host: str = "127.0.0.1", port: int = 0) -> None:
        """Run the receiver with uvicorn until cancelled."""
        try:
            import uvicorn
        except ImportError as e:
            msg = "uvicorn required. Install with: pip install uvicorn"
            raise ImportError(msg) from e
        config = uvicorn.Config(self, host=host, port=port, log_level="warning")
        await uvicorn.Server(config).serve()


__all__ = [
    "JWKS_PATH",
    "NOTIFICATION_TOKEN_HEADER",
    "SIGNATURE_HEADER",
    "PushNotification",
    "PushNotificationDispatcher",
    "PushNotificationError",
    "PushNotificationReceiver",
    "PushNotificationSettings",
    "PushNotificationSigner",
    "PushNotificationStore",
    "PushVerificationError",
    "validate_push_url",
    "verify_signature",
]
