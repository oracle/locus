# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""httpx.Auth wrapper that signs requests with an OCI signer.

Used by :class:`OCIChatCompletionsModel` for IAM authentication against OCI's
OpenAI-compatible ``/openai/v1`` endpoint.

We deliberately don't depend on ``requests``, ``oci-openai``, or
``oci-genai-auth-python`` for this. The OCI signer interface only reads
``method``, ``url``, ``path_url``, ``headers``, and ``body`` from the
prepared request, so we duck-type those from an ``httpx.Request`` and skip
the ``requests`` round-trip entirely.

Reference (not vendored): oracle-samples/oci-genai-auth-python (UPL-1.0).
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Generator
from typing import TYPE_CHECKING, Any

import httpx


if TYPE_CHECKING:
    from oci.signer import AbstractBaseSigner


_logger = logging.getLogger(__name__)


class _PreparedRequestProxy:
    """Duck-typed stand-in for ``requests.PreparedRequest``.

    The OCI signer reads ``method``, ``url``, ``path_url``, ``headers``,
    and ``body`` and mutates ``headers`` in place. That's all this proxy
    has to expose.
    """

    __slots__ = ("body", "headers", "method", "path_url", "url")

    def __init__(self, request: httpx.Request, content: bytes) -> None:
        self.method = request.method
        self.url = str(request.url)
        # ``raw_path`` already includes ``?query`` if present. Falls back to
        # "/" for empty paths to match ``requests.PreparedRequest.path_url``.
        raw = request.url.raw_path.decode("ascii")
        self.path_url = raw or "/"
        self.headers = dict(request.headers)
        self.body = content


class OCIRequestSigner(httpx.Auth):
    """Signs every httpx request with an OCI signer.

    Works with any ``oci.signer.AbstractBaseSigner`` subclass — user
    principal (API key signing), security token (session), instance
    principal, or resource principal — so a single ``http_client`` can
    be reused regardless of which IAM mode the caller picked.

    For token-based signers that rotate (session, instance/resource
    principal), pass a ``refresh_signer`` callback that refreshes the
    underlying signer. The auth flow will:

    - Periodically refresh on a timer (``refresh_interval`` seconds).
    - Refresh and retry once on a 401 response.

    The callback can either mutate the current signer in place (typical
    for instance/resource principals whose ``refresh_security_token``
    method updates the signer's internal token) **or** return a brand-new
    signer that should replace the current one (typical for session-token
    signers that are immutable but need to be rebuilt from a refreshed
    ``security_token_file`` on disk). Return-value handling: if the
    callback returns anything that exposes ``do_request_sign``, the
    wrapper swaps to it; otherwise the existing in-place mutation is
    assumed.
    """

    requires_request_body = True

    def __init__(
        self,
        signer: AbstractBaseSigner,
        compartment_id: str | None = None,
        refresh_signer: Callable[[], Any] | None = None,
        refresh_interval: float = 3600.0,
    ) -> None:
        self._signer = signer
        self._compartment_id = compartment_id
        self._refresh = refresh_signer
        self._refresh_interval = refresh_interval
        self._last_refresh = time.monotonic()
        self._lock = threading.Lock()
        # Publicly readable last-refresh-error attribute. None means either
        # "no refresh has run yet" or "the most recent refresh succeeded".
        # Operators can inspect this to spot silent refresh failures (the
        # auth flow keeps using the old signer rather than crashing).
        self._last_refresh_error: Exception | None = None

    @property
    def last_refresh_error(self) -> Exception | None:
        """The exception from the most recent failed refresh, or ``None``.

        Reset to ``None`` on every successful refresh. Useful for health
        checks: if this returns non-None for an extended window, the
        underlying signer's credentials are likely stale and the next
        401 will be unrecoverable.
        """
        return self._last_refresh_error

    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
        self._maybe_refresh()
        self._sign(request)
        response = yield request

        if response.status_code == 401 and self._refresh is not None:
            self._do_refresh()
            self._sign(request)
            yield request

    def _maybe_refresh(self) -> None:
        if self._refresh is None:
            return
        if time.monotonic() - self._last_refresh < self._refresh_interval:
            return
        self._do_refresh()

    def _do_refresh(self) -> None:
        if self._refresh is None:
            return
        with self._lock:
            try:
                result = self._refresh()
            except Exception as exc:  # noqa: BLE001 — keep using the old signer if refresh fails
                self._last_refresh_error = exc
                _logger.warning(
                    "OCI signer refresh failed; continuing with previous "
                    "credentials. Inspect OCIRequestSigner.last_refresh_error "
                    "for the cause: %s",
                    exc,
                )
                return
            # If the callback returned a new signer instance, swap to it.
            # Session-token signers are immutable, so refresh has to
            # rebuild them; instance/resource principal signers mutate
            # in place and typically return None.
            if result is not None and hasattr(result, "do_request_sign"):
                self._signer = result
            self._last_refresh = time.monotonic()
            self._last_refresh_error = None
            _logger.debug("OCI signer refresh completed successfully")

    def _sign(self, request: httpx.Request) -> None:
        # Drop any Authorization the openai SDK injected — OCI signing
        # replaces it.
        request.headers.pop("Authorization", None)
        request.headers.pop("X-Api-Key", None)

        # OCI requires opc-compartment-id on /openai/v1/chat/completions
        # under IAM auth. Adding before signing so it's part of the signed
        # payload.
        if self._compartment_id is not None:
            request.headers["opc-compartment-id"] = self._compartment_id

        try:
            content = request.content
        except httpx.RequestNotRead:
            content = request.read()

        proxy = _PreparedRequestProxy(request, content)
        self._signer.do_request_sign(proxy)

        for key, value in proxy.headers.items():
            request.headers[key] = value


__all__ = ["OCIRequestSigner"]
