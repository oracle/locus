# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Coverage tests for ``locus.models.providers.oci._signing``.

The signer is deliberately stand-alone so it doesn't pull a real OCI
signer here — we use a fake signer that copies the request and asserts
the expected mutations. ``httpx.Request`` is real (no transport).
"""

from __future__ import annotations

import time
from typing import Any

import httpx
import pytest
import requests

from locus.models.providers.oci._signing import OCIRequestSigner


# ---------------------------------------------------------------------------
# Fake signer
# ---------------------------------------------------------------------------


class _FakeSigner:
    """Mimics ``oci.signer.AbstractBaseSigner`` for the bits we use."""

    def __init__(self) -> None:
        self.calls = 0
        self.last_proxy: Any = None

    def do_request_sign(self, prepared: Any) -> None:
        self.calls += 1
        self.last_proxy = prepared
        # Simulate the signer adding headers (this is what the real one does).
        prepared.headers["x-content-sha256"] = "fakehash"
        prepared.headers["Authorization"] = 'Signature keyId="x", signature="y"'


# ---------------------------------------------------------------------------
# Signs against a real requests.PreparedRequest (oracle-samples practice)
# ---------------------------------------------------------------------------


class TestSignUsesPreparedRequest:
    """``_sign`` must hand ``do_request_sign`` a real
    ``requests.PreparedRequest`` — the object the OCI signer is written and
    tested against — mirroring oracle-samples/oci-genai-auth-python rather
    than a hand-rolled duck-type."""

    def test_signer_receives_prepared_request_with_method_url_body(self) -> None:
        signer = _FakeSigner()
        auth = OCIRequestSigner(signer)  # type: ignore[arg-type]
        req = httpx.Request(
            "POST",
            "https://example.com/openai/v1/chat?stream=1",
            headers={"X-Test": "1"},
            content=b'{"x":1}',
        )
        next(auth.auth_flow(req))
        prepared = signer.last_proxy
        assert isinstance(prepared, requests.PreparedRequest)
        assert prepared.method == "POST"
        assert prepared.url == "https://example.com/openai/v1/chat?stream=1"
        assert prepared.path_url == "/openai/v1/chat?stream=1"
        assert prepared.body == b'{"x":1}'


# ---------------------------------------------------------------------------
# OCIRequestSigner.auth_flow
# ---------------------------------------------------------------------------


class TestAuthFlowSign:
    def test_sign_adds_compartment_and_clears_auth(self) -> None:
        signer = _FakeSigner()
        auth = OCIRequestSigner(
            signer,  # type: ignore[arg-type]
            compartment_id="ocid1.compartment.oc1..xx",
        )
        req = httpx.Request(
            "POST",
            "https://example.com/openai/v1/chat",
            headers={
                "Authorization": "Bearer should-be-stripped",
                "X-Api-Key": "should-also-be-stripped",
            },
            content=b'{"q":1}',
        )

        gen = auth.auth_flow(req)
        signed_request = next(gen)
        # The same request object is yielded back, but now with the
        # signer's headers applied.
        assert signed_request is req
        assert "authorization" in req.headers
        assert req.headers["opc-compartment-id"] == "ocid1.compartment.oc1..xx"
        assert req.headers["x-content-sha256"] == "fakehash"
        # Send a 200 response — flow should terminate.
        try:
            gen.send(httpx.Response(200))
        except StopIteration:
            pass

    def test_no_compartment_id_skips_header(self) -> None:
        signer = _FakeSigner()
        auth = OCIRequestSigner(signer)  # type: ignore[arg-type]
        req = httpx.Request("POST", "https://example.com/v1/x", content=b"")
        gen = auth.auth_flow(req)
        next(gen)
        assert "opc-compartment-id" not in req.headers

    def test_401_triggers_refresh_and_retry(self) -> None:
        refreshed = {"count": 0}

        def refresh() -> None:
            refreshed["count"] += 1

        signer = _FakeSigner()
        auth = OCIRequestSigner(
            signer,  # type: ignore[arg-type]
            refresh_signer=refresh,
        )
        req = httpx.Request("POST", "https://example.com/v1/x", content=b"")
        gen = auth.auth_flow(req)
        next(gen)  # first sign + yield
        # Send a 401 response — flow should refresh and re-sign.
        try:
            retry = gen.send(httpx.Response(401))
            # Got a second yielded request — that's the retry path.
            assert retry is req
        except StopIteration:
            pytest.fail("Expected a retry on 401 but generator stopped")
        assert refreshed["count"] == 1
        # Two sign calls: original + retry
        assert signer.calls == 2
        # Drain the generator
        try:
            gen.send(httpx.Response(200))
        except StopIteration:
            pass

    def test_401_without_refresh_does_not_retry(self) -> None:
        signer = _FakeSigner()
        auth = OCIRequestSigner(signer)  # type: ignore[arg-type]
        req = httpx.Request("POST", "https://example.com/v1/x", content=b"")
        gen = auth.auth_flow(req)
        next(gen)  # initial sign
        # Send 401, but no refresh callback → generator should stop.
        with pytest.raises(StopIteration):
            gen.send(httpx.Response(401))
        assert signer.calls == 1


# ---------------------------------------------------------------------------
# Periodic refresh
# ---------------------------------------------------------------------------


class TestPeriodicRefresh:
    def test_refresh_skipped_inside_window(self) -> None:
        called = {"n": 0}

        def refresh() -> None:
            called["n"] += 1

        signer = _FakeSigner()
        auth = OCIRequestSigner(
            signer,  # type: ignore[arg-type]
            refresh_signer=refresh,
            refresh_interval=600.0,
        )
        # First sign — initialised _last_refresh to now, so no refresh yet.
        req = httpx.Request("POST", "https://example.com/v1/x", content=b"")
        next(auth.auth_flow(req))
        assert called["n"] == 0

    def test_refresh_triggered_when_window_elapsed(self) -> None:
        called = {"n": 0}

        def refresh() -> None:
            called["n"] += 1

        signer = _FakeSigner()
        auth = OCIRequestSigner(
            signer,  # type: ignore[arg-type]
            refresh_signer=refresh,
            refresh_interval=0.0,  # always elapsed
        )
        # Force the last refresh to be in the past
        auth._last_refresh = time.monotonic() - 100.0
        req = httpx.Request("POST", "https://example.com/v1/x", content=b"")
        next(auth.auth_flow(req))
        assert called["n"] == 1

    def test_refresh_failure_swallowed(self) -> None:
        def refresh() -> None:
            raise RuntimeError("refresh broke")

        signer = _FakeSigner()
        auth = OCIRequestSigner(
            signer,  # type: ignore[arg-type]
            refresh_signer=refresh,
            refresh_interval=0.0,
        )
        auth._last_refresh = time.monotonic() - 100.0
        req = httpx.Request("POST", "https://example.com/v1/x", content=b"")
        # Must not raise — the refresh failure is swallowed.
        next(auth.auth_flow(req))


# ---------------------------------------------------------------------------
# _sign reads body when not yet read
# ---------------------------------------------------------------------------


class TestSignReadBody:
    def test_unread_request_reads_body_on_demand(self) -> None:
        signer = _FakeSigner()
        auth = OCIRequestSigner(signer)  # type: ignore[arg-type]
        # Build a request that has not been ``read`` yet — passing bytes via
        # ``content=`` already finalises it, so use an iterable instead.

        def _iterable() -> bytes:
            yield b'{"a":1}'

        req = httpx.Request(
            "POST",
            "https://example.com/v1/x",
            headers={"content-type": "application/json"},
            content=_iterable(),
        )
        # Triggers the ``except httpx.RequestNotRead`` branch.
        next(auth.auth_flow(req))
        assert signer.calls == 1
