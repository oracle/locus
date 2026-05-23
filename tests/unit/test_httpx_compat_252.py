# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Regression test for #252 — httpx 1.0 prerelease incompatibility.

``OCIRequestSigner`` (``locus.models.providers.oci._signing``) and
``BearerAuth`` (``locus.integrations.fastmcp``, defined inside
``__init__``) both subclass ``httpx.Auth`` — the public re-export
that httpx 1.0.dev3 dropped. Resolving with prereleases allowed
pulled in 1.0.dev3 and broke class-definition time on import.

The fix is a constraint cap (``httpx>=0.27,<1.0``) in ``pyproject.toml``.
This test asserts the cap is in place so a careless ``>=`` cleanup
later doesn't silently regress.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import httpx


def _pyproject_path() -> Path:
    return Path(__file__).resolve().parents[2] / "pyproject.toml"


def test_httpx_dependency_is_capped_below_1_0():
    """``pyproject.toml`` must keep ``httpx<1.0`` until both ``httpx.Auth``
    subclasses are ported to the httpx 1.0 auth API."""
    data = tomllib.loads(_pyproject_path().read_text())
    deps = data["project"]["dependencies"]
    httpx_pins = [d for d in deps if d.startswith("httpx")]
    assert httpx_pins, "httpx must remain in [project.dependencies]"
    assert len(httpx_pins) == 1, f"unexpected duplicate httpx pins: {httpx_pins!r}"
    pin = httpx_pins[0]
    assert "<1.0" in pin or "<1" in pin, (
        f"httpx pin {pin!r} is missing the <1.0 upper bound — see #252. "
        "httpx 1.0.dev3 drops the top-level httpx.Auth re-export, which "
        "breaks OCIRequestSigner and BearerAuth."
    )


def test_oci_request_signer_imports_and_subclasses_httpx_auth():
    """The class that prompted the cap must still import + remain a
    real subclass of ``httpx.Auth`` under the resolved httpx version."""
    from locus.models.providers.oci._signing import OCIRequestSigner

    assert issubclass(OCIRequestSigner, httpx.Auth)


def test_fastmcp_module_imports_cleanly():
    """Smoke: the ``BearerAuth`` defined inside ``MCPClient.__init__``
    can't be reached at module top level, but the surrounding module
    must still import — that's the regression surface from #252."""
    import locus.integrations.fastmcp  # noqa: F401 — import is the assertion


# =============================================================================
# Actually exercise the httpx.Auth contract — the classes don't just need to
# import, they need to keep working as httpx.Auth handlers. Without this, a
# silent break in the auth_flow signature (which can happen across httpx
# major versions) would slip through unnoticed.
# =============================================================================


class _DummyOCISigner:
    """Stand-in for an ``oci.signer.AbstractBaseSigner``.

    The OCI signer interface only exposes ``do_request_sign(prepared)``;
    real signers mutate ``prepared.headers`` in place with an
    ``Authorization`` (and a few ``x-content-sha256`` / ``date`` headers).
    The dummy mimics that contract.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def do_request_sign(self, prepared: object) -> None:
        # The proxy exposes mutable ``headers`` and read-only ``method`` /
        # ``url`` / ``path_url`` / ``body``. Capture for the assertion and
        # then mutate exactly like a real signer.
        self.calls.append(
            {
                "method": getattr(prepared, "method", None),
                "url": getattr(prepared, "url", None),
                "path_url": getattr(prepared, "path_url", None),
                "body": getattr(prepared, "body", None),
            }
        )
        prepared.headers["Authorization"] = "Signature dummy-sig"  # type: ignore[attr-defined]
        prepared.headers["x-content-sha256"] = "dummy-hash"  # type: ignore[attr-defined]


def test_oci_request_signer_auth_flow_signs_request_through_httpx():
    """``auth_flow`` is the httpx.Auth contract method. Drive it the same
    way ``httpx.Client.send`` would, then assert the signer was called
    and the signed headers landed on the outgoing request."""
    from locus.models.providers.oci._signing import OCIRequestSigner

    signer = _DummyOCISigner()
    auth = OCIRequestSigner(signer=signer, compartment_id="ocid1.compartment..xyz")

    request = httpx.Request(
        "POST",
        "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/openai/v1/chat/completions",
        content=b'{"hello":"world"}',
        headers={"Authorization": "Bearer should-be-stripped", "X-Api-Key": "drop-me"},
    )

    # Drive the auth_flow generator manually — same protocol httpx.Client
    # follows. First yield = the signed request to send; we feed a 200 back
    # and confirm the generator terminates (no retry).
    flow = auth.auth_flow(request)
    signed = next(flow)
    assert signed is request
    fake_response = httpx.Response(200, content=b"{}", request=request)
    try:
        flow.send(fake_response)
    except StopIteration:
        pass

    assert len(signer.calls) == 1
    call = signer.calls[0]
    assert call["method"] == "POST"
    assert call["body"] == b'{"hello":"world"}'
    # The previous Authorization header was stripped before signing — the
    # one the test sees now came from the signer.
    assert request.headers["Authorization"] == "Signature dummy-sig"
    assert "X-Api-Key" not in request.headers
    assert request.headers["x-content-sha256"] == "dummy-hash"
    # Compartment id gets attached as part of the signed payload.
    assert request.headers["opc-compartment-id"] == "ocid1.compartment..xyz"


def test_oci_request_signer_retries_once_on_401_with_refresh():
    """When a refresh callback is configured and the first attempt 401s,
    the auth_flow refreshes the signer and replays the request — the
    second yield is the same (re-signed) request."""
    from locus.models.providers.oci._signing import OCIRequestSigner

    signer = _DummyOCISigner()
    refreshed: list[int] = []

    def refresh() -> None:
        refreshed.append(1)

    auth = OCIRequestSigner(signer=signer, refresh_signer=refresh)

    request = httpx.Request(
        "POST",
        "https://example.invalid/openai/v1/chat/completions",
        content=b"{}",
    )

    flow = auth.auth_flow(request)
    first = next(flow)
    assert first is request
    # Simulate a 401 — the signer should refresh + retry once.
    second = flow.send(httpx.Response(401, content=b"", request=request))
    assert second is request
    # Generator terminates on the next .send().
    try:
        flow.send(httpx.Response(200, content=b"{}", request=request))
    except StopIteration:
        pass

    # Refresh fired once. The signer's been called twice (original + retry).
    assert refreshed == [1]
    assert len(signer.calls) == 2


def test_oci_request_signer_swallows_refresh_exception():
    """The refresh callback raising must NOT crash the auth_flow — the
    previous credentials stay in use and the exception is exposed via
    ``last_refresh_error`` for health-check inspection."""
    from locus.models.providers.oci._signing import OCIRequestSigner

    signer = _DummyOCISigner()
    boom = RuntimeError("token endpoint down")

    def refresh() -> None:
        raise boom

    auth = OCIRequestSigner(
        signer=signer,
        refresh_signer=refresh,
        refresh_interval=0.0,  # force refresh on next call
    )

    request = httpx.Request("GET", "https://example.invalid/openai/v1/models")
    flow = auth.auth_flow(request)
    next(flow)  # triggers _maybe_refresh which calls refresh()
    try:
        flow.send(httpx.Response(200, content=b"{}", request=request))
    except StopIteration:
        pass

    assert auth.last_refresh_error is boom
    # Signer still ran (with the old credentials, as documented).
    assert len(signer.calls) == 1


def test_bearer_auth_pattern_works_under_current_httpx():
    """Mirror the ``BearerAuth`` class that ``MCPClient.__init__`` builds
    inline. The pattern (subclass httpx.Auth, yield a request with the
    Bearer header set) must keep working under whichever httpx the cap
    resolves to."""

    class BearerAuth(httpx.Auth):
        def __init__(self, token: str):
            self.token = token

        def auth_flow(self, request):  # type: ignore[no-untyped-def]
            request.headers["Authorization"] = f"Bearer {self.token}"
            yield request

    auth = BearerAuth("test-token")
    request = httpx.Request("GET", "https://example.invalid/v1/whatever")
    flow = auth.auth_flow(request)
    signed = next(flow)
    assert signed is request
    assert request.headers["Authorization"] == "Bearer test-token"
    try:
        flow.send(httpx.Response(200, content=b"{}", request=request))
    except StopIteration:
        pass
