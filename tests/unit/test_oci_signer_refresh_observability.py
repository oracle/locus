# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for OCIRequestSigner refresh observability + session-token disk reload.

Covers two gaps closed in v0.2.0b14:

1. `OCIRequestSigner.last_refresh_error` surfaces refresh failures
   instead of swallowing them silently.
2. The refresh callback can return a *new* signer (not just mutate in
   place), and the wrapper swaps it in. That's what makes session-token
   disk-reload work end-to-end through the auth flow.
"""

from __future__ import annotations

from typing import Any

from locus.models.providers.oci._signing import OCIRequestSigner


class _StubSigner:
    """Minimal AbstractBaseSigner stand-in.

    Only needs `do_request_sign` for the wrapper to consider it valid.
    The label lets us assert which signer instance is currently active.
    """

    def __init__(self, label: str) -> None:
        self.label = label

    def do_request_sign(self, request: Any) -> None:  # pragma: no cover
        pass


def test_last_refresh_error_starts_none() -> None:
    wrapper = OCIRequestSigner(_StubSigner("v1"))
    assert wrapper.last_refresh_error is None


def test_last_refresh_error_captures_exception() -> None:
    boom = RuntimeError("token endpoint unreachable")

    def _refresh_that_fails() -> None:
        raise boom

    wrapper = OCIRequestSigner(
        _StubSigner("v1"),
        refresh_signer=_refresh_that_fails,
        refresh_interval=0.0,  # force refresh on every call
    )
    wrapper._do_refresh()
    assert wrapper.last_refresh_error is boom


def test_last_refresh_error_resets_on_success() -> None:
    state = {"calls": 0}

    def _refresh_flaky() -> None:
        state["calls"] += 1
        if state["calls"] == 1:
            raise RuntimeError("transient")
        # second call succeeds

    wrapper = OCIRequestSigner(
        _StubSigner("v1"),
        refresh_signer=_refresh_flaky,
        refresh_interval=0.0,
    )

    wrapper._do_refresh()
    assert isinstance(wrapper.last_refresh_error, RuntimeError)

    wrapper._do_refresh()
    assert wrapper.last_refresh_error is None


def test_refresh_callable_returning_new_signer_swaps_in_place() -> None:
    """The b14 contract: if refresh returns a new signer, the wrapper
    replaces self._signer with it. That's what makes session-token
    disk-reload work end-to-end."""
    new_signer = _StubSigner("v2")

    def _refresh_returning_new() -> _StubSigner:
        return new_signer

    wrapper = OCIRequestSigner(
        _StubSigner("v1"),
        refresh_signer=_refresh_returning_new,
        refresh_interval=0.0,
    )
    assert wrapper._signer.label == "v1"
    wrapper._do_refresh()
    assert wrapper._signer is new_signer
    assert wrapper._signer.label == "v2"
    assert wrapper.last_refresh_error is None


def test_refresh_callable_returning_none_keeps_existing_signer() -> None:
    """In-place mutation (instance/resource principal pattern) — the
    refresh callback's `refresh_security_token` returns None, and the
    wrapper must keep using the same signer instance (which has now
    been mutated internally)."""
    original = _StubSigner("v1")

    def _refresh_in_place() -> None:
        # Pretend we mutated the signer's internal state. Wrapper
        # shouldn't swap it.
        original.label = "v1-rotated"

    wrapper = OCIRequestSigner(original, refresh_signer=_refresh_in_place, refresh_interval=0.0)
    wrapper._do_refresh()
    assert wrapper._signer is original
    assert wrapper._signer.label == "v1-rotated"


def test_refresh_callable_returning_non_signer_is_ignored() -> None:
    """If the refresh callback returns something that doesn't quack like
    a signer (no `do_request_sign`), we treat it as the in-place case
    and keep the existing signer."""
    original = _StubSigner("v1")

    def _refresh_returning_garbage() -> str:
        return "not a signer"  # type: ignore[return-value]

    wrapper = OCIRequestSigner(
        original, refresh_signer=_refresh_returning_garbage, refresh_interval=0.0
    )
    wrapper._do_refresh()
    assert wrapper._signer is original
    # Successful exit path — return type just isn't a signer.
    assert wrapper.last_refresh_error is None


def test_session_token_disk_reload_picks_up_new_file_content(tmp_path) -> None:
    """Full session-token round-trip: write token to disk, build refresh
    callable, mutate the file, call refresh, assert the resulting
    signer carries the new token."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    from locus.models.providers.oci.openai_compat import (
        _build_signer_from_profile,
        _refresh_callable_for,
    )

    # Set up a temp OCI config profile pointing at a temp key + token file
    key_path = tmp_path / "private.pem"
    token_path = tmp_path / "token"
    config_path = tmp_path / "config"

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    key_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    token_path.write_text("TOKEN_v1")
    config_path.write_text(
        f"[TEST]\n"
        f"fingerprint = aa:bb:cc\n"
        f"tenancy = ocid1.tenancy.oc1..x\n"
        f"region = us-chicago-1\n"
        f"key_file = {key_path}\n"
        f"security_token_file = {token_path}\n"
    )

    initial = _build_signer_from_profile("TEST", str(config_path))
    refresh = _refresh_callable_for(initial, profile="TEST", config_file=str(config_path))
    assert refresh is not None  # session-token signer must get a refresh callable

    # Rotate the token on disk (simulating `oci session refresh`)
    token_path.write_text("TOKEN_v2")
    rebuilt = refresh()
    assert rebuilt is not initial
    assert "TOKEN_v2" in rebuilt.api_key
    assert "TOKEN_v1" in initial.api_key  # original signer unchanged


def test_session_token_refresh_returns_none_without_profile() -> None:
    """Without profile/config_file, we can't rebuild the session signer,
    so the refresh callable should be None and the auth flow stays
    dormant (rather than crashing on a missing config)."""
    from locus.models.providers.oci.openai_compat import _refresh_callable_for

    class _SessionLike:
        # Looks like SecurityTokenSigner: no refresh_security_token.
        def do_request_sign(self, request: Any) -> None:  # pragma: no cover
            pass

    assert _refresh_callable_for(_SessionLike()) is None


def test_user_principal_signer_returns_no_refresh() -> None:
    """API-key signers have no expiring credential — refresh callable
    should be None."""
    from locus.models.providers.oci.openai_compat import _refresh_callable_for

    class _ApiKeyLike:
        def do_request_sign(self, request: Any) -> None:  # pragma: no cover
            pass

    assert _refresh_callable_for(_ApiKeyLike(), profile=None, config_file=None) is None
