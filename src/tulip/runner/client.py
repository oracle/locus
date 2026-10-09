# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""The box runner's connection to the Tulip gateway that started it.

A runner is an agent loop inside a sandbox box. It holds no policy, no model
key and no audit chain of its own: every decision, every gateway-side tool and
every record goes back to the gateway that opened the box. This module is the
one HTTP client all of those calls share.

The gateway hands the box three values in its environment (the same names a
bring-your-own box already gets):

- ``TULIP_ADMIT_URL`` — the gateway's base URL, reachable from the box;
- ``TULIP_ADMIT_TOKEN`` — the box's workload token. It is sent as a bearer
  token on every call and is never logged, repr'd or put in an error message;
- ``TULIP_RUN_ID`` — the run this box serves.

:class:`GatewayClient` raises :class:`GatewayUnavailable` when the gateway
cannot be reached (connection refused, timeout, 5xx) and :class:`GatewayError`
when it answered and refused (4xx). Callers that must survive an outage —
:class:`~tulip.runner.events.GatewayEvents` — tell the two apart; everything
else lets them propagate.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import httpx


__all__ = [
    "ADMIT_TOKEN_VAR",
    "ADMIT_URL_VAR",
    "RUN_ID_VAR",
    "GatewayClient",
    "GatewayError",
    "GatewayUnavailable",
    "RunnerConfig",
]

ADMIT_URL_VAR = "TULIP_ADMIT_URL"
ADMIT_TOKEN_VAR = "TULIP_ADMIT_TOKEN"  # noqa: S105 — a variable name, not a secret
RUN_ID_VAR = "TULIP_RUN_ID"

#: Seconds a call may take before it counts as the gateway being unreachable.
DEFAULT_TIMEOUT = 30.0


class GatewayError(Exception):
    """The gateway answered and refused the call (a 4xx).

    ``status`` is the HTTP status and ``detail`` the gateway's own reason.
    """

    def __init__(self, status: int, detail: str, *, path: str) -> None:
        self.status = status
        self.detail = detail
        self.path = path
        super().__init__(f"gateway refused {path} ({status}): {detail}")


class GatewayUnavailable(Exception):  # noqa: N818 — a condition, like ConnectionError
    """The gateway could not be reached, or failed (5xx). Retrying may succeed."""


@dataclass(frozen=True)
class RunnerConfig:
    """Where the runner's gateway is and how it proves which box it is.

    ``token`` never appears in ``repr()``.
    """

    url: str
    run_id: str
    token: str = field(repr=False)
    timeout: float = DEFAULT_TIMEOUT

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> RunnerConfig:
        """Read the three values the gateway sets in every box.

        Raises:
            ValueError: one of them is missing or empty. The message names the
                variable, never a value.
        """
        source = os.environ if env is None else env
        values: dict[str, str] = {}
        for name in (ADMIT_URL_VAR, ADMIT_TOKEN_VAR, RUN_ID_VAR):
            value = (source.get(name) or "").strip()
            if not value:
                raise ValueError(f"{name} is not set: this process was not started by a gateway")
            values[name] = value
        return cls(
            url=values[ADMIT_URL_VAR].rstrip("/"),
            run_id=values[RUN_ID_VAR],
            token=values[ADMIT_TOKEN_VAR],
        )


#: How many times a call the box's network dropped is sent again, and the first wait (s).
GATEWAY_RETRIES = 3
GATEWAY_RETRY_DELAY = 0.25


def _safe_to_repeat(method: str, path: str, exc: httpx.TransportError) -> bool:
    """Whether a call that failed with ``exc`` may be sent again.

    Always when it never connected (``ConnectError``/``ConnectTimeout``: nothing was
    sent). Otherwise only when repeating it changes nothing: a read, a checkpoint (a
    ``PUT`` of the same state) or an event batch (the gateway drops what it has by
    ``seq_from``). An admission, a gateway tool or a result is never sent twice: the
    first may have arrived.
    """
    if isinstance(exc, httpx.ConnectError | httpx.ConnectTimeout):
        return True
    if method.upper() in ("GET", "PUT"):
        return True
    return method.upper() == "POST" and path.rstrip("/").endswith("/events")


class GatewayClient:
    """JSON over HTTP to the gateway, authenticated as this box.

    Args:
        config: The gateway's address, the run and the workload token.
        transport: An ``httpx`` transport to use instead of the network
            (tests pass an ``httpx.MockTransport``).
    """

    def __init__(
        self, config: RunnerConfig, *, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        self.config = config
        #: The transport given, for clients this one starts (a subagent's).
        self.transport = transport
        self._http = httpx.AsyncClient(
            base_url=config.url,
            timeout=config.timeout,
            transport=transport,
            # No kept-alive connections: inside an OpenShell box a pooled tunnel is closed
            # once the box's policy generation moves on, and the next call would fail.
            limits=httpx.Limits(max_keepalive_connections=0),
            headers={"authorization": f"Bearer {config.token}"},
        )

    @property
    def run_id(self) -> str:
        """The run this box serves."""
        return self.config.run_id

    def __repr__(self) -> str:
        return f"GatewayClient(url={self.config.url!r}, run_id={self.config.run_id!r})"

    async def request(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        params: Mapping[str, Any] | None = None,
        allow_404: bool = False,
    ) -> Any:
        """Send one call and return its decoded JSON body (``None`` when empty).

        With ``allow_404`` a 404 returns ``None`` instead of raising.

        Raises:
            GatewayUnavailable: no answer, a timeout, or a 5xx.
            GatewayError: a 4xx.
        """
        attempt = 0
        while True:
            try:
                response = await self._http.request(
                    method,
                    path,
                    json=json,
                    params=dict(params) if params else None,
                )
                break
            except httpx.TransportError as exc:
                # Inside an OpenShell box a connection is cut whenever the box's policy
                # generation moves on. Asked again: a call that is safe to repeat, and any
                # call that never connected (nothing reached the gateway).
                if attempt < GATEWAY_RETRIES and _safe_to_repeat(method, path, exc):
                    attempt += 1
                    await asyncio.sleep(GATEWAY_RETRY_DELAY * attempt)
                    continue
                # The exception's text can carry the URL but never the token: the
                # token is a header, and httpx does not echo headers in errors.
                raise GatewayUnavailable(f"{method} {path}: {type(exc).__name__}") from None
        if response.status_code >= 500:  # noqa: PLR2004 — HTTP status classes
            raise GatewayUnavailable(f"{method} {path}: gateway answered {response.status_code}")
        if response.status_code == 404 and allow_404:  # noqa: PLR2004
            return None
        if response.status_code >= 400:  # noqa: PLR2004
            raise GatewayError(response.status_code, _detail(response), path=path)
        if not response.content:
            return None
        return response.json()

    async def aclose(self) -> None:
        """Close the connection pool."""
        await self._http.aclose()


def _detail(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:200]
    if isinstance(body, dict) and "detail" in body:
        return str(body["detail"])[:500]
    return str(body)[:500]
