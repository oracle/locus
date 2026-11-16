# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""The AgentServer app behind an OCI Function.

OCI Functions run a container that speaks the Fn FDK protocol; this module
is the bridge. ``func.py`` in the image is two lines::

    from locus.deploy.functions import asgi_handler

    handler = asgi_handler("server:app")

Each invocation becomes one request on the ASGI app, run in-process through
``httpx.ASGITransport``. Behind an HTTP trigger or API Gateway the method,
path and headers are the caller's (``LOCUS_FN_PATH_PREFIX`` strips the gateway's
deployment prefix from the path); a direct invocation (``oci fn function
invoke``) has none, so it is treated as ``POST /invoke`` with the body as sent
and the server's own ``LOCUS_SERVER_API_KEY`` as the bearer: the Functions API
already authenticated the caller through IAM. Streaming routes return their
whole body at once, since a function returns one response.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import httpx

from locus.deploy.entrypoint import resolve_secrets


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

__all__ = ["PATH_PREFIX_VARIABLE", "asgi_handler", "request_for"]

#: Set it to the API Gateway deployment prefix (``/agent``) so ``/agent/invoke`` reaches ``/invoke``.
PATH_PREFIX_VARIABLE = "LOCUS_FN_PATH_PREFIX"

_DIRECT_PATH = "/invoke"


def request_for(
    ctx: Any, body: bytes, environ: dict[str, str] | None = None
) -> tuple[str, str, dict[str, str]]:
    """``(method, path_with_query, headers)`` for an FDK invocation context."""
    env = os.environ if environ is None else environ
    method = _call(ctx, "Method")
    url = _call(ctx, "RequestURL")
    headers: dict[str, str] = {}
    raw_headers = _call(ctx, "HTTPHeaders") or {}
    for name, value in dict(raw_headers).items():
        headers[str(name)] = value[0] if isinstance(value, list) else str(value)
    if method and url:
        parts = urlsplit(str(url))
        path = parts.path or "/"
        prefix = env.get(PATH_PREFIX_VARIABLE, "").rstrip("/")
        if prefix and path.startswith(prefix):
            path = path[len(prefix) :] or "/"
        if parts.query:
            path = f"{path}?{parts.query}"
        headers.setdefault("content-type", "application/json")
        return str(method).upper(), path, headers
    headers = {"content-type": "application/json"}
    key = env.get("LOCUS_SERVER_API_KEY")
    if key:
        headers["authorization"] = f"Bearer {key}"
    del body
    return "POST", _DIRECT_PATH, headers


def _fdk_response() -> Any:
    from fdk import response  # type: ignore[import-not-found]  # the Fn FDK, an optional dependency

    return response


def _call(ctx: Any, name: str) -> Any:
    method = getattr(ctx, name, None)
    return method() if callable(method) else None


def asgi_handler(app: str | Any) -> Callable[..., Awaitable[Any]]:
    """An FDK handler that serves ``app`` (an ASGI app or its ``MODULE:ATTR``)."""
    state: dict[str, Any] = {}

    def prepare() -> Any:
        if "app" not in state:
            resolve_secrets(os.environ)
            target = app
            if isinstance(target, str):
                from locus.cli import load_object

                target = load_object(target)
            state["app"] = target
        return state["app"]

    async def handler(ctx: Any, data: Any = None) -> Any:
        body = data.getvalue() if data is not None and hasattr(data, "getvalue") else (data or b"")
        if isinstance(body, str):
            body = body.encode("utf-8")
        method, path, headers = request_for(ctx, body)
        transport = httpx.ASGITransport(app=prepare())
        async with httpx.AsyncClient(transport=transport, base_url="http://function") as client:
            response = await client.request(method, path, content=body, headers=headers)
        return _fdk_response().Response(
            ctx,
            response_data=response.content,
            headers={"content-type": response.headers.get("content-type", "application/json")},
            status_code=response.status_code,
        )

    return handler
