# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Web-fetch provider protocol + ``httpx`` implementation.

The protocol is one method: ``async fetch(url) -> WebPage``. The default
:class:`HTTPXWebFetcher` implementation uses the ``httpx`` dep that's
already in core, plus a tiny stdlib HTML→text shim so we don't pull in
``html2text`` / ``beautifulsoup`` for the common case.
"""

from __future__ import annotations

import asyncio
import html
import ipaddress
import re
from html.parser import HTMLParser
from typing import Protocol, runtime_checkable
from urllib.parse import urlsplit

from tulip.providers.types import WebPage


class WebFetchError(Exception):
    """A fetch was refused before/while connecting — SSRF guard or bad scheme.

    Raised by :class:`HTTPXWebFetcher` when a URL's scheme isn't http(s), or
    when its host resolves to a private / link-local / loopback / metadata
    address. It is a *refusal*, surfaced to the model as a tool error — never a
    silent fetch of an internal target.
    """


#: Extra IPv4/IPv6 ranges to refuse that Python's ``ipaddress`` category flags
#: don't already cover: CGNAT (100.64/10), IETF benchmarking (198.18/15), and
#: IPv4-mapped IPv6 (``::ffff:0:0/96`` — otherwise ``::ffff:169.254.169.254``
#: would slip past the IPv4 checks). Loopback/private/link-local/reserved/
#: multicast/unspecified are handled by the category properties below.
_EXTRA_BLOCKED_NETS = tuple(
    ipaddress.ip_network(n) for n in ("100.64.0.0/10", "198.18.0.0/15", "::ffff:0:0/96")
)


def _ip_is_blocked(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """True for any address a server-side fetcher must never reach.

    Covers loopback, RFC-1918 private, link-local (incl. the cloud metadata
    endpoint 169.254.169.254 and fe80::/10), reserved, multicast, unspecified,
    plus the :data:`_EXTRA_BLOCKED_NETS`. An IPv4-mapped IPv6 address is
    unwrapped so ``::ffff:10.0.0.1`` is judged on its IPv4 value.
    """
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        ip = mapped
    return bool(
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
        or any(ip in net for net in _EXTRA_BLOCKED_NETS)
    )


async def _assert_public_destination(url: str) -> None:
    """Refuse ``url`` unless it is http(s) to a host that resolves to only
    public addresses. Called on the initial URL and again on every redirect.

    Interim SSRF guard (the durable fix is the egress-proxy allow-list — see
    DESIGN-appsec-threat-model.md A3): a hostile task or prompt-injected page
    can make the model emit ``web_fetch("http://169.254.169.254/…")`` from a
    process that holds real credentials. This blocks the obvious internal
    targets. Residual risk: DNS rebinding between this resolve and httpx's own
    connect-time resolve (TOCTOU) — the egress proxy, which connects to a
    validated, pinned address, is the real closure.
    """
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise WebFetchError(
            f"web_fetch: scheme {parts.scheme or '(none)'!r} is not allowed — http/https only"
        )
    host = parts.hostname
    if not host:
        raise WebFetchError("web_fetch: URL has no host")
    # A bare IP literal is judged directly (no DNS); a name is fully resolved
    # and refused if ANY address it maps to is internal.
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        if _ip_is_blocked(literal):
            raise WebFetchError(
                f"web_fetch: destination {host} is a blocked address "
                "(private/link-local/loopback/metadata ranges are refused)"
            )
        return
    port = parts.port or (443 if parts.scheme == "https" else 80)
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(host, port)
    except OSError as exc:
        raise WebFetchError(f"web_fetch: cannot resolve host {host!r}") from exc
    for info in infos:
        addr = info[4][0]
        if _ip_is_blocked(ipaddress.ip_address(addr)):
            raise WebFetchError(
                f"web_fetch: host {host!r} resolves to a blocked address ({addr}) — "
                "private/link-local/loopback/metadata ranges are refused"
            )


@runtime_checkable
class BaseWebFetchProvider(Protocol):
    """Protocol every web-fetch provider must implement."""

    async def fetch(
        self,
        url: str,
        *,
        max_chars: int = 50_000,
        keep_html: bool = False,
    ) -> WebPage:
        """Fetch ``url`` and return a normalized :class:`WebPage`.

        Implementations should follow redirects, time out within a
        reasonable budget, and cap the returned ``text`` at ``max_chars``
        to keep it agent-context friendly.
        """
        ...


class _HTMLToText(HTMLParser):
    """Minimal HTML → plain-text converter.

    Skips the contents of ``<script>`` / ``<style>`` blocks, collapses
    runs of whitespace, and emits one line per block-level element. This
    is sufficient for an agent reading a page; it doesn't preserve layout
    or tables. Production users who need richer extraction should ship
    a custom :class:`BaseWebFetchProvider` that wraps ``trafilatura`` or
    ``html2text``.
    """

    _BLOCK_TAGS = frozenset(
        {
            "p",
            "div",
            "br",
            "li",
            "tr",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "section",
            "article",
            "header",
            "footer",
            "main",
            "nav",
            "blockquote",
        }
    )
    _SKIP_TAGS = frozenset({"script", "style", "noscript", "iframe"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._buf: list[str] = []
        self._title: str | None = None
        self._in_title = False
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
            return
        if tag == "title":
            self._in_title = True
            return
        if tag in self._BLOCK_TAGS:
            self._buf.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if tag == "title":
            self._in_title = False
            return
        if tag in self._BLOCK_TAGS:
            self._buf.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self._title = (self._title or "") + data
            return
        self._buf.append(data)

    def text(self) -> str:
        joined = "".join(self._buf)
        joined = html.unescape(joined)
        joined = re.sub(r"[ \t\r\f\v]+", " ", joined)
        joined = re.sub(r"\n[ \t]+", "\n", joined)
        joined = re.sub(r"\n{3,}", "\n\n", joined)
        return joined.strip()

    def title(self) -> str:
        return (self._title or "").strip()


class HTTPXWebFetcher:
    """Default web-fetch provider using ``httpx`` + a stdlib HTML→text shim.

    Args:
        timeout_seconds: Per-request timeout. Default 10s.
        user_agent: ``User-Agent`` header. Default ``tulip-web-fetch/1.0``.
        follow_redirects: Whether to follow redirects. Default True.
        max_redirects: Redirect hops to follow (each re-validated). Default 10.
        block_private: Refuse hosts resolving to private/link-local/loopback/
            metadata addresses — the SSRF guard. Default True. Set False ONLY in
            a trusted network where fetching internal hosts is intended (it
            re-opens the SSRF class, incl. cloud-metadata credential theft).
    """

    def __init__(
        self,
        *,
        timeout_seconds: float = 10.0,
        user_agent: str = "tulip-web-fetch/1.0",
        follow_redirects: bool = True,
        max_redirects: int = 10,
        block_private: bool = True,
    ) -> None:
        self._timeout = timeout_seconds
        self._ua = user_agent
        self._follow = follow_redirects
        self._max_redirects = max_redirects
        self._block_private = block_private

    async def fetch(
        self,
        url: str,
        *,
        max_chars: int = 50_000,
        keep_html: bool = False,
    ) -> WebPage:
        import httpx

        if self._block_private:
            await _assert_public_destination(url)

        # Redirects are followed MANUALLY so every hop is re-validated — httpx's
        # own follow_redirects would jump to an internal Location without a
        # second SSRF check. Kept off; we loop and validate each target.
        async with httpx.AsyncClient(
            timeout=self._timeout,
            follow_redirects=False,
            headers={"User-Agent": self._ua},
        ) as client:
            resp = await client.get(url)
            hops = 0
            while self._follow and resp.is_redirect and hops < self._max_redirects:
                location = resp.headers.get("location")
                if not location:
                    break
                next_url = str(resp.url.join(location))
                if self._block_private:
                    await _assert_public_destination(next_url)
                resp = await client.get(next_url)
                hops += 1

        body = resp.text or ""
        ctype = resp.headers.get("content-type", "")
        if "html" in ctype.lower():
            parser = _HTMLToText()
            try:
                parser.feed(body)
                parser.close()
                text = parser.text()
                title = parser.title()
            except (ValueError, TypeError, AttributeError):
                # Defensive: stdlib HTMLParser can choke on truly malformed
                # markup. Fall back to the raw body.
                text = body
                title = ""
        else:
            text = body
            title = ""

        truncated = False
        if len(text) > max_chars:
            text = text[:max_chars]
            truncated = True

        return WebPage(
            url=str(resp.url),
            status=resp.status_code,
            title=title,
            text=text,
            html=body if keep_html else None,
            truncated=truncated,
        )


__all__ = ["BaseWebFetchProvider", "HTTPXWebFetcher", "WebFetchError"]
