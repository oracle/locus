# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""``Target`` — a uniform handle to the AI system under assessment.

This is the keystone of the agentic-AI-security SDK: the thing a
red-team / assurance job points at. Whether the subject is a remote
LLM/agent HTTP endpoint, an in-process :class:`tulip.Agent`, an A2A peer,
or an arbitrary async callable, a :class:`Target` exposes the same
minimal contract — :meth:`Target.send` takes a prompt and returns the
target's text response. Probes and assessments operate on that contract
and never need to know what is behind it.

    from tulip.security import Target, red_team

    target = Target.endpoint("https://bot.example/chat", auth=("user", "tok"))
    findings = await red_team(target, suite="owasp-asi")

The offline-friendly :meth:`Target.from_callable` wraps any (sync or
async) ``str -> str`` function, which is what the test-suite and the
bundled demos use to run end-to-end with no network.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from tulip.security.adapter import as_json


# A target's whole contract: a prompt in, a text response out.
Sender = Callable[[str], Awaitable[str]]

_RESPONSE_KEYS = ("response", "output", "text", "content", "message", "answer", "completion")


def _extract_text(body: Any, path: str | None) -> str:
    """Pull the assistant text out of a decoded JSON response body.

    With an explicit dotted ``path`` (e.g. ``"choices.0.message.content"``)
    we walk it; otherwise we try the common single-field shapes and the
    OpenAI chat/completions shape, falling back to the raw JSON so the
    caller always gets *something* to inspect.
    """
    if path:
        cur: Any = body
        for key in path.split("."):
            if isinstance(cur, Mapping) and key in cur:
                cur = cur[key]
            elif isinstance(cur, list) and key.isdigit() and int(key) < len(cur):
                cur = cur[int(key)]
            else:
                return as_json(body)
        return cur if isinstance(cur, str) else as_json(cur)
    if isinstance(body, str):
        return body
    if isinstance(body, Mapping):
        for key in _RESPONSE_KEYS:
            val = body.get(key)
            if isinstance(val, str):
                return val
        # OpenAI-compatible chat / completions shape.
        choices = body.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, Mapping):
                msg = first.get("message")
                if isinstance(msg, Mapping) and isinstance(msg.get("content"), str):
                    return str(msg["content"])
                if isinstance(first.get("text"), str):
                    return str(first["text"])
    return as_json(body)


@dataclass(frozen=True)
class Target:
    """A uniform handle to the AI system under assessment.

    Construct one with a classmethod rather than directly; each builds the
    appropriate :data:`Sender`. ``kind`` records the variant for telemetry
    / finding provenance; ``metadata`` is free-form context (model name,
    owner, environment) that probes may surface in evidence.
    """

    name: str
    kind: str
    _send: Sender
    metadata: Mapping[str, str] = field(default_factory=dict)

    async def send(self, prompt: str) -> str:
        """Send ``prompt`` to the target and return its text response."""
        return await self._send(prompt)

    @classmethod
    def from_callable(
        cls,
        fn: Callable[[str], Awaitable[str] | str],
        *,
        name: str = "callable",
        metadata: Mapping[str, str] | None = None,
    ) -> Target:
        """Wrap any (sync or async) ``str -> str`` function as a target."""

        async def _send(prompt: str) -> str:
            result = fn(prompt)
            text = await result if inspect.isawaitable(result) else result
            return str(text)

        return cls(name=name, kind="callable", _send=_send, metadata=dict(metadata or {}))

    @classmethod
    def endpoint(
        cls,
        url: str,
        *,
        name: str | None = None,
        method: str = "POST",
        auth: Any = None,
        headers: Mapping[str, str] | None = None,
        prompt_field: str = "prompt",
        build_payload: Callable[[str], dict[str, Any]] | None = None,
        response_path: str | None = None,
        timeout: float = 30.0,
        transport: Any = None,
        metadata: Mapping[str, str] | None = None,
    ) -> Target:
        """Target a remote LLM/agent HTTP endpoint.

        By default the prompt is POSTed as ``{prompt_field: prompt}`` and the
        response text is extracted heuristically (common single-field shapes
        and the OpenAI chat shape). Override ``build_payload`` for a custom
        request body and ``response_path`` (dotted, e.g.
        ``"choices.0.message.content"``) for a custom response shape. ``auth``
        / ``headers`` / ``transport`` are passed straight to ``httpx``.
        """

        async def _send(prompt: str) -> str:
            import httpx  # local: keep the module importable without a live client

            payload = build_payload(prompt) if build_payload is not None else {prompt_field: prompt}
            async with httpx.AsyncClient(timeout=timeout, transport=transport) as client:
                resp = await client.request(method, url, json=payload, headers=headers, auth=auth)
                resp.raise_for_status()
                try:
                    body = resp.json()
                except ValueError:
                    return resp.text
            return _extract_text(body, response_path)

        return cls(
            name=name or url,
            kind="endpoint",
            _send=_send,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def agent(
        cls,
        agent: Any,
        *,
        name: str = "agent",
        metadata: Mapping[str, str] | None = None,
    ) -> Target:
        """Target an in-process :class:`tulip.Agent` (or anything with a
        compatible async ``run(prompt)`` event stream).

        Drives the agent's async run and returns the final assistant message
        (the last event carrying a ``final_message``), so an attacker prompt
        flows through the full agent loop — tools and all.
        """

        async def _send(prompt: str) -> str:
            final = ""
            async for event in agent.run(prompt):
                msg = getattr(event, "final_message", None)
                if isinstance(msg, str) and msg:
                    final = msg
            return final

        return cls(name=name, kind="agent", _send=_send, metadata=dict(metadata or {}))

    @classmethod
    def a2a(
        cls,
        sender: Callable[[str], Awaitable[Any]],
        *,
        name: str = "a2a",
        metadata: Mapping[str, str] | None = None,
    ) -> Target:
        """Target an A2A peer via its async send coroutine.

        Thin adapter: pass the coroutine that delivers a message to the peer
        and yields its reply (e.g. bound off your ``tulip.a2a`` client). The
        full protocol client can be wired here as the A2A surface settles.
        """

        async def _send(prompt: str) -> str:
            return str(await sender(prompt))

        return cls(name=name, kind="a2a", _send=_send, metadata=dict(metadata or {}))


__all__ = ["Sender", "Target"]
