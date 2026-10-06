# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""The provider seam, and its zero-infra default: any OpenAI-compatible server.

:class:`DecisionProvider` is the interface. :class:`LogprobDecider` is the free,
local implementation: it asks a chat-completions endpoint for **one** token with
its top log-probabilities and reads every listed answer's probability from that
single forward pass. Nothing is generated beyond the letter, so a 0.6B head on a
GPU answers in milliseconds and on a CPU in tens of them.

It works with any server that returns ``logprobs`` on chat completions — vLLM,
llama.cpp's ``llama-server``, LM Studio, a LiteLLM proxy in front of any of
them — and with any model, trained head or not: a general instruction model
answers the same rendered question zero-shot, only less sharply.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Callable, Coroutine, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

from tulip.core.loop_bound import loop_bound
from tulip.decision.fields import (
    SYSTEM_PROMPT,
    Answer,
    Decision,
    DecisionError,
    Field,
    answer_from_logprobs,
    check_fields,
)


if TYPE_CHECKING:
    import httpx


__all__ = ["DecisionProvider", "LogprobDecider", "decide_sync"]

T = TypeVar("T")


@runtime_checkable
class DecisionProvider(Protocol):
    """Answers typed fields about one input, each with a probability per answer.

    Implementations: :class:`LogprobDecider` (any OpenAI-compatible server), a
    :class:`~tulip.decision.TenantDecisionRouter` over per-tenant heads, or your
    own — a local classifier, a hosted head behind a private endpoint. Raise
    :class:`~tulip.decision.DecisionError` rather than return a guess.
    """

    async def decide(self, text: str, fields: Sequence[Field]) -> Decision:
        """Answer every field of ``fields`` about ``text``."""
        ...


def _run_coroutine_sync(factory: Callable[[], Coroutine[Any, Any, T]]) -> T:
    """Run a coroutine from synchronous code, inside a running loop or not.

    :func:`~tulip.control.policy.approve` is synchronous and is itself called
    from async code (``admit()``), so a sync advisor cannot ``asyncio.run`` on
    the caller's thread. A short-lived thread with a loop of its own can.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(factory())

    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["value"] = asyncio.run(factory())
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread
            box["error"] = exc

    worker = threading.Thread(target=target, name="tulip-decision", daemon=True)
    worker.start()
    worker.join()
    if "error" in box:
        raise box["error"]
    return box["value"]  # type: ignore[no-any-return]


def decide_sync(provider: DecisionProvider, text: str, fields: Sequence[Field]) -> Decision:
    """Call ``provider.decide`` from synchronous code.

    Uses the provider's own ``decide_sync`` when it has one (no thread, no
    loop), and otherwise runs the coroutine on a loop of its own.
    """
    own = getattr(provider, "decide_sync", None)
    if callable(own):
        result = own(text, fields)
        if not isinstance(result, Decision):
            raise DecisionError(f"{type(provider).__name__}.decide_sync returned no Decision")
        return result
    return _run_coroutine_sync(lambda: provider.decide(text, fields))


class LogprobDecider:
    """Decisions from any OpenAI-compatible chat-completions server, by logprobs.

    One request per field, sent concurrently; each asks for ``max_tokens=1``
    with ``logprobs`` and reads the listed letters' probabilities from the top
    log-probabilities at that one position (see
    :func:`~tulip.decision.answer_from_logprobs`).

    Args:
        base_url: The server's OpenAI base URL, e.g. ``http://127.0.0.1:8000/v1``.
        model: The served model name (for vLLM with LoRA, the adapter's name).
        api_key: Sent as a bearer token when given.
        timeout: Seconds per request.
        extra_body: Merged into every request body *under* the fixed keys, so it
            can add options (``{"chat_template_kwargs": {"enable_thinking":
            False}}`` for a Qwen3 head) but cannot change the question, the
            answer length or the logprobs.
        top_logprobs: How many alternatives to ask for. 20 is the OpenAI and
            vLLM maximum; a field with more answers than this can lose mass.
        concurrency: Most requests in flight at once for one decision.
        transport: An ``httpx`` async transport (tests pass
            ``httpx.MockTransport``).
        sync_transport: The same, for :meth:`decide_sync`.
    """

    provider_name = "logprob"

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        timeout: float = 10.0,
        extra_body: Mapping[str, Any] | None = None,
        top_logprobs: int = 20,
        concurrency: int = 8,
        transport: httpx.AsyncBaseTransport | None = None,
        sync_transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("base_url is required")
        if not model:
            raise ValueError("model is required")
        if top_logprobs < 1:
            raise ValueError("top_logprobs must be at least 1")
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._api_key = api_key
        self.timeout = timeout
        self.extra_body = dict(extra_body or {})
        self.top_logprobs = top_logprobs
        self.concurrency = concurrency
        self._transport = transport
        self._sync_transport = sync_transport
        self._client: httpx.AsyncClient | None = None
        self._sync_client: httpx.Client | None = None
        self._sync_lock = threading.Lock()

    def __repr__(self) -> str:
        # The key never appears in a repr, a log line or an exception.
        return f"LogprobDecider(base_url={self.base_url!r}, model={self.model!r})"

    # -- the request ---------------------------------------------------------

    def request_body(self, text: str, question: Field) -> dict[str, Any]:
        """The JSON body sent for one field. Public so a server can be checked against it."""
        return {
            **self.extra_body,
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question.render(text)},
            ],
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": self.top_logprobs,
        }

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    @property
    def _url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _read(self, question: Field, status: int, payload: bytes) -> Answer:
        if status >= 400:
            raise DecisionError(f"field {question.name!r}: the server answered HTTP {status}")
        try:
            body = json.loads(payload)
            first = body["choices"][0]["logprobs"]["content"][0]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise DecisionError(
                f"field {question.name!r}: the response carries no logprobs "
                "(does the server support logprobs on chat completions?)"
            ) from exc
        scores: list[tuple[str, float]] = []
        for alt in first.get("top_logprobs") or []:
            token, logprob = alt.get("token"), alt.get("logprob")
            if isinstance(token, str) and isinstance(logprob, int | float):
                scores.append((token, float(logprob)))
        # The sampled token is normally among the alternatives; count it only
        # when a server left it out, so it is never counted twice.
        token, logprob = first.get("token"), first.get("logprob")
        if (
            isinstance(token, str)
            and isinstance(logprob, int | float)
            and all(token != seen for seen, _ in scores)
        ):
            scores.append((token, float(logprob)))
        return answer_from_logprobs(question, scores)

    def _decision(self, answers: Sequence[Answer], started: float) -> Decision:
        return Decision(
            answers={a.field: a for a in answers},
            model=self.model,
            provider=self.provider_name,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    # -- async ---------------------------------------------------------------

    def _build_client(self) -> httpx.AsyncClient:
        import httpx  # noqa: PLC0415

        return httpx.AsyncClient(timeout=self.timeout, transport=self._transport)

    async def _ask(self, text: str, question: Field, gate: asyncio.Semaphore) -> Answer:
        import httpx  # noqa: PLC0415

        client = loop_bound(self, "_client", self._build_client)
        async with gate:
            try:
                response = await client.post(
                    self._url, json=self.request_body(text, question), headers=self._headers()
                )
            except httpx.HTTPError as exc:
                raise DecisionError(
                    f"field {question.name!r}: request failed ({type(exc).__name__})"
                ) from exc
        return self._read(question, response.status_code, response.content)

    async def decide(self, text: str, fields: Sequence[Field]) -> Decision:
        """Answer every field about ``text``, one concurrent request per field."""
        check_fields(fields)
        started = time.perf_counter()
        gate = asyncio.Semaphore(self.concurrency)
        answers = await asyncio.gather(*(self._ask(text, f, gate) for f in fields))
        return self._decision(answers, started)

    async def aclose(self) -> None:
        """Close the async client, while its loop is still running."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # -- sync ----------------------------------------------------------------

    def decide_sync(self, text: str, fields: Sequence[Field]) -> Decision:
        """The same, from synchronous code (fields one after another)."""
        import httpx  # noqa: PLC0415

        check_fields(fields)
        started = time.perf_counter()
        with self._sync_lock:
            if self._sync_client is None:
                self._sync_client = httpx.Client(
                    timeout=self.timeout, transport=self._sync_transport
                )
            client = self._sync_client
        answers = []
        for question in fields:
            try:
                response = client.post(
                    self._url, json=self.request_body(text, question), headers=self._headers()
                )
            except httpx.HTTPError as exc:
                raise DecisionError(
                    f"field {question.name!r}: request failed ({type(exc).__name__})"
                ) from exc
            answers.append(self._read(question, response.status_code, response.content))
        return self._decision(answers, started)

    def close(self) -> None:
        """Close the synchronous client."""
        if self._sync_client is not None:
            self._sync_client.close()
            self._sync_client = None
