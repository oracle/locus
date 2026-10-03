# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0
#
# Portions of this module (the observation id scheme, the head/tail placeholder
# and the paged recall with its UTF-8 and line limits) are adapted from
# ObservationPack in SoL-Pi (https://github.com/NVlabs/SoL-Pi,
# src/sol-pi/extensions/observation-pack/), which carries this notice:
#
#   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
#   All rights reserved.
#   SPDX-License-Identifier: MIT
#
#   Permission is hereby granted, free of charge, to any person obtaining a
#   copy of this software and associated documentation files (the "Software"),
#   to deal in the Software without restriction, including without limitation
#   the rights to use, copy, modify, merge, publish, distribute, sublicense,
#   and/or sell copies of the Software, and to permit persons to whom the
#   Software is furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in
#   all copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#   DEALINGS IN THE SOFTWARE.

"""ObservationPack: large old tool results leave the request, not the record.

A coding agent reads a 20 KB file, acts on it, and then sends those 20 KB
again on every later request of the run although it rarely looks at them
again. Clearing old output (stage 1 of :class:`~tulip.memory.compaction.ContextCompactor`)
frees the room but loses the bytes: the model has to call the tool again and
may get a different answer. ObservationPack (from SoL-Pi, arXiv 2609.20519)
frees the room losslessly:

* A text tool result over ``threshold_bytes`` is sent in full for its first
  ``full_sends`` provider requests. After that, the *request* shows a
  placeholder instead: a stable id, the size, and about a kilobyte of its
  first and last complete lines. The run's state and its checkpoints keep the
  full result; only the list handed to the provider changes.
* The exact bytes are archived per session, content addressed, and the
  ``obs_recall`` tool pages them back by byte offset or line, at most
  ``recall_max_bytes`` / ``recall_max_lines`` per call.
* Anything that fails (a full disk, a symlinked archive) fails open: the
  request keeps the full result.

**Prompt caching.** Providers cache a request's prefix, and a long run here
reads about nine tenths of every request back from that cache because the
history only ever grows at its end. Swapping one result for its placeholder
rewrites the request from that message on, so a swap per request would cost
more than it saves. Swaps are therefore *batched* and *priced*
(:class:`SwapCostModel`): results that are due wait until together they free
``min_batch_bytes`` and the cache-read savings over the expected rest of the
run outweigh rewriting the cached suffix — or until the prefix breaks anyway
(a compaction, a message window sliding), when a swap behind the break costs
nothing. Once swapped, a result stays swapped, so the new prefix is stable.

Every batch and recall is counted (:class:`ObservationStats`), logged to the
session's ``ledger.jsonl`` beside the archive, and recorded in the run's
:class:`~tulip.observability.mechanisms.MechanismLedger` as ``observation_pack``:
``swap`` per batch, ``placeholders`` per request that sent any (with the bytes
not resent), ``recall``, ``cleared_recallable`` and ``fail_open``.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import stat
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from tulip.core.media import has_images
from tulip.core.messages import Message, Role
from tulip.memory.compactor import _char_count_tokens
from tulip.observability.mechanisms import OBSERVATION_PACK, record_mechanism


logger = logging.getLogger(__name__)

__all__ = [
    "OBSERVATION_ID_KEY",
    "Observation",
    "ObservationArchive",
    "ObservationPack",
    "ObservationStats",
    "SwapCostModel",
    "SwapDecision",
    "is_observation_id",
    "session_directory_name",
]

#: ``Message.metadata`` key naming the archived observation a message stands
#: for: set on placeholders and on outputs compaction cleared into a recallable
#: stub, so neither is packed a second time.
OBSERVATION_ID_KEY = "tulip_observation_id"

#: The marker compaction puts on outputs it cleared. Spelled out rather than
#: imported, because compaction imports this module.
_CLEARED_OUTPUT_KEY = "tulip_compaction_cleared"

_ID_PATTERN = re.compile(r"^obs_[a-f0-9]{24}$")
_CHARS_PER_TOKEN = 4

#: Bytes of each recall reserved for its two header lines.
_RECALL_HEADER_BYTES = 512
_RECALL_HEADER_LINES = 2

#: Placeholder texts and per-message facts kept between requests. A run rarely
#: has more large outputs than this; a miss only costs hashing one again.
_PLACEHOLDER_CACHE_LIMIT = 4_096
_OBSERVED_CACHE_LIMIT = 16_384


def is_observation_id(value: str) -> bool:
    """Whether ``value`` is an id this module issues (and safe as a file name)."""
    return bool(_ID_PATTERN.match(value))


def session_directory_name(session: str) -> str:
    """The directory name for ``session``: what the file checkpointer names a thread."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in session) or "_"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _count_lines(data: bytes) -> int:
    if not data:
        return 0
    return data.count(b"\n") + (0 if data.endswith(b"\n") else 1)


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SwapDecision:
    """Whether a batch of due results is swapped now, and why."""

    swap: bool
    #: ``free``: the prefix breaks before the batch anyway. ``pays``: the
    #: savings over the horizon beat the rewrite. ``batching``: too little is
    #: due yet. ``costly``: the rewrite costs more than it would save.
    #: ``nothing``: the placeholders would not be smaller.
    reason: str
    saved_tokens: int
    rewrite_tokens: int
    benefit: float
    cost: float


@dataclass(frozen=True)
class SwapCostModel:
    """When swapping due results for placeholders is worth breaking the cache.

    Prices are relative to an uncached input token. A swap rewrites the request
    from its first swapped message to the end of what the provider already
    cached (``rewrite_tokens``): those tokens are written again instead of
    read, costing ``cache_write_cost - cache_read_cost`` each, once. In return
    every later request reads ``saved_tokens`` fewer cached tokens, saving
    ``cache_read_cost`` each, over a horizon of ``horizon_requests`` requests.

    Args:
        cache_read_cost: Price of a cached input token. 0.1 on Anthropic,
            OpenAI and DeepSeek-style automatic caches.
        cache_write_cost: Price of re-sending a token that was cached: 1.0
            where a cache miss is billed as plain input, 1.25 on Anthropic's
            five-minute cache writes.
        horizon_requests: Requests a swap is expected to keep saving on.
            ``None`` takes the requests the session has made so far (at least
            ``min_horizon_requests``): a run that has gone on for n requests
            is, without other knowledge, as likely as not to go on for n more.
            Either way it is capped by the requests left before compaction is
            due (when the caller knows the threshold): compaction clears old
            outputs anyway, so a swap stops paying there.
        min_horizon_requests: The adaptive horizon's floor.
        min_batch_bytes: Bytes a batch must free before a swap that breaks a
            cached prefix is considered. Bounds the prefix breaks ObservationPack
            causes at one per ``min_batch_bytes`` of packed output. A swap behind
            a break that happens anyway is free and ignores it.
    """

    cache_read_cost: float = 0.1
    cache_write_cost: float = 1.0
    horizon_requests: int | None = None
    min_horizon_requests: int = 4
    min_batch_bytes: int = 32 * 1024

    def __post_init__(self) -> None:
        if not 0.0 <= self.cache_read_cost <= self.cache_write_cost:
            raise ValueError("need 0 <= cache_read_cost <= cache_write_cost")
        if self.horizon_requests is not None and self.horizon_requests < 1:
            raise ValueError("horizon_requests must be at least 1")
        if self.min_horizon_requests < 1:
            raise ValueError("min_horizon_requests must be at least 1")
        if self.min_batch_bytes < 0:
            raise ValueError("min_batch_bytes must be non-negative")

    def horizon(self, requests_so_far: int, requests_left: float | None = None) -> float:
        """Requests a swap made now is expected to save on."""
        horizon = float(
            self.horizon_requests
            if self.horizon_requests is not None
            else max(self.min_horizon_requests, requests_so_far)
        )
        if requests_left is not None:
            horizon = min(horizon, max(0.0, requests_left))
        return horizon

    def decide(
        self,
        *,
        saved_bytes: int,
        saved_tokens: int,
        rewrite_tokens: int,
        requests_so_far: int,
        requests_left: float | None = None,
    ) -> SwapDecision:
        """Price one batch: swap it now, or keep sending it whole for now.

        ``requests_left`` is how many more requests fit before compaction is
        due, when that is known.
        """
        benefit = saved_tokens * self.cache_read_cost * self.horizon(requests_so_far, requests_left)
        cost = rewrite_tokens * (self.cache_write_cost - self.cache_read_cost)

        def decision(reason: str, *, swap: bool) -> SwapDecision:
            return SwapDecision(swap, reason, saved_tokens, rewrite_tokens, benefit, cost)

        if saved_tokens <= 0:
            return decision("nothing", swap=False)
        if rewrite_tokens <= 0:
            return decision("free", swap=True)
        if saved_bytes < self.min_batch_bytes:
            return decision("batching", swap=False)
        if benefit >= cost:
            return decision("pays", swap=True)
        return decision("costly", swap=False)


# ---------------------------------------------------------------------------
# Observations and their archive
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Observation:
    """One tool output, as archived."""

    id: str
    content_hash: str
    tool_name: str
    text: str
    size: int
    lines: int

    @property
    def data(self) -> bytes:
        return self.text.encode("utf-8", "surrogatepass")

    @property
    def tokens(self) -> int:
        return -(-len(self.text) // _CHARS_PER_TOKEN)

    @classmethod
    def of(cls, message: Message) -> Observation:
        """The observation for a tool message's text."""
        text = message.content or ""
        data = text.encode("utf-8", "surrogatepass")
        content_hash = _sha256(data)
        tool = message.name or "tool"
        key = f"{tool}\0{message.tool_call_id or ''}\0{content_hash}".encode()
        return cls(
            id=f"obs_{_sha256(key)[:24]}",
            content_hash=content_hash,
            tool_name=tool,
            text=text,
            size=len(data),
            lines=_count_lines(data),
        )


class ObservationArchive:
    """One session's archived outputs: ``objects/<id>.txt``, the swapped set, the ledger.

    Objects are written once, exclusively, without following symlinks, and an
    existing one is reused only when its bytes hash to the same content.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.objects = root / "objects"

    def _ensure_dir(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        info = os.lstat(path)
        if not stat.S_ISDIR(info.st_mode):
            raise OSError(f"observation archive {path} is not a directory")

    def path_of(self, observation_id: str) -> Path:
        if not is_observation_id(observation_id):
            raise ValueError(f"unknown observation id: {observation_id}")
        return self.objects / f"{observation_id}.txt"

    def store(self, observation: Observation) -> None:
        """Write ``observation``'s bytes, or check the copy already there."""
        self._ensure_dir(self.objects)
        path = self.path_of(observation.id)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(path, flags, 0o600)
        except FileExistsError:
            existing = self.read(observation.id)
            if _sha256(existing) != observation.content_hash:
                raise OSError(f"archived {observation.id} does not match its content") from None
            return
        with os.fdopen(fd, "wb") as handle:
            handle.write(observation.data)

    def read(self, observation_id: str) -> bytes:
        """An archived object's exact bytes."""
        path = self.path_of(observation_id)
        try:
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        except FileNotFoundError:
            raise ValueError(f"unknown observation id: {observation_id}") from None
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd)
            raise OSError(f"archived {observation_id} is not a regular file")
        with os.fdopen(fd, "rb") as handle:
            return handle.read()

    def swapped(self) -> set[str]:
        """Ids this session already sends as placeholders (survives a restart)."""
        try:
            text = (self.root / "swapped.txt").read_text(encoding="utf-8")
        except FileNotFoundError:
            return set()
        return {line for line in text.split() if is_observation_id(line)}

    def add_swapped(self, ids: Sequence[str]) -> None:
        self._ensure_dir(self.root)
        with (self.root / "swapped.txt").open("a", encoding="utf-8") as handle:
            handle.writelines(f"{i}\n" for i in ids)

    def log(self, entry: dict[str, Any]) -> None:
        """Append one record to the session's ledger."""
        self._ensure_dir(self.root)
        line = json.dumps({"timestamp": time.time(), **entry}, default=str)
        with (self.root / "ledger.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _utf8_floor(data: bytes, end: int) -> int:
    """``end`` moved back off a UTF-8 continuation byte."""
    while 0 < end < len(data) and (data[end] & 0xC0) == 0x80:
        end -= 1
    return end


def _excerpt(data: bytes, budget: int, *, from_end: bool) -> str:
    """Complete lines from one end of ``data`` within ``budget`` bytes.

    A first (or last) line longer than the budget — minified JSON, a long log
    line — is cut at a character boundary instead, so the excerpt is never
    empty for a non-empty output.
    """
    lines = data.splitlines(keepends=True)
    picked: list[bytes] = []
    size = 0
    for line in reversed(lines) if from_end else lines:
        if size + len(line) > budget:
            break
        picked.append(line)
        size += len(line)
    if from_end:
        picked.reverse()
    if not picked and data:
        if from_end:
            start = len(data) - budget
            while start < len(data) and (data[start] & 0xC0) == 0x80:
                start += 1
            return data[start:].decode("utf-8", "replace")
        return data[: _utf8_floor(data, budget)].decode("utf-8", "replace")
    return b"".join(picked).decode("utf-8", "replace")


def placeholder_for(observation: Observation, excerpt_bytes: int) -> str:
    """The text a large output is sent as once it is packed.

    Deterministic in the observation alone, so every request that carries it
    carries the same bytes and the provider's cache keeps serving it.
    """
    head_budget = excerpt_bytes // 2
    tail_budget = excerpt_bytes - head_budget
    head = _excerpt(observation.data, head_budget, from_end=False)
    tail = _excerpt(observation.data, tail_budget, from_end=True)
    recall = json.dumps({"id": observation.id, "offset": 0})
    return "\n".join(
        [
            "[large tool output archived to save context; its exact text is recallable]",
            f"id: {observation.id}",
            f"tool: {observation.tool_name}",
            f"original_bytes: {observation.size}",
            f"original_lines: {observation.lines}",
            f"estimated_tokens: {observation.tokens}",
            f"retrieve: call obs_recall with {recall}; continue with the returned next_offset",
            f"[first complete lines, up to {head_budget} bytes]",
            head.rstrip("\n"),
            f"[middle omitted; last complete lines, up to {tail_budget} bytes]",
            tail.rstrip("\n"),
            "[end of excerpt]",
        ]
    )


def recall_stub_for(observation: Observation, label: str) -> str:
    """A cleared output's one-line stub that still leads back to its exact text."""
    recall = json.dumps({"id": observation.id, "offset": 0})
    return (
        f"[output cleared to free context: {label} returned {len(observation.text)} "
        f"characters. Its exact text is archived as {observation.id}: call obs_recall "
        f"with {recall} to read it.]"
    )


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class ObservationStats:
    """What ObservationPack did for one session, in this process."""

    #: Provider requests projected.
    requests: int = 0
    #: Outputs swapped for placeholders.
    swaps: int = 0
    #: Swap batches: each one rewrote the request from its first swap on.
    batches: int = 0
    #: Batches that cost a cached prefix (the rest rode on a break that
    #: happened anyway).
    prefix_breaks: int = 0
    #: Placeholders sent, summed over requests.
    placeholders_sent: int = 0
    #: Bytes not sent because a placeholder went instead, summed over requests.
    bytes_saved: int = 0
    #: Outputs compaction cleared into a recallable stub instead of a lossy one.
    cleared_recallable: int = 0
    recalls: int = 0
    recalled_bytes: int = 0
    #: Times something failed and the full output was sent instead.
    fail_open: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass
class _Session:
    archive: ObservationArchive
    swapped: set[str]
    stats: ObservationStats = field(default_factory=ObservationStats)
    #: Digests of the last request sent, to find where the next one departs.
    last_digests: list[int] | None = None
    #: Estimated tokens of the last request sent, and the smoothed growth per
    #: request, to tell how soon compaction is due.
    last_tokens: int | None = None
    growth: float | None = None


def _digest(message: Message) -> int:
    """A cheap fingerprint: Python caches a string's hash on the string."""
    return hash(
        (
            message.role,
            message.content,
            message.tool_call_id,
            message.name,
            tuple((call.id, call.name) for call in message.tool_calls),
        )
    )


def _common_prefix(before: Sequence[int], after: Sequence[int]) -> int:
    n = 0
    for a, b in zip(before, after, strict=False):
        if a != b:
            break
        n += 1
    return n


# ---------------------------------------------------------------------------
# The mechanism
# ---------------------------------------------------------------------------


EventSink = Callable[[dict[str, Any]], None]


class ObservationPack:
    """Send large old tool outputs as recallable placeholders.

    One instance serves every run of an agent; what it remembers is kept per
    session (a thread id, or the run id of a run without one), in memory and
    in ``<directory>/<session>/observation-pack/``.

    Args:
        directory: Where per-session archives live. ``None`` uses
            ``$TMPDIR/tulip-observation-pack``.
        threshold_bytes: Outputs larger than this (UTF-8 bytes) are packed.
        full_sends: Requests that carry an output whole before it is due.
        excerpt_bytes: Head plus tail excerpt in a placeholder.
        recall_max_bytes: Most bytes one ``obs_recall`` returns, header included.
        recall_max_lines: Most lines one ``obs_recall`` returns, header included.
        cost_model: When a batch of due outputs is swapped.
    """

    def __init__(
        self,
        *,
        directory: str | Path | None = None,
        threshold_bytes: int = 10 * 1024,
        full_sends: int = 2,
        excerpt_bytes: int = 1024,
        recall_max_bytes: int = 16 * 1024,
        recall_max_lines: int = 400,
        cost_model: SwapCostModel | None = None,
    ) -> None:
        if threshold_bytes < 0:
            raise ValueError("threshold_bytes must be non-negative")
        if full_sends < 0:
            raise ValueError("full_sends must be non-negative")
        if excerpt_bytes < 0:
            raise ValueError("excerpt_bytes must be non-negative")
        if recall_max_bytes <= _RECALL_HEADER_BYTES:
            raise ValueError(f"recall_max_bytes must exceed {_RECALL_HEADER_BYTES}")
        if recall_max_lines <= _RECALL_HEADER_LINES:
            raise ValueError(f"recall_max_lines must exceed {_RECALL_HEADER_LINES}")
        if directory is None:
            import tempfile  # noqa: PLC0415 — only for the default

            directory = Path(tempfile.gettempdir()) / "tulip-observation-pack"
        self.directory = Path(directory).expanduser()
        self.threshold_bytes = threshold_bytes
        self.full_sends = full_sends
        self.excerpt_bytes = excerpt_bytes
        self.recall_max_bytes = recall_max_bytes
        self.recall_max_lines = recall_max_lines
        self.cost_model = cost_model or SwapCostModel()
        self._sessions: dict[str, _Session] = {}
        self._lock = threading.Lock()
        self._observed: dict[tuple[Any, ...], tuple[str, str, int, int]] = {}
        self._placeholders: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    @staticmethod
    def session_key(thread_id: str | None, run_id: str | None) -> str:
        """The session a run belongs to: its thread, or itself without one."""
        return thread_id or run_id or "default"

    def _session(self, session: str) -> _Session:
        with self._lock:
            found = self._sessions.get(session)
            if found is None:
                root = self.directory / session_directory_name(session) / "observation-pack"
                archive = ObservationArchive(root)
                try:
                    swapped = archive.swapped()
                except OSError:
                    swapped = set()
                found = _Session(archive=archive, swapped=swapped)
                self._sessions[session] = found
            return found

    def stats(self, session: str) -> ObservationStats:
        """What the mechanism did for ``session`` in this process."""
        return self._session(session).stats

    def _log(self, state: _Session, entry: dict[str, Any]) -> None:
        try:
            state.archive.log(entry)
        except OSError as exc:  # the ledger is a record, never a reason to fail
            logger.debug("observation ledger write failed: %s", exc)

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _packable(self, message: Message, threshold: int) -> Observation | None:
        """The observation for ``message`` when it may be packed, else ``None``."""
        if message.role != Role.TOOL or not message.content:
            return None
        metadata = message.metadata
        if metadata.get(OBSERVATION_ID_KEY) or metadata.get(_CLEARED_OUTPUT_KEY):
            return None
        content = message.content
        # Cheap rejections first: UTF-8 is at most four bytes a character.
        if len(content) * 4 <= threshold or has_images(content):
            return None
        key = (message.tool_call_id, message.name, len(content), hash(content))
        known = self._observed.get(key)
        if known is None:
            observation = Observation.of(message)
            if len(self._observed) > _OBSERVED_CACHE_LIMIT:
                self._observed.clear()
            # Only the small facts are cached: the text is the message's own.
            self._observed[key] = (
                observation.id,
                observation.content_hash,
                observation.size,
                observation.lines,
            )
        else:
            observation_id, content_hash, size, lines = known
            observation = Observation(
                observation_id, content_hash, message.name or "tool", content, size, lines
            )
        return observation if observation.size > threshold else None

    def _placeholder(self, observation: Observation) -> str:
        text = self._placeholders.get(observation.id)
        if text is None:
            text = placeholder_for(observation, self.excerpt_bytes)
            if len(self._placeholders) > _PLACEHOLDER_CACHE_LIMIT:
                self._placeholders.clear()
            self._placeholders[observation.id] = text
        return text

    def _swap(self, message: Message, observation: Observation) -> Message:
        return message.model_copy(
            update={
                "content": self._placeholder(observation),
                "metadata": {**message.metadata, OBSERVATION_ID_KEY: observation.id},
            }
        )

    def _apply(
        self, messages: Sequence[Message], swapped: set[str]
    ) -> tuple[list[Message], int, int]:
        """``messages`` with every swapped output as its placeholder; bytes saved; swaps shown."""
        out = list(messages)
        saved = 0
        shown = 0
        if not swapped:
            return out, 0, 0
        for index, message in enumerate(out):
            observation = self._packable(message, self.threshold_bytes)
            if observation is not None and observation.id in swapped:
                out[index] = self._swap(message, observation)
                saved += observation.size - len(out[index].content or "")
                shown += 1
        return out, saved, shown

    # ------------------------------------------------------------------
    # The request-time transform
    # ------------------------------------------------------------------

    def view(self, messages: Sequence[Message], *, session: str) -> list[Message]:
        """``messages`` as the next request would show them, without deciding anything.

        For measuring the context (compaction's trigger): outputs already
        swapped count at their placeholder's size.
        """
        try:
            return self._apply(messages, self._session(session).swapped)[0]
        except Exception:  # noqa: BLE001 — fail open: measure the full outputs
            logger.debug("observation view failed", exc_info=True)
            return list(messages)

    def project(
        self,
        messages: Sequence[Message],
        *,
        session: str,
        on_event: EventSink | None = None,
        context_limit: int | None = None,
    ) -> list[Message]:
        """The list to send for this request; ``messages`` itself is never changed.

        Swaps every due output for its placeholder when :class:`SwapCostModel`
        says the batch is worth it, then shows every swapped output as its
        placeholder. Any failure sends ``messages`` as they are.

        ``context_limit`` is the estimated message tokens at which compaction
        starts, when the caller compacts: a swap is priced only over the
        requests left before then.
        """
        state = self._session(session)
        try:
            return self._project(messages, state, on_event, context_limit)
        except Exception as exc:  # noqa: BLE001 — fail open: the full outputs go
            state.stats.fail_open += 1
            error = f"{type(exc).__name__}: {exc}"
            logger.warning("ObservationPack failed open: %s", error)
            self._log(state, {"event": "fail_open", "error": error})
            record_mechanism(
                OBSERVATION_PACK, triggered=False, outcome="fail_open", detail={"error": error}
            )
            return list(messages)

    def _project(
        self,
        messages: Sequence[Message],
        state: _Session,
        on_event: EventSink | None,
        context_limit: int | None,
    ) -> list[Message]:
        # How many requests each message has already been in: the assistant
        # replies that follow it. Derived from the history, so it survives a
        # restart and needs no bookkeeping.
        after: list[int] = [0] * len(messages)
        replies = 0
        for index in range(len(messages) - 1, -1, -1):
            after[index] = replies
            if messages[index].role == Role.ASSISTANT:
                replies += 1

        due: list[tuple[int, Observation]] = []
        for index, message in enumerate(messages):
            if after[index] < self.full_sends:
                continue
            observation = self._packable(message, self.threshold_bytes)
            if observation is not None and observation.id not in state.swapped:
                due.append((index, observation))

        view, _, _ = self._apply(messages, state.swapped)
        tokens = sum(_char_count_tokens(m) for m in view)
        if state.last_tokens is not None and tokens >= state.last_tokens:
            step = float(tokens - state.last_tokens)
            state.growth = step if state.growth is None else 0.7 * state.growth + 0.3 * step
        if due:
            requests_left = None
            if context_limit is not None and state.growth:
                requests_left = (context_limit - tokens) / state.growth
            view = self._consider_batch(
                messages, view, due, replies, state, on_event, requests_left
            )

        view, saved, shown = self._apply(messages, state.swapped)
        stats = state.stats
        stats.requests += 1
        stats.bytes_saved += saved
        stats.placeholders_sent += shown
        if saved > 0:
            # Per request, because that is where the saving happens: these
            # bytes were not sent this time, and would have been.
            record_mechanism(
                OBSERVATION_PACK,
                outcome="placeholders",
                bytes_saved=saved,
                tokens_saved=saved // _CHARS_PER_TOKEN,
                detail={"placeholders": shown},
            )
        state.last_digests = [_digest(m) for m in view]
        state.last_tokens = sum(_char_count_tokens(m) for m in view)
        return view

    def _consider_batch(
        self,
        messages: Sequence[Message],
        view: list[Message],
        due: list[tuple[int, Observation]],
        requests_so_far: int,
        state: _Session,
        on_event: EventSink | None,
        requests_left: float | None,
    ) -> list[Message]:
        digests = [_digest(m) for m in view]
        # Where this request already departs from the last one sent: before
        # it the provider holds a cached prefix; from it on the request is
        # written again whatever happens. With no last request in this
        # process (a resumed session), assume all of it is cached.
        cached_end = (
            _common_prefix(state.last_digests, digests)
            if state.last_digests is not None
            else len(view)
        )
        tentative = list(view)
        saved_bytes = 0
        for index, observation in due:
            tentative[index] = self._swap(messages[index], observation)
            saved_bytes += observation.size - len(tentative[index].content or "")
        first = min(index for index, _ in due)
        rewrite = (
            sum(_char_count_tokens(m) for m in tentative[first:cached_end])
            if first < cached_end
            else 0
        )
        decision = self.cost_model.decide(
            saved_bytes=saved_bytes,
            saved_tokens=saved_bytes // _CHARS_PER_TOKEN,
            rewrite_tokens=rewrite,
            requests_so_far=requests_so_far,
            requests_left=requests_left,
        )
        if not decision.swap:
            return view

        stored: list[Observation] = []
        for _, observation in due:
            try:
                state.archive.store(observation)
            except (OSError, ValueError) as exc:
                # Fail open for this output: it stays whole in every request.
                state.stats.fail_open += 1
                logger.warning("Could not archive %s: %s", observation.id, exc)
                self._log(state, {"event": "fail_open", "id": observation.id, "error": str(exc)})
                record_mechanism(
                    OBSERVATION_PACK,
                    triggered=False,
                    outcome="fail_open",
                    detail={"id": observation.id, "error": str(exc)},
                )
                continue
            stored.append(observation)
        if not stored:
            return view
        ids = [o.id for o in stored]
        state.swapped.update(ids)
        with contextlib.suppress(OSError):
            state.archive.add_swapped(ids)
        stats = state.stats
        stats.swaps += len(stored)
        stats.batches += 1
        if decision.reason != "free":
            stats.prefix_breaks += 1
        entry = {
            "event": "swap",
            "reason": decision.reason,
            "ids": ids,
            "tools": [o.tool_name for o in stored],
            "original_bytes": sum(o.size for o in stored),
            "saved_tokens": decision.saved_tokens,
            "rewrite_tokens": decision.rewrite_tokens,
            "benefit": round(decision.benefit, 1),
            "cost": round(decision.cost, 1),
            "request": requests_so_far + 1,
        }
        self._log(state, entry)
        record_mechanism(
            OBSERVATION_PACK,
            outcome="swap",
            detail={
                "reason": decision.reason,
                "swapped": len(stored),
                "original_bytes": entry["original_bytes"],
                "saved_tokens_per_request": decision.saved_tokens,
                "rewrite_tokens": decision.rewrite_tokens,
                "prefix_break": decision.reason != "free",
            },
        )
        if on_event is not None:
            on_event({**entry, "stats": stats.as_dict()})
        return view

    # ------------------------------------------------------------------
    # Compaction: clear into a recallable stub
    # ------------------------------------------------------------------

    def archive_for(self, session: str) -> SessionArchive:
        """What compaction uses to clear outputs losslessly in ``session``."""
        return SessionArchive(self, self._session(session))

    # ------------------------------------------------------------------
    # Recall
    # ------------------------------------------------------------------

    def recall(
        self,
        session: str,
        observation_id: str,
        *,
        offset: int = 0,
        line: int | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """One page of an archived output: the tool's text, and what it returned.

        ``offset`` is a byte offset into the exact original; ``line`` (1-based)
        starts at a line instead. A page ends at ``recall_max_bytes``,
        ``recall_max_lines`` or the end, never inside a UTF-8 character.
        """
        state = self._session(session)
        observation_id = observation_id.strip()
        data = state.archive.read(observation_id)
        if line is not None:
            if line < 1:
                raise ValueError("line must be at least 1")
            offset = 0
            for _ in range(line - 1):
                found = data.find(b"\n", offset)
                if found < 0:
                    raise ValueError(f"line {line} is past the end ({_count_lines(data)} lines)")
                offset = found + 1
        if offset < 0 or offset > len(data):
            raise ValueError(f"offset {offset} is outside the output ({len(data)} bytes)")

        max_bytes = self.recall_max_bytes - _RECALL_HEADER_BYTES
        max_lines = self.recall_max_lines - _RECALL_HEADER_LINES
        window = data[offset : offset + max_bytes]
        end = len(window)
        newlines = 0
        for position, byte in enumerate(window):
            if byte == 0x0A:
                newlines += 1
                if newlines == max_lines:
                    end = position + 1
                    break
        end = _utf8_floor(data, offset + end) - offset
        chunk = data[offset : offset + end]
        next_offset = offset + len(chunk)
        eof = next_offset >= len(data)
        first_line = data.count(b"\n", 0, offset) + 1
        header = (
            f"[obs_recall id={observation_id} offset={offset} next_offset={next_offset} "
            f"eof={str(eof).lower()} first_line={first_line} total_bytes={len(data)}]\n"
            f"[chunk_bytes={len(chunk)} chunk_lines={_count_lines(chunk)}; "
            + ("end of output]" if eof else "call again with offset=next_offset to continue]")
        )
        details = {
            "id": observation_id,
            "offset": offset,
            "next_offset": next_offset,
            "eof": eof,
            "bytes": len(chunk),
            "lines": _count_lines(chunk),
        }
        stats = state.stats
        stats.recalls += 1
        stats.recalled_bytes += len(chunk)
        self._log(state, {"event": "recall", **details})
        record_mechanism(OBSERVATION_PACK, outcome="recall", detail=details)
        return f"{header}\n{chunk.decode('utf-8', 'surrogatepass')}", {
            **details,
            "stats": stats.as_dict(),
        }


class SessionArchive:
    """One session's archive, as compaction sees it."""

    def __init__(self, pack: ObservationPack, state: _Session) -> None:
        self._pack = pack
        self._state = state

    def _archived(self, message: Message) -> Observation | None:
        """Archive ``message``'s output whatever its size; ``None`` when it cannot be."""
        existing = message.metadata.get(OBSERVATION_ID_KEY)
        if message.role != Role.TOOL or not message.content or existing:
            return None
        if has_images(message.content):
            return None
        observation = Observation.of(message)
        try:
            self._state.archive.store(observation)
        except (OSError, ValueError) as exc:
            self._state.stats.fail_open += 1
            logger.warning("Could not archive %s: %s", observation.id, exc)
            return None
        return observation

    def clear(self, message: Message, label: str) -> Message | None:
        """``message`` cleared to a one-line stub that leads back to its exact text.

        A stub, not the placeholder with its excerpt: clearing is for room,
        and on a long run a kilobyte per cleared output adds up to the window
        it was meant to free. ``None`` means the output could not be archived;
        compaction then clears it the lossy way.
        """
        observation = self._archived(message)
        if observation is None:
            return None
        self._state.stats.cleared_recallable += 1
        # The room it frees is the compaction's own record; this one says the
        # clearing kept the bytes.
        record_mechanism(
            OBSERVATION_PACK, outcome="cleared_recallable", detail={"id": observation.id}
        )
        return message.model_copy(
            update={
                "content": recall_stub_for(observation, label),
                "metadata": {**message.metadata, OBSERVATION_ID_KEY: observation.id},
            }
        )

    def recall_id(self, message: Message) -> str | None:
        """The archive id of a tool output, archiving it first when needed."""
        existing = message.metadata.get(OBSERVATION_ID_KEY)
        if isinstance(existing, str) and is_observation_id(existing):
            return existing
        observation = self._archived(message)
        return observation.id if observation is not None else None
