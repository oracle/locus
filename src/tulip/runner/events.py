# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""A box runner's progress, sent to the gateway's run stream.

The runner may describe what it is doing — tokens, thinking, a tool starting
and finishing, a command's output — and nothing else. Decisions, holds,
approvals, sandbox and egress events, the workspace diff and the run's end are
written by the gateway alone, from what it decided and observed itself.
:data:`ALLOWED_TYPES` is that list; :meth:`GatewayEvents.emit` refuses any
other type before it leaves the box, and the gateway refuses it again.

Events are numbered from 0 in the order they were emitted and sent in batches::

    POST /internal/v1/runs/{run_id}/events
    {"seq_from": 12, "events": [{"type": "token", ...}, ...]}

``seq_from`` is the number of the batch's first event, so a batch the gateway
already has (a retry after a lost answer) is recognised and not recorded twice.

**Outages.** When the gateway cannot be reached, a batch is appended to a
spool file (``/sandbox/.tulip/events.jsonl`` by default) instead of being
lost; the next successful :meth:`~GatewayEvents.flush` sends what is spooled
first, in order, then truncates the file. The spool lives on the box's
persistent volume, so it also survives the runner parking and coming back.
A refusal (4xx) is not an outage: it is raised, because resending the same
batch would be refused again.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tulip.runner.client import GatewayClient, GatewayUnavailable


__all__ = ["ALLOWED_TYPES", "DEFAULT_SPOOL", "GatewayEvents"]

#: The event types a runner may report.
ALLOWED_TYPES = frozenset(
    {
        "token",
        "think",
        "tool_start",
        "tool_complete",
        "tool.sandbox.output",
        "harness.exec",
        "playbook_step",
        "playbook_progress",
    }
)

#: Where undelivered batches wait, on the box's persistent volume.
DEFAULT_SPOOL = Path("/sandbox/.tulip/events.jsonl")

#: Events per request.
DEFAULT_BATCH = 64


class GatewayEvents:
    """Batched, numbered, spooled event delivery to the gateway.

    Args:
        client: The runner's gateway client.
        spool: The spool file for batches the gateway could not take.
        batch_size: Events per request; :meth:`emit` flushes when a batch is full.
        start_seq: The number of the next event (a resumed runner continues
            where it stopped, from what the gateway reported).
    """

    def __init__(
        self,
        client: GatewayClient,
        *,
        spool: Path | str = DEFAULT_SPOOL,
        batch_size: int = DEFAULT_BATCH,
        start_seq: int = 0,
    ) -> None:
        self._client = client
        self._spool = Path(spool)
        self._batch_size = max(1, batch_size)
        self._seq = start_seq
        self._pending: list[dict[str, Any]] = []

    @property
    def path(self) -> str:
        return f"/internal/v1/runs/{self._client.run_id}/events"

    @property
    def next_seq(self) -> int:
        """The number the next emitted event will get."""
        return self._seq + len(self._pending)

    @property
    def spooled(self) -> int:
        """How many batches are waiting in the spool."""
        if not self._spool.exists():
            return 0
        return sum(1 for line in self._spool.read_text().splitlines() if line.strip())

    async def emit(self, event: dict[str, Any]) -> None:
        """Queue one event; send the batch when it is full.

        Raises:
            ValueError: the event has a type a runner may not report.
        """
        kind = event.get("type")
        if kind not in ALLOWED_TYPES:
            raise ValueError(f"a runner may not report {kind!r} events")
        self._pending.append(dict(event))
        if len(self._pending) >= self._batch_size:
            await self.flush()

    async def flush(self) -> bool:
        """Send the spool, then the queued batch. True when nothing is left undelivered.

        Raises:
            GatewayError: the gateway refused a batch.
        """
        batch = {"seq_from": self._seq, "events": self._pending}
        if self._pending:
            self._seq += len(self._pending)
            self._pending = []
        else:
            batch = {}
        if not await self._drain_spool():
            if batch:
                self._append(batch)
            return False
        if not batch:
            return True
        try:
            await self._client.request("POST", self.path, json=batch)
        except GatewayUnavailable:
            self._append(batch)
            return False
        return True

    async def _drain_spool(self) -> bool:
        if not self._spool.exists():
            return True
        batches = [
            json.loads(line) for line in self._spool.read_text().splitlines() if line.strip()
        ]
        for index, batch in enumerate(batches):
            try:
                await self._client.request("POST", self.path, json=batch)
            except GatewayUnavailable:
                self._rewrite(batches[index:])
                return False
        self._spool.unlink()
        return True

    def _append(self, batch: dict[str, Any]) -> None:
        self._spool.parent.mkdir(parents=True, exist_ok=True)
        with self._spool.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(batch, separators=(",", ":")) + "\n")

    def _rewrite(self, batches: list[dict[str, Any]]) -> None:
        text = "".join(json.dumps(batch, separators=(",", ":")) + "\n" for batch in batches)
        self._spool.write_text(text, encoding="utf-8")
