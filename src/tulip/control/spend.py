# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Spend ledgers: what a scope has already spent, so policy can cap it.

An :class:`~tulip.security.policy.Action` carries ``cost_usd``, what performing
it will spend. :class:`~tulip.security.policy.ControlPolicy` can require a
person above a per-action cost (``require_human_over_usd``) and deny once a
scope's cumulative spend would cross a limit (``spend_limit_usd``). The ledger
supplies the cumulative figure::

    ledger = FileSpendLedger("spend.json")
    refund = gate_tool(
        issue_refund,
        policy=ControlPolicy(spend_limit_usd=5_000, require_human_over_usd=500),
        action=lambda name, args: Action(
            name=name, asset=args["order_id"], cost_usd=args["amount_usd"]
        ),
        ledger=ledger,
        spend_scope=lambda name, args: f"customer:{args['customer_id']}",
    )

A scope is any string: a thread, a customer, a tenant, a month. Spend is
recorded only after the action has run, so a refused or failed action costs
nothing. The check and the record are not one transaction: two processes
writing one ledger can each pass the check before either records. Use one
writer per scope, or a database-backed ledger with a transactional check.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SpendLedger(Protocol):
    """Cumulative spend per scope."""

    def spent(self, scope: str) -> float:
        """USD recorded against ``scope`` so far; 0 for an unknown scope."""
        ...

    def record(self, scope: str, amount_usd: float, *, action: str = "") -> float:
        """Add ``amount_usd`` to ``scope`` and return the new total."""
        ...


def _check_amount(amount_usd: float) -> None:
    if amount_usd < 0:
        raise ValueError(f"spend cannot be negative: {amount_usd}")


class InMemorySpendLedger:
    """A spend ledger in this process only. Gone on restart; for tests and demos."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._totals: dict[str, float] = {}

    def spent(self, scope: str) -> float:
        with self._lock:
            return self._totals.get(scope, 0.0)

    def record(self, scope: str, amount_usd: float, *, action: str = "") -> float:
        _check_amount(amount_usd)
        with self._lock:
            self._totals[scope] = self._totals.get(scope, 0.0) + amount_usd
            return self._totals[scope]


class FileSpendLedger:
    """A spend ledger in one JSON file, with every entry kept.

    Re-read on every call and written atomically (temporary file, then rename),
    like :class:`~tulip.control.FileApprovals`. Writers are serialised within a
    process only.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def _load(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        data: dict[str, dict[str, Any]] = json.loads(self.path.read_text(encoding="utf-8"))
        return data

    def _save(self, data: dict[str, dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            dir=self.path.parent, prefix=f".{self.path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            Path(tmp).replace(self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def spent(self, scope: str) -> float:
        with self._lock:
            return float(self._load().get(scope, {}).get("spent_usd", 0.0))

    def record(self, scope: str, amount_usd: float, *, action: str = "") -> float:
        _check_amount(amount_usd)
        with self._lock:
            data = self._load()
            entry = data.setdefault(scope, {"spent_usd": 0.0, "entries": []})
            entry["spent_usd"] = float(entry["spent_usd"]) + amount_usd
            entry["entries"].append(
                {"at": datetime.now(UTC).isoformat(), "action": action, "amount_usd": amount_usd}
            )
            self._save(data)
            return float(entry["spent_usd"])

    def entries(self, scope: str) -> list[dict[str, Any]]:
        """Every recorded spend for ``scope``, oldest first."""
        with self._lock:
            return list(self._load().get(scope, {}).get("entries", []))


__all__ = ["FileSpendLedger", "InMemorySpendLedger", "SpendLedger"]
