# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Write ``fixtures/golden.json`` from tulip-gateway's engine (the reference).

Run once, from a checkout of the gateway taken BEFORE it imports the engine from this
package, so the golden trace is the engine as the gateway ran it::

    PYTHONPATH=<gateway>/src:<this repo>/src python tests/unit/playbooks_v2/make_golden.py <gateway sha>

Not a test (no ``test_`` prefix): ``test_golden.py`` replays the same scenarios on
:mod:`tulip.playbooks.v2` and compares.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tests.unit.playbooks_v2.golden_scenarios import FIXTURES, Engine, run_all  # noqa: E402


def gateway_engine() -> Engine:
    from tulip_gateway import playbook_v2, tool_dispatch, when  # type: ignore[import-not-found]

    def make(pb: Any, *, audit: list[dict[str, Any]], **kwargs: Any) -> Any:
        class Sink:
            def emit(self, event: Any) -> None:
                audit.append(event.model_dump(exclude={"tenant"}))

        return playbook_v2.PlaybookRuntime(pb, audit=Sink(), **kwargs)

    def mode(pb: Any, deployment: str) -> str:
        os.environ["TULIP_GATEWAY_PLAYBOOK_ENFORCEMENT"] = deployment
        try:
            return str(playbook_v2.enforcement_mode(pb))
        finally:
            del os.environ["TULIP_GATEWAY_PLAYBOOK_ENFORCEMENT"]

    return Engine(engine=playbook_v2, results=tool_dispatch, when=when, make=make, mode=mode)


def main() -> None:
    sha = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    golden = {"source": f"tulip-gateway {sha}", "scenarios": run_all(gateway_engine())}
    out = FIXTURES / "golden.json"
    out.write_text(json.dumps(golden, indent=1, sort_keys=True, ensure_ascii=False) + "\n", "utf-8")
    print(f"wrote {out} ({out.stat().st_size} bytes)")  # noqa: T201


if __name__ == "__main__":
    main()
