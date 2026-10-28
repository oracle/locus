#!/usr/bin/env python3
# ruff: noqa: T201
"""Capture what a notebook actually prints, and keep the page honest about it.

72 of the 75 notebook pages were prose plus a source include, and the three
that looked like exceptions were showing ASCII architecture diagrams rather
than output. So a site visitor never saw what running an example looks like —
which is most of the persuasive value for the examples whose entire point is
the printed decision trail. `notebook_86` ends on ``chain intact
(tamper-evident): True`` and none of that reached the page.

Output on a page rots the moment the example changes, and rotted output is
worse than none: a reader who runs it and gets something else has no way to
tell which of the two is wrong. So this does both jobs from one place ::

    python scripts/notebook_output.py --update   # re-capture into the pages
    python scripts/notebook_output.py --check    # CI: still matches?

``--check`` runs in the docs build job, which is the one that has the SDK
installed. It pins the same checkout the pages render from, so the output
shown is the output of the source shown.

Only deterministic, offline notebooks belong here. Every entry in
:data:`PAGES` was verified to produce byte-identical output across runs before
being added; one that starts varying will fail ``--check`` loudly rather than
drifting quietly, which is the intended outcome.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs" / "notebooks"

#: Notebook stem -> why its output earns space on the page.
#:
#: Deliberately not "all of them". These are the examples whose printed trail
#: *is* the point: a gate holding, a decision recorded, a probe refused.
PAGES: dict[str, str] = {
    "notebook_83_agent_red_team": "what a red-team run reports",
    "notebook_87_soc_alert_triage": "the triage an analyst would otherwise do",
    "notebook_71_payment_refund_gate": "a small refund paid, a large one held",
    "notebook_72_infra_deploy_gate": "a staging deploy allowed and a production rollback held",
    "notebook_73_support_account_gate": "a small credit applied, a plan upgrade held, and the trail that records both",
    "notebook_74_data_deletion_gate": "a GDPR erasure, and the chain that proves it",
    "notebook_75_cloud_resource_gate": "a staging resize allowed, then a production terminate and an IAM grant held",
}

_MARKER = "## Output"

#: The block this script owns, so ``--update`` replaces rather than accretes.
_BLOCK = re.compile(
    r"\n## Output\n.*?<!-- notebook-output:end -->\n",
    re.DOTALL,
)


def sdk_dir() -> Path:
    """Where the SDK is checked out, matching the docs build's own lookup."""
    for candidate in (
        os.environ.get("LOCUS_SDK_DIR"),
        ROOT / ".sdk",
        ROOT.parent / "locus-sdk",
    ):
        if candidate and Path(candidate).is_dir():
            # Resolved, because the subprocess below runs with ``cwd`` set to
            # this directory: a relative path would then be interpreted
            # relative to itself. CI passes ``LOCUS_SDK_DIR=./.sdk``, which
            # turned into ``.sdk/.sdk/examples/...`` and failed only there.
            return Path(candidate).resolve()
    raise SystemExit("SDK checkout not found — set LOCUS_SDK_DIR")


#: The security-domain notebooks ship with the locus-sdk-security
#: distribution, which lives in the same checkout under packages/.
SECURITY_PKG = Path("packages") / "locus-sdk-security"


def notebook_script(stem: str, sdk: Path) -> Path:
    """The notebook's source: the core examples first, then the security package's."""
    for base in (sdk, sdk / SECURITY_PKG):
        script = base / "examples" / f"{stem}.py"
        if script.is_file():
            return script
    return sdk / "examples" / f"{stem}.py"


def run_notebook(stem: str, sdk: Path) -> str:
    """Run one notebook offline and return exactly what it printed."""
    script = notebook_script(stem, sdk)
    if not script.is_file():
        raise SystemExit(f"{script} not found — is the SDK checkout current?")
    completed = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=script.parent.parent,
        # No provider set: the notebooks fall back to the bundled mock, which
        # is what makes this reproducible and keyless.
        env={
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", ""),
            "PYTHONPATH": os.pathsep.join([str(sdk / "src"), str(sdk / SECURITY_PKG / "src")]),
        },
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(f"{stem} exited {completed.returncode}:\n{completed.stderr[-2000:]}")
    return completed.stdout.rstrip("\n")


def uses_model(stem: str, sdk: Path) -> bool:
    """Whether the notebook builds a model at all.

    Most gate notebooks call ``admit()`` directly and never touch a model, so a
    lead-in claiming a "bundled mock model" would describe something the reader
    cannot find in the source. Read it from the source rather than asserting it.
    """
    script = notebook_script(stem, sdk)
    return script.is_file() and "get_model" in script.read_text()


def render(stem: str, why: str, output: str, *, with_model: bool = True) -> str:
    """The page section, with an end marker so ``--update`` can find it again."""
    how = "no credentials, bundled mock model" if with_model else "no model and no credentials"
    return (
        f"\n{_MARKER}\n\n"
        f"Running it offline — {how} — prints "
        f"{why}:\n\n"
        f"```text\n{output}\n```\n"
        f"<!-- notebook-output:end -->\n"
    )


def page_for(stem: str) -> Path:
    path = DOCS / f"{stem}.md"
    if not path.is_file():
        raise SystemExit(f"no page for {stem} at {path}")
    return path


def apply(stem: str, why: str, output: str, *, write: bool, with_model: bool = True) -> bool:
    """Update or verify one page. Returns True when it already matched."""
    path = page_for(stem)
    text = path.read_text()
    section = render(stem, why, output, with_model=with_model)

    if _BLOCK.search(text):
        updated = _BLOCK.sub(lambda _: section, text, count=1)
    else:
        # New section goes before "## Source": read what it does, then read it.
        anchor = "\n## Source\n"
        if anchor not in text:
            raise SystemExit(f"{path.name} has no '## Source' section to anchor against")
        updated = text.replace(anchor, section + anchor, 1)

    if updated == text:
        return True
    if write:
        path.write_text(updated)
    return False


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="fail if a page is stale")
    group.add_argument("--update", action="store_true", help="re-capture into the pages")
    args = parser.parse_args(argv[1:])

    sdk = sdk_dir()
    stale = []
    for stem, why in PAGES.items():
        output = run_notebook(stem, sdk)
        if not apply(stem, why, output, write=args.update, with_model=uses_model(stem, sdk)):
            stale.append(stem)

    if args.update:
        print(f"updated {len(stale)} page(s); {len(PAGES) - len(stale)} already current")
        return 0

    print(f"checked {len(PAGES)} notebook page(s)")
    if not stale:
        print("output on every page matches a real run")
        return 0
    print("\nthese pages show output the notebook no longer produces:\n")
    for stem in stale:
        print(f"  docs/notebooks/{stem}.md")
    print("\nRefresh them with: python scripts/notebook_output.py --update")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
