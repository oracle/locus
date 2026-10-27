#!/usr/bin/env python3
"""Run Locus notebooks with a specific model provider.

This script configures the environment and runs notebooks to verify
they work correctly with real LLM providers.

Usage:
    # Run all notebooks with mock (default):
    python scripts/run_notebooks.py

    # Run all notebooks with OpenAI:
    python scripts/run_notebooks.py --provider openai

    # Run specific notebook:
    python scripts/run_notebooks.py --provider openai --notebook 06

    # Any other provider the model registry knows (needs --model):
    python scripts/run_notebooks.py --provider gemini --model gemini-2.5-flash

    # List available notebooks:
    python scripts/run_notebooks.py --list
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


#: Exit status a notebook uses to say it cannot run here — its prerequisites
#: (an OCI profile, a service endpoint) are missing. The automake/meson
#: convention for a skipped test; Python itself never exits with it.
SKIP_EXIT_CODE = 77


# Provider configurations.
PROVIDERS = {
    "mock": {
        "LOCUS_MODEL_PROVIDER": "mock",
    },
    "openai": {
        "LOCUS_MODEL_PROVIDER": "openai",
        "LOCUS_MODEL_ID": "gpt-4o",
    },
    "anthropic": {
        "LOCUS_MODEL_PROVIDER": "anthropic",
        "LOCUS_MODEL_ID": "claude-sonnet-4-6",
    },
}


def get_notebooks() -> list[Path]:
    """Get list of notebook files."""
    examples_dir = Path(__file__).parent.parent / "examples"
    notebooks = sorted(examples_dir.glob("notebook_*.py"))
    return notebooks


def run_notebook(notebook: Path, env: dict[str, str], timeout: int = 120) -> str:
    """Run a single notebook.

    Returns ``"PASS"``, ``"SKIP"`` (it exited with :data:`SKIP_EXIT_CODE`) or
    ``"FAIL"``.
    """
    print(f"\n{'=' * 60}")
    print(f"Running: {notebook.name}")
    print(f"{'=' * 60}")

    try:
        result = subprocess.run(
            [sys.executable, str(notebook)],
            env={**os.environ, **env},
            timeout=timeout,
            capture_output=False,
            check=False,
        )
        if result.returncode == 0:
            return "PASS"
        if result.returncode == SKIP_EXIT_CODE:
            return "SKIP"
        return "FAIL"
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {notebook.name} exceeded {timeout}s")
        return "FAIL"
    except Exception as e:
        print(f"ERROR: {e}")
        return "FAIL"


def main():
    parser = argparse.ArgumentParser(description="Run Locus notebooks")
    parser.add_argument(
        "--provider",
        default="mock",
        help=(
            f"Model provider: {', '.join(PROVIDERS)}, or any prefix in "
            "locus.models.registry.list_providers() together with --model"
        ),
    )
    parser.add_argument(
        "--model",
        help="Model ID (overrides default for provider)",
    )
    parser.add_argument(
        "--notebook",
        help="Run specific notebook (e.g., '01' or '01,02,03')",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available notebooks",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help=(
            "Timeout per notebook in seconds. Some notebooks "
            "(orchestrator/specialist/multi-agent/RAG) make many model "
            "calls — 300s gives them headroom."
        ),
    )
    args = parser.parse_args()

    notebooks = get_notebooks()

    if args.list:
        print("Available notebooks:")
        for t in notebooks:
            print(f"  {t.name}")
        return

    # Get provider config
    if args.provider in PROVIDERS:
        env = PROVIDERS[args.provider].copy()
    elif args.model:
        env = {"LOCUS_MODEL_PROVIDER": args.provider}
    else:
        parser.error(f"--provider {args.provider} needs --model")

    # Apply overrides
    if args.model:
        env["LOCUS_MODEL_ID"] = args.model

    print(f"Provider: {args.provider}")
    print(f"Config: {env}")

    # Filter notebooks if specified
    if args.notebook:
        numbers = args.notebook.split(",")
        notebooks = [
            t for t in notebooks if any(f"notebook_{n.zfill(2)}" in t.name for n in numbers)
        ]

    if not notebooks:
        print("No notebooks found matching criteria")
        return

    # Run notebooks
    results = {}
    for notebook in notebooks:
        results[notebook.name] = run_notebook(notebook, env, args.timeout)

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    passed = sum(1 for s in results.values() if s == "PASS")
    skipped = sum(1 for s in results.values() if s == "SKIP")
    failed = len(results) - passed - skipped

    for name, status in results.items():
        print(f"  [{status}] {name}")

    print(f"\nTotal: {passed} passed, {skipped} skipped, {failed} failed")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
