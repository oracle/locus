#!/usr/bin/env python3
"""Run Tulip notebooks with a specific model provider.

This script configures the environment and runs notebooks to verify
they work correctly with real LLM providers.

Usage:
    # Run all notebooks with mock (default):
    python scripts/run_notebooks.py

    # Run all notebooks with OpenAI:
    python scripts/run_notebooks.py --provider openai

    # Run specific notebook:
    python scripts/run_notebooks.py --provider openai --notebook 06

    # List available notebooks:
    python scripts/run_notebooks.py --list
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


# Provider configurations.
PROVIDERS = {
    "mock": {
        "TULIP_MODEL_PROVIDER": "mock",
    },
    "openai": {
        "TULIP_MODEL_PROVIDER": "openai",
        "TULIP_MODEL_ID": "gpt-4o",
    },
    "anthropic": {
        "TULIP_MODEL_PROVIDER": "anthropic",
        "TULIP_MODEL_ID": "claude-sonnet-4-6",
    },
}


def get_notebooks() -> list[Path]:
    """Get list of notebook files."""
    examples_dir = Path(__file__).parent.parent / "examples"
    notebooks = sorted(examples_dir.glob("notebook_*.py"))
    return notebooks


def run_notebook(notebook: Path, env: dict[str, str], timeout: int = 120) -> bool:
    """Run a single notebook.

    Returns True if successful, False otherwise.
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
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {notebook.name} exceeded {timeout}s")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run Tulip notebooks")
    parser.add_argument(
        "--provider",
        choices=list(PROVIDERS.keys()),
        default="mock",
        help="Model provider to use",
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
    env = PROVIDERS[args.provider].copy()

    # Apply overrides
    if args.model:
        env["TULIP_MODEL_ID"] = args.model

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
        success = run_notebook(notebook, env, args.timeout)
        results[notebook.name] = success

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    passed = sum(1 for s in results.values() if s)
    failed = len(results) - passed

    for name, success in results.items():
        status = "PASS" if success else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\nTotal: {passed} passed, {failed} failed")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
