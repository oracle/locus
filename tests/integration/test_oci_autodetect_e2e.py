# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""End-to-end integration tests for OCI auto-detection.

The unit tests in ``tests/unit/test_examples_config_oci_autodetect.py``
exercise the resolver under ``pytest`` (in-process, ``importlib`` loaded,
sometimes with mocked model classes). These integration tests run the
harness inside a *real* Python subprocess against a *real* (fake)
``~/.oci/config`` — that's the path notebook users actually take when
they invoke ``python examples/notebook_NN.py`` from the CLI.

Closes #276.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_harness_probe(
    home: Path,
    *,
    extra_env: dict[str, str] | None = None,
    snippet: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a Python subprocess that loads ``examples/config.py`` and prints
    a JSON probe of the harness state.

    The default snippet exercises the full auto-detect chain
    (``_auto_detect_provider`` then ``_resolve_oci_profile`` when OCI is
    picked). Pass a custom ``snippet`` to probe other entry points.
    """
    default_snippet = textwrap.dedent(
        """
        import importlib.util, json, sys
        spec = importlib.util.spec_from_file_location(
            "harness", "examples/config.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        provider = mod._auto_detect_provider()
        out = {"provider": provider}
        if provider == "oci":
            try:
                out["profile"] = mod._resolve_oci_profile()
            except RuntimeError as exc:
                out["error"] = str(exc)
        print(json.dumps(out))
        """
    ).strip()

    env = {
        # Isolate from the user's real env: keep PATH/PYTHONPATH/etc but
        # strip everything that influences provider/profile resolution.
        "PATH": os.environ["PATH"],
        "HOME": str(home),
        "PYTHONPATH": str(_REPO_ROOT),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    # virtualenv / uv markers some shells leak in.
    for keep in ("VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"):
        if keep in os.environ:
            env[keep] = os.environ[keep]
    if extra_env:
        env.update(extra_env)

    return subprocess.run(
        [sys.executable, "-c", snippet or default_snippet],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def _write_oci_config(home: Path, content: str) -> Path:
    oci_dir = home / ".oci"
    oci_dir.mkdir(parents=True, exist_ok=True)
    cfg = oci_dir / "config"
    cfg.write_text(content)
    return cfg


# --- E2E auto-detect against a fake HOME ------------------------------------


class TestAutoDetectE2E:
    def test_no_config_picks_mock_in_subprocess(self, tmp_path):
        result = _run_harness_probe(tmp_path)
        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout)
        assert payload == {"provider": "mock"}

    def test_default_section_resolves_to_default(self, tmp_path):
        _write_oci_config(tmp_path, "[DEFAULT]\nuser=u\n")
        result = _run_harness_probe(tmp_path)
        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout)
        assert payload == {"provider": "oci", "profile": "DEFAULT"}

    def test_single_named_profile_resolves_to_that_name(self, tmp_path):
        _write_oci_config(tmp_path, "[BOAT-OC1]\nuser=u\n")
        result = _run_harness_probe(tmp_path)
        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout)
        assert payload == {"provider": "oci", "profile": "BOAT-OC1"}

    def test_multi_profile_no_default_returns_friendly_error(self, tmp_path):
        _write_oci_config(
            tmp_path,
            "[ALPHA]\nuser=u\n[BETA]\nuser=u\n",
        )
        result = _run_harness_probe(tmp_path)
        # The probe catches RuntimeError and prints it as JSON, so
        # returncode is 0 — the subprocess didn't crash, the harness
        # raised the friendly error we want users to see.
        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout)
        assert payload["provider"] == "oci"
        assert "error" in payload
        assert "ALPHA" in payload["error"]
        assert "BETA" in payload["error"]
        assert "OCI_PROFILE" in payload["error"]

    def test_oci_profile_env_overrides_file(self, tmp_path):
        _write_oci_config(
            tmp_path,
            "[ALPHA]\nuser=u\n[BETA]\nuser=u\n",
        )
        result = _run_harness_probe(tmp_path, extra_env={"OCI_PROFILE": "BETA"})
        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout)
        assert payload == {"provider": "oci", "profile": "BETA"}

    def test_locus_oci_profile_env_overrides_file(self, tmp_path):
        _write_oci_config(
            tmp_path,
            "[ALPHA]\nuser=u\n[BETA]\nuser=u\n",
        )
        result = _run_harness_probe(tmp_path, extra_env={"LOCUS_OCI_PROFILE": "ALPHA"})
        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout)
        assert payload == {"provider": "oci", "profile": "ALPHA"}


# --- E2E: real notebook executes against the mock provider ------------------


class TestNotebookE2E:
    """When ``LOCUS_MODEL_PROVIDER=mock``, the harness must short-circuit
    before touching OCI — even if the dev box has a multi-profile
    ``~/.oci/config`` without ``[DEFAULT]`` (the bug-report scenario).

    This is the regression guard: pre-fix, the path was the same; post-fix,
    we need to confirm the new resolver isn't accidentally called when the
    provider is hard-set to mock.
    """

    def test_mock_provider_bypasses_resolver_even_with_ambiguous_config(self, tmp_path):
        _write_oci_config(
            tmp_path,
            "[FOO]\nuser=u\n[BAR]\nuser=u\n[BAZ]\nuser=u\n",
        )
        snippet = textwrap.dedent(
            """
            import importlib.util, json
            spec = importlib.util.spec_from_file_location(
                "harness", "examples/config.py"
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            # LOCUS_MODEL_PROVIDER=mock means get_model() must return
            # MockModel without going through OCI resolution at all.
            m = mod.get_model()
            print(json.dumps({"class": type(m).__name__}))
            """
        ).strip()
        result = _run_harness_probe(
            tmp_path,
            extra_env={"LOCUS_MODEL_PROVIDER": "mock"},
            snippet=snippet,
        )
        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout)
        assert payload == {"class": "MockModel"}

    def test_real_notebook_runs_with_mock_provider(self, tmp_path):
        """Smoke test: run an actual ``examples/notebook_*.py`` end-to-end
        with the new harness, under the same multi-profile config that
        would have produced an opaque SDK trace pre-fix.
        """
        _write_oci_config(
            tmp_path,
            "[ALPHA]\nuser=u\n[BETA]\nuser=u\n",
        )

        # Pick a notebook that exists today and is fast under MockModel.
        # 13_sse_streaming is small, deterministic, and uses get_model().
        notebook = _REPO_ROOT / "examples" / "notebook_13_sse_streaming.py"
        if not notebook.exists():
            pytest.skip(f"notebook fixture not found: {notebook}")

        env = {
            "PATH": os.environ["PATH"],
            "HOME": str(tmp_path),
            "PYTHONPATH": str(_REPO_ROOT),
            "LOCUS_MODEL_PROVIDER": "mock",
        }
        for keep in ("VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"):
            if keep in os.environ:
                env[keep] = os.environ[keep]

        result = subprocess.run(
            [sys.executable, str(notebook)],
            cwd=_REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert result.returncode == 0, (
            f"notebook failed under multi-profile config + mock provider.\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
