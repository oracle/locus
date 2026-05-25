# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for ``examples/litellm-gateway/``.

These tests don't call out anywhere — they only parse the sample files
shipped in the repo and assert their shape. The goal is to catch the
class of error where someone edits the example config and accidentally
breaks one of the documented invariants (env-var wiring, alias /
upstream-model parity, the OKE pod hardening flags, etc.) without
needing a live gateway or OCI tenancy.

The live end-to-end smoke test lives at
``tests/integration/test_litellm_gateway_live.py`` and is gated on
``LITELLM_GATEWAY_URL`` / ``LITELLM_GATEWAY_KEY`` env vars.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
GATEWAY_DIR = REPO_ROOT / "examples" / "litellm-gateway"
CONFIG_YAML = GATEWAY_DIR / "config.yaml"
COMPOSE_YAML = GATEWAY_DIR / "docker-compose.yml"
HELM_YAML = GATEWAY_DIR / "helm-values.yaml"
README_MD = GATEWAY_DIR / "README.md"


# Env vars the sample config consumes via ``os.environ/...`` lookups. If
# any of these is missing from docker-compose.yml the gateway starts but
# every upstream OCI call 401s, which is exactly the failure mode this
# test exists to catch.
_REQUIRED_OCI_ENV = {
    "OCI_REGION",
    "OCI_USER",
    "OCI_FINGERPRINT",
    "OCI_TENANCY",
    "OCI_KEY_FILE",
    "OCI_COMPARTMENT_ID",
}

# Aliases the how-to / notebook 71 advertise as available. Drift between
# these and config.yaml breaks the documented quickstart, so the test
# asserts they remain in sync.
_DOCUMENTED_ALIASES = {
    "oci-cohere-command",
    "oci-grok",
    "oci-gpt5-mini",
    "oci-llama-4-maverick",
    "oci-gemini-2.5-flash",
    "oci-cohere-embed",
}


@pytest.fixture(scope="module")
def gateway_dir_exists() -> Path:
    if not GATEWAY_DIR.is_dir():
        pytest.fail(f"examples/litellm-gateway/ missing at {GATEWAY_DIR}")
    return GATEWAY_DIR


@pytest.fixture(scope="module")
def config(gateway_dir_exists: Path) -> dict[str, Any]:
    return yaml.safe_load(CONFIG_YAML.read_text())


@pytest.fixture(scope="module")
def compose(gateway_dir_exists: Path) -> dict[str, Any]:
    return yaml.safe_load(COMPOSE_YAML.read_text())


@pytest.fixture(scope="module")
def helm_values(gateway_dir_exists: Path) -> dict[str, Any]:
    return yaml.safe_load(HELM_YAML.read_text())


# ---------------------------------------------------------------------------
# config.yaml — the gateway's model catalog
# ---------------------------------------------------------------------------


class TestConfigYaml:
    def test_has_model_list(self, config: dict[str, Any]):
        assert "model_list" in config, "config.yaml missing top-level model_list"
        assert isinstance(config["model_list"], list)
        assert len(config["model_list"]) >= 1

    def test_aliases_match_documentation(self, config: dict[str, Any]):
        """Every alias the docs / notebook 71 advertise must exist in the
        config, and every config alias must be one we document — otherwise
        callers see ``404 not found`` or the docs say a model exists that
        doesn't."""
        aliases = {entry["model_name"] for entry in config["model_list"]}
        assert aliases == _DOCUMENTED_ALIASES, (
            f"alias drift between config.yaml and docs/how-to/litellm-gateway.md: "
            f"only-in-config={aliases - _DOCUMENTED_ALIASES}, "
            f"only-in-docs={_DOCUMENTED_ALIASES - aliases}"
        )

    def test_aliases_unique(self, config: dict[str, Any]):
        aliases = [entry["model_name"] for entry in config["model_list"]]
        assert len(aliases) == len(set(aliases)), f"duplicate aliases: {aliases}"

    def test_every_entry_points_at_oci(self, config: dict[str, Any]):
        """The sample is OCI-fronted. Every entry's upstream ``model`` must
        be an OCI catalog id — otherwise the gateway routes off to a
        provider whose credentials aren't configured."""
        for entry in config["model_list"]:
            model = entry["litellm_params"]["model"]
            assert model.startswith("oci/"), (
                f"alias {entry['model_name']!r} routes to non-OCI model {model!r}; "
                "the OCI-fronting sample should only declare oci/* upstream ids."
            )

    def test_every_entry_pulls_oci_env_vars(self, config: dict[str, Any]):
        """Every model entry must reference all six OCI_* env vars via
        ``os.environ/...`` lookups. Forgetting one means the gateway
        rejects upstream calls with ``Missing required parameters`` at
        runtime — which the live smoke test would catch, but later than
        we want."""
        for entry in config["model_list"]:
            params = entry["litellm_params"]
            referenced = {
                v[len("os.environ/") :]
                for v in params.values()
                if isinstance(v, str) and v.startswith("os.environ/")
            }
            missing = _REQUIRED_OCI_ENV - referenced
            assert not missing, (
                f"alias {entry['model_name']!r} doesn't read these env vars from the "
                f"environment: {sorted(missing)}. Add ``os.environ/<NAME>`` to "
                "litellm_params so the gateway picks them up at runtime."
            )

    def test_drop_params_enabled(self, config: dict[str, Any]):
        """drop_params=True is the documented default; otherwise OpenAI
        kwargs the upstream OCI provider doesn't accept 400 instead of
        being stripped, breaking cross-vendor agent loops."""
        assert config.get("litellm_settings", {}).get("drop_params") is True

    def test_master_key_is_env_sourced(self, config: dict[str, Any]):
        """master_key must NOT be inlined in YAML — it should come from
        the environment so the sample doesn't ship a fixed secret."""
        mk = config["general_settings"]["master_key"]
        assert isinstance(mk, str), "master_key must be a string"
        assert mk.startswith("os.environ/"), (
            "general_settings.master_key must read from the environment "
            "(via ``os.environ/LITELLM_MASTER_KEY``); never inline a key."
        )

    def test_fallback_chains_reference_declared_aliases(self, config: dict[str, Any]):
        """If a fallback target isn't declared in model_list, the
        fallback silently no-ops — exactly the bug class this catches."""
        fallbacks = config.get("router_settings", {}).get("fallbacks", [])
        declared = {entry["model_name"] for entry in config["model_list"]}
        for chain in fallbacks:
            for source, targets in chain.items():
                assert source in declared, f"fallback source {source!r} not declared"
                for t in targets:
                    assert t in declared, f"fallback target {t!r} not declared"


# ---------------------------------------------------------------------------
# docker-compose.yml — local-dev wiring
# ---------------------------------------------------------------------------


class TestComposeYaml:
    def test_uses_official_image(self, compose: dict[str, Any]):
        svc = compose["services"]["litellm"]
        # The official LiteLLM proxy image. Other community forks exist
        # but the docs explicitly pin to the upstream one.
        assert svc["image"].startswith("ghcr.io/berriai/litellm:")

    def test_exposes_port_4000(self, compose: dict[str, Any]):
        svc = compose["services"]["litellm"]
        ports = svc.get("ports", [])
        # Accept either "4000:4000" string or [4000, 4000] sequence;
        # docker-compose normalises both. The point is: 4000 must be
        # reachable from the host, because that's what every doc snippet
        # tells the reader to curl.
        joined = " ".join(str(p) for p in ports)
        assert "4000:4000" in joined or "4000" in joined

    def test_env_var_wiring_is_strict(self, compose: dict[str, Any]):
        """``${OCI_REGION:?...}`` form (with ``?``) causes compose to
        refuse to start when the env var is missing. That's the correct
        failure mode — better than booting the gateway with empty
        credentials and 401-ing every upstream call."""
        env = compose["services"]["litellm"]["environment"]
        # Read environment as a dict regardless of compose normalisation
        # (it ships as a mapping in our sample).
        assert isinstance(env, dict)
        for var in (
            "OCI_REGION",
            "OCI_USER",
            "OCI_FINGERPRINT",
            "OCI_TENANCY",
            "OCI_COMPARTMENT_ID",
        ):
            value = env[var]
            assert ":?" in value, (
                f"{var} should use ${{{var}:?…}} form so compose refuses to "
                f"start without it, got {value!r}"
            )

    def test_key_file_mounted_read_only(self, compose: dict[str, Any]):
        """The OCI private key gets mounted into the container. It must
        be read-only (``:ro``) — the gateway should never be able to
        modify the signing material."""
        svc = compose["services"]["litellm"]
        oci_key_mount = next(
            (v for v in svc["volumes"] if "oci-keys" in v or "OCI_KEY_FILE" in v),
            None,
        )
        assert oci_key_mount is not None, "no OCI key volume mounted"
        assert oci_key_mount.endswith(":ro"), (
            f"OCI key mount must be read-only, got {oci_key_mount!r}"
        )

    def test_container_oci_key_file_points_at_mount(self, compose: dict[str, Any]):
        """Inside the container OCI_KEY_FILE must point at the mounted
        path, not the host path — otherwise the gateway tries to open a
        path that doesn't exist inside the container."""
        env = compose["services"]["litellm"]["environment"]
        assert env["OCI_KEY_FILE"] == "/oci-keys/key.pem"

    def test_healthcheck_present(self, compose: dict[str, Any]):
        svc = compose["services"]["litellm"]
        assert "healthcheck" in svc


# ---------------------------------------------------------------------------
# helm-values.yaml — OKE wiring
# ---------------------------------------------------------------------------


class TestHelmValues:
    def test_uses_official_image_repo(self, helm_values: dict[str, Any]):
        assert helm_values["image"]["repository"] == "ghcr.io/berriai/litellm"

    def test_clusterip_only(self, helm_values: dict[str, Any]):
        """The gateway carries OCI signing material — must not be exposed
        via a LoadBalancer in the sample. Production operators can flip
        to an internal LB / Ingress + mTLS deliberately."""
        assert helm_values["service"]["type"] == "ClusterIP"

    def test_pod_hardening(self, helm_values: dict[str, Any]):
        """Run as non-root, read-only root FS, no privilege escalation,
        drop all capabilities. These are the four flags that move you
        from a default Pod security posture to a hardened one."""
        psc = helm_values["podSecurityContext"]
        assert psc["runAsNonRoot"] is True

        sc = helm_values["securityContext"]
        assert sc["readOnlyRootFilesystem"] is True
        assert sc["allowPrivilegeEscalation"] is False
        # ALL caps dropped. Different YAML loaders normalise the list
        # bracket differently, so accept either form.
        drop = sc["capabilities"]["drop"]
        assert "ALL" in drop

    def test_secrets_sourced_not_inlined(self, helm_values: dict[str, Any]):
        """No literal OCI / master-key values in the sample — must come
        from Kubernetes Secrets via envFrom."""
        env_from = helm_values["envFrom"]
        secret_names = {ref["secretRef"]["name"] for ref in env_from if "secretRef" in ref}
        # The README documents these secret names; if anyone renames one
        # without updating the README the docs go stale.
        assert "oci-credentials" in secret_names
        assert "litellm-master" in secret_names


# ---------------------------------------------------------------------------
# README cross-references
# ---------------------------------------------------------------------------


class TestReadme:
    def test_mentions_all_three_artifacts(self):
        text = README_MD.read_text()
        for filename in ("config.yaml", "docker-compose.yml", "helm-values.yaml"):
            assert filename in text, f"README must reference {filename} so readers can find it"

    def test_points_at_the_howto_page(self):
        text = README_MD.read_text()
        # Either a relative link or the slug shows up — both are acceptable.
        assert "docs/how-to/litellm-gateway.md" in text or "litellm-gateway.md" in text
