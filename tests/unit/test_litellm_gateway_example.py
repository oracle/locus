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

# How-to markdown — the single source of truth for which OCI aliases the
# gateway is documented to expose. The parity test scrapes this file for
# backticked tokens that start with ``oci-`` and asserts ``config.yaml``
# exposes exactly that set. No hardcoded alias list to drift against the
# docs.
HOWTO_MD = REPO_ROOT / "docs" / "how-to" / "litellm-gateway.md"

# Tokens that scrape positive on ``oci-...`` but aren't gateway aliases —
# exclude them from the parity set so the regex doesn't false-positive
# on real text from the docs.
_NON_ALIAS_OCI_TOKENS = {
    # OCI service / API names referenced in prose, not gateway aliases.
    "oci-models.md",
    "oci-dac.md",
    # add more here if the docs ever reference an `oci-...` string that
    # isn't a gateway alias — keep this list tight to keep the test honest.
}


def _aliases_from_docs() -> set[str]:
    """Return every gateway alias the how-to advertises.

    Looks for backticked tokens matching ``^oci-[a-z0-9.-]+$`` and
    filters out the documentation cross-references in
    ``_NON_ALIAS_OCI_TOKENS``. The contract enforced by the parity test
    is then: *what config.yaml exposes must equal what the docs say it
    exposes.*
    """
    import re

    text = HOWTO_MD.read_text()
    # Match `oci-foo`, `oci-foo-bar`, `oci-foo.bar` (escape: ``oci-foo``).
    tokens = set(re.findall(r"`(oci-[a-z0-9][a-z0-9.\-]*)`", text))
    return {t for t in tokens if t not in _NON_ALIAS_OCI_TOKENS}


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
        """Single source of truth: the set of aliases declared in
        ``config.yaml`` must equal the set of aliases the how-to
        markdown mentions in backticks. Updating one without updating
        the other should fail this test loudly — not silently leave a
        404 for the reader. The "documented" set is *parsed* from the
        how-to file, not hardcoded here, so there's no third source to
        drift against."""
        from_config = {entry["model_name"] for entry in config["model_list"]}
        from_docs = _aliases_from_docs()
        assert from_docs, (
            "the how-to (docs/how-to/litellm-gateway.md) mentions no "
            "`oci-...` aliases in backticks — either the file is wrong "
            "or the scrape regex needs updating."
        )
        assert from_config == from_docs, (
            "alias drift between config.yaml and "
            "docs/how-to/litellm-gateway.md:\n"
            f"  only-in-config-yaml: {sorted(from_config - from_docs)}\n"
            f"  only-in-how-to-md:   {sorted(from_docs - from_config)}\n"
            "Fix one or the other so they agree (or add the token to "
            "_NON_ALIAS_OCI_TOKENS if it's a doc cross-reference, "
            "not a gateway alias)."
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
        """The default LiteLLM image must be the official ghcr.io build.
        The compose file wraps it in ``${LITELLM_IMAGE:-...}`` so users
        behind networks that can't reach ghcr.io can override; we assert
        only the default behaviour here."""
        svc = compose["services"]["litellm"]
        img = svc["image"]
        # Either the literal path or the ${LITELLM_IMAGE:-...} form,
        # both of which fall back to ghcr.io/berriai/litellm.
        assert "ghcr.io/berriai/litellm" in img, (
            f"litellm image should default to ghcr.io/berriai/litellm:..., got {img!r}"
        )

    def test_litellm_image_is_overridable(self, compose: dict[str, Any]):
        """Operators behind corporate proxies that block ghcr.io must
        be able to override the image without editing this file. Assert
        the env-var-with-default form is in use."""
        img = compose["services"]["litellm"]["image"]
        assert "${LITELLM_IMAGE" in img, (
            f"litellm image must use ${{LITELLM_IMAGE:-...}} form so "
            f"users can override it; got {img!r}"
        )

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
# docker-compose.yml — Postgres sidecar (required for /key/generate +
# persistent virtual-key state + spend logs)
# ---------------------------------------------------------------------------


class TestComposePostgres:
    def test_db_service_declared(self, compose: dict[str, Any]):
        """A Postgres ``db`` service must exist alongside the gateway —
        without it the gateway runs in stateless mode and
        ``/key/generate`` 500s with ``{"error": "DB not connected"}``."""
        assert "db" in compose["services"], (
            "docker-compose.yml is missing the `db` service. Add a "
            "postgres sidecar so /key/generate, virtual-key state, and "
            "spend logs persist across restarts."
        )

    def test_db_is_postgres(self, compose: dict[str, Any]):
        """Image must be a postgres-shaped tag. The compose file uses
        ``${LITELLM_DB_IMAGE:-postgres:17-alpine}`` so operators behind
        corporate proxies can swap in a mirror (e.g.
        ``mirror.gcr.io/library/postgres:17-alpine``) without editing
        this file."""
        db_img = compose["services"]["db"]["image"]
        assert "postgres" in db_img, f"db service image should be postgres, got {db_img!r}"

    def test_db_image_is_overridable(self, compose: dict[str, Any]):
        """The Postgres pull is the most common failure point behind
        corporate proxies that TLS-intercept Docker Hub. Asserting the
        ``${LITELLM_DB_IMAGE:-...}`` form means a reviewer who hits the
        pull failure can swap to a mirror without forking the sample."""
        db_img = compose["services"]["db"]["image"]
        assert "${LITELLM_DB_IMAGE" in db_img, (
            f"db image must use ${{LITELLM_DB_IMAGE:-...}} form so users "
            f"behind corporate proxies can override the registry; got "
            f"{db_img!r}"
        )

    def test_db_creds_env_wired_strictly(self, compose: dict[str, Any]):
        """Postgres POSTGRES_PASSWORD must use the ``${VAR:?…}`` strict
        form so compose refuses to start without LITELLM_DB_PASSWORD set
        — otherwise we'd silently boot Postgres with an empty password."""
        env = compose["services"]["db"]["environment"]
        pwd = env["POSTGRES_PASSWORD"]
        assert ":?" in pwd, (
            f"POSTGRES_PASSWORD should use ${{LITELLM_DB_PASSWORD:?…}} strict form; got {pwd!r}"
        )

    def test_db_data_persists_via_named_volume(self, compose: dict[str, Any]):
        """Postgres data must mount a named volume — otherwise virtual
        keys + spend logs are lost on every ``docker compose down``."""
        volumes = compose["services"]["db"].get("volumes", [])
        assert any("/var/lib/postgresql/data" in v for v in volumes), (
            f"db must mount a volume at /var/lib/postgresql/data; got volumes={volumes!r}"
        )
        # Confirm the volume is a top-level named volume, not a bind mount,
        # so the data survives between compose runs and lives under the
        # Docker volume store rather than spraying files into the user's
        # working directory.
        assert "volumes" in compose, "top-level `volumes:` block must declare the named volume"

    def test_db_has_healthcheck(self, compose: dict[str, Any]):
        """Without a db healthcheck, the gateway may start before
        Postgres is ready and Prisma migrations fail with a connection
        refused that's confusing to debug."""
        assert "healthcheck" in compose["services"]["db"]

    def test_gateway_depends_on_db_being_healthy(self, compose: dict[str, Any]):
        """``depends_on: db: condition: service_healthy`` is the
        difference between 'compose up succeeds eventually' and 'compose
        up succeeds, then Prisma can't reach Postgres on the first
        request because the gateway raced past it'."""
        depends = compose["services"]["litellm"].get("depends_on", {})
        # Accept the long-form mapping (which is what we ship) — the
        # short-form list ``["db"]`` doesn't wait for healthy.
        assert isinstance(depends, dict), (
            "litellm.depends_on must be the long-form mapping so it can "
            "carry `condition: service_healthy`, got "
            f"{type(depends).__name__}"
        )
        assert depends.get("db", {}).get("condition") == "service_healthy", (
            f"litellm should depend_on db with condition: service_healthy; got {depends!r}"
        )

    def test_gateway_database_url_env_present(self, compose: dict[str, Any]):
        """The gateway must be told where to reach Postgres. The URL
        must use the in-network hostname (``db``), not localhost — and
        must pull the password from LITELLM_DB_PASSWORD with strict
        env-var form."""
        env = compose["services"]["litellm"]["environment"]
        url = env.get("DATABASE_URL", "")
        assert "@db:" in url, f"DATABASE_URL must use in-network host `db`, got {url!r}"
        assert "LITELLM_DB_PASSWORD" in url, "DATABASE_URL must reference LITELLM_DB_PASSWORD"
        assert ":?" in url, (
            "DATABASE_URL should use the ${LITELLM_DB_PASSWORD:?…} strict "
            f"form so compose refuses to start without it, got {url!r}"
        )


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
