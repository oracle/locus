# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""The LiteLLM gateway sample under ``examples/litellm-gateway/`` stays coherent.

``config.yaml`` and ``config.platform.yaml`` must describe the same gateway
apart from the response cache and the observability callback, which only the
platform file turns on; ``docker-compose.yml`` must select the platform file
on ``REDIS_URL`` and keep the redis sidecar behind the ``cache`` profile; and
``helm-values.yaml`` must use the keys the LiteLLM chart reads. These tests
read the shipped files and need no gateway.
"""

from __future__ import annotations

import pathlib

import pytest
import yaml


ROOT = pathlib.Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "litellm-gateway"

PLATFORM_ONLY = {"cache", "cache_params", "callbacks"}


def _load(name: str) -> dict:
    return yaml.safe_load((EXAMPLE / name).read_text())


@pytest.fixture(scope="module")
def base() -> dict:
    return _load("config.yaml")


@pytest.fixture(scope="module")
def platform() -> dict:
    return _load("config.platform.yaml")


@pytest.fixture(scope="module")
def compose() -> dict:
    return _load("docker-compose.yml")


def test_platform_config_is_the_base_config_plus_cache_and_callbacks(base, platform):
    assert set(platform) == set(base)
    for key in ("model_list", "router_settings", "guardrails", "general_settings"):
        assert platform[key] == base[key], f"{key} drifted between the two configs"

    settings = platform["litellm_settings"]
    assert settings["callbacks"] == ["otel"]
    assert settings["cache"] is True
    assert settings["cache_params"]["type"] == "redis"
    assert settings["cache_params"]["url"] == "os.environ/REDIS_URL"
    assert settings["cache_params"]["ttl"] > 0
    shared = {k: v for k, v in settings.items() if k not in PLATFORM_ONLY}
    assert shared == base["litellm_settings"]


def test_base_config_needs_no_backing_service(base):
    settings = base["litellm_settings"]
    assert PLATFORM_ONLY.isdisjoint(settings)
    assert "success_callback" not in settings
    assert "failure_callback" not in settings


def test_pii_guardrail_runs_in_process_and_masks_before_the_upstream_call(base):
    (guardrail,) = base["guardrails"]
    assert guardrail["guardrail_name"] == "pii-mask"
    params = guardrail["litellm_params"]
    assert params["guardrail"] == "litellm_content_filter"
    assert params["mode"] == "pre_call"
    assert params["default_on"] is True
    patterns = {(p["pattern_type"], p["pattern_name"], p["action"]) for p in params["patterns"]}
    assert patterns == {("prebuilt", "us_ssn", "MASK"), ("prebuilt", "credit_card", "MASK")}


def test_compose_selects_the_platform_config_on_redis_url(compose):
    litellm = compose["services"]["litellm"]
    assert "/app/config${REDIS_URL:+.platform}.yaml" in litellm["command"]
    mounted = {volume.split(":")[1] for volume in litellm["volumes"]}
    assert {"/app/config.yaml", "/app/config.platform.yaml"} <= mounted

    env = litellm["environment"]
    # Inherited from the host when set, absent from the container otherwise.
    assert env["REDIS_URL"] is None
    assert env["OTEL_EXPORTER_OTLP_ENDPOINT"] is None
    # OTLP/HTTP with an endpoint; empty without one, which is LiteLLM's console exporter.
    assert env["OTEL_EXPORTER_OTLP_PROTOCOL"] == "${OTEL_EXPORTER_OTLP_ENDPOINT:+http/protobuf}"
    for name in (
        "OTEL_EXPORTER_OTLP_HEADERS",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_HOST",
    ):
        assert name in env
        assert env[name] is None


def test_compose_redis_sidecar_is_behind_the_cache_profile(compose):
    redis = compose["services"]["redis"]
    assert redis["profiles"] == ["cache"]
    assert "redis" not in compose["services"]["litellm"].get("depends_on", {})
    assert "volumes" not in redis, "the cache is ephemeral on purpose"


def test_helm_values_use_the_keys_the_chart_reads():
    values = _load("helm-values.yaml")
    assert "envFrom" not in values, "the LiteLLM chart ignores envFrom; use environmentSecrets"
    assert values["environmentSecrets"] == ["provider-credentials", "litellm-master"]
    assert values["masterkeySecretName"] == "litellm-master"
    assert values["masterkeySecretKey"] == "LITELLM_MASTER_KEY"
    assert values["db"]["useExisting"] is True
    assert values["db"]["deployStandalone"] is False
    assert "url" not in values["db"], "the chart's default URL names the env vars it sets"
    assert values["service"]["type"] == "ClusterIP"
    assert values["securityContext"]["readOnlyRootFilesystem"] is True
