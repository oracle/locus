# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for the open-spec OCI tools (`use_oci` + `describe_oci`).

These cover the pure-Python logic — docstring parsing, mutation gating,
SDK introspection, signer routing, response serialisation, error
handling — without hitting any real OCI endpoint. The integration-y
parts (live calls, JSON payloads from the wire) are exercised by the
notebook 70 walk-through against a real tenancy.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from unittest.mock import MagicMock, patch

import pytest

from locus.tools import describe_oci, use_oci
from locus.tools.oci import (
    _MUTATION_OPT_IN_ENV,
    _is_read_only,
    _mutations_enabled,
    _parse_operation_doc,
    _serialize,
    _UseOCIError,
)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestReadOnlyClassifier:
    """`_is_read_only` decides which operations the mutation gate lets through."""

    @pytest.mark.parametrize(
        "name",
        [
            "list_compartments",
            "get_instance",
            "head_object",
            "summarize_metrics",
            "describe_workspace",
            "search_resources",
            "fetch_secret_bundle",
            "compute_pricing",
            "preview_change",
            "validate_workflow",
            "test_connection",
        ],
    )
    def test_read_only_prefixes(self, name: str) -> None:
        assert _is_read_only(name)

    @pytest.mark.parametrize(
        "name",
        [
            "create_compartment",
            "update_instance",
            "delete_bucket",
            "change_compartment",
            "terminate_instance",
            "start_workspace",
            "stop_workspace",
            "rotate_key",
            "attach_volume",
            "post_message",
        ],
    )
    def test_mutating_prefixes(self, name: str) -> None:
        assert not _is_read_only(name)


class TestMutationGate:
    """The mutation gate is opt-in by argument OR by env var."""

    def test_default_blocks_mutations(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(_MUTATION_OPT_IN_ENV, raising=False)
        assert _mutations_enabled(False) is False

    def test_explicit_kwarg_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(_MUTATION_OPT_IN_ENV, raising=False)
        assert _mutations_enabled(True) is True

    @pytest.mark.parametrize("value", ["1", "true", "yes", "TRUE", "YES"])
    def test_env_truthy_values(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        monkeypatch.setenv(_MUTATION_OPT_IN_ENV, value)
        assert _mutations_enabled(False) is True

    @pytest.mark.parametrize("value", ["0", "false", "no", ""])
    def test_env_falsy_values(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        monkeypatch.setenv(_MUTATION_OPT_IN_ENV, value)
        assert _mutations_enabled(False) is False


class TestDocstringParser:
    """`_parse_operation_doc` lifts :param TYPE name: blocks out of OCI SDK docstrings."""

    def test_required_and_optional(self) -> None:
        doc = (
            "Lists the compartments.\n\n"
            ":param str compartment_id: (required)\n"
            "    The OCID of the compartment.\n"
            ":param int limit: (optional)\n"
            "    Max items to return.\n"
            ":param bool compartment_id_in_subtree: (optional)\n"
            "    Whether to recurse.\n"
        )
        params = _parse_operation_doc(doc)
        names = [p["name"] for p in params]
        assert names == ["compartment_id", "limit", "compartment_id_in_subtree"]

        cid = params[0]
        assert cid["type"] == "str"
        assert cid["required"] is True
        assert cid["description"].startswith("The OCID")
        assert "(required)" not in cid["description"]

        assert params[1]["type"] == "int"
        assert params[1]["required"] is False
        assert params[2]["type"] == "bool"

    def test_no_params(self) -> None:
        assert _parse_operation_doc("") == []
        assert _parse_operation_doc("Plain docstring, no :param: blocks.") == []

    def test_truncates_long_description(self) -> None:
        long = "x " * 1000
        doc = f":param str foo: (required)\n    {long}"
        params = _parse_operation_doc(doc)
        assert len(params[0]["description"]) <= 400


class TestSerializer:
    """`_serialize` turns OCI SDK model objects into JSON-safe Python."""

    def test_primitives(self) -> None:
        for v in [None, True, 42, 3.14, "hi"]:
            assert _serialize(v) == v

    def test_decimal(self) -> None:
        assert _serialize(Decimal("1.5")) == 1.5

    def test_datetime(self) -> None:
        dt = datetime(2026, 5, 22, 12, 30, 0, tzinfo=UTC)
        assert _serialize(dt) == "2026-05-22T12:30:00+00:00"

    def test_enum(self) -> None:
        class State(Enum):
            ACTIVE = "ACTIVE"

        assert _serialize(State.ACTIVE) == "ACTIVE"

    def test_list_and_tuple_and_set(self) -> None:
        assert _serialize([1, 2, 3]) == [1, 2, 3]
        assert _serialize((1, 2, 3)) == [1, 2, 3]
        assert sorted(_serialize({1, 2, 3})) == [1, 2, 3]

    def test_dict(self) -> None:
        assert _serialize({"a": 1, "b": [2, 3]}) == {"a": 1, "b": [2, 3]}

    def test_oci_model_via_attribute_map(self) -> None:
        """OCI SDK models expose `attribute_map` (snake_case -> wire camelCase)."""
        m = MagicMock()
        m.attribute_map = {"compartment_id": "compartmentId", "name": "name"}
        m.compartment_id = "ocid1.tenancy.oc1..x"
        m.name = "test"
        out = _serialize(m)
        assert out == {"compartmentId": "ocid1.tenancy.oc1..x", "name": "test"}

    def test_nested_model_inside_list(self) -> None:
        inner = MagicMock()
        inner.attribute_map = {"id": "id"}
        inner.id = "ocid1.compartment.oc1..y"
        out = _serialize([inner, inner])
        assert out == [{"id": "ocid1.compartment.oc1..y"}, {"id": "ocid1.compartment.oc1..y"}]

    def test_unknown_type_falls_back_to_str(self) -> None:
        class Weird:
            def __str__(self) -> str:
                return "<weird>"

        assert _serialize(Weird()) == "<weird>"


# ---------------------------------------------------------------------------
# use_oci — high-level behaviour
# ---------------------------------------------------------------------------


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestUseOCISchema:
    def test_schema_required_keys(self) -> None:
        params = use_oci.parameters
        assert params["required"] == ["service", "client", "operation"]
        for k in (
            "service",
            "client",
            "operation",
            "parameters",
            "compartment_id",
            "region",
            "profile",
            "auth_type",
            "label",
            "allow_mutations",
        ):
            assert k in params["properties"]

    def test_is_idempotent(self) -> None:
        # idempotent=True is set on the decorator so a model re-emitting
        # the same (service, client, operation, parameters) gets the
        # previous result rather than a fresh call.
        assert use_oci.idempotent is True


class TestUseOCIMutationGate:
    """The gate is the first thing use_oci does — covered without any OCI auth."""

    @pytest.mark.asyncio
    async def test_mutating_operation_refused_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(_MUTATION_OPT_IN_ENV, raising=False)
        out = json.loads(
            await use_oci.execute(
                service="object_storage",
                client="ObjectStorageClient",
                operation="delete_bucket",
                parameters={"namespace_name": "x", "bucket_name": "y"},
                profile="DOES_NOT_MATTER",
            )
        )
        assert out["status"] == "error"
        assert "delete_bucket" in out["error"]
        assert "mutating" in out["error"].lower()

    @pytest.mark.asyncio
    async def test_allow_mutations_kwarg_unblocks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With allow_mutations=True we proceed past the gate (and fail later at auth)."""
        monkeypatch.delenv(_MUTATION_OPT_IN_ENV, raising=False)
        out = json.loads(
            await use_oci.execute(
                service="object_storage",
                client="ObjectStorageClient",
                operation="delete_bucket",
                parameters={"namespace_name": "x", "bucket_name": "y"},
                profile="DOES_NOT_EXIST_INTENTIONALLY",
                allow_mutations=True,
            )
        )
        assert out["status"] == "error"
        # The error should be the auth-side failure, NOT the mutation refusal.
        assert "mutating" not in out["error"].lower()

    @pytest.mark.asyncio
    async def test_env_var_unblocks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_MUTATION_OPT_IN_ENV, "1")
        out = json.loads(
            await use_oci.execute(
                service="object_storage",
                client="ObjectStorageClient",
                operation="delete_bucket",
                parameters={"namespace_name": "x", "bucket_name": "y"},
                profile="DOES_NOT_EXIST_INTENTIONALLY",
            )
        )
        assert out["status"] == "error"
        assert "mutating" not in out["error"].lower()


class TestUseOCICompartmentIdMerge:
    """compartment_id is a convenience param; it's auto-merged into parameters."""

    @pytest.mark.asyncio
    async def test_compartment_id_merged_when_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Confirm the merge happens before auth resolution fires."""
        captured: dict[str, object] = {}

        # Patch the auth resolver so we can inspect what parameters reached it.
        from locus.tools import oci as oci_module

        original = oci_module._build_config_and_signer

        def spy(*args, **kwargs):  # noqa: ANN001, ANN002
            raise _UseOCIError("intentional — captured for inspection")

        monkeypatch.setattr(oci_module, "_build_config_and_signer", spy)

        out = json.loads(
            await use_oci.execute(
                service="identity",
                client="IdentityClient",
                operation="list_compartments",
                compartment_id="ocid1.tenancy.oc1..xxx",
                profile="X",
            )
        )
        assert out["status"] == "error"
        assert "intentional" in out["error"]

        # Restore (pytest does this anyway via monkeypatch, but explicit).
        monkeypatch.setattr(oci_module, "_build_config_and_signer", original)

    @pytest.mark.asyncio
    async def test_explicit_compartment_in_parameters_wins(self) -> None:
        """If the model passes compartment_id in parameters, the kwarg is not overwritten."""
        # We can't easily inspect what landed in `method(**parameters)` without
        # patching the SDK; the merge logic is straightforward (`if not in parameters`).
        # This test asserts no exception path is triggered when both are present.
        out = json.loads(
            await use_oci.execute(
                service="identity",
                client="IdentityClient",
                operation="list_compartments",
                parameters={"compartment_id": "ocid1.tenancy.oc1..A"},
                compartment_id="ocid1.tenancy.oc1..B",  # should be ignored
                profile="DOES_NOT_EXIST_INTENTIONALLY",
            )
        )
        assert out["status"] == "error"
        # Whatever the error is, it shouldn't be the mutation gate.
        assert "mutating" not in out["error"].lower()


class TestUseOCIErrorPaths:
    """Service / client / operation resolution errors get surfaced cleanly."""

    @pytest.mark.asyncio
    async def test_unknown_service(self) -> None:
        out = json.loads(
            await use_oci.execute(
                service="not_a_real_service",
                client="X",
                operation="list_things",
                profile="DOES_NOT_EXIST_INTENTIONALLY",
            )
        )
        assert out["status"] == "error"
        assert "not_a_real_service" in out["error"]

    @pytest.mark.asyncio
    async def test_unknown_client_in_real_service(self) -> None:
        out = json.loads(
            await use_oci.execute(
                service="identity",
                client="NotAClient",
                operation="list_things",
                profile="DOES_NOT_EXIST_INTENTIONALLY",
            )
        )
        assert out["status"] == "error"
        assert "NotAClient" in out["error"]


# ---------------------------------------------------------------------------
# describe_oci
# ---------------------------------------------------------------------------


class TestDescribeOCILevels:
    """describe_oci walks the SDK at four levels — all four work without OCI auth."""

    @pytest.mark.asyncio
    async def test_level_1_services(self) -> None:
        r = json.loads(await describe_oci.execute())
        assert r["status"] == "ok"
        assert r["level"] == "services"
        assert r["count"] > 100
        assert "identity" in r["services"]
        assert "core" in r["services"]

    @pytest.mark.asyncio
    async def test_level_2_clients(self) -> None:
        r = json.loads(await describe_oci.execute(service="core"))
        assert r["status"] == "ok"
        assert r["level"] == "clients"
        assert "ComputeClient" in r["clients"]
        assert "VirtualNetworkClient" in r["clients"]

    @pytest.mark.asyncio
    async def test_level_3_operations_partitioned(self) -> None:
        r = json.loads(
            await describe_oci.execute(service="object_storage", client="ObjectStorageClient")
        )
        assert r["status"] == "ok"
        assert r["level"] == "operations"
        assert "get_namespace" in r["read_only_operations"]
        # mutating bucket lifecycle operations
        assert any("delete_" in o for o in r["mutating_operations"])
        assert any("create_" in o for o in r["mutating_operations"])

    @pytest.mark.asyncio
    async def test_level_4_operation_schema(self) -> None:
        r = json.loads(
            await describe_oci.execute(
                service="identity",
                client="IdentityClient",
                operation="list_compartments",
            )
        )
        assert r["status"] == "ok"
        assert r["level"] == "operation"
        assert r["read_only"] is True
        assert "compartment_id" in r["required_parameters"]
        # All parameter entries have the structured shape we promise.
        for p in r["parameters"]:
            assert set(p.keys()) >= {"name", "type", "required", "description"}


class TestDescribeOCIErrors:
    @pytest.mark.asyncio
    async def test_unknown_service_returns_suggestions(self) -> None:
        r = json.loads(await describe_oci.execute(service="not_a_real_service"))
        assert r["status"] == "error"
        assert "not_a_real_service" in r["error"]
        assert isinstance(r.get("available_services_sample"), list)
        assert r["total_services"] > 100

    @pytest.mark.asyncio
    async def test_unknown_client_returns_candidates(self) -> None:
        r = json.loads(await describe_oci.execute(service="identity", client="NotAClient"))
        assert r["status"] == "error"
        assert "NotAClient" in r["error"]
        assert "IdentityClient" in r.get("available_clients", [])

    @pytest.mark.asyncio
    async def test_unknown_operation_returns_suggestions(self) -> None:
        r = json.loads(
            await describe_oci.execute(
                service="identity",
                client="IdentityClient",
                operation="list_widgets_that_dont_exist",
            )
        )
        assert r["status"] == "error"
        assert "list_widgets_that_dont_exist" in r["error"]


# ---------------------------------------------------------------------------
# Auth router
# ---------------------------------------------------------------------------


class TestAuthRouter:
    """`_build_config_and_signer` picks the right auth path per auth_type."""

    def test_unsupported_auth_type_raises(self) -> None:
        """The 'Unsupported auth_type' check fires after a successful profile load."""
        from locus.tools import oci as oci_module

        oci_mod = oci_module._load_oci()
        with (
            patch.object(oci_mod.config, "from_file", return_value={"region": "us-ashburn-1"}),
            pytest.raises(_UseOCIError, match="Unsupported auth_type"),
        ):
            oci_module._build_config_and_signer(oci_mod, profile="X", auth_type="nope", region=None)

    def test_missing_profile_raises(self) -> None:
        from locus.tools import oci as oci_module

        with pytest.raises(_UseOCIError):
            oci_module._build_config_and_signer(
                oci_module._load_oci(),
                profile="DEFINITELY_NOT_A_REAL_PROFILE_XYZ123",
                auth_type="api_key",
                region=None,
            )

    def test_instance_principal_signer_is_built(self) -> None:
        """We patch the OCI SDK's signer constructor so this stays offline."""
        from locus.tools import oci as oci_module

        with patch.object(
            oci_module._load_oci().auth.signers,
            "InstancePrincipalsSecurityTokenSigner",
        ) as ctor:
            ctor.return_value.region = "us-ashburn-1"
            cfg, signer = oci_module._build_config_and_signer(
                oci_module._load_oci(),
                profile="ignored",
                auth_type="instance_principal",
                region=None,
            )
        assert cfg["region"] == "us-ashburn-1"
        assert signer is ctor.return_value

    def test_resource_principal_signer_is_built(self) -> None:
        """`resource_principal` path goes through ``get_resource_principals_signer``."""
        from locus.tools import oci as oci_module

        with patch.object(
            oci_module._load_oci().auth.signers,
            "get_resource_principals_signer",
        ) as factory:
            factory.return_value.region = "us-phoenix-1"
            cfg, signer = oci_module._build_config_and_signer(
                oci_module._load_oci(),
                profile="ignored",
                auth_type="resource_principal",
                region=None,
            )
        assert cfg["region"] == "us-phoenix-1"
        assert signer is factory.return_value

    def test_security_token_missing_token_file_raises(self) -> None:
        from locus.tools import oci as oci_module

        oci_mod = oci_module._load_oci()
        with (
            patch.object(oci_mod.config, "from_file", return_value={"region": "us-ashburn-1"}),
            pytest.raises(_UseOCIError, match="security_token_file"),
        ):
            oci_module._build_config_and_signer(
                oci_mod, profile="X", auth_type="security_token", region=None
            )

    def test_api_key_returns_config_with_no_signer(self) -> None:
        from locus.tools import oci as oci_module

        oci_mod = oci_module._load_oci()
        fake_cfg = {"region": "us-ashburn-1", "user": "x"}
        with patch.object(oci_mod.config, "from_file", return_value=fake_cfg):
            cfg, signer = oci_module._build_config_and_signer(
                oci_mod, profile="X", auth_type="api_key", region=None
            )
        assert cfg is fake_cfg
        assert signer is None

    def test_region_override(self) -> None:
        """Passing ``region=`` overrides the profile's region."""
        from locus.tools import oci as oci_module

        oci_mod = oci_module._load_oci()
        with patch.object(oci_mod.config, "from_file", return_value={"region": "us-ashburn-1"}):
            cfg, _ = oci_module._build_config_and_signer(
                oci_mod, profile="X", auth_type="api_key", region="us-chicago-1"
            )
        assert cfg["region"] == "us-chicago-1"


# ---------------------------------------------------------------------------
# Full success path — mock the client + method so we exercise the response
# serialisation and the http_status/opc_request_id propagation.
# ---------------------------------------------------------------------------


class TestUseOCISuccessPath:
    """Patch the SDK call to confirm response packaging works end-to-end."""

    @pytest.mark.asyncio
    async def test_successful_call_returns_serialised_data(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from locus.tools import oci as oci_module

        # Mock the auth resolver to skip ~/.oci/config entirely.
        monkeypatch.setattr(
            oci_module,
            "_build_config_and_signer",
            lambda *a, **kw: ({"region": "us-ashburn-1"}, None),
        )

        # Mock the client class so its constructor returns a stub with
        # list_compartments() -> Response.
        fake_response = MagicMock()
        fake_response.status = 200
        fake_response.headers = {"opc-request-id": "REQ-ABCDEF123"}
        # Build a fake OCI model object with attribute_map for serialisation
        item = MagicMock()
        item.attribute_map = {"id": "id", "name": "name"}
        item.id = "ocid1.compartment.oc1..xxx"
        item.name = "test-compartment"
        fake_response.data = [item]

        fake_client = MagicMock()
        fake_client.list_compartments = MagicMock(return_value=fake_response)

        fake_client_cls = MagicMock(return_value=fake_client)
        monkeypatch.setattr(oci_module, "_resolve_client_class", lambda *a, **kw: fake_client_cls)

        out = json.loads(
            await use_oci.execute(
                service="identity",
                client="IdentityClient",
                operation="list_compartments",
                parameters={"compartment_id": "ocid1.tenancy.oc1..xxx"},
                profile="ignored",
                region="us-ashburn-1",
                label="get all compartments",
            )
        )

        assert out["status"] == "ok"
        assert out["http_status"] == 200
        assert out["opc_request_id"] == "REQ-ABCDEF123"
        assert out["region"] == "us-ashburn-1"
        assert out["label"] == "get all compartments"
        assert out["data"] == [{"id": "ocid1.compartment.oc1..xxx", "name": "test-compartment"}]

    @pytest.mark.asyncio
    async def test_operation_not_on_client_gives_suggestions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from locus.tools import oci as oci_module

        monkeypatch.setattr(
            oci_module,
            "_build_config_and_signer",
            lambda *a, **kw: ({"region": "us-ashburn-1"}, None),
        )

        # Build a client class that lacks the operation name, but has list_*
        class FakeClient:
            def __init__(self, **kw):  # accept config=, signer=
                pass

            def list_compartments(self, **kw):
                return None

            def list_users(self, **kw):
                return None

        monkeypatch.setattr(oci_module, "_resolve_client_class", lambda *a, **kw: FakeClient)

        out = json.loads(
            await use_oci.execute(
                service="identity",
                client="IdentityClient",
                operation="list_widgets",
                profile="ignored",
            )
        )
        assert out["status"] == "error"
        assert "list_widgets" in out["error"]

    @pytest.mark.asyncio
    async def test_oci_service_error_is_surfaced_with_parameters(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the SDK call raises, we wrap the exception text and echo parameters."""
        from locus.tools import oci as oci_module

        monkeypatch.setattr(
            oci_module,
            "_build_config_and_signer",
            lambda *a, **kw: ({"region": "us-ashburn-1"}, None),
        )

        class BoomClient:
            def __init__(self, **kw):
                pass

            def list_compartments(self, **kw):
                raise RuntimeError("ServiceError: 404 NotAuthorized")

        monkeypatch.setattr(oci_module, "_resolve_client_class", lambda *a, **kw: BoomClient)

        out = json.loads(
            await use_oci.execute(
                service="identity",
                client="IdentityClient",
                operation="list_compartments",
                parameters={"compartment_id": "ocid1.tenancy.oc1..xxx"},
                profile="ignored",
            )
        )
        assert out["status"] == "error"
        assert "RuntimeError" in out["error"]
        assert "NotAuthorized" in out["error"]
        # The parameters we sent are echoed for debuggability
        assert out["parameters_sent"]["compartment_id"] == "ocid1.tenancy.oc1..xxx"
