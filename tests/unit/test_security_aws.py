# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the spec-driven, read-only AWS tools."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from tulip.security.aws import (
    _session,
    describe_aws,
    describe_aws_tool,
    is_readonly_operation,
    use_aws,
    use_aws_tool,
)


class TestReadonlyGate:
    @pytest.mark.parametrize(
        "op",
        [
            "DescribeInstances",
            "ListBuckets",
            "GetAccountSummary",
            "SearchResources",
            "BatchGetItem",
            "ScanTable",
            "SimulatePrincipalPolicy",
            "LookupEvents",
        ],
    )
    def test_read_ops_admitted(self, op: str) -> None:
        assert is_readonly_operation(op)

    @pytest.mark.parametrize(
        "op",
        [
            "CreateUser",
            "PutObject",
            "DeleteBucket",
            "RunInstances",
            "AttachUserPolicy",
            "ModifyInstanceAttribute",
            "UpdateStack",
            "TerminateInstances",
        ],
    )
    def test_write_ops_rejected(self, op: str) -> None:
        assert not is_readonly_operation(op)


def test_use_aws_refuses_write_before_touching_aws() -> None:
    with patch("tulip.security.aws._session") as sess:
        with pytest.raises(PermissionError, match="not a read-only"):
            use_aws("iam", "CreateUser", {"UserName": "nope"})
        sess.assert_not_called()  # the gate fires before any boto3 call


def test_describe_aws_lists_services() -> None:
    with patch("tulip.security.aws._session") as sess:
        sess.return_value.get_available_services.return_value = ["s3", "ec2", "iam"]
        assert describe_aws()["services"] == ["s3", "ec2", "iam"]


def test_describe_aws_filters_operations_to_readonly() -> None:
    model = MagicMock()
    model.operation_names = ["DescribeInstances", "RunInstances", "ListBuckets", "CreateUser"]
    with patch("tulip.security.aws._session") as sess:
        sess.return_value.client.return_value.meta.service_model = model
        out = describe_aws("ec2")
        assert set(out["readonly_operations"]) == {"DescribeInstances", "ListBuckets"}
        assert out["count"] == 2


def test_describe_aws_operation_parameters() -> None:
    member = MagicMock()
    member.type_name = "list"
    shape = MagicMock()
    shape.members = {"InstanceIds": member}
    shape.required_members = []
    op_model = MagicMock()
    op_model.input_shape = shape
    model = MagicMock()
    model.operation_model.return_value = op_model
    with patch("tulip.security.aws._session") as sess:
        sess.return_value.client.return_value.meta.service_model = model
        out = describe_aws("ec2", "DescribeInstances")
        assert out["operation"] == "DescribeInstances"
        assert out["readonly"] is True
        assert out["parameters"]["InstanceIds"]["type"] == "list"


def test_use_aws_executes_and_strips_response_metadata() -> None:
    client = MagicMock()
    client.describe_instances.return_value = {
        "Reservations": [],
        "ResponseMetadata": {"RequestId": "x"},
    }
    with patch("tulip.security.aws._session") as sess:
        sess.return_value.client.return_value = client
        out = use_aws("ec2", "DescribeInstances")
        assert "ResponseMetadata" not in out
        assert out["Reservations"] == []


async def test_use_aws_tool_refused_write_returns_error_json() -> None:
    # No boto3 needed — the read-only gate fires first.
    payload = json.loads(await use_aws_tool("iam", "CreateUser", {"UserName": "nope"}))
    assert "not a read-only" in payload["error"]


def test_session_uses_profile_and_region_from_env() -> None:
    with patch.dict(
        "os.environ", {"TULIP_AWS_PROFILE": "myprof", "TULIP_AWS_REGION": "eu-west-1"}
    ), patch("boto3.Session") as session:
        _session()
        session.assert_called_once_with(profile_name="myprof", region_name="eu-west-1")


async def test_describe_aws_tool_returns_services_json() -> None:
    with patch("tulip.security.aws._session") as sess:
        sess.return_value.get_available_services.return_value = ["s3", "ec2"]
        payload = json.loads(await describe_aws_tool())
        assert payload["services"] == ["s3", "ec2"]


async def test_use_aws_tool_executes_readonly_and_returns_json() -> None:
    client = MagicMock()
    client.list_buckets.return_value = {"Buckets": [], "ResponseMetadata": {"x": 1}}
    with patch("tulip.security.aws._session") as sess:
        sess.return_value.client.return_value = client
        payload = json.loads(await use_aws_tool("s3", "ListBuckets"))
        assert payload["Buckets"] == []
        assert "ResponseMetadata" not in payload
