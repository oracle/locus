# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Generic, spec-driven AWS tools for the cloud-posture agent.

Instead of hand-writing one tool per AWS service, the agent gets two
generic tools driven by **botocore's service models** — AWS's canonical
API specification, shipped with boto3:

- :func:`describe_aws` — introspect the spec. With no arguments it lists
  every AWS service (the *shape* of AWS); given a service it lists that
  service's read-only operations; given a service + operation it returns
  the operation's parameters. This is how the agent discovers what it can
  call.
- :func:`use_aws` — execute *any* read-only AWS operation
  (``service`` + ``operation`` + ``parameters``) and return the raw API
  response, which becomes the **evidence** behind a grounded finding.

Both are **read-only by construction**: an operation is admitted only if
its name starts with a read verb (Describe/List/Get/…), and the agent is
expected to run under a read-only IAM identity (``SecurityAudit`` /
``ViewOnlyAccess``) so the IAM policy is the hard backstop. ``use_aws``
will refuse a non-read operation before any call is made.

Credentials follow boto3's standard chain; the default profile is
``tulip-security-audit`` (override via ``TULIP_AWS_PROFILE``), and the
region via ``TULIP_AWS_REGION`` (default ``us-east-1``).
"""

from __future__ import annotations

import json
import os
from typing import Any, cast

from tulip.tools.decorator import tool


# Read-only operation verbs. botocore operation names are PascalCase
# (``DescribeInstances``, ``ListBuckets``); an operation is admitted only
# if it begins with one of these. The read-only IAM policy is the hard
# enforcement — this gate is defense-in-depth and makes intent explicit.
READONLY_PREFIXES: tuple[str, ...] = (
    "Describe",
    "List",
    "Get",
    "Lookup",
    "Search",
    "BatchGet",
    "Select",
    "Head",
    "Estimate",
    "Simulate",  # SimulatePrincipalPolicy / SimulateCustomPolicy are read-only
    "Preview",
    "Query",  # DynamoDB read
    "Scan",  # DynamoDB read
    "Retrieve",
)


def is_readonly_operation(operation: str) -> bool:
    """Whether ``operation`` (a botocore PascalCase op name) only reads."""
    return operation.startswith(READONLY_PREFIXES)


def _profile() -> str:
    return os.environ.get("TULIP_AWS_PROFILE", "tulip-security-audit")


def _region(region: str | None) -> str:
    return region or os.environ.get("TULIP_AWS_REGION", "us-east-1")


def _session(region: str | None = None) -> Any:
    """A boto3 Session on the read-only profile (lazy boto3 import)."""
    try:
        import boto3
    except ImportError as exc:  # pragma: no cover - optional dep
        msg = "AWS tools require boto3 — install with: pip install 'tulip-agents[aws]'"
        raise ImportError(msg) from exc
    return boto3.Session(profile_name=_profile(), region_name=_region(region))


def aws_services(region: str | None = None) -> list[str]:
    """Every AWS service available in the spec — the *shape* of AWS."""
    return list(_session(region).get_available_services())


def describe_aws(
    service: str | None = None,
    operation: str | None = None,
    region: str | None = None,
) -> dict[str, Any]:
    """Introspect the AWS API spec (botocore service models).

    - ``service is None`` → ``{"services": [...]}`` (all services).
    - ``service`` only → that service's read-only operations.
    - ``service`` + ``operation`` → the operation's parameter shape.
    """
    if service is None:
        return {"services": aws_services(region)}

    client = _session(region).client(service)
    model = client.meta.service_model
    if operation is None:
        ops = sorted(o for o in model.operation_names if is_readonly_operation(o))
        return {"service": service, "readonly_operations": ops, "count": len(ops)}

    op_model = model.operation_model(operation)
    inp = op_model.input_shape
    params: dict[str, Any] = {}
    if inp is not None:
        required = set(getattr(inp, "required_members", []) or [])
        for name, shape in inp.members.items():
            params[name] = {"type": shape.type_name, "required": name in required}
    return {
        "service": service,
        "operation": operation,
        "readonly": is_readonly_operation(operation),
        "parameters": params,
    }


def use_aws(
    service: str,
    operation: str,
    parameters: dict[str, Any] | None = None,
    region: str | None = None,
) -> dict[str, Any]:
    """Execute a **read-only** AWS operation and return the raw response.

    Refuses any operation that is not a read verb (see
    :data:`READONLY_PREFIXES`) before making a call. The returned response
    is the evidence a grounded finding cites.
    """
    if not is_readonly_operation(operation):
        msg = (
            f"refused: {service}:{operation} is not a read-only operation. "
            "use_aws only runs describe/list/get-style calls."
        )
        raise PermissionError(msg)

    from botocore import xform_name  # noqa: PLC0415 — lazy, boto3 optional

    client = _session(region).client(service)
    method = getattr(client, xform_name(operation))
    response = cast("dict[str, Any]", method(**(parameters or {})))
    response.pop("ResponseMetadata", None)
    return response


# --- Agent-facing @tool wrappers (string output) ---------------------------


@tool(
    name="describe_aws",
    description=(
        "Introspect the AWS API: no args lists all services; a service lists "
        "its read-only operations; service+operation lists the parameters."
    ),
)
async def describe_aws_tool(service: str = "", operation: str = "") -> str:
    """Discover the shape of AWS from the API spec."""
    result = describe_aws(service or None, operation or None)
    return json.dumps(result, default=str)


@tool(
    name="use_aws",
    description=(
        "Call a read-only AWS API (service + operation + parameters) and return "
        "the raw response as evidence. Only describe/list/get-style operations."
    ),
)
async def use_aws_tool(
    service: str, operation: str, parameters: dict[str, Any] | None = None
) -> str:
    """Execute one read-only AWS operation."""
    try:
        result = use_aws(service, operation, parameters)
    except PermissionError as exc:
        return json.dumps({"error": str(exc)})
    return json.dumps(result, default=str)
