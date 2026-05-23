# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""``use_oci`` — open-spec dispatch to any OCI Python SDK operation.

Instead of writing a separate ``@tool`` for every OCI resource (one
for compartments, one for ADBs, one for buckets, …), the agent loop
calls a single tool and the LLM picks the right
``oci.<service>.<Client>.<operation>(...)`` call from
natural-language intent.

Concretely the agent emits a structured tool call of the form::

    use_oci(
        service="identity",
        client="IdentityClient",
        operation="list_compartments",
        parameters={"compartment_id": "ocid1.tenancy.oc1..xxx"},
        profile="DEFAULT",
        region="us-ashburn-1",
        label="Find every compartment in the tenancy",
    )

and the tool resolves that into ``oci.identity.IdentityClient(config)
.list_compartments(**parameters)``, serializes the response, and
returns it as JSON. The OCI SDK has ~190 services, so a single
``use_oci`` covers essentially the whole control plane without any
per-service plumbing.

Safety: read-only by default. Mutating operations (``create_*``,
``update_*``, ``delete_*``, ``change_*``, ``terminate_*``, …) are
refused unless the caller opts in via either the ``allow_mutations``
parameter or the ``LOCUS_USE_OCI_ALLOW_MUTATIONS=1`` env var.
Enforced in code rather than via an interactive prompt — locus tools
must be non-interactive.
"""

from __future__ import annotations

import datetime as _dt
import importlib
import inspect
import json
import os
import re
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, cast

from locus.tools.decorator import tool


# OCI SDK methods follow strict naming. Anything *not* on this prefix
# allowlist is treated as potentially side-effecting and gated behind
# the mutation opt-in. Keep this list conservative — it's far better
# to over-block and force an explicit allow_mutations=True than to
# silently let the model run a delete.
_READ_ONLY_PREFIXES: tuple[str, ...] = (
    "list_",
    "get_",
    "head_",
    "summarize_",
    "preview_",
    "validate_",
    "test_",
    "search_",
    "describe_",
    "fetch_",
    "compute_",
)

_MUTATION_OPT_IN_ENV = "LOCUS_USE_OCI_ALLOW_MUTATIONS"


class _UseOCIError(RuntimeError):
    """Raised for any user-facing failure inside ``use_oci``.

    Surfaces as the tool's error string so the model gets a readable
    message instead of a traceback.
    """


def _is_read_only(operation: str) -> bool:
    return any(operation.startswith(p) for p in _READ_ONLY_PREFIXES)


def _mutations_enabled(explicit: bool) -> bool:
    if explicit:
        return True
    return os.environ.get(_MUTATION_OPT_IN_ENV, "").strip().lower() in {"1", "true", "yes"}


def _load_oci() -> Any:
    try:
        return importlib.import_module("oci")
    except ImportError as exc:
        raise _UseOCIError(
            "The 'oci' SDK is not installed. Install with: "
            "pip install 'locus-sdk[oci]'  (or pip install oci)."
        ) from exc


def _resolve_client_class(oci_mod: Any, service: str, client: str) -> type:
    try:
        service_mod = importlib.import_module(f"oci.{service}")
    except ImportError as exc:
        # Help the model self-correct: list a few service names so it
        # can re-emit the call with a valid one.
        import pkgutil  # noqa: PLC0415

        services = sorted(
            m.name for m in pkgutil.iter_modules(oci_mod.__path__) if not m.name.startswith("_")
        )
        raise _UseOCIError(
            f"Unknown OCI service '{service}'. The OCI Python SDK has "
            f"{len(services)} service modules under oci.* — examples: "
            f"{', '.join(services[:15])}, … "
            f"Pass the snake_case module name (e.g. 'identity', 'core', "
            f"'database', 'object_storage')."
        ) from exc

    client_cls = getattr(service_mod, client, None)
    if client_cls is None or not isinstance(client_cls, type):
        # List candidate clients in this service to nudge the model.
        candidates = sorted(
            name
            for name in dir(service_mod)
            if name.endswith("Client") and not name.startswith("_")
        )
        raise _UseOCIError(
            f"OCI service 'oci.{service}' has no client class '{client}'. "
            f"Available clients: {', '.join(candidates) or '(none)'}."
        )
    # mypy 3.12 needs the cast to narrow Any → type (no-any-return);
    # mypy 3.14 sees the cast as redundant after the isinstance check.
    # Suppressing both codes keeps both versions green.
    return cast("type", client_cls)  # type: ignore[unused-ignore, redundant-cast]


def _build_config_and_signer(
    oci_mod: Any,
    profile: str,
    auth_type: str,
    region: str | None,
) -> tuple[dict[str, Any], Any]:
    """Return ``(config, signer)`` suitable for an OCI client constructor.

    For ``api_key`` auth the SDK reads everything from the config dict;
    ``signer`` is ``None``. For ``security_token`` (delegation token
    from ``oci session authenticate``) we build a ``SecurityTokenSigner``
    over the private key referenced by ``key_file`` in the profile.
    """
    auth_type = (auth_type or "api_key").lower()

    if auth_type == "instance_principal":
        signer = oci_mod.auth.signers.InstancePrincipalsSecurityTokenSigner()
        config: dict[str, Any] = {"region": region or signer.region}
        return config, signer

    if auth_type == "resource_principal":
        signer = oci_mod.auth.signers.get_resource_principals_signer()
        config = {"region": region or signer.region}
        return config, signer

    # File-based profiles (api_key or security_token)
    try:
        config = oci_mod.config.from_file(profile_name=profile)
    except oci_mod.exceptions.ConfigFileNotFound as exc:
        raise _UseOCIError(
            f"OCI config file not found. Looked for profile '{profile}' in ~/.oci/config."
        ) from exc
    except oci_mod.exceptions.ProfileNotFound as exc:
        raise _UseOCIError(f"Profile '{profile}' is not present in ~/.oci/config.") from exc

    if region:
        config["region"] = region

    if auth_type == "security_token":
        token_file = config.get("security_token_file")
        if not token_file:
            raise _UseOCIError(
                f"Profile '{profile}' has no 'security_token_file' — "
                f"either run `oci session authenticate --profile-name "
                f"{profile}` first, or use auth_type='api_key'."
            )
        token = Path(token_file).expanduser().read_text(encoding="utf-8").strip()
        private_key = oci_mod.signer.load_private_key_from_file(config["key_file"])
        signer = oci_mod.auth.signers.SecurityTokenSigner(token, private_key)
        return config, signer

    if auth_type != "api_key":
        raise _UseOCIError(
            f"Unsupported auth_type '{auth_type}'. Use one of: "
            f"api_key, security_token, instance_principal, resource_principal."
        )
    return config, None


def _serialize(value: Any) -> Any:
    """Recursively turn an OCI SDK model object into JSON-safe data.

    OCI SDK response models expose an ``attribute_map`` (snake_case
    Python attr → camelCase wire name); we walk it to mirror the
    REST response shape.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (_dt.date, _dt.datetime)):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (list, tuple, set)):
        return [_serialize(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _serialize(v) for k, v in value.items()}
    attr_map = getattr(value, "attribute_map", None)
    if isinstance(attr_map, dict):
        return {
            wire_name: _serialize(getattr(value, py_attr, None))
            for py_attr, wire_name in attr_map.items()
        }
    return str(value)


@tool(idempotent=True)
def use_oci(
    service: str,
    client: str,
    operation: str,
    parameters: dict[str, Any] | None = None,
    compartment_id: str | None = None,
    region: str | None = None,
    profile: str = "DEFAULT",
    auth_type: str = "api_key",
    label: str = "",
    allow_mutations: bool = False,
) -> dict[str, Any]:
    """Call any operation on any OCI service via the OCI Python SDK.

    This is a single open-specification tool that dispatches to the
    full OCI control plane. Pick the service module, the client class
    inside it, the operation (SDK method name), and the parameters —
    the tool builds the right client with auth, invokes the method,
    and returns the serialized response.

    Read-only by default. Mutating operations (anything that isn't a
    list/get/head/summarize/preview/validate/test/search/describe/fetch/
    compute_) are refused unless ``allow_mutations=True`` is passed
    explicitly, or the ``LOCUS_USE_OCI_ALLOW_MUTATIONS=1`` env var
    is set. This is intentional — the model will sometimes pick a
    mutating verb when a read would do.

    Args:
        service: OCI service module under ``oci.*`` in snake_case.
            Examples: ``"identity"``, ``"core"``, ``"database"``,
            ``"object_storage"``, ``"vault"``, ``"bastion"``,
            ``"generative_ai"``, ``"functions"``, ``"resource_search"``.
            The OCI Python SDK has ~190 such modules.
        client: Client class inside that module (CamelCase, ends in
            ``Client``). Most services have one — e.g.
            ``"IdentityClient"``, ``"DatabaseClient"``,
            ``"ObjectStorageClient"``. ``core`` has three:
            ``ComputeClient``, ``BlockstorageClient``,
            ``VirtualNetworkClient``.
        operation: SDK method name in snake_case — mirrors the REST
            operation id. Examples: ``"list_compartments"``,
            ``"get_autonomous_database"``, ``"list_buckets"``,
            ``"list_instances"``.
        parameters: Keyword arguments to pass to the operation. For
            list/get calls this is typically ``{"compartment_id": "..."}``
            plus optional filters like ``{"lifecycle_state": "AVAILABLE"}``.
            For most ``get_*`` calls it's the resource OCID, e.g.
            ``{"autonomous_database_id": "ocid1.autonomousdatabase..."}``.
        compartment_id: Convenience — if the operation accepts
            ``compartment_id`` and you didn't put it in ``parameters``,
            it's merged in automatically. Pass the tenancy OCID for
            tenancy-scoped listings, or a specific compartment OCID.
        region: OCI region (e.g. ``"us-ashburn-1"``, ``"us-chicago-1"``,
            ``"ca-toronto-1"``). Falls back to the profile's home region.
        profile: ``~/.oci/config`` profile name. Defaults to ``DEFAULT``.
        auth_type: One of ``api_key`` (default, permanent key),
            ``security_token`` (session token from
            ``oci session authenticate``), ``instance_principal``
            (running on an OCI compute instance with a dynamic group),
            or ``resource_principal`` (running as an OCI Function or
            other resource principal).
        label: Free-text hint describing the user-facing intent of the
            call. Surfaces in logs and traces — not sent to OCI.
        allow_mutations: Explicit opt-in to run a write operation
            (anything that isn't ``list_/get_/head_/summarize_/...``).
            Without this AND without the env var, write operations
            return an error instead of executing.

    Returns:
        A dict with ``status`` (``"ok"`` or ``"error"``), ``operation``,
        ``service``, ``client``, ``region``, and either ``data`` (the
        serialized response) or ``error`` (a human-readable message).

    Examples:
        List every compartment in a tenancy::

            use_oci(
                service="identity",
                client="IdentityClient",
                operation="list_compartments",
                parameters={
                    "compartment_id": "ocid1.tenancy.oc1..xxx",
                    "compartment_id_in_subtree": True,
                },
                profile="DEFAULT",
            )

        Get one Autonomous Database::

            use_oci(
                service="database",
                client="DatabaseClient",
                operation="get_autonomous_database",
                parameters={"autonomous_database_id": "ocid1.autonomousdatabase..."},
                region="ca-toronto-1",
            )

        Free-form search across all resources in a compartment::

            use_oci(
                service="resource_search",
                client="ResourceSearchClient",
                operation="search_resources",
                parameters={
                    "search_details": {
                        "type": "Structured",
                        "query": "query all resources where compartmentId = '...'",
                    }
                },
            )
    """
    parameters = dict(parameters or {})
    if compartment_id and "compartment_id" not in parameters:
        parameters["compartment_id"] = compartment_id

    if not _is_read_only(operation) and not _mutations_enabled(allow_mutations):
        return {
            "status": "error",
            "service": service,
            "client": client,
            "operation": operation,
            "error": (
                f"Operation '{operation}' looks mutating (it doesn't start "
                f"with a read-only prefix like list_/get_/head_/summarize_/"
                f"describe_). Refusing by default. Pass allow_mutations=True "
                f"or set {_MUTATION_OPT_IN_ENV}=1 to override."
            ),
        }

    try:
        oci_mod = _load_oci()
        client_cls = _resolve_client_class(oci_mod, service, client)
        config, signer = _build_config_and_signer(oci_mod, profile, auth_type, region)

        client_kwargs: dict[str, Any] = {"config": config}
        if signer is not None:
            client_kwargs["signer"] = signer
        instance = client_cls(**client_kwargs)

        method = getattr(instance, operation, None)
        if method is None or not callable(method):
            available = sorted(
                name
                for name in dir(instance)
                if not name.startswith("_") and callable(getattr(instance, name, None))
            )
            close = [a for a in available if operation.split("_", 1)[0] in a][:15]
            raise _UseOCIError(
                f"Client '{client}' has no operation '{operation}'. "
                f"Did you mean one of: {', '.join(close) or available[:15]}?"
            )

        response = method(**parameters)
        data = _serialize(getattr(response, "data", response))
        status_code = getattr(response, "status", None)
        request_id = (
            response.headers.get("opc-request-id")
            if hasattr(response, "headers") and response.headers
            else None
        )

        result: dict[str, Any] = {
            "status": "ok",
            "service": service,
            "client": client,
            "operation": operation,
            "region": config.get("region"),
            "data": data,
        }
        if status_code is not None:
            result["http_status"] = status_code
        if request_id:
            result["opc_request_id"] = request_id
        if label:
            result["label"] = label
        return result

    except _UseOCIError as exc:
        return {
            "status": "error",
            "service": service,
            "client": client,
            "operation": operation,
            "error": str(exc),
        }
    except Exception as exc:  # noqa: BLE001 — bubble OCI ServiceError text up
        # OCI's ServiceError prints a structured one-liner with the
        # REST status, opc-request-id, and message — better to surface
        # that verbatim than to swallow it.
        return {
            "status": "error",
            "service": service,
            "client": client,
            "operation": operation,
            "error": f"{type(exc).__name__}: {exc}",
            "parameters_sent": _serialize(parameters),
        }


def _json_str(obj: Any) -> str:
    """Test helper — invoke ``use_oci`` and return its result as JSON."""
    return json.dumps(obj, indent=2, default=str)


# ---------------------------------------------------------------------------
# describe_oci — runtime introspection of the OCI Python SDK
# ---------------------------------------------------------------------------

# OCI docstrings use Sphinx-style param lines:
#   :param str compartment_id: (required)
#       The OCID of the compartment...
#   :param int limit: (optional)
#       Maximum items...
# Type is between ``:param `` and the next space; the ``(required)`` or
# ``(optional)`` marker (when present) is at the start of the description.
_PARAM_RE = re.compile(
    r":param\s+([^\s:]+)\s+([A-Za-z_][A-Za-z0-9_]*):\s*(.*?)(?=\n:|\Z)", re.DOTALL
)


def _parse_operation_doc(doc: str) -> list[dict[str, Any]]:
    """Pull a structured parameter list out of an OCI SDK method docstring."""
    params: list[dict[str, Any]] = []
    for match in _PARAM_RE.finditer(doc or ""):
        type_str, name, description = match.group(1), match.group(2), match.group(3)
        description = inspect.cleandoc(description).strip()
        required = description.lower().startswith("(required)")
        # Strip the leading (required)/(optional) marker for cleanliness.
        description = re.sub(r"^\((required|optional)\)\s*", "", description, flags=re.IGNORECASE)
        params.append(
            {
                "name": name,
                "type": type_str,
                "required": required,
                "description": description[:400],  # truncate verbose descriptions
            }
        )
    return params


def _list_service_modules(oci_mod: Any) -> list[str]:
    import pkgutil  # noqa: PLC0415

    return sorted(
        m.name for m in pkgutil.iter_modules(oci_mod.__path__) if not m.name.startswith("_")
    )


def _list_clients_in_service(service_mod: Any) -> list[str]:
    return sorted(
        name
        for name in dir(service_mod)
        if name.endswith("Client")
        and not name.startswith("_")
        and inspect.isclass(getattr(service_mod, name, None))
    )


def _list_operations_on_client(client_cls: type) -> list[str]:
    return sorted(
        name
        for name, member in inspect.getmembers(client_cls, predicate=inspect.isfunction)
        if not name.startswith("_")
    )


@tool(idempotent=True)
def describe_oci(
    service: str | None = None,
    client: str | None = None,
    operation: str | None = None,
) -> dict[str, Any]:
    """Introspect the OCI Python SDK — discover services, clients, operations, params.

    Use this *before* ``use_oci`` whenever you're unsure what shape the
    call needs to take. It progressively zooms in:

    - No args: list every service module under ``oci.*`` (~190 of them).
    - ``service`` only: list every ``*Client`` class inside that service.
    - ``service`` + ``client``: list every operation (method) on that
      client, separated into read-only and mutating.
    - ``service`` + ``client`` + ``operation``: return the operation's
      parameter schema — name, type, required/optional, description —
      parsed from the OCI SDK docstring, plus a snippet of the human
      description and whether the operation is read-only.

    Args:
        service: Snake-case service module name (e.g. ``"identity"``,
            ``"core"``, ``"database"``, ``"object_storage"``).
        client: CamelCase client class name (e.g. ``"IdentityClient"``,
            ``"ComputeClient"``).
        operation: Snake-case operation name (e.g. ``"list_compartments"``).

    Returns:
        A dict whose shape depends on how deep you drilled. Always
        includes ``status``: ``"ok"`` or ``"error"``.

    Examples:
        Discover all services::

            describe_oci()
            # {"status": "ok", "level": "services", "services": ["access_governance_cp", ...]}

        Find clients in a service::

            describe_oci(service="core")
            # {"status": "ok", "level": "clients",
            #  "clients": ["BlockstorageClient", "ComputeClient", "VirtualNetworkClient"]}

        See operation parameter schema::

            describe_oci(
                service="identity",
                client="IdentityClient",
                operation="list_compartments",
            )
            # {"status": "ok", "level": "operation",
            #  "read_only": True,
            #  "parameters": [
            #      {"name": "compartment_id", "type": "str", "required": True, ...},
            #      {"name": "compartment_id_in_subtree", "type": "bool", ...},
            #      ...
            #  ]}
    """
    try:
        oci_mod = _load_oci()
    except _UseOCIError as exc:
        return {"status": "error", "error": str(exc)}

    if service is None:
        services = _list_service_modules(oci_mod)
        return {
            "status": "ok",
            "level": "services",
            "count": len(services),
            "services": services,
        }

    try:
        service_mod = importlib.import_module(f"oci.{service}")
    except ImportError:
        services = _list_service_modules(oci_mod)
        return {
            "status": "error",
            "error": f"No OCI service module named '{service}'.",
            "available_services_sample": services[:30],
            "total_services": len(services),
        }

    if client is None:
        clients = _list_clients_in_service(service_mod)
        return {
            "status": "ok",
            "level": "clients",
            "service": service,
            "clients": clients,
        }

    client_cls = getattr(service_mod, client, None)
    if client_cls is None or not inspect.isclass(client_cls):
        return {
            "status": "error",
            "error": f"OCI service 'oci.{service}' has no client class '{client}'.",
            "available_clients": _list_clients_in_service(service_mod),
        }

    if operation is None:
        ops = _list_operations_on_client(client_cls)
        read = [o for o in ops if _is_read_only(o)]
        write = [o for o in ops if not _is_read_only(o)]
        return {
            "status": "ok",
            "level": "operations",
            "service": service,
            "client": client,
            "count": len(ops),
            "read_only_operations": read,
            "mutating_operations": write,
        }

    method = getattr(client_cls, operation, None)
    if method is None or not callable(method):
        ops = _list_operations_on_client(client_cls)
        close = [o for o in ops if operation.split("_", 1)[0] in o][:20]
        return {
            "status": "error",
            "error": f"Client '{client}' has no operation '{operation}'.",
            "suggestions": close,
        }

    doc = inspect.getdoc(method) or ""
    summary = doc.split("\n\n", 1)[0].strip() if doc else ""
    parameters = _parse_operation_doc(doc)
    try:
        sig = str(inspect.signature(method))
    except (TypeError, ValueError):
        sig = ""

    return {
        "status": "ok",
        "level": "operation",
        "service": service,
        "client": client,
        "operation": operation,
        "signature": f"{operation}{sig}" if sig else operation,
        "read_only": _is_read_only(operation),
        "summary": summary[:600],
        "parameters": parameters,
        "required_parameters": [p["name"] for p in parameters if p["required"]],
    }


__all__ = ["describe_oci", "use_oci"]
