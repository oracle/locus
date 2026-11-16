# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""The resource-principal signer for the platform a container runs on.

OCI Container Instances and OCI Functions hand the container a resource
principal session token (``OCI_RESOURCE_PRINCIPAL_RPST`` and
``OCI_RESOURCE_PRINCIPAL_PRIVATE_PEM``), which the OCI SDK's
``get_resource_principals_signer`` reads. OKE workload identity hands a pod
no such token: it sets ``OCI_RESOURCE_PRINCIPAL_VERSION`` and the region and
the pod exchanges its Kubernetes service-account token through
``get_oke_workload_identity_resource_principal_signer``; the generic signer
fails there with ``session_token was not provided``. Every ``auth_type``
switch in the SDK builds its resource-principal signer here, so
``resource_principal`` works on all three platforms and
``oke_workload_identity`` names the OKE path outright.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "OKE_WORKLOAD_IDENTITY",
    "RESOURCE_PRINCIPAL",
    "RESOURCE_PRINCIPAL_AUTH_TYPES",
    "SERVICE_ACCOUNT_TOKEN_PATH",
    "on_oke_workload_identity",
    "resource_principal_signer",
]

#: ``auth_type`` for the generic resource principal (Container Instances, Functions).
RESOURCE_PRINCIPAL = "resource_principal"

#: ``auth_type`` that selects the OKE workload-identity signer outright.
OKE_WORKLOAD_IDENTITY = "oke_workload_identity"

#: The auth types :func:`resource_principal_signer` accepts.
RESOURCE_PRINCIPAL_AUTH_TYPES: frozenset[str] = frozenset(
    {RESOURCE_PRINCIPAL, OKE_WORKLOAD_IDENTITY}
)

#: Where Kubernetes mounts the pod's service-account token.
SERVICE_ACCOUNT_TOKEN_PATH = Path("/var/run/secrets/kubernetes.io/serviceaccount/token")  # noqa: S105 — a path, not a secret

_SESSION_TOKEN_VARIABLES = ("OCI_RESOURCE_PRINCIPAL_RPST", "OCI_RESOURCE_PRINCIPAL_PRIVATE_PEM")


def on_oke_workload_identity(
    environ: Mapping[str, str] | None = None,
    token_path: Path | None = None,
) -> bool:
    """Whether this process is a pod with a service-account token and no session token.

    That is the shape of OKE workload identity; Container Instances and
    Functions always carry ``OCI_RESOURCE_PRINCIPAL_RPST``. ``token_path``
    defaults to :data:`SERVICE_ACCOUNT_TOKEN_PATH`.
    """
    env = os.environ if environ is None else environ
    if any(env.get(name) for name in _SESSION_TOKEN_VARIABLES):
        return False
    return (SERVICE_ACCOUNT_TOKEN_PATH if token_path is None else token_path).is_file()


def resource_principal_signer(
    auth_type: str = RESOURCE_PRINCIPAL,
    *,
    oci_module: Any = None,
    environ: Mapping[str, str] | None = None,
    token_path: Path | None = None,
) -> Any:
    """The OCI signer for ``resource_principal`` or ``oke_workload_identity``.

    ``oke_workload_identity`` always returns the OKE workload-identity signer.
    ``resource_principal`` returns it too when the process looks like an OKE
    pod (see :func:`on_oke_workload_identity`), so a deployment that says
    ``resource_principal`` authenticates on every OCI platform; otherwise it
    returns the generic resource-principal signer. ``oci_module`` is the
    imported ``oci`` package; it is imported here when not given.
    """
    if auth_type not in RESOURCE_PRINCIPAL_AUTH_TYPES:
        raise ValueError(
            f"auth_type must be {RESOURCE_PRINCIPAL} or {OKE_WORKLOAD_IDENTITY}, got {auth_type!r}"
        )
    if oci_module is None:
        import oci  # noqa: PLC0415 — the SDK is optional and imported on use

        oci_module = oci
    signers = oci_module.auth.signers
    if auth_type == OKE_WORKLOAD_IDENTITY or on_oke_workload_identity(environ, token_path):
        return signers.get_oke_workload_identity_resource_principal_signer()
    return signers.get_resource_principals_signer()
