# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Resolve OCI Vault secrets into the environment, then start the server.

The images ``locus deploy`` builds start with::

    python -m locus.deploy.entrypoint uvicorn server:app --host 0.0.0.0 --port 8080

Every variable named ``<VAR>_SECRET_OCID`` is read from OCI Vault with the
principal the platform gives the container (``LOCUS_OCI_AUTH_TYPE``:
``resource_principal`` on Container Instances, Functions and OKE workload
identity; ``instance_principal`` on OKE node identity; ``api_key`` with
``OCI_PROFILE`` for a local run) and its content becomes ``<VAR>``. A ``<VAR>``
that is already set wins, so a value injected another way is never
overwritten. With ``LOCUS_APM_DATA_KEY`` resolved and no
``OTEL_EXPORTER_OTLP_HEADERS``, the Application Performance Monitoring
authorization header is set from it.
"""

from __future__ import annotations

import base64
import os
import sys
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Callable, MutableMapping, Sequence

__all__ = ["SECRET_SUFFIX", "main", "resolve_secrets", "vault_fetcher"]

#: Environment variables ending in this name a Vault secret for the variable before it.
SECRET_SUFFIX = "_SECRET_OCID"  # noqa: S105 — a variable-name suffix, not a secret

_APM_KEY = "LOCUS_APM_DATA_KEY"
_OTLP_HEADERS = "OTEL_EXPORTER_OTLP_HEADERS"


def _signer(auth_type: str, environ: MutableMapping[str, str]) -> tuple[dict[str, Any], Any]:
    import oci

    if auth_type == "resource_principal":
        signer = oci.auth.signers.get_resource_principals_signer()
        return {"region": signer.region}, signer
    if auth_type == "instance_principal":
        signer = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
        return {"region": signer.region}, signer
    if auth_type == "api_key":
        config = oci.config.from_file(profile_name=environ.get("OCI_PROFILE", "DEFAULT"))
        return dict(config), oci.signer.Signer.from_config(config)
    raise ValueError(
        "LOCUS_OCI_AUTH_TYPE must be resource_principal, instance_principal or api_key, "
        f"got {auth_type!r}"
    )


def vault_fetcher(environ: MutableMapping[str, str]) -> Callable[[str], str]:
    """A function that returns the current content of a Vault secret, as text."""
    import oci

    auth_type = environ.get("LOCUS_OCI_AUTH_TYPE") or (
        "resource_principal"
        if environ.get("OCI_RESOURCE_PRINCIPAL_VERSION")
        else "instance_principal"
    )
    config, signer = _signer(auth_type, environ)
    client = oci.secrets.SecretsClient(config, signer=signer)

    def fetch(secret_id: str) -> str:
        bundle = client.get_secret_bundle(secret_id).data
        content = bundle.secret_bundle_content.content
        return base64.b64decode(content).decode("utf-8")

    return fetch


def resolve_secrets(
    environ: MutableMapping[str, str],
    fetch: Callable[[str], str] | None = None,
) -> list[str]:
    """Set every ``<VAR>`` whose ``<VAR>_SECRET_OCID`` is in ``environ``; return the names set.

    ``fetch`` maps a secret OCID to its content; the default reads OCI Vault
    with the principal :func:`vault_fetcher` picks. Nothing is fetched when no
    variable asks for it, so an image without secrets never imports the OCI SDK.
    """
    pending = {
        key[: -len(SECRET_SUFFIX)]: secret_id
        for key, secret_id in environ.items()
        if key.endswith(SECRET_SUFFIX) and secret_id and len(key) > len(SECRET_SUFFIX)
    }
    resolved: list[str] = []
    todo = {name: secret_id for name, secret_id in pending.items() if not environ.get(name)}
    if todo:
        fetch = fetch or vault_fetcher(environ)
        for name in sorted(todo):
            environ[name] = fetch(todo[name])
            resolved.append(name)
    if environ.get(_APM_KEY) and not environ.get(_OTLP_HEADERS):
        environ[_OTLP_HEADERS] = f"Authorization=dataKey%20{environ[_APM_KEY]}"
    return resolved


def main(argv: Sequence[str] | None = None) -> int:
    """Resolve secrets, then replace this process with the command in ``argv``."""
    command = list(sys.argv[1:] if argv is None else argv)
    if not command:
        print("usage: python -m locus.deploy.entrypoint COMMAND [ARG ...]", file=sys.stderr)
        return 2
    resolved = resolve_secrets(os.environ)
    if resolved:
        print(
            f"locus.deploy.entrypoint: resolved {', '.join(resolved)} from OCI Vault",
            file=sys.stderr,
        )
    os.execvp(command[0], command)  # noqa: S606 - the point of the entrypoint is to hand over to the server
    return 0  # pragma: no cover - execvp does not return


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
