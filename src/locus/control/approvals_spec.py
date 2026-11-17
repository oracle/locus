# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Name an approval store in one string, so a command line can open the agent's.

A hold is placed by the agent and decided somewhere else: ``locus approvals``,
an operator's script, a service of its own. Each needs to open the same store,
and a URL is how a store is named across processes::

    memory://                                   InMemoryApprovals (this process only)
    file://approvals.json   or   approvals.json  FileApprovals
    postgresql://user:pw@host:5432/db?table=ops.approvals
    redis://host:6379/0?prefix=billing:approvals:
    oracle://user:pw@host:1522/service?wallet=/opt/wallet&table=approvals
    oracle://user:pw@/?dsn=locusdb_low&wallet=/opt/wallet
    oracle://user:pw@/?dsn=locusdb_low&config_dir=/opt/tns     (TLS-only, no wallet)

A database spec with nothing after the scheme falls back to the environment the
integration tests already read: ``POSTGRES_HOST`` / ``POSTGRES_PORT`` /
``POSTGRES_DB`` / ``POSTGRES_USER`` / ``POSTGRES_PASSWORD``, ``REDIS_URL``, and
``ORACLE_DSN`` / ``ORACLE_USER`` / ``ORACLE_PASSWORD`` / ``ORACLE_CONFIG_DIR`` (else
``TNS_ADMIN``) / ``ORACLE_WALLET`` / ``ORACLE_WALLET_PASSWORD``. The spec itself can come from
``LOCUS_APPROVALS_STORE``, which is what ``locus approvals`` reads when
``--store`` is not given.

Every store knows its own spec (``store.spec``), with any password removed, and
the gate puts it on the hold it writes into the thread's checkpoint, so a process
that finds a held thread can open the store the hold is in.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit

from locus.control.approvals import ApprovalStore, FileApprovals, InMemoryApprovals


if TYPE_CHECKING:
    from collections.abc import Callable

    from locus.control.approvals import ApprovalAuthority, ApprovalRecord


__all__ = ["SPEC_ENV", "approval_store", "store_spec"]

#: The environment variable ``locus approvals`` reads the spec from.
SPEC_ENV = "LOCUS_APPROVALS_STORE"

_Authority = "ApprovalAuthority | Callable[[ApprovalRecord], ApprovalAuthority | None] | None"


def store_spec(store: Any) -> str | None:
    """``store``'s spec, when it has one; the SDK's stores all do."""
    spec = getattr(store, "spec", None)
    return spec if isinstance(spec, str) else None


def _split_options(spec: str, *names: str) -> tuple[str, dict[str, str]]:
    """``spec`` without the query parameters in ``names``, and those parameters."""
    parts = urlsplit(spec)
    query = parse_qs(parts.query, keep_blank_values=True)
    taken = {name: query.pop(name)[-1] for name in names if name in query}
    rest = urlencode({k: v[-1] for k, v in query.items()})
    return urlunsplit((parts.scheme, parts.netloc, parts.path, rest, parts.fragment)), taken


def _table(options: dict[str, str], default_schema: str | None) -> dict[str, Any]:
    """``table=`` as ``schema_name`` / ``table_name`` keyword arguments."""
    kwargs: dict[str, Any] = {}
    table = options.get("table")
    if table:
        schema, dot, name = table.rpartition(".")
        kwargs["table_name"] = name
        if dot:
            kwargs["schema_name"] = schema
    if "schema" in options:
        kwargs["schema_name"] = options["schema"]
    if default_schema is not None:
        kwargs.setdefault("schema_name", default_schema)
    return kwargs


def _postgres(spec: str, authority: Any) -> ApprovalStore:
    from locus.control.approvals_postgres import PostgresApprovals  # noqa: PLC0415

    dsn, options = _split_options(spec, "table", "schema")
    kwargs = _table(options, "public")
    if urlsplit(dsn).netloc or urlsplit(dsn).path.strip("/"):
        return PostgresApprovals(dsn=dsn, authority=authority, **kwargs)
    host = os.environ.get("POSTGRES_HOST")
    if not host:
        raise ValueError(
            "a bare postgresql:// spec needs POSTGRES_HOST (and POSTGRES_PORT / POSTGRES_DB / "
            "POSTGRES_USER / POSTGRES_PASSWORD) in the environment"
        )
    return PostgresApprovals(
        host=host,
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        database=os.environ.get("POSTGRES_DB", "locus"),
        user=os.environ.get("POSTGRES_USER", "postgres"),
        password=os.environ.get("POSTGRES_PASSWORD", ""),
        authority=authority,
        **kwargs,
    )


def _redis(spec: str, authority: Any) -> ApprovalStore:
    from locus.control.approvals_redis import RedisApprovals  # noqa: PLC0415

    url, options = _split_options(spec, "prefix")
    kwargs: dict[str, Any] = {}
    if "prefix" in options:
        kwargs["prefix"] = options["prefix"]
    if not urlsplit(url).netloc:
        url = os.environ.get("REDIS_URL", "")
        if not url:
            raise ValueError("a bare redis:// spec needs REDIS_URL in the environment")
    return RedisApprovals(url, authority=authority, **kwargs)


def _oracle(spec: str, authority: Any) -> ApprovalStore:
    from locus.control.approvals_oracle import OracleApprovals  # noqa: PLC0415
    from locus.memory.backends._oracle_config import config_dir_from_env  # noqa: PLC0415

    url, options = _split_options(
        spec,
        "table",
        "schema",
        "dsn",
        "config_dir",
        "wallet",
        "wallet_password",
        "user",
        "password",
    )
    parts = urlsplit(url)
    kwargs: dict[str, Any] = _table(options, None)
    user = parts.username or options.get("user") or os.environ.get("ORACLE_USER", "admin")
    password = parts.password or options.get("password") or os.environ.get("ORACLE_PASSWORD", "")
    wallet = options.get("wallet") or os.environ.get("ORACLE_WALLET")
    wallet_password = options.get("wallet_password")
    if wallet_password is None and wallet and os.environ.get("ORACLE_WALLET_PASSWORD") is not None:
        wallet_password = os.environ.get("ORACLE_WALLET_PASSWORD")
    config_dir = options.get("config_dir") or config_dir_from_env(wallet)
    if parts.hostname and parts.path.strip("/"):
        kwargs.update(
            host=parts.hostname, port=parts.port or 1521, service_name=parts.path.strip("/")
        )
    else:
        dsn = options.get("dsn") or os.environ.get("ORACLE_DSN")
        if not dsn:
            raise ValueError(
                "an oracle:// spec names host:port/service, or dsn=<tns alias> "
                "(or ORACLE_DSN in the environment)"
            )
        kwargs["dsn"] = dsn
    return OracleApprovals(
        user=user,
        password=password,
        config_dir=config_dir,
        wallet_location=wallet,
        wallet_password=wallet_password,
        authority=authority,
        **kwargs,
    )


def approval_store(
    spec: str | None = None,
    authority: ApprovalAuthority
    | Callable[[ApprovalRecord], ApprovalAuthority | None]
    | None = None,
) -> ApprovalStore:
    """The store a spec names (see the module), with ``authority`` checked on decisions.

    ``spec=None`` reads ``LOCUS_APPROVALS_STORE``.

    Raises:
        ValueError: No spec, an unknown scheme, or a database spec that names
            neither a server nor an environment to find one in.
    """
    spec = spec or os.environ.get(SPEC_ENV, "")
    if not spec:
        raise ValueError(f"no approval store: pass a spec or set {SPEC_ENV}")
    scheme = urlsplit(spec).scheme.lower() if "://" in spec else ""
    if scheme in ("", "file"):
        path = spec[len("file://") :] if scheme == "file" else spec
        return FileApprovals(path, authority)
    if scheme == "memory":
        return InMemoryApprovals(authority)
    if scheme in ("postgresql", "postgres"):
        return _postgres(spec, authority)
    if scheme in ("redis", "rediss"):
        return _redis(spec, authority)
    if scheme == "oracle":
        return _oracle(spec, authority)
    raise ValueError(
        f"unknown approval store scheme {scheme!r}: "
        "expected memory://, file://, postgresql://, redis:// or oracle://"
    )
