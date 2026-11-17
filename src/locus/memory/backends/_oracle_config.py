# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Oracle backend configuration + identifier validator.

Pulled out of ``oracle.py`` so the connection-envelope shape and its
SQL identifier safety check live in one focused module. Both
:class:`locus.memory.backends.oracle.OracleBackend` and the
forthcoming :class:`locus.memory.store_backends.oracle.OracleStore`
share these symbols — keeps the validator regex in exactly one
place.
"""

from __future__ import annotations

import os
import re
from typing import Any, Protocol

from pydantic import BaseModel, SecretStr


_SAFE_SQL_IDENTIFIER = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_$#]{0,127}$")


def validate_sql_identifier(value: str, field_name: str) -> str:
    """Reject anything that isn't a safe Oracle SQL identifier.

    Used everywhere locus splices a table / schema / column name into
    DDL or DML. Oracle identifiers max out at 128 characters, must
    start with a letter or underscore, and may contain only letters,
    digits, underscore, dollar, and hash. Anything else (dots, quotes,
    spaces, semicolons) is an injection risk and gets rejected.
    """
    if not _SAFE_SQL_IDENTIFIER.match(value):
        msg = (
            f"Invalid {field_name}: {value!r}. "
            "Must start with a letter or underscore and contain only "
            "alphanumeric characters, underscores, $, or # (max 128 chars)."
        )
        raise ValueError(msg)
    return value


class OracleConfig(BaseModel):
    """Connection envelope for an Oracle Database 23ai/26ai pool.

    Shared by the checkpointer backend and the long-term store. Three
    ways to address a database:

    * **Connect string** — set ``dsn`` to the full connect descriptor
      (``(description=...(protocol=tcps)...)``) or an Easy Connect
      string. Needs no directory and, over one-way TLS, no wallet.
    * **TNS alias** — set ``dsn`` to a ``tnsnames.ora`` entry (e.g.
      ``mydb_low``) and name the directory holding the file:
      ``config_dir`` for a TLS-only database, or ``wallet_location``
      for mTLS (the wallet directory carries its own ``tnsnames.ora``).
      With neither, python-oracledb reads ``TNS_ADMIN``.
    * **Host triplet** — set ``host`` + ``port`` + ``service_name``
      and locus will assemble the connect string at pool creation.

    An Autonomous Database without mutual TLS has no wallet: the client
    checks the server certificate against the public CAs, so ``dsn``
    plus ``user`` / ``password`` is enough. With mTLS, point
    ``wallet_location`` at the unzipped wallet directory.
    ``wallet_password`` is only required when the wallet bundle is
    encrypted (rare for the instance wallet, common for the regional
    wallet).
    """

    # Connection options
    dsn: str | None = None  # TNS name or connection string
    user: str = "admin"
    password: SecretStr = SecretStr("")

    # Directory holding tnsnames.ora, for a TNS alias without a wallet
    config_dir: str | None = None

    # For Autonomous Database with wallet
    wallet_location: str | None = None
    wallet_password: SecretStr | None = None

    # Connection string components (alternative to DSN)
    host: str | None = None
    port: int = 1521
    service_name: str | None = None

    # Table settings
    table_name: str = "locus_checkpoints"
    schema_name: str | None = None  # Uses user's default schema if None

    # Pool settings
    min_pool_size: int = 1
    max_pool_size: int = 5

    def model_post_init(self, __context: Any) -> None:
        """Validate SQL identifiers to prevent injection."""
        validate_sql_identifier(self.table_name, "table_name")
        if self.schema_name is not None:
            validate_sql_identifier(self.schema_name, "schema_name")


class _ConnectionEnvelope(Protocol):
    """The fields :func:`connect_params` reads; every Oracle config has them."""

    config_dir: str | None
    wallet_location: str | None
    wallet_password: SecretStr | None


def connect_params(config: _ConnectionEnvelope) -> dict[str, Any]:
    """The ``config_dir`` / wallet keyword arguments for ``oracledb.create_pool*``.

    ``config_dir`` is passed when set. ``wallet_location`` is passed only
    when a wallet is given — python-oracledb then insists on finding
    ``ewallet.pem`` or ``cwallet.sso`` there, which a TLS-only database
    has no use for — and it doubles as ``config_dir`` when no
    ``config_dir`` is set, so wallet users keep resolving aliases from
    the wallet directory. Nothing is passed when neither is set: the
    driver reads ``TNS_ADMIN`` for an alias, and a full connect string
    needs no directory at all.
    """
    params: dict[str, Any] = {}
    if config.config_dir:
        params["config_dir"] = config.config_dir
    if config.wallet_location:
        params.setdefault("config_dir", config.wallet_location)
        params["wallet_location"] = config.wallet_location
        # ``is not None`` (not truthiness): SecretStr("") is falsy in
        # Pydantic v2, and "" is the python-oracledb idiom for an
        # auto-login (cwallet.sso) wallet — a truthy guard would drop it
        # and force the encrypted ewallet.pem path. See issue #289.
        if config.wallet_password is not None:
            params["wallet_password"] = config.wallet_password.get_secret_value()
    return params


def config_dir_from_env(wallet_location: str | None = None) -> str | None:
    """The TNS directory the environment names: ``ORACLE_CONFIG_DIR``, else ``TNS_ADMIN``.

    ``TNS_ADMIN`` is only read when no wallet is given, so a wallet
    directory keeps resolving its own aliases as before.
    """
    config_dir = os.environ.get("ORACLE_CONFIG_DIR")
    if config_dir or wallet_location:
        return config_dir or None
    return os.environ.get("TNS_ADMIN") or None
