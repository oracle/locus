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

import re
from typing import Any

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

    Shared by the checkpointer backend and the long-term store. Two
    ways to address a database:

    * **TNS alias** — set ``dsn`` to a tnsnames entry inside the
      wallet (e.g. ``mydb_low``). Most common path for Autonomous
      Database.
    * **Host triplet** — set ``host`` + ``port`` + ``service_name``
      and locus will assemble the connect string at pool creation.

    For Autonomous Database with mTLS, point ``wallet_location`` at
    the unzipped wallet directory. ``wallet_password`` is only
    required when the wallet bundle is encrypted (rare for the
    instance wallet, common for the regional wallet).
    """

    # Connection options
    dsn: str | None = None  # TNS name or connection string
    user: str = "admin"
    password: SecretStr = SecretStr("")

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
