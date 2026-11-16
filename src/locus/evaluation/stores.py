# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Reports kept somewhere, and the trend across them.

A report on disk is one build's artefact. A regression is a question about
several: did the pass count move since last week, did the average slide a
hundredth a day, did the suite itself change under the numbers. A
:class:`ReportStore` keeps :class:`~locus.evaluation.EvalReport` objects by
suite and by fingerprint, and :func:`trend` reads the last N of one suite
back, newest first, and says where each moved against the one before it.

Four stores share one interface:

- :class:`FileReportStore`: a directory, one JSON file per report.
- :class:`OracleReportStore`: a table on an Oracle AI Database, the report
  as JSON with its headline figures in columns, so a query can trend a
  suite without reading every report.
- :class:`OCIBucketReportStore`: an OCI Object Storage bucket.
- :class:`S3ReportStore`: any S3-compatible bucket.

:func:`report_store` opens one from a spec string, which is how
``locus eval run --store`` and ``locus eval trend`` name theirs::

    file:///var/lib/locus/evals          or a bare path
    oracle://user:pw@/?dsn=mydb_low&wallet=/opt/wallet&table=LOCUS_EVAL_REPORTS
    oci://<namespace>/<bucket>/<prefix>?profile=DEFAULT&region=eu-frankfurt-1
    s3://<bucket>/<prefix>?endpoint_url=https://...

Every report is stored under a suite name; the fingerprint on the report
is kept beside it, and ``list()`` can be narrowed to one fingerprint so a
trend never mixes two different suites.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

from pydantic import BaseModel, Field

from locus.evaluation.framework import EvalReport


if TYPE_CHECKING:
    from collections.abc import Sequence


__all__ = [
    "SPEC_ENV",
    "FileReportStore",
    "OCIBucketReportStore",
    "OracleReportStore",
    "ReportStore",
    "S3ReportStore",
    "StoredReport",
    "Trend",
    "TrendPoint",
    "report_store",
    "trend",
    "trend_of",
]


#: The environment variable ``locus eval`` reads a store spec from.
SPEC_ENV = "LOCUS_EVAL_STORE"

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
_INLINE_LIMIT = 32_000
_TIMESTAMP = "%Y%m%dT%H%M%S%fZ"


def _safe(name: str) -> str:
    cleaned = _SAFE_NAME.sub("-", name.strip()).strip("-.")
    return cleaned or "suite"


def _stamp(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime(_TIMESTAMP)


def _unstamp(text: str) -> datetime:
    return datetime.strptime(text, _TIMESTAMP).replace(tzinfo=UTC)


def _utc_naive(moment: datetime) -> datetime:
    """``moment`` as UTC wall-clock time with no zone, the form the table stores."""
    if moment.tzinfo is None:
        return moment
    return moment.astimezone(UTC).replace(tzinfo=None)


def _utc_aware(value: Any) -> datetime:
    """A stored ``created_at`` read back as an aware UTC datetime."""
    moment = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment.astimezone(UTC)


class StoredReport(BaseModel):
    """What a store knows about one report without loading it."""

    report_id: str
    suite: str
    fingerprint: str | None = None
    created_at: datetime
    total_cases: int = 0
    passed: int = 0
    failed: int = 0
    not_graded: int = 0
    avg_score: float = 0.0
    location: str = Field(default="", description="Where the store keeps it: a path, a key")

    @classmethod
    def of(
        cls, report: EvalReport, *, report_id: str, suite: str, location: str = ""
    ) -> StoredReport:
        return cls(
            report_id=report_id,
            suite=suite,
            fingerprint=report.fingerprint,
            created_at=report.created_at,
            total_cases=report.total_cases,
            passed=report.passed,
            failed=report.failed,
            not_graded=report.not_graded,
            avg_score=report.avg_score,
            location=location,
        )


@runtime_checkable
class ReportStore(Protocol):
    """Where reports are kept. Every store knows its ``spec``, with no password in it."""

    spec: str

    def put(self, report: EvalReport, *, suite: str, report_id: str | None = None) -> StoredReport:
        """Keep ``report`` under ``suite``; returns what was recorded about it."""

    def get(self, report_id: str) -> EvalReport | None:
        """The report with ``report_id``, or ``None``."""

    def list(
        self, *, suite: str | None = None, fingerprint: str | None = None, limit: int = 20
    ) -> list[StoredReport]:
        """Reports newest first, narrowed to a suite and a fingerprint when given."""


def _object_name(prefix: str, suite: str, report: EvalReport, report_id: str) -> str:
    return (
        f"{prefix}{_safe(suite)}/{_stamp(report.created_at)}-"
        f"{report.fingerprint or 'none'}-{report_id}.json"
    )


_NAME = re.compile(
    r"(?P<stamp>\d{8}T\d{12}Z)-(?P<fingerprint>[0-9a-f]{12}|none)-(?P<id>[0-9a-f]{32})\.json$"
)


def _parse_name(name: str, *, prefix: str) -> tuple[str, str, str | None, datetime] | None:
    """``(suite, report_id, fingerprint, created_at)`` from an object name, or ``None``."""
    if not name.startswith(prefix):
        return None
    rest = name[len(prefix) :]
    suite, sep, filename = rest.rpartition("/")
    if not sep:
        return None
    match = _NAME.match(filename)
    if match is None:
        return None
    fingerprint = match.group("fingerprint")
    return (
        suite,
        match.group("id"),
        None if fingerprint == "none" else fingerprint,
        _unstamp(match.group("stamp")),
    )


class _ObjectStore:
    """The shared half of the bucket-backed stores: names, listing, headline rows."""

    prefix: str
    spec: str

    def _put_bytes(self, name: str, body: bytes) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def _get_bytes(self, name: str) -> bytes | None:  # pragma: no cover - abstract
        raise NotImplementedError

    def _list_names(self, prefix: str) -> list[str]:  # pragma: no cover - abstract
        raise NotImplementedError

    def put(self, report: EvalReport, *, suite: str, report_id: str | None = None) -> StoredReport:
        report_id = report_id or uuid4().hex
        name = _object_name(self.prefix, suite, report, report_id)
        self._put_bytes(name, report.model_dump_json(indent=2).encode("utf-8"))
        return StoredReport.of(report, report_id=report_id, suite=suite, location=name)

    def get(self, report_id: str) -> EvalReport | None:
        for name in self._list_names(self.prefix):
            parsed = _parse_name(name, prefix=self.prefix)
            if parsed is not None and parsed[1] == report_id:
                body = self._get_bytes(name)
                return None if body is None else EvalReport.model_validate_json(body)
        return None

    def list(
        self, *, suite: str | None = None, fingerprint: str | None = None, limit: int = 20
    ) -> list[StoredReport]:
        scope = f"{self.prefix}{_safe(suite)}/" if suite else self.prefix
        found: list[StoredReport] = []
        for name in self._list_names(scope):
            parsed = _parse_name(name, prefix=self.prefix)
            if parsed is None:
                continue
            found_suite, report_id, found_fingerprint, created_at = parsed
            if fingerprint is not None and found_fingerprint != fingerprint:
                continue
            found.append(
                StoredReport(
                    report_id=report_id,
                    suite=found_suite,
                    fingerprint=found_fingerprint,
                    created_at=created_at,
                    location=name,
                )
            )
        found.sort(key=lambda item: (item.created_at, item.report_id), reverse=True)
        chosen = found[: max(0, limit)]
        # The name carries when and which suite; the headline figures take a read.
        for item in chosen:
            body = self._get_bytes(item.location)
            if body is None:
                continue
            report = EvalReport.model_validate_json(body)
            item.total_cases = report.total_cases
            item.passed = report.passed
            item.failed = report.failed
            item.not_graded = report.not_graded
            item.avg_score = report.avg_score
        return chosen


class FileReportStore(_ObjectStore):
    """Reports as JSON files under a directory, one sub-directory per suite."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory).expanduser()
        self.prefix = ""
        self.spec = f"file://{self.directory}"

    def _put_bytes(self, name: str, body: bytes) -> None:
        path = self.directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)

    def _get_bytes(self, name: str) -> bytes | None:
        path = self.directory / name
        return path.read_bytes() if path.is_file() else None

    def _list_names(self, prefix: str) -> list[str]:
        if not self.directory.is_dir():
            return []
        return sorted(
            str(path.relative_to(self.directory)).replace(os.sep, "/")
            for path in self.directory.rglob("*.json")
            if str(path.relative_to(self.directory)).replace(os.sep, "/").startswith(prefix)
        )


class OCIBucketReportStore(_ObjectStore):
    """Reports in an OCI Object Storage bucket.

    Authentication follows the OCI checkpointer: ``profile_name`` from
    ``config_file`` with ``auth_type`` ``api_key`` (default),
    ``security_token``, ``instance_principal`` or ``resource_principal``.

    Args:
        bucket_name: The bucket.
        namespace: The bucket's Object Storage namespace.
        prefix: Object name prefix; reports go under ``<prefix><suite>/``.
        profile_name: Profile in ``config_file`` (default ``DEFAULT``).
        config_file: OCI config file (default ``~/.oci/config``).
        auth_type: ``api_key``, ``security_token``, ``instance_principal`` or
            ``resource_principal``.
        region: Region override for the profile.
        _client: An ``ObjectStorageClient`` to use instead of building one; the
            seam tests inject a double through.
    """

    def __init__(
        self,
        bucket_name: str,
        namespace: str,
        *,
        prefix: str = "locus/eval/",
        profile_name: str = "DEFAULT",
        config_file: str = "~/.oci/config",
        auth_type: str = "api_key",
        region: str | None = None,
        _client: Any = None,
    ) -> None:
        self.bucket_name = bucket_name
        self.namespace = namespace
        self.prefix = prefix
        self._profile_name = profile_name
        self._config_file = config_file
        self._auth_type = auth_type
        self._region = region
        self._client: Any = _client
        self.spec = f"oci://{namespace}/{bucket_name}/{prefix}"

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        from locus.memory.backends.oci_bucket import (  # noqa: PLC0415
            OCIBucketConfig,
            build_object_storage_client,
        )

        self._client = build_object_storage_client(
            OCIBucketConfig(
                bucket_name=self.bucket_name,
                namespace=self.namespace,
                prefix=self.prefix,
                profile_name=self._profile_name,
                config_file=self._config_file,
                auth_type=self._auth_type,
                region=self._region,
            )
        )
        return self._client

    def _put_bytes(self, name: str, body: bytes) -> None:
        self._get_client().put_object(
            namespace_name=self.namespace,
            bucket_name=self.bucket_name,
            object_name=name,
            put_object_body=body,
            content_type="application/json",
        )

    def _get_bytes(self, name: str) -> bytes | None:
        try:
            response = self._get_client().get_object(
                namespace_name=self.namespace, bucket_name=self.bucket_name, object_name=name
            )
        except Exception as exc:
            if "ObjectNotFound" in str(exc) or "404" in str(exc):
                return None
            raise
        content: bytes = response.data.content
        return content

    def _list_names(self, prefix: str) -> list[str]:
        client = self._get_client()
        names: list[str] = []
        start: str | None = None
        while True:
            kwargs: dict[str, Any] = {
                "namespace_name": self.namespace,
                "bucket_name": self.bucket_name,
                "prefix": prefix,
                "fields": "name",
                "limit": 1000,
            }
            if start:
                kwargs["start"] = start
            response = client.list_objects(**kwargs)
            names.extend(obj.name for obj in (response.data.objects or []))
            start = getattr(response.data, "next_start_with", None)
            if not start:
                return names


class S3ReportStore(_ObjectStore):
    """Reports in an S3-compatible bucket; ``endpoint_url`` for a non-AWS service.

    Args:
        bucket: The bucket.
        prefix: Key prefix; reports go under ``<prefix><suite>/``.
        endpoint_url: Endpoint of a non-AWS S3-compatible service.
        region_name: Region, when the endpoint needs one.
        aws_access_key_id: Explicit access key; the default credential chain
            applies when omitted.
        aws_secret_access_key: Secret key, paired with ``aws_access_key_id``.
        _client: A pre-built S3 client to use instead; the seam tests inject a
            double through.
    """

    def __init__(
        self,
        bucket: str,
        *,
        prefix: str = "locus/eval/",
        endpoint_url: str | None = None,
        region_name: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        _client: Any = None,
    ) -> None:
        self.bucket = bucket
        self.prefix = prefix
        self._endpoint_url = endpoint_url
        self._region_name = region_name
        self._aws_access_key_id = aws_access_key_id
        self._aws_secret_access_key = aws_secret_access_key
        self._client: Any = _client
        self.spec = f"s3://{bucket}/{prefix}"

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import boto3  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                'boto3 is not installed. Install with: pip install "locus-sdk[s3]"'
            ) from exc
        self._client = boto3.client(
            "s3",
            endpoint_url=self._endpoint_url,
            region_name=self._region_name,
            aws_access_key_id=self._aws_access_key_id,
            aws_secret_access_key=self._aws_secret_access_key,
        )
        return self._client

    def _put_bytes(self, name: str, body: bytes) -> None:
        self._get_client().put_object(
            Bucket=self.bucket, Key=name, Body=body, ContentType="application/json"
        )

    def _get_bytes(self, name: str) -> bytes | None:
        try:
            response = self._get_client().get_object(Bucket=self.bucket, Key=name)
        except Exception as exc:
            code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
            if code in ("NoSuchKey", "404") or "NoSuchKey" in str(exc):
                return None
            raise
        data: bytes = response["Body"].read()
        return data

    def _list_names(self, prefix: str) -> list[str]:
        client = self._get_client()
        names: list[str] = []
        kwargs: dict[str, Any] = {"Bucket": self.bucket, "Prefix": prefix}
        while True:
            response = client.list_objects_v2(**kwargs)
            names.extend(obj["Key"] for obj in response.get("Contents", []))
            if not response.get("IsTruncated"):
                return names
            kwargs["ContinuationToken"] = response.get("NextContinuationToken")


class OracleReportStore:
    """Reports in a table on an Oracle AI Database.

    One row per report: the headline figures in columns, so ``list()`` is a
    query rather than a read of every report, and the report itself as JSON.
    Connections come from a small synchronous ``oracledb`` pool built with the
    connection arguments the Oracle checkpointer takes (``dsn`` or
    ``host``/``port``/``service_name``, ``wallet_location``,
    ``wallet_password``). Needs ``oracledb`` (``pip install "locus-sdk[oracledb]"``).

    Args:
        dsn: TNS alias or connect string. Alternatively ``host``, ``port`` and
            ``service_name``.
        user: Schema user.
        password: The schema user's password.
        wallet_location: Wallet directory, for Autonomous Database with mTLS.
        wallet_password: Wallet password, when the wallet is encrypted.
        host: Database host, when not using ``dsn``.
        port: Listener port (default 1521).
        service_name: Service name, paired with ``host``.
        table_name: The table; created on first use when ``auto_create_table``
            is on. Must be a safe SQL identifier.
        schema_name: Owner of the table when it is not the connecting user's.
        auto_create_table: Create the table when it does not exist.
        min_pool_size: Smallest size of the pool.
        max_pool_size: Largest size of the pool.
        _connect: A factory returning a connection, instead of the pool; the
            seam tests inject a double through.
    """

    def __init__(
        self,
        *,
        dsn: str | None = None,
        user: str = "admin",
        password: str = "",
        wallet_location: str | None = None,
        wallet_password: str | None = None,
        host: str | None = None,
        port: int = 1521,
        service_name: str | None = None,
        table_name: str = "locus_eval_reports",
        schema_name: str | None = None,
        auto_create_table: bool = True,
        min_pool_size: int = 1,
        max_pool_size: int = 2,
        _connect: Callable[[], Any] | None = None,
    ) -> None:
        from pydantic import SecretStr  # noqa: PLC0415

        from locus.memory.backends._oracle_config import OracleConfig  # noqa: PLC0415

        self.config: OracleConfig = OracleConfig(
            dsn=dsn,
            user=user,
            password=SecretStr(password),
            wallet_location=wallet_location,
            wallet_password=SecretStr(wallet_password) if wallet_password is not None else None,
            host=host,
            port=port,
            service_name=service_name,
            table_name=table_name,
            schema_name=schema_name,
            min_pool_size=min_pool_size,
            max_pool_size=max_pool_size,
        )
        self._auto_create = auto_create_table
        self._connect = _connect
        self._pool: Any = None
        self._table_ready = False
        if dsn:
            where = f"oracle://{user}@/?dsn={dsn}"
        else:
            where = f"oracle://{user}@{host}:{port}/{service_name}?"
        self.spec = f"{where}&table={table_name}" if "?" in where else f"{where}?table={table_name}"

    # -- connection ----------------------------------------------------------

    @property
    def _full_table_name(self) -> str:
        if self.config.schema_name:
            return f"{self.config.schema_name}.{self.config.table_name}"
        return self.config.table_name

    def _build_pool(self) -> Any:
        try:
            import oracledb  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise ImportError(
                'OracleReportStore needs the oracledb package:\n    pip install "locus-sdk[oracledb]"'
            ) from exc
        cfg = self.config
        dsn = cfg.dsn
        if dsn is None and cfg.host and cfg.service_name:
            dsn = oracledb.makedsn(cfg.host, cfg.port, service_name=cfg.service_name)
        params: dict[str, Any] = {}
        if cfg.wallet_location:
            params["config_dir"] = cfg.wallet_location
            params["wallet_location"] = cfg.wallet_location
            if cfg.wallet_password is not None:
                params["wallet_password"] = cfg.wallet_password.get_secret_value()
        return oracledb.create_pool(
            user=cfg.user,
            password=cfg.password.get_secret_value(),
            dsn=dsn,
            min=cfg.min_pool_size,
            max=cfg.max_pool_size,
            **params,
        )

    def _run(self, op: Callable[[Any], Any]) -> Any:
        """Run ``op(conn)`` on a connection, rebuilding the pool once on a dead socket."""
        if self._connect is not None:
            conn = self._connect()
            return op(conn)
        from locus._oracle_pool_cache import is_reconnectable  # noqa: PLC0415

        pool = self._pool if self._pool is not None else self._build_pool()
        self._pool = pool
        try:
            with pool.acquire() as conn:
                return op(conn)
        except Exception as exc:
            if not is_reconnectable(exc):
                raise
            self.close()
            self._pool = self._build_pool()
            with self._pool.acquire() as conn:
                return op(conn)

    def close(self) -> None:
        """Close the connection pool. The next call opens a new one."""
        if self._pool is not None:
            try:
                self._pool.close(force=True)
            except Exception:  # noqa: BLE001 — best-effort teardown
                pass
            self._pool = None
        self._table_ready = False

    # -- table ---------------------------------------------------------------

    def ddl(self) -> str:
        """The ``CREATE TABLE`` this store runs, or expects a DBA to have run.

        ``created_at`` is stored in UTC without a zone, so what is written and
        what is read back do not depend on the session's time zone.
        """
        return (
            f"CREATE TABLE {self._full_table_name} (\n"
            "  report_id VARCHAR2(64) PRIMARY KEY,\n"
            "  suite VARCHAR2(255) NOT NULL,\n"
            "  fingerprint VARCHAR2(32),\n"
            "  created_at TIMESTAMP NOT NULL,\n"
            "  total_cases NUMBER NOT NULL,\n"
            "  passed NUMBER NOT NULL,\n"
            "  failed NUMBER NOT NULL,\n"
            "  not_graded NUMBER NOT NULL,\n"
            "  avg_score NUMBER NOT NULL,\n"
            "  report CLOB CHECK (report IS JSON)\n"
            ")"
        )

    def _ensure_table(self, conn: Any) -> None:
        if self._table_ready:
            return
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM all_tables WHERE table_name = UPPER(:name) "
                "AND owner = UPPER(NVL(:owner, SYS_CONTEXT('USERENV', 'CURRENT_SCHEMA')))",
                {"name": self.config.table_name, "owner": self.config.schema_name},
            )
            row = cur.fetchone()
            exists = bool(row and row[0])
            if not exists:
                if not self._auto_create:
                    raise RuntimeError(
                        f"table {self._full_table_name} does not exist and auto_create_table is off"
                    )
                cur.execute(self.ddl())
                cur.execute(
                    f"CREATE INDEX {self.config.table_name}_suite_idx "
                    f"ON {self._full_table_name} (suite, created_at)"
                )
                conn.commit()
        self._table_ready = True

    # -- the interface -------------------------------------------------------

    def put(self, report: EvalReport, *, suite: str, report_id: str | None = None) -> StoredReport:
        report_id = report_id or uuid4().hex
        body = report.model_dump_json()

        def _insert(conn: Any) -> None:
            self._ensure_table(conn)
            payload: Any = body
            if len(body) > _INLINE_LIMIT:
                import oracledb  # noqa: PLC0415

                payload = conn.createlob(oracledb.DB_TYPE_CLOB)
                payload.write(body)
            with conn.cursor() as cur:
                cur.execute(
                    f"INSERT INTO {self._full_table_name} "  # noqa: S608 — validated identifier
                    "(report_id, suite, fingerprint, created_at, total_cases, passed, failed, "
                    "not_graded, avg_score, report) VALUES (:report_id, :suite, :fingerprint, "
                    ":created_at, :total_cases, :passed, :failed, :not_graded, :avg_score, :report)",
                    {
                        "report_id": report_id,
                        "suite": suite,
                        "fingerprint": report.fingerprint,
                        "created_at": _utc_naive(report.created_at),
                        "total_cases": report.total_cases,
                        "passed": report.passed,
                        "failed": report.failed,
                        "not_graded": report.not_graded,
                        "avg_score": report.avg_score,
                        "report": payload,
                    },
                )
            conn.commit()

        self._run(_insert)
        return StoredReport.of(
            report,
            report_id=report_id,
            suite=suite,
            location=f"{self._full_table_name}:{report_id}",
        )

    def get(self, report_id: str) -> EvalReport | None:
        def _select(conn: Any) -> Any:
            self._ensure_table(conn)
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT report FROM {self._full_table_name} WHERE report_id = :id",  # noqa: S608
                    {"id": report_id},
                )
                return cur.fetchone()

        row = self._run(_select)
        if row is None:
            return None
        body = row[0]
        if hasattr(body, "read"):
            body = body.read()
        if isinstance(body, bytes):
            body = body.decode("utf-8")
        if isinstance(body, dict):
            return EvalReport.model_validate(body)
        return EvalReport.model_validate_json(body)

    def list(
        self, *, suite: str | None = None, fingerprint: str | None = None, limit: int = 20
    ) -> list[StoredReport]:
        clauses: list[str] = []
        binds: dict[str, Any] = {}
        if suite is not None:
            clauses.append("suite = :suite")
            binds["suite"] = suite
        if fingerprint is not None:
            clauses.append("fingerprint = :fingerprint")
            binds["fingerprint"] = fingerprint
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        binds["limit"] = max(0, limit)

        def _select(conn: Any) -> list[tuple[Any, ...]]:
            self._ensure_table(conn)
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT report_id, suite, fingerprint, created_at, total_cases, passed, "  # noqa: S608
                    f"failed, not_graded, avg_score FROM {self._full_table_name}{where} "
                    "ORDER BY created_at DESC, report_id DESC FETCH FIRST :limit ROWS ONLY",
                    binds,
                )
                return list(cur.fetchall())

        rows = self._run(_select)
        return [
            StoredReport(
                report_id=row[0],
                suite=row[1],
                fingerprint=row[2],
                created_at=_utc_aware(row[3]),
                total_cases=int(row[4]),
                passed=int(row[5]),
                failed=int(row[6]),
                not_graded=int(row[7]),
                avg_score=float(row[8]),
                location=f"{self._full_table_name}:{row[0]}",
            )
            for row in rows
        ]

    def __repr__(self) -> str:
        return f"OracleReportStore(table={self._full_table_name!r})"


# ---------------------------------------------------------------------------
# opening a store from a spec
# ---------------------------------------------------------------------------


def _options(spec: str) -> dict[str, str]:
    return {key: values[-1] for key, values in parse_qs(urlsplit(spec).query).items()}


def _oracle(spec: str) -> OracleReportStore:
    parts = urlsplit(spec)
    options = _options(spec)
    kwargs: dict[str, Any] = {}
    table = options.get("table")
    if table:
        schema, dot, name = table.rpartition(".")
        kwargs["table_name"] = name
        if dot:
            kwargs["schema_name"] = schema
    if "schema" in options:
        kwargs["schema_name"] = options["schema"]
    user = parts.username or options.get("user") or os.environ.get("ORACLE_USER", "admin")
    password = parts.password or options.get("password") or os.environ.get("ORACLE_PASSWORD", "")
    wallet = options.get("wallet") or os.environ.get("ORACLE_WALLET")
    wallet_password = options.get("wallet_password")
    if wallet_password is None and wallet and os.environ.get("ORACLE_WALLET_PASSWORD") is not None:
        wallet_password = os.environ.get("ORACLE_WALLET_PASSWORD")
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
    return OracleReportStore(
        user=user,
        password=password,
        wallet_location=wallet,
        wallet_password=wallet_password,
        **kwargs,
    )


def _oci(spec: str) -> OCIBucketReportStore:
    parts = urlsplit(spec)
    options = _options(spec)
    namespace = parts.netloc
    bucket, _, prefix = parts.path.lstrip("/").partition("/")
    if not namespace or not bucket:
        raise ValueError("an oci:// spec is oci://<namespace>/<bucket>/<prefix>")
    kwargs: dict[str, Any] = {}
    if prefix:
        kwargs["prefix"] = prefix if prefix.endswith("/") else prefix + "/"
    for key, option in (
        ("profile_name", "profile"),
        ("auth_type", "auth"),
        ("region", "region"),
        ("config_file", "config_file"),
    ):
        if option in options:
            kwargs[key] = options[option]
    return OCIBucketReportStore(bucket, namespace, **kwargs)


def _s3(spec: str) -> S3ReportStore:
    parts = urlsplit(spec)
    options = _options(spec)
    bucket = parts.netloc
    if not bucket:
        raise ValueError("an s3:// spec is s3://<bucket>/<prefix>")
    prefix = parts.path.lstrip("/")
    kwargs: dict[str, Any] = {}
    if prefix:
        kwargs["prefix"] = prefix if prefix.endswith("/") else prefix + "/"
    for key, option in (("endpoint_url", "endpoint_url"), ("region_name", "region")):
        if option in options:
            kwargs[key] = options[option]
    return S3ReportStore(bucket, **kwargs)


def report_store(spec: str | None = None) -> ReportStore:
    """The store a spec names (see the module); ``None`` reads ``LOCUS_EVAL_STORE``.

    Raises:
        ValueError: No spec, an unknown scheme, or a database spec that names
            neither a server nor an environment to find one in.
    """
    spec = spec or os.environ.get(SPEC_ENV, "")
    if not spec:
        raise ValueError(f"no report store: pass a spec or set {SPEC_ENV}")
    scheme = urlsplit(spec).scheme.lower() if "://" in spec else ""
    if scheme in ("", "file"):
        path = spec[len("file://") :] if scheme == "file" else spec
        return FileReportStore(path)
    if scheme == "oracle":
        return _oracle(spec)
    if scheme == "oci":
        return _oci(spec)
    if scheme == "s3":
        return _s3(spec)
    raise ValueError(
        f"unknown report store scheme {scheme!r}: expected file://, oracle://, oci:// or s3://"
    )


# ---------------------------------------------------------------------------
# the trend
# ---------------------------------------------------------------------------


class TrendPoint(BaseModel):
    """One report in the trend and how it moved against the one before it."""

    report_id: str
    created_at: datetime
    fingerprint: str | None = None
    total_cases: int = 0
    passed: int = 0
    failed: int = 0
    not_graded: int = 0
    avg_score: float = 0.0
    pass_rate: float = Field(default=0.0, description="Passed over graded cases")
    pass_delta: int | None = Field(default=None, description="Passed minus the previous point's")
    avg_score_delta: float | None = None
    suite_changed: bool = Field(
        default=False, description="The fingerprint differs from the previous point's"
    )

    @property
    def regressed(self) -> bool:
        """Fewer passed, or a lower average, than the previous point of the same suite."""
        if self.suite_changed or self.pass_delta is None or self.avg_score_delta is None:
            return False
        return self.pass_delta < 0 or self.avg_score_delta < -1e-9


class Trend(BaseModel):
    """The last reports of one suite, oldest first, each against the one before."""

    suite: str
    points: list[TrendPoint] = Field(default_factory=list)

    @property
    def latest(self) -> TrendPoint | None:
        return self.points[-1] if self.points else None

    @property
    def regressed(self) -> bool:
        """Whether the newest report is worse than the one before it."""
        return self.latest is not None and self.latest.regressed

    def summary(self) -> str:
        """A readable account, one line per report."""
        if not self.points:
            return f"Trend for {self.suite}: no reports"
        lines = [f"Trend for {self.suite}: {len(self.points)} report(s), oldest first", ""]
        for point in self.points:
            when = point.created_at.astimezone(UTC).strftime("%Y-%m-%d %H:%M")
            moved = ""
            if point.pass_delta is not None and point.avg_score_delta is not None:
                moved = f"  ({point.pass_delta:+d} passed, {point.avg_score_delta:+.2f} avg)"
            if point.suite_changed:
                moved += "  suite changed"
            lines.append(
                f"  {when}  {point.passed}/{point.total_cases} passed  avg {point.avg_score:.2f}"
                f"  {point.fingerprint or '-'}{moved}"
            )
        if self.regressed:
            lines.extend(["", "the newest report is worse than the one before it"])
        return "\n".join(lines)

    def to_markdown(self) -> str:
        """The trend as a Markdown table."""
        rows = [
            "| When | Passed | Avg score | Suite | Moved |",
            "|---|---|---|---|---|",
        ]
        for point in self.points:
            when = point.created_at.astimezone(UTC).strftime("%Y-%m-%d %H:%M")
            moved = ""
            if point.pass_delta is not None and point.avg_score_delta is not None:
                moved = f"{point.pass_delta:+d} / {point.avg_score_delta:+.2f}"
            if point.suite_changed:
                moved = (moved + " · suite changed").strip(" ·")
            rows.append(
                f"| {when} | {point.passed}/{point.total_cases} | {point.avg_score:.2f} | "
                f"`{point.fingerprint or '-'}` | {moved} |"
            )
        return "\n".join(rows)


def trend(
    store: ReportStore,
    *,
    suite: str,
    last: int = 10,
    fingerprint: str | None = None,
) -> Trend:
    """The last ``last`` reports of ``suite`` from ``store``, each against the one before.

    ``fingerprint`` keeps the trend to one version of the suite; without it,
    a point whose fingerprint differs from the previous one is marked
    ``suite_changed`` and its deltas are still shown, so a jump can be read
    for what it is.
    """
    stored = store.list(suite=suite, fingerprint=fingerprint, limit=max(1, last))
    return trend_of(stored, suite=suite)


def trend_of(stored: Sequence[StoredReport], *, suite: str) -> Trend:
    """A :class:`Trend` from report rows already listed (newest first or any order)."""
    ordered = sorted(stored, key=lambda item: (item.created_at, item.report_id))
    points: list[TrendPoint] = []
    previous: StoredReport | None = None
    for item in ordered:
        graded = item.total_cases - item.not_graded
        point = TrendPoint(
            report_id=item.report_id,
            created_at=item.created_at,
            fingerprint=item.fingerprint,
            total_cases=item.total_cases,
            passed=item.passed,
            failed=item.failed,
            not_graded=item.not_graded,
            avg_score=item.avg_score,
            pass_rate=item.passed / graded if graded else 0.0,
        )
        if previous is not None:
            point.pass_delta = item.passed - previous.passed
            point.avg_score_delta = round(item.avg_score - previous.avg_score, 4)
            point.suite_changed = item.fingerprint != previous.fingerprint
        points.append(point)
        previous = item
    return Trend(suite=suite, points=points)
