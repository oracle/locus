# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""An OCI GenAI Sandbox as a workspace.

OCI Enterprise AI sandboxes (``pip install "locus-sdk[oci-sandbox]"``, the
``oci`` SDK from 2.189.0) are isolated, expiring Linux runtimes that belong to
a Generative AI project. The SDK's ``GenerativeAiSandboxClient`` can run a
command and read and write a file, which is exactly
:class:`~locus.harness.session.SessionLike`, so :class:`OCISandboxSession`
is a thin adapter and everything else is
:class:`~locus.harness.session.SessionBackend`.

Four facts about the service shape the adapter (measured against the GA
service in ``eu-frankfurt-1``):

- **Commands stream.** ``run_sandbox_command_and_stream`` answers in about a
  second; ``run_sandbox_command_and_wait`` takes about three. Events carry a
  sequence number, so standard output and standard error are joined in the
  order the command wrote them.
- **The service's timeout does not end the stream.** A command past its
  deadline is reported with exit code ``-1`` only when it would have
  finished anyway. So every command runs under ``timeout`` inside the
  sandbox, which stops its process group at the deadline (status 124); the
  service's own timeout is a backstop a little later.
- **Large output is lost.** Somewhere above a megabyte of output a command
  returns nothing at all. Every command's output therefore goes to a file in
  the sandbox, and at most ``max_output_bytes`` of it come back: the head and
  the most recent tail, with the same "dropped here" note
  :class:`~locus.harness.session.SessionBackend` uses.
- **Only ``/workspace`` is writable.** ``/tmp`` is read-only, so background
  commands keep their state in ``/workspace/.locus-harness``
  (:data:`~locus.harness.backend.SKIP_DIRS` keeps it out of recursive walks).

Files move with the file API: writes carry a base64 SHA-256 of the body, and
a write larger than :data:`UPLOAD_CHUNK` goes in parts joined in the sandbox,
because the service refuses very large bodies.

The backend needs the ``oci`` package (2.189.0 or later) installed: every
command and every lifecycle call builds one of the SDK's request models
(``RunSandboxCommandDetails`` and the like). It is imported lazily, on first
use, so importing :mod:`locus.harness` never needs it.
:meth:`OCISandboxBackend.connect` and :meth:`OCISandboxBackend.create` also
build the client and its authentication. To configure the client yourself,
pass a ``GenerativeAiSandboxClient`` (or anything with the methods
:class:`OCISandboxClient` names) to the constructor.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import re
import shlex
import time
import uuid
from collections.abc import Callable, Iterable, Mapping
from typing import Any, Protocol

from locus.harness.local import HEAD_KEEP, MAX_BACKGROUND
from locus.harness.session import SessionBackend


__all__ = ["OCISandboxBackend", "OCISandboxClient", "OCISandboxSession"]

#: The workspace directory of an OCI sandbox; the only writable one.
SANDBOX_ROOT = "/workspace"

#: Where background commands and captured output live inside the sandbox.
STATE_DIR = "/workspace/.locus-harness"

#: The most output one command returns. The service loses all output a
#: little above one megabyte; this stays well under it.
MAX_OUTPUT_BYTES = 900_000

#: Bytes per file write. The service took 10 MB and refused 50 MB.
UPLOAD_CHUNK = 8 * 1024 * 1024

#: Seconds the sandbox gives a command between the stop signal and a kill.
_KILL_GRACE = 5

#: Seconds the service's own timeout runs past the in-sandbox one.
_SERVICE_GRACE = 30

#: The longest lifetime a sandbox may be created with: 48 hours.
MAX_EXPIRATION_MINUTES = 48 * 60

_ISO_DURATION = re.compile(
    r"^P(?:(?P<d>\d+)D)?(?:T(?:(?P<h>\d+)H)?(?:(?P<m>\d+)M)?(?:(?P<s>\d+)S)?)?$"
)

#: States a sandbox does not come back from to ``RUNNING``.
_TERMINAL = frozenset({"FAILED", "STOPPING", "STOPPED", "DELETING", "DELETED"})

_q = shlex.quote


class OCISandboxClient(Protocol):
    """The part of ``oci.generative_ai_sandbox.GenerativeAiSandboxClient`` the session uses."""

    def run_sandbox_command_and_stream(
        self, sandbox_id: str, run_sandbox_command_details: Any, **kwargs: Any
    ) -> Any:
        """Run a command; the response's ``data`` yields NDJSON command events."""
        ...

    def write_sandbox_file(
        self,
        sandbox_id: str,
        file_path: str,
        write_sandbox_file_content_details: Any,
        **kwargs: Any,
    ) -> Any:
        """Create or replace ``file_path`` with the bytes of a binary stream."""
        ...

    def read_sandbox_file(self, sandbox_id: str, file_path: str, **kwargs: Any) -> Any:
        """Read ``file_path``; the response's ``data`` carries its bytes."""
        ...


def _models() -> Any:
    """The SDK's request models; imported on first use."""
    try:
        from oci.generative_ai_sandbox import models
    except ImportError as exc:
        raise ImportError(
            "OCISandboxBackend needs the OCI SDK with GenAI Sandboxes (oci>=2.189.0): "
            'pip install "locus-sdk[oci-sandbox]"'
        ) from exc
    return models


def _duration(seconds: float) -> str:
    """Whole seconds as an ISO 8601 duration, at least one second."""
    return f"PT{max(1, math.ceil(seconds))}S"


def _minutes(expiration: str) -> int:
    """An ISO 8601 duration as whole minutes, rounded up; between one minute and 48 hours.

    The service takes a sandbox's lifetime as ``expiresInMinutes``; a value
    that never expires (``-1``) is refused, because the expiry is the backstop
    for a sandbox a crashed process leaves behind.
    """
    match = _ISO_DURATION.match(expiration.strip().upper()) if isinstance(expiration, str) else None
    if match is None:
        raise ValueError(
            f"expiration must be an ISO 8601 duration such as 'PT1H', got {expiration!r}"
        )
    days, hours, minutes, seconds = (int(match.group(k) or 0) for k in ("d", "h", "m", "s"))
    total = math.ceil((((days * 24 + hours) * 60 + minutes) * 60 + seconds) / 60)
    if not 1 <= total <= MAX_EXPIRATION_MINUTES:
        raise ValueError(f"expiration must be between one minute and 48 hours, got {expiration!r}")
    return total


def _lines(data: Any) -> Iterable[bytes | str]:
    """The NDJSON lines of a streamed response, whatever shape the SDK hands back."""
    if hasattr(data, "iter_lines"):
        return data.iter_lines()  # type: ignore[no-any-return]
    raw = getattr(data, "raw", None)
    if raw is not None:
        return raw.read().splitlines()  # type: ignore[no-any-return]
    if isinstance(data, bytes | str):
        return data.splitlines()
    return data  # type: ignore[no-any-return]


def _content(data: Any) -> bytes:
    """The bytes of a ``read_sandbox_file`` response."""
    content = getattr(data, "content", None)
    if content is None:
        raw = getattr(data, "raw", None)
        content = raw.read() if raw is not None else data
    if isinstance(content, str):
        return content.encode()
    return bytes(content)


def _check_output_limit(max_output_bytes: int) -> None:
    """The output cap must leave room for a tail after the kept head."""
    if max_output_bytes <= HEAD_KEEP:
        raise ValueError(f"max_output_bytes must exceed {HEAD_KEEP}, got {max_output_bytes}")


def _status(exc: BaseException) -> int | None:
    status = getattr(exc, "status", None)
    return status if isinstance(status, int) else None


class OCISandboxSession:
    """One OCI GenAI Sandbox as a :class:`~locus.harness.session.SessionLike`.

    Args:
        client: A ``GenerativeAiSandboxClient``, or anything with the methods
            :class:`OCISandboxClient` names.
        sandbox_id: The sandbox's id.
        state_dir: Where captured output is written inside the sandbox.
        max_output_bytes: The most output one command returns.
    """

    def __init__(
        self,
        client: OCISandboxClient,
        sandbox_id: str,
        *,
        state_dir: str = STATE_DIR,
        max_output_bytes: int = MAX_OUTPUT_BYTES,
    ) -> None:
        _check_output_limit(max_output_bytes)
        self.client = client
        self.sandbox_id = sandbox_id
        self.state_dir = state_dir
        self.max_output_bytes = max_output_bytes

    def _script(self, command: str, seconds: int) -> str:
        """``command`` under an in-sandbox deadline, its output captured and capped."""
        head = HEAD_KEEP
        tail = self.max_output_bytes - HEAD_KEEP
        return (
            f'd={_q(self.state_dir)}; mkdir -p "$d" || exit 125; f="$d/out-$$"; '
            f'timeout -k {_KILL_GRACE} {seconds} sh -c {_q(command)} > "$f" 2>&1 < /dev/null; '
            'rc=$?; n=$(wc -c < "$f" | tr -d " "); '
            f'if [ "$n" -gt {self.max_output_bytes} ]; then head -c {head} "$f"; '
            f'printf "\\n... %s bytes of output were dropped here ...\\n" $((n - {head + tail})); '
            f'tail -c {tail} "$f"; else cat "$f"; fi; rm -f "$f"; exit $rc'
        )

    def run(
        self,
        command: str,
        *,
        timeout: float,
        on_output: Callable[[bytes], None] | None = None,
    ) -> tuple[int | None, bytes]:
        """Run ``command`` with ``sh -c``; its status and its (capped) output.

        The command runs under ``timeout`` inside the sandbox, its output goes
        to a file there, and at most ``max_output_bytes`` come back.

        Args:
            command: A POSIX shell command line.
            timeout: Seconds before the sandbox stops the command.
            on_output: Called with each chunk of output, in order.

        Returns:
            The exit status and the output. The status is ``None`` when the
            command did not finish: it ran past ``timeout``, or the service
            ended it without a status.

        Raises:
            Exception: The transport failed before ``timeout`` elapsed.
        """
        seconds = max(1, math.ceil(timeout))
        details = _models().RunSandboxCommandDetails(
            command=self._script(command, seconds),
            timeout=_duration(seconds + _KILL_GRACE + _SERVICE_GRACE),
        )
        started = time.monotonic()
        try:
            response = self.client.run_sandbox_command_and_stream(self.sandbox_id, details)
            events = [json.loads(line) for line in _lines(response.data) if line]
        except Exception:
            # Past the deadline a dropped stream is a timeout; before it, the
            # transport failed, and that is the caller's to hear about.
            if time.monotonic() - started < timeout:
                raise
            return None, b""
        events.sort(key=lambda event: event.get("seq") or 0)
        parts: list[bytes] = []
        code: int | None = None
        for event in events:
            kind = event.get("event")
            if kind in ("STDOUT", "STDERR"):
                chunk = str(event.get("data") or "").encode()
                parts.append(chunk)
                if on_output is not None and chunk:
                    on_output(chunk)
            elif kind == "FINISHED":
                raw = event.get("exitCode", event.get("exit_code"))
                code = raw if isinstance(raw, int) and raw >= 0 else None
        if code == 128 + 9 and time.monotonic() - started >= timeout:
            code = None  # killed after the grace period: still a timeout
        return code, b"".join(parts)

    def exec(self, command: str, *, timeout: float) -> tuple[int | None, bytes]:
        """Run ``command`` with ``sh -c``; the :class:`SessionLike` verb.

        Args:
            command: A POSIX shell command line.
            timeout: Seconds before the sandbox stops it.

        Returns:
            The exit status (``None`` when it did not finish) and the
            command's standard output and standard error, joined and capped
            at ``max_output_bytes``.
        """
        return self.run(command, timeout=timeout)

    def upload_file(self, path: str, data: bytes) -> None:
        """Create or replace the file at ``path`` with ``data``.

        Each write carries a base64 SHA-256 of its body, which the service
        checks. Data larger than :data:`UPLOAD_CHUNK` is written in parts
        next to ``path`` and joined with ``cat`` in the sandbox, which keeps
        an existing file's mode.

        Args:
            path: An absolute path under ``/workspace``.
            data: The file's new contents.

        Raises:
            OSError: The parts of a large upload could not be joined; they
                are removed.
            Exception: The service refused a write (the SDK's
                ``ServiceError``).
        """
        if len(data) <= UPLOAD_CHUNK:
            self._write(path, data)
            return
        parts = []
        for index, start in enumerate(range(0, len(data), UPLOAD_CHUNK)):
            part = f"{path}.locus-part-{index:04d}"
            self._write(part, data[start : start + UPLOAD_CHUNK])
            parts.append(part)
        quoted = " ".join(_q(p) for p in parts)
        code, out = self.run(f"cat {quoted} > {_q(path)} && rm -f {quoted}", timeout=120)
        if code != 0:
            self.run(f"rm -f {quoted}", timeout=60)
            raise OSError(f"upload of {path} failed: {out.decode(errors='replace').strip()}")

    def _write(self, path: str, data: bytes) -> None:
        checksum = base64.b64encode(hashlib.sha256(data).digest()).decode()
        self.client.write_sandbox_file(
            self.sandbox_id, path, io.BytesIO(data), opc_content_sha256=checksum
        )

    def download_file(self, path: str) -> bytes:
        """The contents of the file at ``path``.

        Args:
            path: An absolute path under ``/workspace``.

        Returns:
            The file's bytes.

        Raises:
            FileNotFoundError: The service answered 404: no such file, or a
                directory.
            Exception: Any other service error, unchanged.
        """
        try:
            response = self.client.read_sandbox_file(self.sandbox_id, path)
        except Exception as exc:
            if _status(exc) == 404:
                raise FileNotFoundError(path) from exc
            raise
        return _content(response.data)


class OCISandboxBackend(SessionBackend):
    """An OCI GenAI Sandbox as a workspace. Isolated; rooted at ``/workspace``.

    Build one around a sandbox that exists with :meth:`connect`, have one
    created with :meth:`create`, or pass a client of your own here.

    Args:
        client: A ``GenerativeAiSandboxClient`` (or anything with the methods
            :class:`OCISandboxClient` names, plus ``stop_sandbox``,
            ``get_sandbox`` and ``delete_sandbox`` if ``owns_sandbox``).
        sandbox_id: The sandbox's id.
        region: The sandbox's region, for the label.
        root: The workspace directory inside the sandbox.
        owns_sandbox: Stop and delete the sandbox on :meth:`close`. Set by
            :meth:`create`.
        env: Variables added to every command's environment.
        max_background: How many background commands may run at once.
        state_dir: Where background commands and captured output live,
            inside the sandbox. It must be writable: under ``/workspace``.
        max_output_bytes: The most output one command returns.
    """

    def __init__(
        self,
        client: OCISandboxClient,
        sandbox_id: str,
        *,
        region: str | None = None,
        root: str = SANDBOX_ROOT,
        owns_sandbox: bool = False,
        env: Mapping[str, str] | None = None,
        max_background: int = MAX_BACKGROUND,
        state_dir: str = STATE_DIR,
        max_output_bytes: int = MAX_OUTPUT_BYTES,
    ) -> None:
        self._oci = OCISandboxSession(
            client, sandbox_id, state_dir=state_dir, max_output_bytes=max_output_bytes
        )
        where = f"{region}/{sandbox_id}" if region else sandbox_id
        super().__init__(
            self._oci,
            root=root,
            label=f"oci-sandbox:{where}",
            isolated=True,
            state_dir=state_dir,
            env=env,
            max_background=max_background,
        )
        self._owns_sandbox = owns_sandbox
        self._jobs_stopped = False
        self._sandbox_deleted = not owns_sandbox

    @property
    def client(self) -> OCISandboxClient:
        """The sandbox client every call goes through."""
        return self._oci.client

    @property
    def sandbox_id(self) -> str:
        """The id of the sandbox this backend works in."""
        return self._oci.sandbox_id

    def _exec_script(
        self,
        script: str,
        timeout: float,
        on_output: Callable[[bytes], None] | None,
    ) -> tuple[int | None, bytes]:
        return self._oci.run(script, timeout=timeout, on_output=on_output)

    def extend(self, duration: str) -> Any:
        """Add ``duration`` to the sandbox's lifetime.

        Args:
            duration: An ISO 8601 duration, e.g. ``"PT30M"``.

        Returns:
            The new expiry time the service reports.
        """
        details = _models().ExtendSandboxDetails(additional_duration=duration)
        sandbox = self.client.extend_sandbox(self.sandbox_id, details).data  # type: ignore[attr-defined]
        return getattr(sandbox, "time_expires", None)

    def close(self) -> None:
        """Stop background commands; stop and delete the sandbox if this backend made it.

        Safe to call again: a step that failed (a transient service error,
        say) is retried by the next call, and one that succeeded is not
        repeated. The sandbox is deleted even when its background commands
        could not be stopped.

        Raises:
            Exception: The error of a step that failed, after every pending
                step was attempted (the delete's, when both failed).
        """
        try:
            if not self._jobs_stopped:
                super().close()
                self._jobs_stopped = True
        finally:
            if not self._sandbox_deleted:
                _stop_and_delete(self.client, self.sandbox_id)
                self._sandbox_deleted = True

    @classmethod
    def connect(
        cls,
        sandbox_id: str,
        *,
        region: str | None = None,
        profile: str = "DEFAULT",
        auth_type: str = "api_key",
        service_endpoint: str | None = None,
        root: str = SANDBOX_ROOT,
        env: Mapping[str, str] | None = None,
        **options: Any,
    ) -> OCISandboxBackend:
        """A backend over a sandbox that already exists. It is never deleted.

        Running commands needs only ``use generative-ai-sandbox``.

        Args:
            sandbox_id: The sandbox's id.
            region: Its region; the profile's when omitted.
            profile: The ``~/.oci/config`` profile, for ``api_key`` and
                ``security_token`` auth.
            auth_type: ``api_key``, ``security_token``,
                ``instance_principal`` or ``resource_principal``.
            service_endpoint: Overrides the regional inference endpoint.
            root: The workspace directory inside the sandbox.
            env: Variables added to every command's environment.
            options: Passed to the backend (``state_dir``,
                ``max_output_bytes``, ``max_background``).
        """
        client, resolved = _client(profile, auth_type, region, service_endpoint)
        return cls(client, sandbox_id, region=resolved, root=root, env=env, **options)

    @classmethod
    def create(
        cls,
        project_id: str,
        *,
        region: str | None = None,
        profile: str = "DEFAULT",
        auth_type: str = "api_key",
        service_endpoint: str | None = None,
        display_name: str | None = None,
        runtime: str = "python-3.11",
        shape: str = "SMALL",
        expiration: str = "PT1H",
        warmpool_name: str | None = None,
        ready_timeout: float = 300.0,
        poll_interval: float = 2.0,
        root: str = SANDBOX_ROOT,
        env: Mapping[str, str] | None = None,
        **options: Any,
    ) -> OCISandboxBackend:
        """Create a sandbox in ``project_id``, wait until it runs, and return a backend over it.

        The backend owns the sandbox: :meth:`close` stops and deletes it. The
        sandbox also expires on its own after ``expiration`` (an ISO 8601
        duration of at most 48 hours, sent to the service as whole minutes),
        so one a crashed process left behind does not run on; a lifetime that
        never ends is refused.
        Needs ``manage generative-ai-sandbox``.

        Raises:
            ValueError: An option is invalid; checked before a sandbox exists.
            RuntimeError: The sandbox failed, or did not run within
                ``ready_timeout``. Whatever fails once the sandbox exists,
                it is stopped and deleted before the error propagates.
        """
        _check_output_limit(options.get("max_output_bytes", MAX_OUTPUT_BYTES))
        minutes = _minutes(expiration)
        client, resolved = _client(profile, auth_type, region, service_endpoint)
        models = _models()
        details = models.CreateSandboxDetails(
            display_name=display_name or f"locus-harness-{uuid.uuid4().hex[:8]}",
            runtime=runtime,
            shape=shape,
            expires_in_minutes=minutes,
            warmpool_name=warmpool_name,
        )
        sandbox_id = client.create_sandbox(project_id, details).data.id
        try:
            _wait_running(client, sandbox_id, ready_timeout, poll_interval)
            return cls(
                client,
                sandbox_id,
                region=resolved,
                root=root,
                env=env,
                owns_sandbox=True,
                **options,
            )
        except BaseException:
            # Whatever failed after the sandbox exists, it must not outlive us.
            _stop_and_delete(client, sandbox_id)
            raise


def _wait_running(client: Any, sandbox_id: str, timeout: float, interval: float) -> None:
    deadline = time.monotonic() + timeout
    state = "UNKNOWN"
    while time.monotonic() < deadline:
        state = str(client.get_sandbox(sandbox_id).data.lifecycle_state).upper()
        if state == "RUNNING":
            return
        if state in _TERMINAL:
            raise RuntimeError(f"OCI sandbox {sandbox_id} is {state}, not RUNNING")
        time.sleep(interval)
    raise RuntimeError(f"OCI sandbox {sandbox_id} did not run within {timeout:g}s (last: {state})")


def _stop_and_delete(client: Any, sandbox_id: str, stop_timeout: float = 120.0) -> None:
    """Stop the sandbox, wait for ``STOPPED`` for a while, delete it. Gone already is fine."""
    try:
        state = str(client.get_sandbox(sandbox_id).data.lifecycle_state).upper()
        if state in {"DELETING", "DELETED"}:
            return
        if state not in {"STOPPED", "STOPPING"}:
            client.stop_sandbox(sandbox_id, _models().StopSandboxDetails(is_force=False))
        deadline = time.monotonic() + stop_timeout
        while state != "STOPPED" and time.monotonic() < deadline:
            time.sleep(2.0)
            state = str(client.get_sandbox(sandbox_id).data.lifecycle_state).upper()
        client.delete_sandbox(sandbox_id)
    except Exception as exc:
        if _status(exc) != 404:
            raise


def _client(
    profile: str, auth_type: str, region: str | None, service_endpoint: str | None
) -> tuple[Any, str | None]:
    """A ``GenerativeAiSandboxClient`` and its region; the only place the SDK is imported."""
    try:
        import oci
        from oci.generative_ai_sandbox import GenerativeAiSandboxClient
    except ImportError as exc:
        raise ImportError(
            "OCISandboxBackend needs the OCI SDK with GenAI Sandboxes (oci>=2.189.0): "
            'pip install "locus-sdk[oci-sandbox]"'
        ) from exc
    from locus.tools.oci import _build_config_and_signer

    config, signer = _build_config_and_signer(oci, profile, auth_type, region)
    kwargs: dict[str, Any] = {}
    if signer is not None:
        kwargs["signer"] = signer
    if service_endpoint:
        kwargs["service_endpoint"] = service_endpoint
    resolved = config.get("region")
    return GenerativeAiSandboxClient(config, **kwargs), resolved
