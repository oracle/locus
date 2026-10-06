# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""An NVIDIA OpenShell sandbox as a workspace.

OpenShell (``pip install "tulip-agents[openshell]"``, the ``openshell``
package on PyPI) runs agents in sandboxes behind a gateway, scoped to a
workspace. Its client can run a command — ``SandboxClient.exec_stream``,
which yields ``ExecChunk`` s as the command writes and an ``ExecResult`` at
the end — and that is all it offers for the sandbox's contents: there is no
file RPC. So :class:`OpenShellSession` moves files over ``exec`` too:

- **Download** is ``tar -c`` of the one file to standard output, read back
  as an archive. The archive says whether the path was a directory, and the
  bytes are exact.
- **Upload** is ``cat >`` with the bytes on standard input. Not ``tar -x``:
  extracting replaces the file with the archive's mode, and an edit to an
  executable script would quietly make it unexecutable. Large files go in
  chunks, so no single request approaches the gateway's message size limit.

Everything else is :class:`~tulip.harness.session.SessionBackend`. Commands
run with ``execution_timeout`` set, so the gateway, not this process, stops
a command that runs too long (it reports exit status 124).

The ``openshell`` package is imported only by :meth:`OpenShellBackend.connect`
and :meth:`OpenShellBackend.create`, the two constructors that build a client
for you. Pass a client of your own — a configured ``SandboxClient``, or
anything with the same methods — and nothing from it is imported at all.
"""

from __future__ import annotations

import io
import math
import posixpath
import shlex
import tarfile
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Protocol

from tulip.harness.local import MAX_BACKGROUND
from tulip.harness.session import SessionBackend


__all__ = ["OpenShellBackend", "OpenShellClient", "OpenShellSession"]

#: The workspace directory in an OpenShell sandbox.
SANDBOX_ROOT = "/sandbox"

#: Bytes per upload request. Well under gRPC's default 4 MiB message limit,
#: with room for the request's other fields.
UPLOAD_CHUNK = 2 * 1024 * 1024

#: Seconds an internal operation may take when the caller set no deadline.
_DEFAULT_TIMEOUT = 60.0

_q = shlex.quote


class OpenShellClient(Protocol):
    """The part of ``openshell.SandboxClient`` this module uses."""

    def exec_stream(
        self,
        sandbox: str,
        command: Sequence[str],
        *,
        workspace: str,
        workdir: str | None = None,
        env: Mapping[str, str] | None = None,
        stdin: bytes | None = None,
        timeout_seconds: int | None = None,
    ) -> Iterable[Any]: ...


class OpenShellSession:
    """One OpenShell sandbox as a :class:`~tulip.harness.session.SessionLike`.

    Args:
        client: An ``openshell.SandboxClient``, or anything with its
            ``exec_stream``.
        sandbox: The sandbox's name.
        workspace: The OpenShell workspace the sandbox belongs to. Every call
            is scoped to it.
    """

    def __init__(self, client: OpenShellClient, sandbox: str, *, workspace: str) -> None:
        self.client = client
        self.sandbox = sandbox
        self.workspace = workspace

    def run(
        self,
        command: str,
        *,
        timeout: float,
        stdin: bytes | None = None,
        on_output: Callable[[bytes], None] | None = None,
    ) -> tuple[int | None, bytes]:
        """Run ``command`` with ``sh -c``; its status and its output.

        Standard output and standard error arrive as separate streams; they
        are joined in the order the gateway delivered them.
        """
        parts: list[bytes] = []
        code: int | None = None
        for item in self.client.exec_stream(
            self.sandbox,
            ["sh", "-c", command],
            workspace=self.workspace,
            stdin=stdin,
            timeout_seconds=max(1, math.ceil(timeout)),
        ):
            data = getattr(item, "data", None)
            if isinstance(data, bytes):
                parts.append(data)
                if on_output is not None:
                    on_output(data)
            elif hasattr(item, "exit_code"):
                code = int(item.exit_code)
        return code, b"".join(parts)

    def exec(self, command: str, *, timeout: float) -> tuple[int | None, bytes]:
        return self.run(command, timeout=timeout)

    def upload_file(self, path: str, data: bytes) -> None:
        directory = posixpath.dirname(path)
        chunks = [data[i : i + UPLOAD_CHUNK] for i in range(0, len(data), UPLOAD_CHUNK)] or [b""]
        for index, chunk in enumerate(chunks):
            op = ">" if index == 0 else ">>"
            code, out = self.run(
                f"mkdir -p {_q(directory)} && cat {op} {_q(path)}",
                timeout=_DEFAULT_TIMEOUT,
                stdin=chunk,
            )
            if code != 0:
                raise OSError(f"upload of {path} failed: {out.decode(errors='replace').strip()}")

    def download_file(self, path: str) -> bytes:
        directory, name = posixpath.split(path)
        # stdout only: tar's own warnings on stderr would corrupt the archive.
        code, out = self._stdout(
            f"tar -chf - -C {_q(directory or '/')} -- {_q(name)} 2>/dev/null",
        )
        if code != 0:
            raise FileNotFoundError(path)
        with tarfile.open(fileobj=io.BytesIO(out), mode="r:") as archive:
            member = archive.next()
            if member is None:
                raise FileNotFoundError(path)
            if member.isdir():
                raise IsADirectoryError(path)
            # ``-h`` followed links, so anything but a regular file here is a
            # socket, a device or a dangling link: nothing to read.
            handle = archive.extractfile(member) if member.isfile() else None
            if handle is None:
                raise FileNotFoundError(path)
            return handle.read()

    def _stdout(self, command: str) -> tuple[int | None, bytes]:
        parts: list[bytes] = []
        code: int | None = None
        for item in self.client.exec_stream(
            self.sandbox,
            ["sh", "-c", command],
            workspace=self.workspace,
            timeout_seconds=int(_DEFAULT_TIMEOUT),
        ):
            if getattr(item, "stream", None) == "stdout":
                parts.append(item.data)
            elif hasattr(item, "exit_code"):
                code = int(item.exit_code)
        return code, b"".join(parts)


class OpenShellBackend(SessionBackend):
    """An OpenShell sandbox as a workspace. Isolated; rooted at ``/sandbox``.

    Build one around a sandbox that exists with :meth:`connect`, have one
    created with :meth:`create`, or pass a client of your own here.

    Args:
        client: An ``openshell.SandboxClient`` (or anything with its
            ``exec_stream``, plus ``delete`` if ``owns_sandbox``).
        sandbox: The sandbox's name.
        workspace: The OpenShell workspace it belongs to.
        root: The workspace directory inside the sandbox.
        owns_sandbox: Delete the sandbox on :meth:`close`. Set by
            :meth:`create`.
        env: Variables added to every command's environment.
        max_background: How many background commands may run at once.
        state_dir: Where background commands keep their state, inside the
            sandbox.
    """

    def __init__(
        self,
        client: OpenShellClient,
        sandbox: str,
        *,
        workspace: str = "default",
        root: str = SANDBOX_ROOT,
        owns_sandbox: bool = False,
        env: Mapping[str, str] | None = None,
        max_background: int = MAX_BACKGROUND,
        state_dir: str = "/tmp/.tulip-harness",  # noqa: S108 - inside the sandbox
    ) -> None:
        self._openshell = OpenShellSession(client, sandbox, workspace=workspace)
        super().__init__(
            self._openshell,
            root=root,
            label=f"openshell:{workspace}/{sandbox}",
            isolated=True,
            state_dir=state_dir,
            env=env,
            max_background=max_background,
        )
        self._owns_sandbox = owns_sandbox

    @property
    def client(self) -> OpenShellClient:
        return self._openshell.client

    def _exec_script(
        self,
        script: str,
        timeout: float,
        on_output: Callable[[bytes], None] | None,
    ) -> tuple[int | None, bytes]:
        parts: list[bytes] = []

        def sink(chunk: bytes) -> None:
            parts.append(chunk)
            if on_output is not None:
                on_output(chunk)

        started = time.monotonic()
        try:
            code, _ = self._openshell.run(script, timeout=timeout, on_output=sink)
        except Exception:
            # A gRPC deadline that lapses first ends the stream with an error
            # rather than an exit event. Past the command's own deadline that
            # is a timeout, and what it printed so far is kept; before it, the
            # transport failed, and that is the caller's to hear about.
            if time.monotonic() - started < timeout:
                raise
            return None, b"".join(parts)
        return code, b"".join(parts)

    def close(self) -> None:
        """Stop background commands; delete the sandbox if this backend made it."""
        super().close()
        if self._owns_sandbox:
            delete = getattr(self.client, "delete", None)
            if delete is not None:
                delete(
                    self._openshell.sandbox, workspace=self._openshell.workspace, allow_missing=True
                )

    @classmethod
    def connect(
        cls,
        sandbox: str,
        *,
        workspace: str = "default",
        endpoint: str | None = None,
        root: str = SANDBOX_ROOT,
        env: Mapping[str, str] | None = None,
        **client_options: Any,
    ) -> OpenShellBackend:
        """A backend over a sandbox that already exists.

        Args:
            sandbox: The sandbox's name.
            workspace: Its OpenShell workspace.
            endpoint: The gateway's ``host:port``. Without one, the client is
                built from the active gateway's local configuration
                (``SandboxClient.from_active_cluster``).
            root: The workspace directory inside the sandbox.
            env: Variables added to every command's environment.
            client_options: Passed to ``SandboxClient`` — ``tls``,
                ``bearer_token``, ``client_credentials``, ``timeout``.
        """
        return cls(
            _client(endpoint, client_options), sandbox, workspace=workspace, root=root, env=env
        )

    @classmethod
    def create(
        cls,
        *,
        workspace: str = "default",
        name: str | None = None,
        endpoint: str | None = None,
        spec: Any = None,
        labels: Mapping[str, str] | None = None,
        ready_timeout: float = 300.0,
        root: str = SANDBOX_ROOT,
        env: Mapping[str, str] | None = None,
        **client_options: Any,
    ) -> OpenShellBackend:
        """Create a sandbox, wait until it is ready, and return a backend over it.

        The backend owns the sandbox: :meth:`close` deletes it.
        """
        client = _client(endpoint, client_options)
        ref = client.create(workspace=workspace, name=name, spec=spec, labels=labels)
        client.wait_ready(ref.name, workspace=workspace, timeout_seconds=ready_timeout)
        return cls(client, ref.name, workspace=workspace, root=root, env=env, owns_sandbox=True)


def _client(endpoint: str | None, options: Mapping[str, Any]) -> Any:
    """An ``openshell.SandboxClient``; the only place the package is imported."""
    try:
        import openshell
    except ImportError as exc:
        raise ImportError(
            'OpenShellBackend needs the openshell package: pip install "tulip-agents[openshell]"'
        ) from exc
    if endpoint is None:
        return openshell.SandboxClient.from_active_cluster(**options)
    return openshell.SandboxClient(endpoint, **options)
