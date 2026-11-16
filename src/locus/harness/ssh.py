# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Any host that runs ``sshd`` as a workspace.

A machine you can SSH into is already most of a sandbox: it runs commands
and moves files, and it is not the machine the harness runs on. ``pip
install "locus-sdk[ssh]"`` adds the ``paramiko`` client, and
:class:`SSHSandboxBackend` makes the host a
:class:`~locus.harness.sandbox.SandboxBackend`. It is also the transport
under :class:`~locus.harness.oci_container.OCIContainerInstanceBackend`,
which provisions the host.

What the backend asks of the host is what any Linux image has — ``sh``,
coreutils or busybox, ``find``, ``grep``, ``awk`` — plus ``sshd`` with the
``sftp`` subsystem, and falls back to ``cat`` over ``exec`` when SFTP is not
there. Three things are decided here rather than left to the host:

- **A command is stopped at its deadline, where it runs.** Closing an SSH
  channel does not kill the process behind it, so every command runs under
  a watchdog on the host: the command in its own process group (``setsid``
  where util-linux has it, POSIX job control elsewhere), a sibling that
  sends the group ``TERM`` at the deadline and ``KILL`` two seconds later.
  A command the watchdog stopped reports exit status 124; the client closes
  the channel a few seconds after that if the host has not.
- **Both streams, in order.** The command's standard error is joined to its
  standard output on the host, so a warning sits where it was written, not
  after everything else. The watchdog's own notices go to the channel's
  separate error stream and are dropped.
- **Host keys are checked.** By default a host must be in the system's or
  the user's ``known_hosts``; ``host_key=`` pins one key for a host whose
  key arrived another way (a container that printed it to its log), and
  ``known_hosts="accept-unknown"`` is the explicit opt-out for a sandbox
  with no other channel.

``paramiko`` is imported by :meth:`SSHSandboxBackend.connect` only. Pass a
connected ``SSHClient`` — or anything with ``get_transport`` and
``open_sftp`` — to :class:`SSHSession` yourself and nothing is imported.
"""

from __future__ import annotations

import base64
import posixpath
import shlex
import socket
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from locus.harness.local import MAX_BACKGROUND
from locus.harness.session import SessionBackend


__all__ = ["SSHJump", "SSHSandboxBackend", "SSHSession"]

#: Seconds between the deadline on the host and the client giving up on the
#: channel: the watchdog's ``TERM`` grace, the ``KILL``, and slack.
_CLIENT_GRACE = 6.0

#: Seconds the watchdog waits between ``TERM`` and ``KILL``.
_KILL_GRACE = 2

#: Seconds an internal operation (a file move, a probe) may take.
_DEFAULT_TIMEOUT = 60.0

#: Bytes read from the channel at a time.
_CHUNK = 64 * 1024

_q = shlex.quote

KnownHosts = Literal["strict", "accept-unknown"]

_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


@dataclass(frozen=True)
class SSHJump:
    """A host to reach the workspace host through: a bastion.

    The jump host is connected first, with its own key and its own host-key
    check, and the workspace host's connection is a ``direct-tcpip`` channel
    through it — ``ssh -J``, without a local port.

    Attributes:
        host: The jump host.
        port: Its SSH port.
        username: The user to log in as; the current user when omitted.
        key_filename: A private key file for it.
        pkey: A loaded ``paramiko`` key, instead of a file.
        password: A password, when the jump host wants one.
        host_key: The jump host's public key, ``"<type> <base64>"``, to pin.
        known_hosts: How an unpinned key is checked.
    """

    host: str
    port: int = 22
    username: str | None = None
    key_filename: str | None = None
    pkey: Any = None
    password: str | None = None
    host_key: str | None = None
    known_hosts: KnownHosts = "strict"


def _deadline_script(command: str, seconds: int) -> str:
    """``command`` under a watchdog that stops it at ``seconds``.

    The command's streams are joined on the host. Its exit status comes
    back unless the watchdog fired, in which case the script exits 124 —
    decided by the watchdog's absence *or* by the clock, because a command
    killed by ``TERM`` returns 143 before the watchdog has gone.
    """
    watchdog = (
        'sleep "$1"; kill -s TERM -- "-$2" 2>/dev/null; '
        f'sleep {_KILL_GRACE}; kill -s KILL -- "-$2" 2>/dev/null'
    )
    return (
        "exec </dev/null; "
        "if command -v setsid >/dev/null 2>&1; then s=setsid; else s=; set -m; fi; "
        "t0=$(date +%s); "
        f"$s sh -c {_q(command)} 2>&1 & p=$!; "
        f'$s sh -c {_q(watchdog)} wd {seconds} "$p" >/dev/null 2>&1 & w=$!; '
        'wait "$p"; rc=$?; '
        'if kill -0 "$w" 2>/dev/null; then kill -s KILL -- "-$w" 2>/dev/null; '
        'kill -s KILL "$w" 2>/dev/null; else rc=124; fi; '
        f"[ $(( $(date +%s) - t0 )) -ge {seconds} ] && rc=124; exit $rc"
    )


class SSHSession:
    """One SSH connection as a :class:`~locus.harness.session.SessionLike`.

    Args:
        client: A connected ``paramiko.SSHClient``, or anything with its
            ``get_transport()`` (giving ``open_session``) and ``open_sftp()``.
        jump: The client connected to a jump host, closed with this one.
        transfer: ``"sftp"`` moves files over the SFTP subsystem, ``"exec"``
            over ``cat``; ``"auto"`` tries SFTP once and falls back.
        path: A ``PATH`` set for every command on the host, the backend's
            own scripts included. The ``PATH`` a non-interactive login gets
            is often short; this is where the host's GNU tools go when
            they are not on it.
    """

    def __init__(
        self,
        client: Any,
        *,
        jump: Any = None,
        transfer: Literal["auto", "sftp", "exec"] = "auto",
        path: str | None = None,
    ) -> None:
        self.client = client
        self.jump = jump
        self._transfer = transfer
        self._sftp: Any = None
        self._prelude = f"export PATH={_q(path)}; " if path else ""
        #: ``user@host[:port]`` for labels; set by :meth:`open`.
        self.where = "ssh"

    @classmethod
    def open(
        cls,
        host: str,
        *,
        port: int = 22,
        username: str | None = None,
        key_filename: str | None = None,
        pkey: Any = None,
        password: str | None = None,
        host_key: str | None = None,
        known_hosts: KnownHosts = "strict",
        via: SSHJump | None = None,
        connect_timeout: float = 30.0,
        transfer: Literal["auto", "sftp", "exec"] = "auto",
        path: str | None = None,
    ) -> SSHSession:
        """Connect to ``host`` with ``paramiko`` and return the session.

        The arguments are those of :meth:`SSHSandboxBackend.connect`. The
        jump host, when there is one, is connected first and closed with the
        session.
        """
        paramiko = _paramiko()
        jump = None
        sock = None
        if via is not None:
            jump = _client(paramiko, via.host, via.port, via.host_key, via.known_hosts)
            jump.connect(
                via.host,
                port=via.port,
                username=via.username,
                key_filename=via.key_filename,
                pkey=via.pkey,
                password=via.password,
                timeout=connect_timeout,
                banner_timeout=connect_timeout,
                auth_timeout=connect_timeout,
                **_only_that_key(via.pkey, via.key_filename),
            )
            sock = jump.get_transport().open_channel(
                "direct-tcpip", (host, port), ("127.0.0.1", 0), timeout=connect_timeout
            )
        client = _client(paramiko, host, port, host_key, known_hosts)
        try:
            client.connect(
                host,
                port=port,
                username=username,
                key_filename=key_filename,
                pkey=pkey,
                password=password,
                sock=sock,
                timeout=connect_timeout,
                banner_timeout=connect_timeout,
                auth_timeout=connect_timeout,
                **_only_that_key(pkey, key_filename),
            )
        except BaseException:
            if jump is not None:
                jump.close()
            raise
        transport = client.get_transport()
        user = username
        if transport is not None:
            transport.set_keepalive(30)
            user = user or transport.get_username()
        session = cls(client, jump=jump, transfer=transfer, path=path)
        where = f"{user}@{host}" if user else host
        session.where = where if port == 22 else f"{where}:{port}"
        return session

    # ------------------------------------------------------------ commands --

    def run(
        self,
        command: str,
        *,
        timeout: float,
        stdin: bytes | None = None,
        on_output: Callable[[bytes], None] | None = None,
    ) -> tuple[int | None, bytes]:
        """Run ``command`` with ``sh -c`` on the host; its status and its output.

        The command runs under a watchdog that stops it at ``timeout``
        seconds (status 124, which the backend reads as a timeout). ``stdin``
        is sent before the command's input is closed. Returns ``None`` for
        the status when the host neither finished the command nor reported
        it by the deadline plus a grace period.
        """
        seconds = max(1, int(timeout + 0.999))
        script = self._prelude + _deadline_script(command, seconds)
        transport = self.client.get_transport()
        if transport is None or not transport.is_active():
            raise ConnectionError("the SSH connection is closed")
        channel = transport.open_session(timeout=_DEFAULT_TIMEOUT)
        parts: list[bytes] = []
        deadline = time.monotonic() + seconds + _CLIENT_GRACE
        try:
            channel.set_combine_stderr(False)
            channel.settimeout(0.5)
            channel.exec_command(f"exec sh -c {_q(script)}")
            if stdin:
                channel.sendall(stdin)
            channel.shutdown_write()
            eof = False
            while True:
                while channel.recv_stderr_ready():
                    channel.recv_stderr(_CHUNK)  # the wrapper's own notices
                if not eof:
                    try:
                        data = channel.recv(_CHUNK)
                    except socket.timeout:
                        data = None
                    if data == b"":
                        eof = True
                    elif data:
                        parts.append(data)
                        if on_output is not None:
                            on_output(data)
                        continue
                if channel.exit_status_ready():
                    while channel.recv_ready():
                        chunk = channel.recv(_CHUNK)
                        parts.append(chunk)
                        if on_output is not None:
                            on_output(chunk)
                    return channel.recv_exit_status(), b"".join(parts)
                if time.monotonic() >= deadline:
                    return None, b"".join(parts)
                if eof:
                    channel.status_event.wait(0.2)
        finally:
            channel.close()

    def exec(self, command: str, *, timeout: float) -> tuple[int | None, bytes]:
        return self.run(command, timeout=timeout)

    # --------------------------------------------------------------- files --

    def _sftp_client(self) -> Any:
        """The SFTP client, opened once; ``None`` when the host has no subsystem."""
        if self._transfer == "exec":
            return None
        if self._sftp is None:
            try:
                self._sftp = self.client.open_sftp()
            except Exception:
                if self._transfer == "sftp":
                    raise
                self._transfer = "exec"
                return None
        return self._sftp

    def upload_file(self, path: str, data: bytes) -> None:
        sftp = self._sftp_client()
        if sftp is not None:
            # ``wb`` truncates an existing file and keeps its mode; a new file
            # gets the host's default, as ``cat >`` would give it.
            with sftp.open(path, "wb") as handle:
                handle.set_pipelined(True)
                handle.write(data)
            return
        code, out = self.run(
            f"mkdir -p {_q(posixpath.dirname(path) or '/')} && cat > {_q(path)}",
            timeout=_DEFAULT_TIMEOUT,
            stdin=data,
        )
        if code != 0:
            raise OSError(f"upload of {path} failed: {out.decode(errors='replace').strip()}")

    def download_file(self, path: str) -> bytes:
        sftp = self._sftp_client()
        if sftp is not None:
            with sftp.open(path, "rb") as handle:
                handle.prefetch()
                return bytes(handle.read())
        code, out = self.run(f"cat {_q(path)} 2>/dev/null", timeout=_DEFAULT_TIMEOUT)
        if code != 0:
            raise FileNotFoundError(path)
        return out

    def close(self) -> None:
        """Close the SFTP channel, the connection and the jump host's."""
        if self._sftp is not None:
            self._sftp.close()
            self._sftp = None
        self.client.close()
        if self.jump is not None:
            self.jump.close()


class SSHSandboxBackend(SessionBackend):
    """A host reached over SSH as a workspace.

    Build one with :meth:`connect`, or pass an :class:`SSHSession` over a
    client you configured.

    Args:
        session: The connection.
        root: The workspace directory on the host; absolute.
        label: Names the workspace in evidence and in the prompt.
        isolated: Whether the host is another machine. A session to this
            machine must say ``False``; :meth:`connect` does so for
            ``localhost``.
        owns_connection: Close the connection on :meth:`close`.
        env: Variables added to every command's environment.
        max_background: How many background commands may run at once.
        state_dir: Where background commands keep their state, on the host.
    """

    def __init__(
        self,
        session: SSHSession,
        *,
        root: str,
        label: str = "ssh",
        isolated: bool = True,
        owns_connection: bool = True,
        env: Mapping[str, str] | None = None,
        max_background: int = MAX_BACKGROUND,
        state_dir: str = "/tmp/.locus-harness",  # noqa: S108 - on the host, per-backend subdir
    ) -> None:
        self._ssh = session
        super().__init__(
            session,
            root=root,
            label=label,
            isolated=isolated,
            state_dir=state_dir,
            env=env,
            max_background=max_background,
        )
        self._owns_connection = owns_connection

    @property
    def ssh(self) -> SSHSession:
        """The connection every call goes through."""
        return self._ssh

    def _exec_script(
        self,
        script: str,
        timeout: float,
        on_output: Callable[[bytes], None] | None,
    ) -> tuple[int | None, bytes]:
        return self._ssh.run(script, timeout=timeout, on_output=on_output)

    def _destroy(self) -> None:
        """Close the connection if this backend opened it."""
        if self._owns_connection:
            self._ssh.close()

    @classmethod
    def connect(
        cls,
        host: str,
        *,
        port: int = 22,
        username: str | None = None,
        key_filename: str | None = None,
        pkey: Any = None,
        password: str | None = None,
        host_key: str | None = None,
        known_hosts: KnownHosts = "strict",
        via: SSHJump | None = None,
        root: str | None = None,
        isolated: bool | None = None,
        env: Mapping[str, str] | None = None,
        connect_timeout: float = 30.0,
        transfer: Literal["auto", "sftp", "exec"] = "auto",
        path: str | None = None,
        **options: Any,
    ) -> SSHSandboxBackend:
        """Connect to ``host`` and return a backend over it.

        Args:
            host: The host name or address.
            port: Its SSH port.
            username: The user to log in as; the current user when omitted.
            key_filename: A private key file. Without one, and without
                ``pkey`` or ``password``, the agent and ``~/.ssh`` keys are
                tried.
            pkey: A loaded ``paramiko`` key.
            password: A password, when the host wants one.
            host_key: The host's public key, ``"<type> <base64>"``, to pin.
            known_hosts: ``"strict"`` requires the host in ``known_hosts``
                unless pinned; ``"accept-unknown"`` accepts any key for a host
                not there, for a sandbox whose key has no other channel.
            via: A jump host to connect through.
            root: The workspace directory on the host, created if missing;
                the login directory when omitted.
            isolated: Whether the host is another machine; ``False`` for
                ``localhost`` when omitted.
            env: Variables added to every command's environment.
            connect_timeout: Seconds to establish the connection.
            transfer: How files move: SFTP, ``cat`` over exec, or SFTP with
                a fallback.
            path: A ``PATH`` for every command on the host, when the
                login's own lacks the tools the backend needs.
            options: Passed to the backend (``state_dir``, ``max_background``).
        """
        session = SSHSession.open(
            host,
            port=port,
            username=username,
            key_filename=key_filename,
            pkey=pkey,
            password=password,
            host_key=host_key,
            known_hosts=known_hosts,
            via=via,
            connect_timeout=connect_timeout,
            transfer=transfer,
            path=path,
        )
        try:
            workdir = _prepare_root(session, root)
        except BaseException:
            session.close()
            raise
        return cls(
            session,
            root=workdir,
            label=f"ssh:{session.where}",
            isolated=host not in _LOCAL_HOSTS if isolated is None else isolated,
            env=env,
            **options,
        )


def _prepare_root(session: SSHSession, root: str | None) -> str:
    """``root`` created on the host, or the login directory when ``None``."""
    if root is None:
        code, out = session.run("pwd", timeout=_DEFAULT_TIMEOUT)
        text = out.decode(errors="replace").strip()
        if code != 0 or not text.startswith("/"):
            raise RuntimeError(f"could not find the login directory on the host: {text!r}")
        return text
    if not root.startswith("/"):
        raise ValueError(f"root must be an absolute path on the host, got {root!r}")
    code, out = session.run(f"mkdir -p {_q(root)}", timeout=_DEFAULT_TIMEOUT)
    if code != 0:
        raise RuntimeError(f"could not create {root} on the host: {out.decode(errors='replace')}")
    return posixpath.normpath(root)


def _only_that_key(pkey: Any, key_filename: str | None) -> dict[str, bool]:
    """With an explicit key, offer that key alone.

    A server that closes the connection after one refused key — a bastion
    does — must see the key the caller meant first and only; the agent and
    ``~/.ssh`` are for a login that named no key.
    """
    explicit = pkey is not None or key_filename is not None
    return {"allow_agent": not explicit, "look_for_keys": not explicit}


def _paramiko() -> Any:
    """The ``paramiko`` package; the only place it is imported."""
    try:
        import paramiko
    except ImportError as exc:
        raise ImportError(
            'SSHSandboxBackend.connect needs paramiko: pip install "locus-sdk[ssh]"'
        ) from exc
    return paramiko


def _client(
    paramiko: Any, host: str, port: int, host_key: str | None, known_hosts: KnownHosts
) -> Any:
    """An ``SSHClient`` with the host-key policy the caller asked for."""
    client = paramiko.SSHClient()
    if host_key is not None:
        kind, key = _public_key(paramiko, host_key)
        client.get_host_keys().add(host if port == 22 else f"[{host}]:{port}", kind, key)
        client.set_missing_host_key_policy(paramiko.RejectPolicy())
        return client
    client.load_system_host_keys()
    try:
        client.load_host_keys(posixpath.expanduser("~/.ssh/known_hosts"))
    except OSError:
        pass
    if known_hosts == "accept-unknown":
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # noqa: S507 - explicit opt-in
    else:
        client.set_missing_host_key_policy(paramiko.RejectPolicy())
    return client


def _public_key(paramiko: Any, line: str) -> tuple[str, Any]:
    """A ``"<type> <base64>"`` line as the key type and a loaded key."""
    fields = line.split()
    if len(fields) < 2:
        raise ValueError(f"host_key must be '<type> <base64>', got {line!r}")
    kind, data = fields[0], base64.b64decode(fields[1])
    return kind, paramiko.PKey.from_type_string(kind, data)
