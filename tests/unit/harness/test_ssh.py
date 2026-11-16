# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""SSHSandboxBackend: the deadline wrapper, the channel loop, file transfer, host keys.

Everything here drives the session with a stand-in for ``paramiko``'s client,
transport and channel; the live test under ``tests/integration`` runs the
conformance kit against a real ``sshd``.
"""

from __future__ import annotations

import socket
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any

import pytest

from locus.harness import SSHJump, SSHSandboxBackend, SSHSession
from locus.harness import ssh as ssh_mod
from locus.harness.ssh import _deadline_script


class FakeChannel:
    """One exec channel: scripted output and status, or a command that never ends."""

    def __init__(self, output: bytes, code: int | None, stderr: bytes = b"") -> None:
        self.chunks = [output] if output else []
        self.code = code
        self.stderr = stderr
        self.commands: list[str] = []
        self.stdin = b""
        self.write_shut = False
        self.closed = False
        self.combine: bool | None = None
        self.status_event = threading.Event()
        if code is not None:
            self.status_event.set()

    def set_combine_stderr(self, flag: bool) -> None:
        self.combine = flag

    def settimeout(self, _t: float) -> None:
        pass

    def exec_command(self, command: str) -> None:
        self.commands.append(command)

    def send(self, data: bytes) -> int:
        self.stdin += bytes(data)
        return len(data)

    def shutdown_write(self) -> None:
        self.write_shut = True

    def recv_stderr_ready(self) -> bool:
        return bool(self.stderr)

    def recv_stderr(self, _n: int) -> bytes:
        data, self.stderr = self.stderr, b""
        return data

    def recv_ready(self) -> bool:
        return bool(self.chunks)

    def recv(self, _n: int) -> bytes:
        if self.chunks:
            return self.chunks.pop(0)
        if self.code is None:
            time.sleep(0.02)
            raise socket.timeout
        return b""

    def exit_status_ready(self) -> bool:
        return self.code is not None and not self.chunks

    def recv_exit_status(self) -> int:
        assert self.code is not None
        return self.code

    def close(self) -> None:
        self.closed = True


class FakeTransport:
    def __init__(self, replies: list[tuple[bytes, int | None]]) -> None:
        self.replies = replies
        self.channels: list[FakeChannel] = []
        self.active = True
        self.keepalive: int | None = None
        self.opened: list[tuple[Any, ...]] = []

    def is_active(self) -> bool:
        return self.active

    def open_session(self, timeout: float | None = None) -> FakeChannel:
        out, code = self.replies.pop(0) if self.replies else (b"", 0)
        channel = FakeChannel(out, code)
        self.channels.append(channel)
        return channel

    def set_keepalive(self, seconds: int) -> None:
        self.keepalive = seconds

    def get_username(self) -> str:
        return "u"

    def open_channel(self, *args: Any, **kwargs: Any) -> str:
        self.opened.append(args)
        return "SOCK"


class FakeSFTPFile:
    def __init__(self, store: dict[str, bytes], path: str, mode: str) -> None:
        self.store, self.path, self.mode = store, path, mode
        if "r" in mode and path not in store:
            raise OSError(2, "No such file")

    def __enter__(self) -> FakeSFTPFile:
        return self

    def __exit__(self, *exc: object) -> None:
        pass

    def set_pipelined(self, _flag: bool) -> None:
        pass

    def prefetch(self) -> None:
        pass

    def write(self, data: bytes) -> None:
        self.store[self.path] = data

    def read(self) -> bytes:
        return self.store[self.path]


class FakeSFTP:
    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}
        self.closed = False

    def open(self, path: str, mode: str) -> FakeSFTPFile:
        return FakeSFTPFile(self.store, path, mode)

    def close(self) -> None:
        self.closed = True


class FakeHostKeys:
    def __init__(self) -> None:
        self.added: list[tuple[str, str, Any]] = []

    def add(self, name: str, kind: str, key: Any) -> None:
        self.added.append((name, kind, key))


class FakeClient:
    """The ``SSHClient`` surface the backend uses."""

    replies: list[tuple[bytes, int | None]] = []
    sftp_error: Exception | None = None
    connect_error: Exception | None = None
    instances: list[FakeClient] = []

    def __init__(self) -> None:
        self.transport = FakeTransport(list(FakeClient.replies))
        self.sftp = FakeSFTP()
        self.connected: dict[str, Any] | None = None
        self.closed = False
        self.policy: Any = None
        self.host_keys = FakeHostKeys()
        self.loaded: list[tuple[str, str | None]] = []
        FakeClient.instances.append(self)

    def get_transport(self) -> FakeTransport:
        return self.transport

    def open_sftp(self) -> FakeSFTP:
        if FakeClient.sftp_error is not None:
            raise FakeClient.sftp_error
        return self.sftp

    def connect(self, host: str, **kwargs: Any) -> None:
        if FakeClient.connect_error is not None:
            raise FakeClient.connect_error
        self.connected = {"host": host, **kwargs}

    def close(self) -> None:
        self.closed = True

    def set_missing_host_key_policy(self, policy: Any) -> None:
        self.policy = policy

    unreadable: dict[str, Exception] = {}

    def load_system_host_keys(self, path: str | None = None) -> None:
        if path in FakeClient.unreadable:
            raise FakeClient.unreadable[path]
        self.loaded.append(("read-only", path))

    def load_host_keys(self, path: str) -> None:
        self.loaded.append(("writable", path))

    def get_host_keys(self) -> FakeHostKeys:
        return self.host_keys


@pytest.fixture(autouse=True)
def _reset_fake_client() -> None:
    FakeClient.replies = []
    FakeClient.sftp_error = None
    FakeClient.connect_error = None
    FakeClient.instances = []
    FakeClient.unreadable = {}


def session(*replies: tuple[bytes, int | None], **options: Any) -> tuple[SSHSession, FakeClient]:
    FakeClient.replies = list(replies)
    client = FakeClient()
    return SSHSession(client, **options), client


# ----------------------------------------------------------------- wrapper --


def test_the_wrapper_runs_the_command_under_a_watchdog() -> None:
    script = _deadline_script("echo 'it''s' here", 7)
    assert "setsid" in script, "a process group where util-linux is"
    assert "set -m" in script, "and job control where it is not"
    assert "sh -c 'echo '\"'\"'it'\"'\"''\"'\"'s'\"'\"' here' <&3 3<&- 2>&1 &" in script
    assert 'wd 7 "$p"' in script, "the watchdog gets the deadline"
    assert "kill -s TERM" in script
    assert "kill -s KILL" in script
    assert "rc=124" in script
    assert "exit $rc" in script


def test_the_wrapper_keeps_the_channels_input_for_the_command() -> None:
    script = _deadline_script("cat", 7)
    assert script.startswith("exec 3<&0 </dev/null; "), (
        "the channel's input moves to descriptor 3 before the wrapper's own is severed"
    )
    assert "sh -c cat <&3 3<&- 2>&1 &" in script, "the command reads it back as its stdin"
    assert 'wd 7 "$p" >/dev/null 2>&1 &' in script, "the watchdog inherits /dev/null"
    assert 'exec 3<&-; wait "$p"' in script, "the wrapper lets go of it once both are started"
    assert "exec </dev/null;" not in script


def test_the_wrapper_passes_stdin_through_to_the_command_on_this_host(tmp_path: Path) -> None:
    import subprocess

    target = tmp_path / "out.bin"
    payload = b"hello watchdog\n" * 1000
    script = _deadline_script(f"cat > {target}; echo rc=$?", 10)
    done = subprocess.run(
        ["/bin/sh", "-c", script], input=payload, capture_output=True, timeout=30, check=False
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout == b"rc=0\n"
    assert target.read_bytes() == payload, "every byte of the input reached the command"


def test_run_returns_the_output_and_status_and_drops_wrapper_noise() -> None:
    s, client = session((b"out\n", 3))
    client.transport.replies = [(b"out\n", 3)]
    channel_stderr = b"sh: line 1: Terminated\n"
    code, out = s.run("echo out; exit 3", timeout=5)
    [channel] = client.transport.channels
    channel.stderr = channel_stderr
    assert (code, out) == (3, b"out\n")
    assert channel.combine is False, "the wrapper's notices stay on the error stream"
    assert channel.commands[0].startswith("exec sh -c ")
    assert "echo out; exit 3" in channel.commands[0]
    assert channel.write_shut, "stdin is closed"
    assert channel.closed


def test_run_streams_and_sends_stdin() -> None:
    s, client = session((b"hello\n", 0))
    seen: list[bytes] = []
    code, out = s.run("cat", timeout=5, stdin=b"hello\n", on_output=seen.append)
    assert (code, out) == (0, b"hello\n")
    assert seen == [b"hello\n"]
    assert client.transport.channels[0].stdin == b"hello\n"


class WindowedChannel:
    """A channel whose send window opens a chunk at a time, as paramiko's does."""

    def __init__(self, script: list[int | Exception]) -> None:
        self.script = script
        self.sent = b""

    def send(self, data: memoryview) -> int:
        step = self.script.pop(0) if self.script else len(data)
        if isinstance(step, Exception):
            raise step
        take = min(step, len(data))
        self.sent += bytes(data[:take])
        return take


def test_input_is_sent_as_the_window_opens() -> None:
    payload = bytes(range(256)) * 1000
    channel = WindowedChannel([1000, socket.timeout(), 5000, socket.timeout(), 100_000])
    ssh_mod._send_input(channel, payload, time.monotonic() + 5)
    assert channel.sent == payload, "a closed window is waited out, partial sends continue"


def test_input_stops_when_the_command_closes_the_channel_or_at_the_deadline() -> None:
    channel = WindowedChannel([3, OSError("Socket is closed")])
    ssh_mod._send_input(channel, b"abcdef", time.monotonic() + 5)
    assert channel.sent == b"abc", "the command ended; its status says what it did"
    late = WindowedChannel([])
    ssh_mod._send_input(late, b"abc", time.monotonic() - 1)
    assert late.sent == b"", "past the deadline nothing more is sent"


def test_a_path_is_exported_before_every_command() -> None:
    s, client = session((b"", 0), path="/opt/gnubin:/usr/bin")
    s.run("true", timeout=5)
    assert "export PATH=/opt/gnubin:/usr/bin; " in client.transport.channels[0].commands[0]


def test_a_command_the_host_never_reports_times_out_on_the_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ssh_mod, "_CLIENT_GRACE", 0.1)
    s, client = session((b"started\n", None))
    started = time.monotonic()
    code, out = s.run("sleep 99", timeout=0.5)
    assert time.monotonic() - started < 5
    assert code is None
    assert out == b"started\n"
    assert client.transport.channels[0].closed


def test_run_on_a_dead_connection_raises() -> None:
    s, client = session()
    client.transport.active = False
    with pytest.raises(ConnectionError, match="closed"):
        s.run("true", timeout=5)


# ------------------------------------------------------------------- files --


def test_files_move_over_sftp_by_default() -> None:
    s, client = session()
    s.upload_file("/w/a.bin", b"\x00\x01")
    assert client.sftp.store["/w/a.bin"] == b"\x00\x01"
    assert s.download_file("/w/a.bin") == b"\x00\x01"
    with pytest.raises(OSError):
        s.download_file("/w/missing")
    s.close()
    assert client.sftp.closed
    assert client.closed


def test_without_sftp_files_move_over_cat() -> None:
    FakeClient.sftp_error = RuntimeError("subsystem request failed")
    s, client = session((b"       7\n", 0), (b"payload", 0), (b"", 1))
    s.upload_file("/w/a.txt", b"payload")
    upload = client.transport.channels[0]
    assert upload.stdin == b"payload"
    assert "mkdir -p /w && cat > /w/a.txt && wc -c < /w/a.txt" in upload.commands[0]
    assert s.download_file("/w/a.txt") == b"payload"
    with pytest.raises(FileNotFoundError):
        s.download_file("/w/missing")


def test_an_upload_over_cat_that_wrote_fewer_bytes_is_an_error() -> None:
    short, _ = session((b"3\n", 0), transfer="exec")
    with pytest.raises(OSError, match=r"wrote 3 of 7 bytes"):
        short.upload_file("/w/a.txt", b"payload")
    silent, _ = session((b"", 0), transfer="exec")
    with pytest.raises(OSError, match=r"wrote -1 of 7 bytes"):
        silent.upload_file("/w/a.txt", b"payload")
    failed, _ = session((b"cat: /w/a.txt: Permission denied\n", 1), transfer="exec")
    with pytest.raises(OSError, match="Permission denied"):
        failed.upload_file("/w/a.txt", b"payload")


def test_a_large_upload_over_cat_gets_time_in_proportion() -> None:
    big = b"x" * (3 * 1024 * 1024)
    s, client = session((str(len(big)).encode(), 0), transfer="exec")
    seen: list[float] = []
    original = s.run

    def spy(command: str, **kwargs: Any) -> tuple[int | None, bytes]:
        seen.append(kwargs["timeout"])
        return original(command, **kwargs)

    s.run = spy  # type: ignore[method-assign]
    s.upload_file("/w/big.bin", big)
    assert client.transport.channels[0].stdin == big
    assert seen == [ssh_mod._DEFAULT_TIMEOUT + len(big) / ssh_mod._UPLOAD_RATE]


def test_sftp_can_be_required_or_refused() -> None:
    FakeClient.sftp_error = RuntimeError("subsystem request failed")
    strict, _ = session(transfer="sftp")
    with pytest.raises(RuntimeError, match="subsystem"):
        strict.upload_file("/w/a", b"x")
    FakeClient.sftp_error = None
    never, client = session((b"1\n", 0), transfer="exec")
    never.upload_file("/w/a", b"x")
    assert client.sftp.store == {}, "exec transfer never opens SFTP"


# ---------------------------------------------------------------- paramiko --


class RejectPolicy:
    pass


class AutoAddPolicy:
    pass


class InvalidHostKey(Exception):  # noqa: N818 - paramiko's name; a plain Exception there too
    pass


class PKey:
    @staticmethod
    def from_type_string(kind: str, data: bytes) -> tuple[str, bytes]:
        return (kind, data)

    @staticmethod
    def from_path(path: str) -> Any:
        return types.SimpleNamespace(
            path=path, get_name=lambda: "ssh-ed25519", get_base64=lambda: "AAAA"
        )


@pytest.fixture
def fake_paramiko(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    module = types.ModuleType("paramiko")
    module.SSHClient = FakeClient  # type: ignore[attr-defined]
    module.RejectPolicy = RejectPolicy  # type: ignore[attr-defined]
    module.AutoAddPolicy = AutoAddPolicy  # type: ignore[attr-defined]
    module.PKey = PKey  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "paramiko", module)
    return module


def test_a_pinned_host_key_is_the_only_one_accepted(fake_paramiko: types.ModuleType) -> None:
    client = ssh_mod._client(fake_paramiko, "box", 2222, "ssh-ed25519 QUJD", "strict")
    assert isinstance(client.policy, RejectPolicy)
    assert client.host_keys.added == [("[box]:2222", "ssh-ed25519", ("ssh-ed25519", b"ABC"))]
    assert client.loaded == [], "a pinned key replaces known_hosts"
    plain = ssh_mod._client(fake_paramiko, "box", 22, "ssh-ed25519 QUJD", "strict")
    assert plain.host_keys.added[0][0] == "box"


def test_known_hosts_policies(
    fake_paramiko: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    strict = ssh_mod._client(fake_paramiko, "box", 22, None, "strict")
    assert isinstance(strict.policy, RejectPolicy)
    assert strict.loaded == [
        ("read-only", "/etc/ssh/ssh_known_hosts"),
        ("read-only", str(tmp_path / ".ssh" / "known_hosts")),
    ], "both files are read as read-only sets: nothing is ever written back"
    loose = ssh_mod._client(fake_paramiko, "box", 22, None, "accept-unknown")
    assert isinstance(loose.policy, AutoAddPolicy)
    assert all(kind == "read-only" for kind, _ in loose.loaded), (
        "an accepted key is kept in memory, not saved to the user's known_hosts"
    )


def test_an_unreadable_known_hosts_file_is_skipped(
    fake_paramiko: types.ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    user_file = str(tmp_path / ".ssh" / "known_hosts")
    FakeClient.unreadable = {
        "/etc/ssh/ssh_known_hosts": FileNotFoundError(2, "No such file"),
        user_file: InvalidHostKey("@cert-authority *.corp.example ssh-ed25519 ..."),
    }
    with caplog.at_level("WARNING"):
        client = ssh_mod._client(fake_paramiko, "box", 22, None, "strict")
    assert isinstance(client.policy, RejectPolicy), "the connection still gets its policy"
    assert client.loaded == [], "a missing file is quiet"
    assert "could not be read" in caplog.text
    assert "@cert-authority" in caplog.text, "an unparsable file is said so, and skipped"


def test_a_jump_hosts_secrets_stay_out_of_its_repr() -> None:
    jump = SSHJump("bastion", username="s", password="hunter2", pkey="PRIVATE-KEY-MATERIAL")  # noqa: S106
    text = repr(jump)
    assert "hunter2" not in text
    assert "PRIVATE-KEY-MATERIAL" not in text
    assert "bastion" in text
    assert jump == SSHJump(
        "bastion",
        username="s",
        password="hunter2",  # noqa: S106 - equality still sees every field
        pkey="PRIVATE-KEY-MATERIAL",
    )


def test_a_malformed_host_key_is_refused(fake_paramiko: types.ModuleType) -> None:
    with pytest.raises(ValueError, match="<type> <base64>"):
        ssh_mod._client(fake_paramiko, "box", 22, "garbage", "strict")


def test_open_connects_with_the_arguments_and_keeps_the_connection_alive(
    fake_paramiko: types.ModuleType,
) -> None:
    s = SSHSession.open("box", port=2222, username="dev", key_filename="/k", connect_timeout=7)
    [client] = FakeClient.instances
    assert client.connected == {
        "host": "box",
        "port": 2222,
        "username": "dev",
        "key_filename": "/k",
        "pkey": None,
        "password": None,
        "sock": None,
        "timeout": 7,
        "banner_timeout": 7,
        "auth_timeout": 7,
        "allow_agent": False,
        "look_for_keys": False,
    }
    assert client.transport.keepalive == 30
    assert s.where == "dev@box:2222"
    assert s.client is client


def test_a_tilde_in_a_key_file_is_expanded_for_the_host_and_the_jump(
    fake_paramiko: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    SSHSession.open(
        "box", key_filename="~/.ssh/agent", via=SSHJump("bastion", key_filename="~/.ssh/jump")
    )
    jump, host = FakeClient.instances
    assert jump.connected is not None
    assert jump.connected["key_filename"] == str(tmp_path / ".ssh" / "jump")
    assert host.connected is not None
    assert host.connected["key_filename"] == str(tmp_path / ".ssh" / "agent")
    assert host.connected["look_for_keys"] is False, "still the explicit key alone"


def test_open_goes_through_a_jump_host(fake_paramiko: types.ModuleType) -> None:
    jump = SSHJump("bastion", username="ocid1.session", host_key="ssh-rsa QUJD")
    s = SSHSession.open("10.0.0.5", via=jump)
    first, second = FakeClient.instances
    assert first.connected is not None
    assert first.connected["host"] == "bastion"
    assert first.connected["username"] == "ocid1.session"
    assert first.connected["allow_agent"] is True, "no key named: the agent and ~/.ssh are tried"
    assert first.host_keys.added[0][0] == "bastion"
    assert first.transport.opened == [("direct-tcpip", ("10.0.0.5", 22), ("127.0.0.1", 0))]
    assert second.connected is not None
    assert second.connected["sock"] == "SOCK"
    assert s.jump is first
    s.close()
    assert first.closed
    assert second.closed


def test_a_failed_connection_closes_the_jump_host(fake_paramiko: types.ModuleType) -> None:
    FakeClient.connect_error = None
    jump = SSHJump("bastion")

    original = FakeClient.connect

    def failing(self: FakeClient, host: str, **kwargs: Any) -> None:
        if host == "10.0.0.5":
            raise OSError("refused")
        original(self, host, **kwargs)

    FakeClient.connect = failing  # type: ignore[method-assign]
    try:
        with pytest.raises(OSError, match="refused"):
            SSHSession.open("10.0.0.5", via=jump)
    finally:
        FakeClient.connect = original  # type: ignore[method-assign]
    assert FakeClient.instances[0].closed


# ----------------------------------------------------------------- backend --


def test_connect_uses_the_login_directory_and_labels_the_host(
    fake_paramiko: types.ModuleType,
) -> None:
    FakeClient.replies = [(b"/home/dev\n", 0)]
    backend = SSHSandboxBackend.connect("box", username="dev")
    assert backend.capabilities.root == "/home/dev"
    assert backend.capabilities.label == "ssh:dev@box"
    assert backend.capabilities.isolated
    assert isinstance(backend.ssh, SSHSession)
    backend.close()
    assert FakeClient.instances[0].closed, "connect owns the connection"
    assert backend.state == "closed"


def test_connect_creates_the_root_and_knows_localhost_is_not_isolated(
    fake_paramiko: types.ModuleType,
) -> None:
    FakeClient.replies = [(b"", 0)]
    backend = SSHSandboxBackend.connect("localhost", root="/srv/ws/", port=2200)
    assert "mkdir -p /srv/ws/" in FakeClient.instances[0].transport.channels[0].commands[0]
    assert backend.capabilities.root == "/srv/ws"
    assert not backend.capabilities.isolated
    assert backend.capabilities.label == "ssh:u@localhost:2200"


def test_connect_refuses_a_relative_root_and_closes(fake_paramiko: types.ModuleType) -> None:
    with pytest.raises(ValueError, match="absolute"):
        SSHSandboxBackend.connect("box", root="relative")
    assert FakeClient.instances[0].closed


def test_connect_fails_when_the_login_directory_is_unknown(
    fake_paramiko: types.ModuleType,
) -> None:
    FakeClient.replies = [(b"", 1)]
    with pytest.raises(RuntimeError, match="login directory"):
        SSHSandboxBackend.connect("box")


def test_a_borrowed_connection_is_left_open() -> None:
    s, client = session((b"", 0))
    backend = SSHSandboxBackend(s, root="/w", owns_connection=False)
    backend.close()
    assert not client.closed


def test_exec_goes_through_the_session_with_streaming() -> None:
    s, client = session((b"streamed\n", 0))
    backend = SSHSandboxBackend(s, root="/w")
    seen: list[bytes] = []
    result = backend.exec("echo streamed", timeout=5, on_output=seen.append)
    assert result.exit_code == 0
    assert seen == [b"streamed\n"]
    assert "cd /w || exit 1" in client.transport.channels[-1].commands[0]


def test_the_hosts_124_is_a_timeout_only_at_the_deadline() -> None:
    s, _ = session((b"started\n", 124))
    backend = SSHSandboxBackend(s, root="/w")
    result = backend.exec("exit 124", timeout=30)
    assert result.exit_code == 124
    assert not result.timed_out, "a quick 124 is the command's own"


def test_without_paramiko_the_error_says_what_to_install(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "paramiko", None)
    with pytest.raises(ImportError, match=r"locus-sdk\[ssh\]"):
        SSHSandboxBackend.connect("box")


def test_a_live_sshd_is_exercised_by_the_integration_test(tmp_path: Path) -> None:
    """A pointer, so the unit file says where the real check is."""
    live = Path(__file__).parents[2] / "integration" / "test_harness_ssh_live.py"
    assert live.exists()
