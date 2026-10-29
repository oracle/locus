# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""OCISandboxBackend: what is particular to it beyond the shared contract.

The contract tests already run on it (see conftest). These pin the parts only
it has: commands streamed as sequenced events and run under an in-sandbox
deadline with their output capped, the service's ``-1`` for an unfinished
command, base64 checksums and chunked uploads, the sandbox lifecycle, and the
constructors that build a client.

The tests marked ``needs_posix`` run the backend's scripts in this machine's
``sh``; the others drive the session with scripted events and run anywhere.
"""

from __future__ import annotations

import base64
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from locus.harness import (
    SKIP_DIRS,
    OCISandboxBackend,
    OCISandboxSession,
    build_harness,
    oci_sandbox,
)
from tests.unit.harness.conftest import (
    REAL_OCI_MODELS,
    FakeOCISandboxClient,
    FakeServiceError,
    needs_posix,
)


class ScriptedClient(FakeOCISandboxClient):
    """Answers every command with the events a test queued; runs nothing."""

    def __init__(self, *replies: list[dict[str, Any]]) -> None:
        super().__init__()
        self.replies = list(replies)
        self.raise_after: float | None = None

    def run_sandbox_command_and_stream(
        self, sandbox_id: str, details: Any, **kwargs: Any
    ) -> SimpleNamespace:
        self.calls.append(
            {"sandbox": sandbox_id, "command": details.command, "timeout": details.timeout}
        )
        if self.raise_after is not None:
            time.sleep(self.raise_after)
            raise ConnectionError("read timed out")
        events = (
            self.replies.pop(0)
            if self.replies
            else [{"event": "FINISHED", "seq": 1, "exitCode": 0}]
        )
        return SimpleNamespace(data=[json.dumps(e).encode() for e in events])


def finished(code: int, seq: int = 9) -> dict[str, Any]:
    return {"event": "FINISHED", "seq": seq, "exitCode": code}


def session(client: FakeOCISandboxClient, **options: Any) -> OCISandboxSession:
    return OCISandboxSession(client, "sbx-7", state_dir="/workspace/.state", **options)


# ------------------------------------------------------------ commands --


def test_events_are_joined_in_sequence_order_and_streamed() -> None:
    client = ScriptedClient(
        [
            {"event": "STARTED", "seq": 1},
            {"event": "STDERR", "seq": 3, "data": "err\n"},
            {"event": "STDOUT", "seq": 2, "data": "out\n"},
            {"event": "STDOUT", "seq": 4, "data": "end\n"},
            finished(0, 5),
        ]
    )
    seen: list[bytes] = []
    code, out = session(client).run("x", timeout=5, on_output=seen.append)
    assert code == 0
    assert out == b"out\nerr\nend\n"
    assert seen == [b"out\n", b"err\n", b"end\n"]


@pytest.mark.parametrize(
    "events",
    [[finished(-1)], [{"event": "STARTED", "seq": 1}, {"event": "STDOUT", "seq": 2, "data": "x"}]],
    ids=["service-timeout", "no-finished-event"],
)
def test_an_unfinished_command_has_no_status(events: list[dict[str, Any]]) -> None:
    code, _ = session(ScriptedClient(events)).run("sleep 99", timeout=5)
    assert code is None


def test_commands_run_under_a_deadline_with_their_output_captured() -> None:
    client = ScriptedClient([finished(0)])
    session(client, max_output_bytes=100_000).run("echo hi", timeout=7.2)
    [call] = client.calls
    script = call["command"]
    assert "timeout -k 5 8 sh -c 'echo hi'" in script, "the sandbox stops it at its own deadline"
    assert '> "$f" 2>&1 < /dev/null' in script, "output goes to a file, not the stream"
    assert "/workspace/.state" in script
    assert "-gt 100000" in script
    assert "bytes of output were dropped here" in script
    assert call["timeout"] == "PT43S", "the service timeout is a later backstop"


def test_the_output_limit_must_leave_room_for_a_tail() -> None:
    with pytest.raises(ValueError, match="max_output_bytes"):
        session(ScriptedClient(), max_output_bytes=10)


def test_a_transport_failure_before_the_deadline_is_raised() -> None:
    client = ScriptedClient()
    client.raise_after = 0.0
    with pytest.raises(ConnectionError):
        session(client).run("true", timeout=5)


def test_a_transport_failure_past_the_deadline_is_a_timeout() -> None:
    client = ScriptedClient()
    client.raise_after = 0.2
    assert session(client).run("sleep 9", timeout=0.1) == (None, b"")


# --------------------------------------------------------------- files --


def test_writes_carry_a_base64_sha256() -> None:
    client = FakeOCISandboxClient()
    seen: dict[str, Any] = {}

    def write(sandbox_id: str, path: str, body: Any, opc_content_sha256: str | None = None) -> None:
        seen.update(path=path, data=body.read(), checksum=opc_content_sha256)

    client.write_sandbox_file = write  # type: ignore[method-assign,assignment]
    session(client).upload_file("/workspace/a.bin", b"payload")
    assert seen["data"] == b"payload"
    assert seen["checksum"] == base64.b64encode(hashlib.sha256(b"payload").digest()).decode()


def test_a_large_upload_goes_in_parts_joined_in_the_sandbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(oci_sandbox, "UPLOAD_CHUNK", 4)
    client = ScriptedClient([finished(0)])
    parts: list[tuple[str, bytes]] = []
    client.write_sandbox_file = lambda sid, path, body, **kw: parts.append((path, body.read()))  # type: ignore[method-assign,assignment,misc]
    session(client).upload_file("/workspace/big.bin", b"0123456789")
    assert [data for _, data in parts] == [b"0123", b"4567", b"89"]
    assert all(path.startswith("/workspace/big.bin.locus-part-") for path, _ in parts)
    join = client.calls[0]["command"]
    assert "cat " in join
    assert "> /workspace/big.bin" in join


def test_a_failed_join_is_an_error_and_leaves_no_parts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(oci_sandbox, "UPLOAD_CHUNK", 4)
    client = ScriptedClient(
        [{"event": "STDOUT", "seq": 1, "data": "disk full"}, finished(1, 2)], [finished(0)]
    )
    client.write_sandbox_file = lambda *a, **kw: None  # type: ignore[method-assign,assignment,misc]
    with pytest.raises(OSError, match="disk full"):
        session(client).upload_file("/workspace/big.bin", b"0123456789")
    assert "rm -f" in client.calls[1]["command"]


def test_a_missing_file_is_not_found_and_other_errors_propagate() -> None:
    client = FakeOCISandboxClient()
    with pytest.raises(FileNotFoundError):
        session(client).download_file("/workspace/nope")
    client.read_sandbox_file = lambda *a, **kw: (_ for _ in ()).throw(FakeServiceError(500, "boom"))  # type: ignore[method-assign,assignment,misc]
    with pytest.raises(FakeServiceError):
        session(client).download_file("/workspace/x")


@pytest.mark.parametrize("content", [b"bytes", "text"])
def test_a_download_returns_bytes(content: bytes | str) -> None:
    client = FakeOCISandboxClient()
    client.read_sandbox_file = lambda *a, **kw: SimpleNamespace(
        data=SimpleNamespace(content=content)
    )  # type: ignore[method-assign,assignment,misc]
    assert session(client).download_file("/workspace/x") in (b"bytes", b"text")


# ----------------------------------------------------------- lifecycle --


def test_it_is_isolated_and_says_where_it_runs() -> None:
    caps = OCISandboxBackend(ScriptedClient(), "sbx-7", region="eu-frankfurt-1").capabilities
    assert caps.isolated
    assert caps.can_exec
    assert caps.root == "/workspace"
    assert caps.label == "oci-sandbox:eu-frankfurt-1/sbx-7"


def test_the_state_directory_is_writable_and_kept_out_of_walks() -> None:
    assert oci_sandbox.STATE_DIR.startswith("/workspace/"), "/tmp is read-only in the sandbox"
    assert oci_sandbox.STATE_DIR.rsplit("/", 1)[-1] in SKIP_DIRS


def install_client(
    monkeypatch: pytest.MonkeyPatch, client: FakeOCISandboxClient
) -> list[tuple[Any, ...]]:
    made: list[tuple[Any, ...]] = []

    def fake(
        profile: str, auth_type: str, region: str | None, endpoint: str | None
    ) -> tuple[Any, str]:
        made.append((profile, auth_type, region, endpoint))
        return client, region or "eu-frankfurt-1"

    monkeypatch.setattr(oci_sandbox, "_client", fake)
    return made


def test_create_waits_for_running_and_owns_the_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeOCISandboxClient(states=["CREATING", "CREATING", "RUNNING"])
    made = install_client(monkeypatch, client)
    b = OCISandboxBackend.create(
        "ocid1.project", region="eu-frankfurt-1", poll_interval=0, expiration="PT30M"
    )
    [(project, details)] = client.created
    assert project == "ocid1.project"
    assert (details.runtime, details.shape, details.expires_in_minutes) == (
        "python-3.11",
        "SMALL",
        30,
    )
    assert made == [("DEFAULT", "api_key", "eu-frankfurt-1", None)]
    assert b.capabilities.label == "oci-sandbox:eu-frankfurt-1/sbx-1"
    monkeypatch.setattr(oci_sandbox.time, "sleep", lambda _: None)
    b._jobs_dir = "/nonexistent"  # nothing to stop in the fake
    b.close()
    b.close()
    assert client.stopped == ["sbx-1"]
    assert client.deleted == ["sbx-1"], "closed once, deleted once"


def test_a_sandbox_that_fails_to_start_is_deleted(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeOCISandboxClient(states=["CREATING", "FAILED"])
    install_client(monkeypatch, client)
    with pytest.raises(RuntimeError, match="FAILED"):
        OCISandboxBackend.create("ocid1.project", poll_interval=0)
    assert client.deleted == ["sbx-1"]


def test_a_sandbox_that_never_runs_times_out_and_is_deleted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeOCISandboxClient(states=["CREATING"])
    install_client(monkeypatch, client)
    monkeypatch.setattr(oci_sandbox.time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="did not run"):
        OCISandboxBackend.create("ocid1.project", ready_timeout=0.05, poll_interval=0)
    assert client.stopped == ["sbx-1"]
    assert client.deleted == ["sbx-1"]


def test_close_leaves_a_sandbox_it_did_not_make(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeOCISandboxClient()
    install_client(monkeypatch, client)
    b = OCISandboxBackend.connect("sbx-9", profile="FRANKFURT")
    b._jobs_dir = "/nonexistent"
    b.close()
    assert client.stopped == []
    assert client.deleted == []
    assert b.sandbox_id == "sbx-9"


def test_a_sandbox_already_gone_is_not_an_error() -> None:
    client = FakeOCISandboxClient()
    client.deleted.append("sbx-7")
    oci_sandbox._stop_and_delete(client, "sbx-7")


def test_extend_adds_time() -> None:
    client = FakeOCISandboxClient()
    b = OCISandboxBackend(client, "sbx-7")
    assert b.extend("PT30M") == "2026-10-25T10:00:00Z"
    assert client.extended == ["PT30M"]


def test_without_the_sdk_the_error_says_what_to_install(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "oci.generative_ai_sandbox", None)
    with pytest.raises(ImportError, match=r"locus-sdk\[oci-sandbox\]"):
        OCISandboxBackend.connect("sbx")


def test_the_client_is_built_from_the_shared_oci_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("oci.generative_ai_sandbox")
    import locus.tools.oci as oci_tools

    built: dict[str, Any] = {}

    def config_and_signer(
        module: Any, profile: str, auth_type: str, region: str | None
    ) -> tuple[dict[str, Any], Any]:
        built.update(profile=profile, auth_type=auth_type, region=region)
        return {
            "region": region,
            "tenancy": "t",
            "user": "u",
            "fingerprint": "f",
            "key_file": "k",
        }, object()

    class Client:
        def __init__(self, config: dict[str, Any], **kwargs: Any) -> None:
            built.update(config=config, kwargs=kwargs)

    monkeypatch.setattr(oci_tools, "_build_config_and_signer", config_and_signer)
    monkeypatch.setattr("oci.generative_ai_sandbox.GenerativeAiSandboxClient", Client)
    client, region = oci_sandbox._client(
        "P", "security_token", "eu-frankfurt-1", "https://example.test"
    )
    assert isinstance(client, Client)
    assert region == "eu-frankfurt-1"
    assert built["profile"] == "P"
    assert built["auth_type"] == "security_token"
    assert set(built["kwargs"]) == {"signer", "service_endpoint"}


# ---------------------------------------------------- in a real shell --


def backend(
    root: Path, client: FakeOCISandboxClient | None = None, **options: Any
) -> OCISandboxBackend:
    return OCISandboxBackend(
        client or FakeOCISandboxClient(),
        "sbx-7",
        region="eu-frankfurt-1",
        root=str(root),
        state_dir=str(root / ".locus-harness"),
        **options,
    )


@needs_posix
def test_the_sandbox_stops_a_command_at_its_deadline(root: Path) -> None:
    started = time.monotonic()
    result = backend(root).exec("sleep 30", timeout=2)
    assert result.timed_out
    assert time.monotonic() - started < 15


@needs_posix
def test_large_output_is_capped_in_the_sandbox(root: Path) -> None:
    result = backend(root, max_output_bytes=200_000).exec(
        "head -c 3000000 /dev/zero | tr '\\0' a; echo; echo END", timeout=30
    )
    assert result.exit_code == 0
    assert len(result.output) < 200_200
    assert b"bytes of output were dropped here" in result.output
    assert result.output.rstrip().endswith(b"END"), "the most recent tail is kept"


@needs_posix
def test_the_status_survives_the_capture(root: Path) -> None:
    result = backend(root).exec("echo out; echo err >&2; exit 7", timeout=10)
    assert result.exit_code == 7
    assert result.output == b"out\nerr\n"


@needs_posix
def test_background_state_stays_out_of_the_listing(root: Path) -> None:
    b = backend(root)
    job = b.start_background("echo hi")
    b.wait_background(job.handle, 10)
    b.write_bytes("app.py", b"x = 1\n")
    paths = [Path(i.path).relative_to(root).as_posix() for i in b.glob("**/*")]
    assert paths == ["app.py"]
    b.close()


@needs_posix
async def test_the_harness_runs_end_to_end_in_the_sandbox(root: Path) -> None:
    h = build_harness(backend(root))
    assert "oci-sandbox:eu-frankfurt-1/sbx-7" in h.prompt_fragment
    assert "not in a sandbox" not in h.prompt_fragment
    await h.tool("write").execute(path="app.py", content="print(40 + 2)\n")
    out = await h.tool("bash").execute(command="python3 app.py")
    assert out == "exit 0\n42"


# ----------------------------------------------------- response shapes --


class _Lines:
    def iter_lines(self) -> list[bytes]:
        return [b'{"event": "FINISHED", "seq": 1, "exitCode": 0}', b""]


class _Raw:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def read(self) -> bytes:
        return self.body


@pytest.mark.parametrize(
    "data",
    [
        _Lines(),
        SimpleNamespace(raw=_Raw(b'{"event": "FINISHED", "seq": 1, "exitCode": 0}\n')),
        b'{"event": "FINISHED", "seq": 1, "exitCode": 0}\n',
        '{"event": "FINISHED", "seq": 1, "exitCode": 0}\n',
    ],
    ids=["iter_lines", "raw", "bytes", "str"],
)
def test_every_stream_shape_the_sdk_returns_is_read(data: Any) -> None:
    client = ScriptedClient()
    client.run_sandbox_command_and_stream = lambda *a, **kw: SimpleNamespace(data=data)  # type: ignore[method-assign,assignment,misc]
    assert session(client).run("true", timeout=5) == (0, b"")


def test_a_download_reads_a_raw_body() -> None:
    client = FakeOCISandboxClient()
    client.read_sandbox_file = lambda *a, **kw: SimpleNamespace(
        data=SimpleNamespace(raw=_Raw(b"raw"))
    )  # type: ignore[method-assign,assignment,misc]
    assert session(client).download_file("/workspace/x") == b"raw"


def test_a_kill_after_the_grace_period_is_a_timeout_only_past_the_deadline() -> None:
    assert session(ScriptedClient([finished(137)])).run("x", timeout=1e-9)[0] is None
    assert session(ScriptedClient([finished(137)])).run("x", timeout=5)[0] == 137


def test_without_the_sdk_a_command_says_what_to_install(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(oci_sandbox, "_models", REAL_OCI_MODELS)
    monkeypatch.setitem(sys.modules, "oci.generative_ai_sandbox", None)
    with pytest.raises(ImportError, match=r"locus-sdk\[oci-sandbox\]"):
        session(ScriptedClient()).run("true", timeout=5)


def test_stop_and_delete_skips_what_is_not_needed() -> None:
    deleting = FakeOCISandboxClient(states=["DELETING"])
    oci_sandbox._stop_and_delete(deleting, "sbx-7")
    assert deleting.deleted == []
    stopped = FakeOCISandboxClient(states=["STOPPED"])
    oci_sandbox._stop_and_delete(stopped, "sbx-7")
    assert stopped.stopped == []
    assert stopped.deleted == ["sbx-7"]


def test_stop_and_delete_raises_what_is_not_a_404() -> None:
    client = FakeOCISandboxClient()
    client.delete_sandbox = lambda *a, **kw: (_ for _ in ()).throw(FakeServiceError(500, "boom"))  # type: ignore[method-assign,assignment,misc]
    client.states = ["STOPPED"]
    with pytest.raises(FakeServiceError):
        oci_sandbox._stop_and_delete(client, "sbx-7")


def test_a_client_without_a_signer_or_endpoint_gets_neither(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("oci.generative_ai_sandbox")
    import locus.tools.oci as oci_tools

    built: dict[str, Any] = {}

    class Client:
        def __init__(self, config: dict[str, Any], **kwargs: Any) -> None:
            built["kwargs"] = kwargs

    monkeypatch.setattr(
        oci_tools, "_build_config_and_signer", lambda *a: ({"region": "eu-frankfurt-1"}, None)
    )
    monkeypatch.setattr("oci.generative_ai_sandbox.GenerativeAiSandboxClient", Client)
    _, region = oci_sandbox._client("DEFAULT", "api_key", None, None)
    assert region == "eu-frankfurt-1"
    assert built["kwargs"] == {}


# --------------------------------------------------- cleanup guarantees --


def test_a_backend_that_cannot_be_built_never_leaves_a_sandbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeOCISandboxClient()
    install_client(monkeypatch, client)
    with pytest.raises(ValueError, match="max_output_bytes"):
        OCISandboxBackend.create("ocid1.project", poll_interval=0, max_output_bytes=10)
    assert client.created == [], "bad options are refused before a sandbox exists"


def test_a_failure_building_the_backend_deletes_the_sandbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeOCISandboxClient()
    install_client(monkeypatch, client)

    def broken(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("constructor failed")

    monkeypatch.setattr(oci_sandbox.SessionBackend, "__init__", broken)
    with pytest.raises(RuntimeError, match="constructor failed"):
        OCISandboxBackend.create("ocid1.project", poll_interval=0)
    assert client.created
    assert client.deleted == ["sbx-1"]


def test_close_retries_a_delete_that_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeOCISandboxClient()
    install_client(monkeypatch, client)
    monkeypatch.setattr(oci_sandbox.time, "sleep", lambda _: None)
    b = OCISandboxBackend.create("ocid1.project", poll_interval=0)
    b._jobs_dir = "/nonexistent"
    real_delete = client.delete_sandbox
    failures = [FakeServiceError(503, "unavailable")]

    def flaky(sandbox_id: str, **kwargs: Any) -> None:
        if failures:
            raise failures.pop()
        real_delete(sandbox_id)

    client.delete_sandbox = flaky  # type: ignore[method-assign]
    with pytest.raises(FakeServiceError):
        b.close()
    assert client.deleted == []
    b.close()
    assert client.deleted == ["sbx-1"], "the second close retries the delete"
    b.close()
    assert client.deleted == ["sbx-1"], "and a third one does nothing"


def test_close_still_deletes_when_stopping_background_commands_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeOCISandboxClient()
    install_client(monkeypatch, client)
    monkeypatch.setattr(oci_sandbox.time, "sleep", lambda _: None)
    b = OCISandboxBackend.create("ocid1.project", poll_interval=0)

    def broken_close(self: Any) -> None:
        raise ConnectionError("sandbox unreachable")

    monkeypatch.setattr(oci_sandbox.SessionBackend, "close", broken_close)
    with pytest.raises(ConnectionError):
        b.close()
    assert client.deleted == ["sbx-1"], "the sandbox goes even if its jobs could not be stopped"


@pytest.mark.parametrize(
    ("expiration", "minutes"),
    [("PT1H", 60), ("PT30M", 30), ("PT90S", 2), ("P2D", 2880), ("pt5m", 5), ("P1DT12H", 2160)],
)
def test_expiration_is_sent_as_whole_minutes(expiration: str, minutes: int) -> None:
    from locus.harness.oci_sandbox import _minutes

    assert _minutes(expiration) == minutes


@pytest.mark.parametrize("expiration", ["-1", "PT0S", "P", "PT49H", "P3D", "1h", "", "PT1.5H"])
def test_a_lifetime_outside_one_minute_to_48_hours_is_refused_before_a_sandbox_exists(
    monkeypatch: pytest.MonkeyPatch, expiration: str
) -> None:
    client = FakeOCISandboxClient(states=["RUNNING"])
    install_client(monkeypatch, client)
    with pytest.raises(ValueError, match="expiration"):
        OCISandboxBackend.create("ocid1.project", poll_interval=0, expiration=expiration)
    assert client.created == []
