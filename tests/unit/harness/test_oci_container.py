# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""OCIContainerInstanceBackend: provisioning, the bootstrap, host keys, the bastion, teardown.

The service clients, the key pair and the SSH connection are stood in; what
is pinned is what the backend asks of each and in what order. The live test
under ``tests/integration`` runs the conformance kit against a real instance.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

from locus.harness import OCIContainerInstanceBackend, ResourceLimits, SandboxBackend, SSHJump
from locus.harness import oci_container as mod


LOG_READY = (
    "LOCUS_HOST_KEY ssh-rsa AAAAB3rsa root@box\n"
    "LOCUS_HOST_KEY ecdsa-sha2-nistp256 AAAAE2ecdsa root@box\n"
    "LOCUS_HOST_KEY ssh-ed25519 AAAAC3ed root@box\n"
    "LOCUS_SSHD_READY\n"
)

#: The log after a restart: the first boot's lines, then the second's.
LOG_REBOOTED = LOG_READY + "LOCUS_HOST_KEY ssh-ed25519 AAAAC3second root@box\nLOCUS_SSHD_READY\n"


def ns(**fields: Any) -> SimpleNamespace:
    return SimpleNamespace(**fields)


class FakeInstances:
    def __init__(self, states: list[str] | None = None, logs: list[str] | None = None) -> None:
        self.states = states or ["CREATING", "ACTIVE"]
        self.logs = logs if logs is not None else ["", LOG_READY]
        self.created: list[Any] = []
        self.deleted: list[str] = []
        self.started: list[str] = []
        self.stopped: list[str] = []
        self.log_reads = 0

    def _state(self) -> str:
        return self.states.pop(0) if len(self.states) > 1 else self.states[0]

    def create_container_instance(self, details: Any) -> SimpleNamespace:
        self.created.append(details)
        return ns(data=ns(id="ocid1.containerinstance.test"))

    def get_container_instance(self, instance_id: str) -> SimpleNamespace:
        return ns(
            data=ns(
                id=instance_id,
                lifecycle_state=self._state(),
                vnics=[ns(vnic_id="ocid1.vnic.test")],
                containers=[ns(container_id="ocid1.container.test")],
            )
        )

    def retrieve_logs(self, container_id: str) -> SimpleNamespace:
        self.log_reads += 1
        text = self.logs.pop(0) if len(self.logs) > 1 else self.logs[0]
        return ns(data=ns(content=text.encode()))

    def delete_container_instance(self, instance_id: str) -> None:
        self.deleted.append(instance_id)
        self.states = ["DELETING", "DELETED"]

    def start_container_instance(self, instance_id: str) -> None:
        self.started.append(instance_id)
        self.states = ["ACTIVE"]

    def stop_container_instance(self, instance_id: str) -> None:
        self.stopped.append(instance_id)
        self.states = ["INACTIVE"]


class FakeNetwork:
    def get_vnic(self, vnic_id: str) -> SimpleNamespace:
        return ns(data=ns(private_ip="10.9.1.7", public_ip="203.0.113.9"))


class FakeIdentity:
    def list_availability_domains(self, compartment_id: str) -> SimpleNamespace:
        return ns(data=[ns(name="AD-1"), ns(name="AD-2")])


class FakeBastion:
    def __init__(self) -> None:
        self.created: list[Any] = []
        self.deleted: list[str] = []
        self.states = ["CREATING", "ACTIVE"]

    def create_session(self, details: Any) -> SimpleNamespace:
        self.created.append(details)
        return ns(data=ns(id="ocid1.bastionsession.test"))

    def get_session(self, session_id: str) -> SimpleNamespace:
        state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
        return ns(
            data=ns(
                id=session_id,
                lifecycle_state=state,
                bastion_user_name=session_id,
                bastion_public_host_key_info="ssh-rsa AAAAB3bastion",
                ssh_metadata={
                    "command": f"ssh -i <privateKey> -N -L <localPort>:10.9.1.7:22 -p 22 "
                    f"{session_id}@host.bastion.eu-frankfurt-1.oci.oraclecloud.com"
                },
            )
        )

    def delete_session(self, session_id: str) -> None:
        self.deleted.append(session_id)


class FakeSSHSession:
    """The SSH session the backend works through once the instance is up."""

    def __init__(self) -> None:
        self.commands: list[str] = []
        self.closed = False
        self.where = "root@10.9.1.7"

    def run(self, command: str, *, timeout: float, **kwargs: Any) -> tuple[int | None, bytes]:
        self.commands.append(command)
        return 0, b""

    def exec(self, command: str, *, timeout: float) -> tuple[int | None, bytes]:
        return self.run(command, timeout=timeout)

    def upload_file(self, path: str, data: bytes) -> None:
        pass

    def download_file(self, path: str) -> bytes:
        return b""

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    clients = mod._Clients(
        instances=FakeInstances(),
        network=FakeNetwork(),
        identity=FakeIdentity(),
        bastion=FakeBastion(),
        region="eu-frankfurt-1",
    )
    opened: list[tuple[Any, ...]] = []
    sessions: list[FakeSSHSession] = []

    def open_ssh(reach: Any, pkey: Any, *, connect_timeout: float, path: str | None) -> Any:
        opened.append((reach, pkey, connect_timeout, path))
        session = FakeSSHSession()
        sessions.append(session)
        return session

    monkeypatch.setattr(mod, "_clients", lambda *args: clients)
    monkeypatch.setattr(mod, "_keypair", lambda: ("PKEY", "ssh-ed25519 AAAAC3locus locus"))
    monkeypatch.setattr(
        mod,
        "_models",
        lambda: ns(
            CreateContainerInstanceDetails=ns,
            CreateContainerInstanceShapeConfigDetails=ns,
            CreateContainerDetails=ns,
            CreateContainerVnicDetails=ns,
        ),
    )
    monkeypatch.setattr(
        mod,
        "_bastion_models",
        lambda: ns(
            CreateSessionDetails=ns,
            PublicKeyDetails=ns,
            CreatePortForwardingSessionTargetResourceDetails=ns,
        ),
    )
    monkeypatch.setattr(mod, "_open_ssh", open_ssh)
    monkeypatch.setattr(mod.time, "sleep", lambda _s: None)
    return {"clients": clients, "opened": opened, "sessions": sessions}


def create(**options: Any) -> OCIContainerInstanceBackend:
    return OCIContainerInstanceBackend.create(
        "ocid1.compartment.test", "ocid1.subnet.test", poll_interval=0, **options
    )


# ------------------------------------------------------------------ create --


def test_create_provisions_a_bootstrapped_instance_and_connects(fakes: dict[str, Any]) -> None:
    backend = create(ocpus=2, memory_gb=4)
    instances = fakes["clients"].instances
    [details] = instances.created
    assert details.compartment_id == "ocid1.compartment.test"
    assert details.availability_domain == "AD-1", "the compartment's first AD when none is given"
    assert details.shape == mod.DEFAULT_SHAPE
    assert (details.shape_config.ocpus, details.shape_config.memory_in_gbs) == (2, 4)
    assert details.container_restart_policy == "NEVER"
    [container] = details.containers
    assert container.image_url == mod.DEFAULT_IMAGE
    assert container.command[:2] == ["/bin/sh", "-c"]
    script = container.command[2]
    assert "openssh-server" in script
    assert "ssh-keygen -A" in script
    assert "LOCUS_HOST_KEY" in script
    assert "LOCUS_SSHD_READY" in script
    assert "PermitRootLogin=prohibit-password" in script
    assert container.environment_variables == {
        "LOCUS_AUTHORIZED_KEY": "ssh-ed25519 AAAAC3locus locus",
        "LOCUS_SSH_PORT": "22",
        "LOCUS_WORKSPACE": "/workspace",
    }
    [vnic] = details.vnics
    assert (vnic.subnet_id, vnic.is_public_ip_assigned) == ("ocid1.subnet.test", False)
    assert details.display_name.startswith("locus-harness-")

    [(reach, pkey, _, _)] = fakes["opened"]
    assert (reach.host, reach.port, pkey) == ("10.9.1.7", 22, "PKEY")
    assert reach.host_keys[0] == "ssh-ed25519 AAAAC3ed", "ed25519 is pinned first"
    assert set(reach.host_keys) == {
        "ssh-ed25519 AAAAC3ed",
        "ssh-rsa AAAAB3rsa",
        "ecdsa-sha2-nistp256 AAAAE2ecdsa",
    }
    assert reach.known_hosts == "strict"
    assert reach.via is None
    assert instances.log_reads == 2, "logs are read until the ready mark appears"

    assert isinstance(backend, SandboxBackend)
    assert backend.capabilities.label == "oci-container:eu-frankfurt-1/ocid1.containerinstance.test"
    assert backend.capabilities.root == "/workspace"
    assert backend.limits == ResourceLimits(cpus=2, memory_mb=4096)
    assert backend.instance_id == "ocid1.containerinstance.test"
    assert backend.client is instances


def test_close_deletes_the_instance_and_waits_for_it(fakes: dict[str, Any]) -> None:
    backend = create()
    backend.close()
    backend.close()
    instances = fakes["clients"].instances
    assert instances.deleted == ["ocid1.containerinstance.test"], "deleted once"
    assert fakes["sessions"][0].closed
    assert backend.state == "closed"


def test_create_through_a_bastion(fakes: dict[str, Any]) -> None:
    backend = create(bastion_id="ocid1.bastion.test", ssh_port=2222)
    bastion = fakes["clients"].bastion
    [details] = bastion.created
    assert details.bastion_id == "ocid1.bastion.test"
    assert details.key_type == "PUB"
    assert details.key_details.public_key_content == "ssh-ed25519 AAAAC3locus locus"
    target = details.target_resource_details
    assert (target.target_resource_private_ip_address, target.target_resource_port) == (
        "10.9.1.7",
        2222,
    )
    [(reach, _, _, _)] = fakes["opened"]
    assert reach.port == 2222
    assert reach.via == SSHJump(
        host="host.bastion.eu-frankfurt-1.oci.oraclecloud.com",
        username="ocid1.bastionsession.test",
        pkey="PKEY",
        host_key="ssh-rsa AAAAB3bastion",
        known_hosts="strict",
    ), "the jump logs in with the key the session was created for"
    backend.close()
    assert bastion.deleted == ["ocid1.bastionsession.test"]
    assert fakes["clients"].instances.deleted == ["ocid1.containerinstance.test"]


def test_a_public_ip_is_used_when_asked(fakes: dict[str, Any]) -> None:
    create(public_ip=True)
    [(reach, _, _, _)] = fakes["opened"]
    assert reach.host == "203.0.113.9"
    [details] = fakes["clients"].instances.created
    assert details.vnics[0].is_public_ip_assigned is True


def test_an_instance_that_fails_is_deleted(fakes: dict[str, Any]) -> None:
    fakes["clients"].instances.states = ["CREATING", "FAILED"]
    with pytest.raises(RuntimeError, match="FAILED"):
        create()
    assert fakes["clients"].instances.deleted == ["ocid1.containerinstance.test"]
    assert fakes["opened"] == []


def test_a_bootstrap_that_cannot_get_sshd_is_reported_and_deleted(fakes: dict[str, Any]) -> None:
    fakes["clients"].instances.logs = [
        "LOCUS_BOOTSTRAP_FAILED no sshd and no known package manager\n"
    ]
    with pytest.raises(RuntimeError, match="could not start sshd"):
        create()
    assert fakes["clients"].instances.deleted == ["ocid1.containerinstance.test"]


def test_an_sshd_that_never_answers_is_a_timeout_and_the_instance_is_deleted(
    fakes: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def refused(*args: Any, **kwargs: Any) -> Any:
        raise OSError("connection refused")

    monkeypatch.setattr(mod, "_open_ssh", refused)
    with pytest.raises(RuntimeError, match="could not reach sshd"):
        create(ready_timeout=0.05)
    assert fakes["clients"].instances.deleted == ["ocid1.containerinstance.test"]


def test_a_bastion_failure_deletes_the_session_and_the_instance(fakes: dict[str, Any]) -> None:
    fakes["clients"].bastion.states = ["CREATING", "FAILED"]
    with pytest.raises(RuntimeError, match="bastion session"):
        create(bastion_id="ocid1.bastion.test")
    assert fakes["clients"].bastion.deleted == ["ocid1.bastionsession.test"]
    assert fakes["clients"].instances.deleted == ["ocid1.containerinstance.test"]


def test_a_log_without_host_keys_connects_unverified_with_a_warning(
    fakes: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    fakes["clients"].instances.logs = ["LOCUS_SSHD_READY\n"]
    with caplog.at_level("WARNING"):
        create()
    [(reach, _, _, _)] = fakes["opened"]
    assert reach.host_keys == ()
    assert reach.known_hosts == "accept-unknown"
    assert "no host key" in caplog.text


# --------------------------------------------------------------- lifecycle --


def test_stop_and_start_restart_the_instance_and_reconnect(fakes: dict[str, Any]) -> None:
    backend = create()
    first = fakes["sessions"][0]
    backend.stop()
    instances = fakes["clients"].instances
    assert instances.stopped == ["ocid1.containerinstance.test"]
    assert first.closed
    assert backend.state == "stopped"
    instances.logs = [LOG_READY, LOG_REBOOTED]
    backend.start()
    assert instances.started == ["ocid1.containerinstance.test"]
    assert len(fakes["sessions"]) == 2, "a new connection to the restarted container"
    assert backend.ssh is fakes["sessions"][1]
    second_reach = fakes["opened"][1][0]
    assert second_reach.host_keys == ("ssh-ed25519 AAAAC3second",), (
        "the restarted container's own keys are pinned, not the first boot's"
    )
    assert backend.state == "running"
    backend.exec("true", timeout=5)
    assert any("true" in c for c in fakes["sessions"][1].commands)
    backend.close()
    assert instances.deleted == ["ocid1.containerinstance.test"]


# ----------------------------------------------------------------- connect --


def test_connect_attaches_to_an_instance_and_never_deletes_it(fakes: dict[str, Any]) -> None:
    pkey = ns(get_name=lambda: "ssh-ed25519", get_base64=lambda: "AAAAmine")
    backend = OCIContainerInstanceBackend.connect(
        "ocid1.containerinstance.mine",
        pkey=pkey,
        host_key="ssh-ed25519 AAAAhost",
        bastion_id="ocid1.bastion.test",
    )
    [(reach, key, _, _)] = fakes["opened"]
    assert key is pkey
    assert (reach.host, reach.username, reach.host_keys) == (
        "10.9.1.7",
        "root",
        ("ssh-ed25519 AAAAhost",),
    )
    assert reach.via is not None
    assert reach.via.username == "ocid1.bastionsession.test"
    assert reach.via.pkey is pkey
    [details] = fakes["clients"].bastion.created
    assert details.key_details.public_key_content == "ssh-ed25519 AAAAmine"
    backend.close()
    assert fakes["clients"].bastion.deleted == ["ocid1.bastionsession.test"]
    assert fakes["clients"].instances.deleted == []


def test_connect_needs_a_key(fakes: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="pkey= or key_filename="):
        OCIContainerInstanceBackend.connect("ocid1.containerinstance.mine")


# ----------------------------------------------------------------- helpers --


def test_the_latest_boot_is_the_one_read() -> None:
    assert mod._last_boot(LOG_REBOOTED).strip().endswith("AAAAC3second root@box")
    assert mod._last_boot(LOG_READY) == LOG_READY.partition("LOCUS_SSHD_READY")[0]
    assert mod._last_boot("no ready mark yet") == "no ready mark yet"


def test_host_keys_are_parsed_and_ed25519_comes_first() -> None:
    keys = mod._host_keys("noise\n" + LOG_READY + "LOCUS_HOST_KEY garbage\n")
    assert keys[0] == "ssh-ed25519 AAAAC3ed"
    assert len(keys) == 3


def test_the_bastion_host_key_is_found_in_whatever_the_service_says() -> None:
    assert mod._public_key_line("ssh-rsa AAAAB3 comment") == "ssh-rsa AAAAB3"
    assert mod._public_key_line("host.bastion ssh-ed25519 AAAAC3") == "ssh-ed25519 AAAAC3"
    assert mod._public_key_line(None) is None
    assert mod._public_key_line("nothing useful") is None


def test_logs_are_read_whatever_shape_the_sdk_returns() -> None:
    assert mod._text(ns(content=b"bytes")) == "bytes"
    assert mod._text(ns(content="text")) == "text"
    assert mod._text(ns(raw=ns(read=lambda: b"raw"))) == "raw"
    assert mod._text("plain") == "plain"


def test_without_the_sdk_the_error_says_what_to_install(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "oci", None)
    with pytest.raises(ImportError, match=r"locus-sdk\[oci-container\]"):
        OCIContainerInstanceBackend.create("ocid1.compartment", "ocid1.subnet")
