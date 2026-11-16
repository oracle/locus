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
import types
from types import SimpleNamespace
from typing import Any

import pytest

from locus.harness import OCIContainerInstanceBackend, ResourceLimits, SandboxBackend, SSHJump
from locus.harness import oci_container as mod


LOG_READY = (
    "LOCUS_HOST_KEY ssh-rsa AAAAB3rsa root@box\n"
    "LOCUS_HOST_KEY ecdsa-sha2-nistp256 AAAAE2ecdsa root@box\n"
    "LOCUS_HOST_KEY ssh-ed25519 AAAAC3ed root@box\n"
    "LOCUS_SSHD_READY boot-1\n"
)

#: The log after a restart: the first boot's lines, then the second's, with
#: new host keys.
LOG_REBOOTED = (
    LOG_READY + "LOCUS_HOST_KEY ssh-ed25519 AAAAC3second root@box\nLOCUS_SSHD_READY boot-2\n"
)

#: The log after a restart of an image that carries its own host keys: the
#: same keys, a new boot.
LOG_REBOOTED_SAME_KEYS = LOG_READY.replace("boot-1", "boot-2")


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


class ServiceError(Exception):
    """The shape of the SDK's error: a ``status`` attribute."""

    def __init__(self, status: int) -> None:
        super().__init__(f"service error {status}")
        self.status = status


class FakeBastion:
    def __init__(self) -> None:
        self.created: list[Any] = []
        self.deleted: list[str] = []
        self.states = ["CREATING", "ACTIVE"]
        self.host_key_info: str | None = "ssh-rsa AAAAB3bastion"
        self.gone: set[str] = set()

    def create_session(self, details: Any) -> SimpleNamespace:
        self.created.append(details)
        return ns(data=ns(id=f"ocid1.bastionsession.test{len(self.created)}"))

    def get_session(self, session_id: str) -> SimpleNamespace:
        if session_id in self.gone:
            raise ServiceError(404)
        state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
        return ns(
            data=ns(
                id=session_id,
                lifecycle_state=state,
                bastion_user_name=session_id,
                bastion_public_host_key_info=self.host_key_info,
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
        self.error: Exception | None = None

    def run(self, command: str, *, timeout: float, **kwargs: Any) -> tuple[int | None, bytes]:
        if self.error is not None:
            raise self.error
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


def reset_instance(fakes: dict[str, Any]) -> None:
    """A fresh instance for a second ``create`` in the same test, after a delete."""
    instances = fakes["clients"].instances
    instances.states = ["CREATING", "ACTIVE"]
    instances.logs = ["", LOG_READY]


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
    assert instances.stopped == [], "a running instance is deleted as it is, not stopped first"
    assert fakes["sessions"][0].closed
    assert backend.state == "closed"


def test_create_through_a_bastion(fakes: dict[str, Any]) -> None:
    backend = create(bastion_id="ocid1.bastion.test", ssh_port=2222, bastion_session_ttl=900)
    bastion = fakes["clients"].bastion
    [details] = bastion.created
    assert details.bastion_id == "ocid1.bastion.test"
    assert details.key_type == "PUB"
    assert details.key_details.public_key_content == "ssh-ed25519 AAAAC3locus locus"
    assert details.session_ttl_in_seconds == 900, "the TTL is the caller's"
    target = details.target_resource_details
    assert (target.target_resource_private_ip_address, target.target_resource_port) == (
        "10.9.1.7",
        2222,
    )
    [(reach, _, _, _)] = fakes["opened"]
    assert reach.port == 2222
    assert reach.via == SSHJump(
        host="host.bastion.eu-frankfurt-1.oci.oraclecloud.com",
        username="ocid1.bastionsession.test1",
        pkey="PKEY",
        host_key="ssh-rsa AAAAB3bastion",
        known_hosts="strict",
    ), "the jump logs in with the key the session was created for"
    backend.close()
    assert bastion.deleted == ["ocid1.bastionsession.test1"]
    assert fakes["clients"].instances.deleted == ["ocid1.containerinstance.test"]


def test_the_default_bastion_ttl_is_three_hours(fakes: dict[str, Any]) -> None:
    create(bastion_id="ocid1.bastion.test")
    [details] = fakes["clients"].bastion.created
    assert details.session_ttl_in_seconds == mod.BASTION_SESSION_TTL == 3 * 60 * 60


def test_a_bastion_without_a_host_key_is_refused_unless_asked(
    fakes: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    bastion = fakes["clients"].bastion
    bastion.host_key_info = None
    with pytest.raises(RuntimeError, match=r'no host key.*known_hosts="accept-unknown"'):
        create(bastion_id="ocid1.bastion.test")
    assert bastion.deleted == ["ocid1.bastionsession.test1"], "the session is not left behind"
    assert fakes["clients"].instances.deleted == ["ocid1.containerinstance.test"]
    assert fakes["opened"] == []

    reset_instance(fakes)
    with caplog.at_level("WARNING"):
        create(bastion_id="ocid1.bastion.test", known_hosts="accept-unknown")
    [(reach, _, _, _)] = fakes["opened"]
    assert reach.via is not None
    assert (reach.via.host_key, reach.via.known_hosts) == (None, "accept-unknown")
    assert "bastion reported no host key" in caplog.text


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
    assert fakes["clients"].bastion.deleted == ["ocid1.bastionsession.test1"]
    assert fakes["clients"].instances.deleted == ["ocid1.containerinstance.test"]


def test_a_log_without_host_keys_is_refused_and_the_instance_deleted(
    fakes: dict[str, Any],
) -> None:
    fakes["clients"].instances.logs = ["LOCUS_SSHD_READY boot-1\n"]
    with pytest.raises(RuntimeError, match=r'no host key.*known_hosts="accept-unknown"'):
        create()
    assert fakes["opened"] == [], "nothing is connected unverified by default"
    assert fakes["clients"].instances.deleted == ["ocid1.containerinstance.test"]


def test_a_log_without_host_keys_connects_unverified_only_when_asked(
    fakes: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    fakes["clients"].instances.logs = ["LOCUS_SSHD_READY boot-1\n"]
    with caplog.at_level("WARNING"):
        create(known_hosts="accept-unknown")
    [(reach, _, _, _)] = fakes["opened"]
    assert reach.host_keys == ()
    assert reach.known_hosts == "accept-unknown"
    assert "no host key" in caplog.text


def test_a_host_key_mismatch_is_not_retried(
    fakes: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    class SSHException(Exception):  # noqa: N818 - paramiko's name is what is matched
        pass

    class BadHostKeyException(SSHException):
        pass

    attempts = 0

    def mismatch(*args: Any, **kwargs: Any) -> Any:
        nonlocal attempts
        attempts += 1
        raise BadHostKeyException("host key for 10.9.1.7 does not match")

    monkeypatch.setattr(mod, "_open_ssh", mismatch)
    with pytest.raises(BadHostKeyException, match="does not match"):
        create(ready_timeout=5)
    assert attempts == 1, "a key that does not match the pinned one is a verdict, not 'not yet'"
    assert fakes["clients"].instances.deleted == ["ocid1.containerinstance.test"]


def test_authentication_not_yet_accepted_is_retried(
    fakes: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    class SSHException(Exception):  # noqa: N818 - paramiko's name is what is matched
        pass

    class AuthenticationException(SSHException):
        pass

    attempts: list[int] = []
    real = mod._open_ssh

    def flaky(*args: Any, **kwargs: Any) -> Any:
        attempts.append(1)
        if len(attempts) < 3:
            raise AuthenticationException("not yet")
        return real(*args, **kwargs)

    monkeypatch.setattr(mod, "_open_ssh", flaky)
    create()
    assert len(attempts) == 3


def test_an_unrelated_error_while_connecting_is_raised_at_once(
    fakes: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*args: Any, **kwargs: Any) -> Any:
        raise TypeError("bad argument")

    monkeypatch.setattr(mod, "_open_ssh", broken)
    with pytest.raises(TypeError, match="bad argument"):
        create()
    assert fakes["clients"].instances.deleted == ["ocid1.containerinstance.test"]


def test_the_ready_timeout_is_a_timeout(fakes: dict[str, Any]) -> None:
    fakes["clients"].instances.logs = ["still installing\n"]
    with pytest.raises(RuntimeError, match="did not start within the ready timeout"):
        create(ready_timeout=0.05)
    assert fakes["clients"].instances.deleted == ["ocid1.containerinstance.test"]


def test_an_instance_without_the_asked_ip_is_an_error(fakes: dict[str, Any]) -> None:
    fakes["clients"].network = ns(
        get_vnic=lambda _vnic_id: ns(data=ns(private_ip="10.9.1.7", public_ip=None))
    )
    with pytest.raises(RuntimeError, match="no public IP"):
        create(public_ip=True)


def test_the_availability_domain_given_is_used(fakes: dict[str, Any]) -> None:
    create(availability_domain="AD-3")
    [details] = fakes["clients"].instances.created
    assert details.availability_domain == "AD-3"


def test_cleanup_failures_on_the_way_out_do_not_hide_the_error(
    fakes: dict[str, Any], caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    clients = fakes["clients"]

    def broken(*args: Any, **kwargs: Any) -> Any:
        raise TypeError("bad argument")

    def failing_delete(_session_id: str) -> None:
        raise ServiceError(500)

    def failing_instance_delete(_instance_id: str) -> None:
        raise ServiceError(500)

    monkeypatch.setattr(mod, "_open_ssh", broken)
    clients.bastion.delete_session = failing_delete
    clients.instances.delete_container_instance = failing_instance_delete
    with caplog.at_level("WARNING"), pytest.raises(TypeError, match="bad argument"):
        create(bastion_id="ocid1.bastion.test")
    assert "could not delete bastion session" in caplog.text
    assert "could not delete container instance" in caplog.text


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
    assert second_reach.boot == "boot-2"
    assert backend.state == "running"
    backend.exec("true", timeout=5)
    assert any("true" in c for c in fakes["sessions"][1].commands)
    backend.close()
    assert instances.deleted == ["ocid1.containerinstance.test"]


def test_a_restart_with_the_same_host_keys_is_accepted(fakes: dict[str, Any]) -> None:
    backend = create()
    backend.stop()
    instances = fakes["clients"].instances
    # The stale log first, then the new boot with the image's own keys.
    instances.logs = [LOG_READY, LOG_REBOOTED_SAME_KEYS]
    backend.start()
    second_reach = fakes["opened"][1][0]
    assert second_reach.host_keys == fakes["opened"][0][0].host_keys, "equal keys are fine"
    assert (fakes["opened"][0][0].boot, second_reach.boot) == ("boot-1", "boot-2")
    assert backend.state == "running"


def test_a_restart_waits_for_the_new_boot_not_the_stale_log(fakes: dict[str, Any]) -> None:
    backend = create(ready_timeout=0.1)
    backend.stop()
    fakes["clients"].instances.logs = [LOG_READY]
    with pytest.raises(RuntimeError, match="did not start within the ready timeout"):
        backend.start()
    assert backend.state == "stopped", "a failed start is retried from where it was"


def test_stop_over_a_dead_connection_still_stops_and_start_reconnects(
    fakes: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    backend = create()
    first = fakes["sessions"][0]
    first.error = ConnectionError("the SSH connection is closed")
    with caplog.at_level("WARNING"):
        backend.stop()
    assert "could not end the commands" in caplog.text
    assert backend.state == "stopped"
    assert first.closed
    assert fakes["clients"].instances.stopped == ["ocid1.containerinstance.test"]
    fakes["clients"].instances.logs = [LOG_REBOOTED]
    backend.start()
    assert backend.state == "running"
    assert backend.exec("true", timeout=5).exit_code == 0
    assert any("true" in c for c in fakes["sessions"][1].commands)


def test_stop_over_a_dead_connection_on_a_borrowed_instance_reconnects_on_start(
    fakes: dict[str, Any],
) -> None:
    pkey = ns(get_name=lambda: "ssh-ed25519", get_base64=lambda: "AAAAmine")
    backend = OCIContainerInstanceBackend.connect(
        "ocid1.containerinstance.mine", pkey=pkey, host_key="ssh-ed25519 AAAAhost"
    )
    fakes["sessions"][0].error = EOFError()
    backend.stop()
    assert backend.state == "stopped"
    assert fakes["clients"].instances.stopped == [], "an instance it does not own is not stopped"
    backend.start()
    assert len(fakes["sessions"]) == 2
    assert backend.exec("true", timeout=5).exit_code == 0


def test_a_stop_that_fails_for_another_reason_keeps_the_connection(
    fakes: dict[str, Any],
) -> None:
    backend = create()
    first = fakes["sessions"][0]
    first.error = RuntimeError("the state directory is read-only")
    with pytest.raises(RuntimeError, match="read-only"):
        backend.stop()
    assert backend.state == "running"
    assert not first.closed, "the connection is fine; the caller retries"
    assert fakes["clients"].instances.stopped == []
    first.error = None
    backend.stop()
    assert backend.state == "stopped"


def test_stop_waits_for_an_instance_the_service_is_updating(fakes: dict[str, Any]) -> None:
    backend = create()
    instances = fakes["clients"].instances
    instances.states = ["UPDATING", "UPDATING", "INACTIVE"]
    backend.stop()
    assert instances.stopped == [], "not told to stop while UPDATING; waited for INACTIVE"
    assert backend.state == "stopped"


def test_start_of_an_instance_that_is_already_active_only_reconnects(
    fakes: dict[str, Any],
) -> None:
    backend = create()
    backend.stop()
    instances = fakes["clients"].instances
    instances.states = ["ACTIVE"]
    instances.logs = [LOG_REBOOTED]
    backend.start()
    assert instances.started == []
    assert backend.state == "running"


def test_a_backend_without_a_lifecycle_or_a_reconnect_says_so(fakes: dict[str, Any]) -> None:
    backend = OCIContainerInstanceBackend(FakeSSHSession(), "ocid1.containerinstance.plain")  # type: ignore[arg-type]
    assert backend.client is None
    backend.stop()
    with pytest.raises(RuntimeError, match="without reconnect="):
        backend.start()
    assert backend.state == "stopped"
    backend.close()
    assert backend.state == "closed"


def test_a_bastion_session_that_cannot_be_deleted_is_logged_unless_gone(
    fakes: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    bastion = fakes["clients"].bastion
    backend = create(bastion_id="ocid1.bastion.test")

    def gone(_session_id: str) -> None:
        raise ServiceError(404)

    bastion.delete_session = gone
    with caplog.at_level("WARNING"):
        backend.close()
    assert "could not delete bastion session" not in caplog.text, "gone already is fine"

    reset_instance(fakes)
    backend = create(bastion_id="ocid1.bastion.test")

    def broken(_session_id: str) -> None:
        raise ServiceError(500)

    bastion.delete_session = broken
    with caplog.at_level("WARNING"):
        backend.close()
    assert "could not delete bastion session" in caplog.text
    assert backend.state == "closed"


def test_a_restart_after_the_bastion_session_expired_makes_a_new_one(
    fakes: dict[str, Any],
) -> None:
    bastion = fakes["clients"].bastion
    backend = create(bastion_id="ocid1.bastion.test")
    backend.stop()
    bastion.gone.add("ocid1.bastionsession.test1")
    bastion.states = ["ACTIVE"]
    fakes["clients"].instances.logs = [LOG_REBOOTED]
    backend.start()
    assert len(bastion.created) == 2, "a new session for the new connection"
    second_reach = fakes["opened"][1][0]
    assert second_reach.via is not None
    assert second_reach.via.username == "ocid1.bastionsession.test2"
    assert second_reach.bastion_session_id == "ocid1.bastionsession.test2"
    backend.close()
    assert bastion.deleted == ["ocid1.bastionsession.test2"], "the new session is the one deleted"


def test_a_restart_within_the_bastion_session_reuses_it(fakes: dict[str, Any]) -> None:
    bastion = fakes["clients"].bastion
    backend = create(bastion_id="ocid1.bastion.test")
    backend.stop()
    bastion.states = ["ACTIVE"]
    fakes["clients"].instances.logs = [LOG_REBOOTED]
    backend.start()
    assert len(bastion.created) == 1
    second_reach = fakes["opened"][1][0]
    assert second_reach.via == fakes["opened"][0][0].via
    assert second_reach.bastion_session_id == "ocid1.bastionsession.test1"
    backend.close()
    assert bastion.deleted == ["ocid1.bastionsession.test1"]


def test_a_deleted_bastion_session_counts_as_expired(fakes: dict[str, Any]) -> None:
    bastion = fakes["clients"].bastion
    bastion.states = ["DELETED"]
    assert mod._bastion_session_alive(fakes["clients"], "ocid1.bastionsession.test1") is False
    bastion.states = ["ACTIVE"]
    assert mod._bastion_session_alive(fakes["clients"], "ocid1.bastionsession.test1") is True

    def broken(_session_id: str) -> None:
        raise ServiceError(500)

    bastion.get_session = broken
    with pytest.raises(ServiceError):
        mod._bastion_session_alive(fakes["clients"], "ocid1.bastionsession.test1")


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
    assert reach.via.username == "ocid1.bastionsession.test1"
    assert reach.via.pkey is pkey
    [details] = fakes["clients"].bastion.created
    assert details.key_details.public_key_content == "ssh-ed25519 AAAAmine"
    assert details.session_ttl_in_seconds == mod.BASTION_SESSION_TTL
    backend.close()
    assert fakes["clients"].bastion.deleted == ["ocid1.bastionsession.test1"]
    assert fakes["clients"].instances.deleted == []


def test_connect_needs_a_key(fakes: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="pkey= or key_filename="):
        OCIContainerInstanceBackend.connect("ocid1.containerinstance.mine")


def test_connect_loads_the_key_file_with_a_tilde_expanded(
    fakes: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    loaded: list[str] = []

    def from_path(path: str) -> Any:
        loaded.append(path)
        return ns(get_name=lambda: "ssh-ed25519", get_base64=lambda: "AAAAfile")

    monkeypatch.setattr(mod, "_paramiko", lambda: ns(PKey=ns(from_path=from_path)))
    backend = OCIContainerInstanceBackend.connect(
        "ocid1.containerinstance.mine",
        key_filename="~/.ssh/sandbox",
        bastion_id="ocid1.bastion.test",
        bastion_session_ttl=1800,
    )
    assert loaded == [str(tmp_path / ".ssh" / "sandbox")]
    [details] = fakes["clients"].bastion.created
    assert details.key_details.public_key_content == "ssh-ed25519 AAAAfile"
    assert details.session_ttl_in_seconds == 1800
    backend.close()


def test_connect_that_cannot_reach_sshd_deletes_its_bastion_session(
    fakes: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def refused(*args: Any, **kwargs: Any) -> Any:
        raise OSError("connection refused")

    monkeypatch.setattr(mod, "_open_ssh", refused)
    pkey = ns(get_name=lambda: "ssh-ed25519", get_base64=lambda: "AAAAmine")
    with pytest.raises(OSError, match="refused"):
        OCIContainerInstanceBackend.connect(
            "ocid1.containerinstance.mine", pkey=pkey, bastion_id="ocid1.bastion.test"
        )
    assert fakes["clients"].bastion.deleted == ["ocid1.bastionsession.test1"]


def test_a_borrowed_instance_restarts_through_a_new_bastion_session_when_needed(
    fakes: dict[str, Any],
) -> None:
    bastion = fakes["clients"].bastion
    pkey = ns(get_name=lambda: "ssh-ed25519", get_base64=lambda: "AAAAmine")
    backend = OCIContainerInstanceBackend.connect(
        "ocid1.containerinstance.mine", pkey=pkey, bastion_id="ocid1.bastion.test"
    )
    backend.stop()
    bastion.gone.add("ocid1.bastionsession.test1")
    backend.start()
    assert len(bastion.created) == 2
    second_reach = fakes["opened"][1][0]
    assert second_reach.via is not None
    assert second_reach.via.username == "ocid1.bastionsession.test2"
    backend.close()
    assert bastion.deleted == ["ocid1.bastionsession.test2"]


# ----------------------------------------------------------------- helpers --


def test_the_latest_boot_is_the_one_read() -> None:
    lines, boot = mod._latest_boot(LOG_REBOOTED)
    assert lines.strip().endswith("AAAAC3second root@box")
    assert boot == "boot-2"
    assert mod._latest_boot(LOG_READY) == (LOG_READY.partition("LOCUS_SSHD_READY")[0], "boot-1")
    assert mod._latest_boot("no ready mark yet") == ("no ready mark yet", "")
    assert mod._latest_boot("LOCUS_SSHD_READY\n")[1] == "", "an old bootstrap without an id"


def test_the_bootstrap_prints_a_boot_id_with_the_ready_mark() -> None:
    assert 'echo "LOCUS_SSHD_READY $(' in mod.BOOTSTRAP


def test_transport_errors_are_told_from_verdicts() -> None:
    class SSHException(Exception):  # noqa: N818 - paramiko's name is what is matched
        pass

    class BadHostKeyException(SSHException):
        pass

    assert mod._transient(OSError("refused"))
    assert mod._transient(EOFError())
    assert mod._transient(SSHException("not yet"))
    assert not mod._transient(BadHostKeyException("mismatch"))
    assert not mod._transient(TypeError("bug"))


def test_the_service_error_status_is_read_when_it_is_one() -> None:
    assert mod._status(ServiceError(404)) == 404
    assert mod._status(ValueError("plain")) is None
    assert mod._status(ns(status="teapot")) is None


def test_waiting_for_a_state_gives_up_at_the_deadline() -> None:
    with pytest.raises(RuntimeError, match=r"did not become ACTIVE within 0\.01s"):
        mod._wait(
            lambda: "creating", "ACTIVE", failed=frozenset(), timeout=0.01, interval=0, what="it"
        )


def test_deleting_an_instance_tolerates_one_that_is_gone_or_going(fakes: dict[str, Any]) -> None:
    clients = fakes["clients"]
    clients.instances.states = ["DELETING", "DELETED"]
    mod._delete_instance(clients, "ocid1.containerinstance.test")
    assert clients.instances.deleted == [], "already on its way out"

    def gone(_instance_id: str) -> Any:
        raise ServiceError(404)

    clients.instances.get_container_instance = gone
    mod._delete_instance(clients, "ocid1.containerinstance.test")

    def broken(_instance_id: str) -> Any:
        raise ServiceError(500)

    clients.instances.get_container_instance = broken
    with pytest.raises(ServiceError):
        mod._delete_instance(clients, "ocid1.containerinstance.test")


def test_open_ssh_pins_the_first_key_and_passes_the_jump(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_open(cls: Any, host: str, **kwargs: Any) -> str:
        calls.append({"host": host, **kwargs})
        return "SESSION"

    monkeypatch.setattr(mod.SSHSession, "open", classmethod(fake_open))
    jump = SSHJump("host.bastion.example", username="s")
    pinned = mod._Reach("10.9.1.7", 2222, ("ssh-ed25519 A", "ssh-rsa B"), via=jump, username="u")
    assert mod._open_ssh(pinned, "PKEY", connect_timeout=3, path="/opt/bin") == "SESSION"
    assert calls[-1] == {
        "host": "10.9.1.7",
        "port": 2222,
        "username": "u",
        "pkey": "PKEY",
        "host_key": "ssh-ed25519 A",
        "known_hosts": "strict",
        "via": jump,
        "connect_timeout": 3,
        "path": "/opt/bin",
    }
    unpinned = mod._Reach("10.9.1.7", 22, (), known_hosts="accept-unknown")
    mod._open_ssh(unpinned, "PKEY", connect_timeout=3, path=None)
    assert (calls[-1]["host_key"], calls[-1]["known_hosts"]) == (None, "accept-unknown")


def test_the_clients_are_built_from_one_config(monkeypatch: pytest.MonkeyPatch) -> None:
    class Client:
        def __init__(self, config: dict[str, Any], **kwargs: Any) -> None:
            self.config = config
            self.kwargs = kwargs

    oci = types.ModuleType("oci")
    container_instances = types.ModuleType("oci.container_instances")
    container_instances.ContainerInstanceClient = Client  # type: ignore[attr-defined]
    container_instances.models = "CI_MODELS"  # type: ignore[attr-defined]
    bastion = types.ModuleType("oci.bastion")
    bastion.BastionClient = Client  # type: ignore[attr-defined]
    bastion.models = "BASTION_MODELS"  # type: ignore[attr-defined]
    oci.container_instances = container_instances  # type: ignore[attr-defined]
    oci.bastion = bastion  # type: ignore[attr-defined]
    oci.core = ns(VirtualNetworkClient=Client)  # type: ignore[attr-defined]
    oci.identity = ns(IdentityClient=Client)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "oci", oci)
    monkeypatch.setitem(sys.modules, "oci.container_instances", container_instances)
    monkeypatch.setitem(sys.modules, "oci.bastion", bastion)
    asked: list[tuple[Any, ...]] = []

    def build(oci_mod: Any, profile: str, auth_type: str, region: str | None) -> tuple[Any, Any]:
        asked.append((oci_mod, profile, auth_type, region))
        return {"region": region or "eu-frankfurt-1"}, (
            "SIGNER" if auth_type != "api_key" else None
        )

    monkeypatch.setattr("locus.tools.oci._build_config_and_signer", build)

    clients = mod._clients("sandbox", "api_key", None)
    assert asked == [(oci, "sandbox", "api_key", None)]
    assert clients.region == "eu-frankfurt-1"
    assert isinstance(clients.instances, Client)
    assert clients.instances.kwargs == {}, "an API key signs through the config"
    assert isinstance(clients.network, Client)
    assert isinstance(clients.identity, Client)
    assert isinstance(clients.bastion, Client)

    signed = mod._clients("sandbox", "security_token", "us-ashburn-1")
    assert signed.region == "us-ashburn-1"
    assert signed.instances.kwargs == {"signer": "SIGNER"}
    assert mod._models() == "CI_MODELS"
    assert mod._bastion_models() == "BASTION_MODELS"


def test_the_key_pair_is_an_ed25519_key_in_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("cryptography")
    seen: list[str] = []

    def from_private_key(handle: Any) -> str:
        seen.append(handle.read())
        return "PKEY"

    monkeypatch.setattr(
        mod, "_paramiko", lambda: ns(Ed25519Key=ns(from_private_key=from_private_key))
    )
    pkey, public = mod._keypair()
    assert pkey == "PKEY"
    [pem] = seen
    # The marker is assembled so the secret scanner does not read a key here.
    assert pem.startswith("-----BEGIN OPENSSH " + "PRIVATE KEY-----")
    assert public.startswith("ssh-ed25519 AAAA")
    assert "\n" not in public


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
