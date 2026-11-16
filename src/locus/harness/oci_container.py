# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""An OCI Container Instance as a workspace: the Oracle-native remote sandbox.

`OCI Container Instances <https://docs.oracle.com/en-us/iaas/Content/container-instances/home.htm>`_
run a container image on serverless compute in your tenancy, on a VNIC in
a subnet you choose, with the CPUs and memory you ask for. The service has
no command API — it starts a container and shows its log — so
:class:`OCIContainerInstanceBackend` provisions the instance with the
``oci`` SDK and works in it over SSH
(:class:`~locus.harness.ssh.SSHSandboxBackend`, the ``paramiko`` client;
``pip install "locus-sdk[oci-container]"`` brings both).

:meth:`OCIContainerInstanceBackend.create` does the whole thing:

1. Generates an SSH key pair for this backend alone, in memory.
2. Creates the container instance from an image that has ``sshd`` or a
   package manager that can install it (the default is Oracle Linux 9 from
   Oracle's registry). The container's command is a bootstrap script that
   installs ``sshd`` if needed, authorizes the generated public key for
   ``root``, creates the workspace directory, **prints the host's public
   keys to the container log** and runs ``sshd`` in the foreground.
3. Waits for the instance to be ``ACTIVE``, reads the host keys from the
   log, and connects with them pinned: the host key is verified against
   what the container itself reported, over the authenticated service API,
   so there is no trust-on-first-use. A log without a key is an error, not
   an unverified connection, unless ``known_hosts="accept-unknown"`` says
   so; the same goes for a bastion that reports no host key.
4. Connects directly to the instance's IP, or through an
   `OCI Bastion <https://docs.oracle.com/en-us/iaas/Content/Bastion/home.htm>`_
   when ``bastion_id`` is given: a port-forwarding session to the
   instance's private IP is created with the same key pair, and the SSH
   connection is a channel through the bastion. That is the shape for an
   instance in a private subnet, which is where a sandbox belongs.

``close()`` ends the harness's background commands, closes the connection,
deletes the bastion session and deletes the instance — a running instance
is deleted as it is, not stopped first; ``stop()`` and ``start()`` stop and
restart the instance, which **loses the container's filesystem**: the
service restarts the container from its image, the bootstrap runs again
and the backend reconnects to a fresh workspace, through a new bastion
session when the previous one has expired.

Everything from the ``oci`` SDK is imported when a backend is built, never
at import time. The container's bootstrap needs network egress to install
``sshd`` unless the image already has it: a NAT or internet gateway on the
subnet's route table.
"""

from __future__ import annotations

import io
import logging
import re
import shlex
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from locus.harness.local import MAX_BACKGROUND
from locus.harness.sandbox import ResourceLimits
from locus.harness.ssh import KnownHosts, SSHJump, SSHSandboxBackend, SSHSession, _paramiko


__all__ = ["OCIContainerInstanceBackend"]

logger = logging.getLogger(__name__)

#: The image the sandbox runs when none is given: Oracle Linux 9 from
#: Oracle's public registry, which has ``dnf`` to install ``sshd``.
DEFAULT_IMAGE = "container-registry.oracle.com/os/oraclelinux:9"

#: The shape when none is given; flexible, so CPUs and memory are chosen.
DEFAULT_SHAPE = "CI.Standard.E4.Flex"

#: The workspace directory in the container; created by the bootstrap.
SANDBOX_ROOT = "/workspace"

#: What the bootstrap prints before each host public key.
HOST_KEY_MARK = "LOCUS_HOST_KEY "

#: What the bootstrap prints when ``sshd`` is about to start, followed by an
#: id for this boot, so a restart is told from the previous boot's log.
READY_MARK = "LOCUS_SSHD_READY"

#: What the bootstrap prints when it could not get ``sshd``.
FAILED_MARK = "LOCUS_BOOTSTRAP_FAILED"

#: Seconds a bastion session lives when ``bastion_session_ttl`` is not
#: given; the backend deletes it sooner on close, and makes a new one when
#: a restart finds it expired.
BASTION_SESSION_TTL = 3 * 60 * 60

#: Seconds a stop or a start of the instance may take.
LIFECYCLE_TIMEOUT = 600.0

_ACTIVE = "ACTIVE"
_INACTIVE = "INACTIVE"
_UPDATING = "UPDATING"
_DELETED = "DELETED"
_FAILED = frozenset({"FAILED", "DELETING", "DELETED"})

_q = shlex.quote

#: The container's command. ``sshd`` is installed with whichever package
#: manager the image has, unless it is already there; the host keys and a
#: ready mark go to the log, which the backend reads through the service.
BOOTSTRAP = r"""
set -e
if ! command -v sshd >/dev/null 2>&1; then
  if command -v dnf >/dev/null 2>&1; then dnf -y -q install openssh-server >/dev/null
  elif command -v microdnf >/dev/null 2>&1; then microdnf -y install openssh-server >/dev/null
  elif command -v apt-get >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get -qq update >/dev/null && apt-get -qq install -y openssh-server >/dev/null
  elif command -v apk >/dev/null 2>&1; then apk add -q openssh
  else echo "LOCUS_BOOTSTRAP_FAILED no sshd and no known package manager"; exit 1; fi
fi
ssh-keygen -A >/dev/null 2>&1
mkdir -p /run/sshd /root/.ssh "$LOCUS_WORKSPACE"
chmod 700 /root/.ssh
printf '%s\n' "$LOCUS_AUTHORIZED_KEY" > /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys
for k in /etc/ssh/ssh_host_*_key.pub; do echo "LOCUS_HOST_KEY $(cut -d' ' -f1,2 < "$k")"; done
echo "LOCUS_SSHD_READY $(cat /proc/sys/kernel/random/uuid 2>/dev/null || date +%s)"
exec "$(command -v sshd)" -D -e -p "$LOCUS_SSH_PORT" \
  -o PermitRootLogin=prohibit-password -o PasswordAuthentication=no -o PubkeyAuthentication=yes
""".strip()


@dataclass
class _Clients:
    """The service clients one backend uses, built from one OCI config."""

    instances: Any
    network: Any
    identity: Any
    bastion: Any
    region: str | None


@dataclass(frozen=True)
class _Reach:
    """How to connect to the instance's ``sshd``: directly, or through a bastion."""

    host: str
    port: int
    host_keys: tuple[str, ...]
    via: SSHJump | None = None
    bastion_session_id: str | None = None
    known_hosts: KnownHosts = "strict"
    username: str = "root"
    #: The id the bootstrap printed with its ready mark; ``""`` when not read.
    boot: str = ""


def _transient(exc: BaseException) -> bool:
    """Whether ``exc`` is a failure of the transport, to retry or to step over.

    The socket's own errors and ``paramiko``'s ``SSHException`` family are,
    except a host key that does not match the pinned one: that is a
    verdict, not a connection that is not up yet.
    """
    names = {cls.__name__ for cls in type(exc).__mro__}
    if "BadHostKeyException" in names:
        return False
    return isinstance(exc, (OSError, EOFError)) or "SSHException" in names


#: Opens a new SSH session after a restart, given the previous boot's host
#: keys: the session, the keys now pinned, and the bastion session in use.
Reconnect = Callable[[tuple[str, ...]], tuple[SSHSession, tuple[str, ...], str | None]]


def _clients(profile: str, auth_type: str, region: str | None) -> _Clients:
    """The clients; the only place the SDK is imported."""
    try:
        import oci
    except ImportError as exc:
        raise ImportError(
            "OCIContainerInstanceBackend needs the OCI SDK and paramiko: "
            'pip install "locus-sdk[oci-container]"'
        ) from exc
    from locus.tools.oci import _build_config_and_signer

    config, signer = _build_config_and_signer(oci, profile, auth_type, region)
    kwargs: dict[str, Any] = {"signer": signer} if signer is not None else {}
    return _Clients(
        instances=oci.container_instances.ContainerInstanceClient(config, **kwargs),
        network=oci.core.VirtualNetworkClient(config, **kwargs),
        identity=oci.identity.IdentityClient(config, **kwargs),
        bastion=oci.bastion.BastionClient(config, **kwargs),
        region=config.get("region"),
    )


def _models() -> Any:
    from oci.container_instances import models

    return models


def _bastion_models() -> Any:
    from oci.bastion import models

    return models


def _keypair() -> tuple[Any, str]:
    """A fresh Ed25519 key for this backend: the ``paramiko`` key and its public line."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    private = Ed25519PrivateKey.generate()
    pem = private.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.OpenSSH,
        serialization.NoEncryption(),
    )
    public = private.public_key().public_bytes(
        serialization.Encoding.OpenSSH, serialization.PublicFormat.OpenSSH
    )
    pkey = _paramiko().Ed25519Key.from_private_key(io.StringIO(pem.decode()))
    return pkey, public.decode()


def _open_ssh(reach: _Reach, pkey: Any, *, connect_timeout: float, path: str | None) -> SSHSession:
    """One SSH connection to the instance; split out so a test can stand it in."""
    return SSHSession.open(
        reach.host,
        port=reach.port,
        username=reach.username,
        pkey=pkey,
        host_key=reach.host_keys[0] if reach.host_keys else None,
        known_hosts=reach.known_hosts,
        via=reach.via,
        connect_timeout=connect_timeout,
        path=path,
    )


def _status(exc: BaseException) -> int | None:
    status = getattr(exc, "status", None)
    return status if isinstance(status, int) else None


def _wait(
    read: Callable[[], str],
    wanted: str,
    *,
    failed: frozenset[str],
    timeout: float,
    interval: float,
    what: str,
) -> None:
    """Poll ``read`` until it returns ``wanted``; raise on a failed state or the deadline."""
    deadline = time.monotonic() + timeout
    state = "UNKNOWN"
    while time.monotonic() < deadline:
        state = read().upper()
        if state == wanted:
            return
        if state in failed:
            raise RuntimeError(f"{what} is {state}, not {wanted}")
        time.sleep(interval)
    raise RuntimeError(f"{what} did not become {wanted} within {timeout:g}s (last: {state})")


def _text(data: Any) -> str:
    """The text of a ``retrieve_logs`` response, whatever shape the SDK hands back."""
    content = getattr(data, "content", None)
    if content is None:
        raw = getattr(data, "raw", None)
        content = raw.read() if raw is not None else data
    if isinstance(content, bytes):
        return content.decode(errors="replace")
    return str(content)


def _host_keys(log: str) -> tuple[str, ...]:
    """The host public keys the bootstrap printed, each ``"<type> <base64>"``."""
    keys: list[str] = []
    for line in log.splitlines():
        # The service may prefix each line (a timestamp, a stream name); the
        # mark is looked for anywhere in it.
        _, mark, rest = line.partition(HOST_KEY_MARK)
        if mark:
            fields = rest.split()
            if len(fields) >= 2 and fields[0].startswith(("ssh-", "ecdsa-", "sk-")):
                keys.append(f"{fields[0]} {fields[1]}")
    # Most recent boot last; the pinned key is the first one, so prefer ed25519.
    keys.sort(key=lambda k: (not k.startswith("ssh-ed25519"), k))
    return tuple(dict.fromkeys(keys))


def _latest_boot(log: str) -> tuple[str, str]:
    """The lines of the latest bootstrap that reached the ready mark, and that boot's id.

    The log may hold several boots after a restart; the latest is the one
    whose keys are current. Without a ready mark the whole log is returned
    with an empty id.
    """
    head, mark, tail = log.rpartition(READY_MARK)
    if not mark:
        return log, ""
    boot = tail.split("\n", 1)[0].strip()
    return head.rpartition(READY_MARK)[2], boot


_BASTION_HOST = re.compile(r"@(host\.bastion\.[a-z0-9.-]+)")


def _public_key_line(info: Any) -> str | None:
    """A ``"<type> <base64>"`` pair out of whatever the service put in ``info``."""
    if not isinstance(info, str):
        return None
    fields = info.split()
    for index, field in enumerate(fields[:-1]):
        if field.startswith(("ssh-", "ecdsa-", "sk-")):
            return f"{field} {fields[index + 1]}"
    return None


class OCIContainerInstanceBackend(SSHSandboxBackend):
    """An OCI Container Instance as a workspace, reached over SSH.

    Build one with :meth:`create` (provisions an instance and owns it) or
    :meth:`connect` (an instance that already runs ``sshd``). The
    constructor takes a connected :class:`~locus.harness.ssh.SSHSession`.

    Args:
        session: The SSH connection to the instance.
        instance_id: The container instance's OCID.
        clients: The service clients, for the lifecycle; ``None`` leaves
            the instance alone on ``stop``/``close``.
        region: The instance's region, for the label.
        owns_instance: Delete the instance on :meth:`close`. Set by
            :meth:`create`.
        reconnect: Opens a new SSH session after a restart, given the host
            keys of the previous boot; returns the session, the new keys
            and the bastion session the connection goes through (``None``
            without one). Set by :meth:`create` and :meth:`connect`.
        host_keys: The container's host public keys, as pinned.
        bastion_session_id: A bastion session to delete on :meth:`close`.
        limits: What the instance was given.
        root: The workspace directory in the container.
        env: Variables added to every command's environment.
        max_background: How many background commands may run at once.
        state_dir: Where background commands keep their state.
    """

    def __init__(
        self,
        session: SSHSession,
        instance_id: str,
        *,
        clients: _Clients | None = None,
        region: str | None = None,
        owns_instance: bool = False,
        reconnect: Reconnect | None = None,
        host_keys: tuple[str, ...] = (),
        bastion_session_id: str | None = None,
        limits: ResourceLimits | None = None,
        root: str = SANDBOX_ROOT,
        env: Mapping[str, str] | None = None,
        max_background: int = MAX_BACKGROUND,
        state_dir: str = "/tmp/.locus-harness",  # noqa: S108 - inside the container
    ) -> None:
        where = f"{region}/{instance_id}" if region else instance_id
        super().__init__(
            session,
            root=root,
            label=f"oci-container:{where}",
            isolated=True,
            owns_connection=True,
            env=env,
            max_background=max_background,
            state_dir=state_dir,
        )
        self._instance_id = instance_id
        self._clients = clients
        self._owns_instance = owns_instance
        self._reconnect = reconnect
        self._host_keys = host_keys
        self._bastion_session_id = bastion_session_id
        self._closing = False
        if limits is not None:
            self.limits = limits

    @property
    def instance_id(self) -> str:
        """The container instance's OCID."""
        return self._instance_id

    @property
    def client(self) -> Any:
        """The ``ContainerInstanceClient``, or ``None`` without a lifecycle."""
        return self._clients.instances if self._clients is not None else None

    # ----------------------------------------------------------- lifecycle --

    def _state(self) -> str:
        assert self._clients is not None  # noqa: S101 - callers check
        instances = self._clients.instances
        return str(instances.get_container_instance(self._instance_id).data.lifecycle_state).upper()

    def _start(self) -> None:
        """Start a stopped instance and reconnect; the workspace is fresh.

        The container restarts from its image, so the bootstrap runs again
        and prints its host keys with a new boot id; the reconnect waits for
        the new boot's lines before it trusts the keys — the same keys as
        before, for an image that carries its own, or new ones, re-pinned.
        """
        if self._clients is not None:
            state = self._state()
            if state == _INACTIVE:
                self._clients.instances.start_container_instance(self._instance_id)
            _wait(
                self._state,
                _ACTIVE,
                failed=_FAILED,
                timeout=LIFECYCLE_TIMEOUT,
                interval=5.0,
                what=f"container instance {self._instance_id}",
            )
        if self._reconnect is None:
            raise RuntimeError("this backend cannot reconnect: it was built without reconnect=")
        session, keys, bastion_session_id = self._reconnect(self._host_keys)
        self._ssh = self._session = session
        self._host_keys = keys
        self._bastion_session_id = bastion_session_id
        self._has_rg = None

    def _stop(self) -> None:
        """End the commands, close the connection and stop the instance.

        A connection that is already gone cannot end the commands, but
        stopping the instance does, and a backend that does not own the
        instance gets a fresh connection from the next ``start``; so a
        transport failure here is logged and the stop goes on, and the
        backend is not left running over a closed connection. Any other
        failure leaves the connection open and the state as it was, for
        the caller to retry.
        """
        try:
            super()._stop()
        except Exception as exc:
            if not _transient(exc):
                raise
            logger.warning("could not end the commands in %s over SSH: %s", self._instance_id, exc)
        self._ssh.close()
        if self._clients is not None and self._owns_instance and not self._closing:
            state = self._state()
            if state == _ACTIVE:
                self._clients.instances.stop_container_instance(self._instance_id)
            if state in (_ACTIVE, _UPDATING):
                _wait(
                    self._state,
                    _INACTIVE,
                    failed=_FAILED,
                    timeout=LIFECYCLE_TIMEOUT,
                    interval=5.0,
                    what=f"container instance {self._instance_id}",
                )

    def close(self) -> None:
        """End the commands, close the connection, delete the session and the instance.

        A running instance is deleted as it is: deleting is one call, and
        stopping it first would wait for the service to stop it.
        """
        self._closing = True
        try:
            super().close()
        finally:
            self._closing = False

    def _destroy(self) -> None:
        """Close the connection; delete the bastion session and the instance this backend made."""
        self._ssh.close()
        if self._clients is None:
            return
        if self._bastion_session_id is not None:
            try:
                self._clients.bastion.delete_session(self._bastion_session_id)
            except Exception as exc:  # noqa: BLE001 - gone already is fine; the session expires anyway
                if _status(exc) != 404:
                    logger.warning("could not delete bastion session: %s", exc)
            self._bastion_session_id = None
        if self._owns_instance:
            _delete_instance(self._clients, self._instance_id)
            self._owns_instance = False

    # -------------------------------------------------------- constructors --

    @classmethod
    def create(
        cls,
        compartment_id: str,
        subnet_id: str,
        *,
        availability_domain: str | None = None,
        image: str = DEFAULT_IMAGE,
        shape: str = DEFAULT_SHAPE,
        ocpus: float = 1.0,
        memory_gb: float = 2.0,
        display_name: str | None = None,
        public_ip: bool = False,
        bastion_id: str | None = None,
        bastion_session_ttl: int = BASTION_SESSION_TTL,
        ssh_port: int = 22,
        known_hosts: KnownHosts = "strict",
        region: str | None = None,
        profile: str = "DEFAULT",
        auth_type: str = "api_key",
        root: str = SANDBOX_ROOT,
        env: Mapping[str, str] | None = None,
        container_env: Mapping[str, str] | None = None,
        freeform_tags: Mapping[str, str] | None = None,
        ready_timeout: float = 600.0,
        poll_interval: float = 5.0,
        connect_timeout: float = 30.0,
        path: str | None = None,
        **options: Any,
    ) -> OCIContainerInstanceBackend:
        """Provision a container instance, wait for ``sshd`` in it, and return a backend.

        The backend owns the instance: :meth:`close` deletes it, and so
        does any failure between creation and the return.

        Args:
            compartment_id: Where the instance is created.
            subnet_id: The subnet of its VNIC. With ``bastion_id``, a
                private subnet the bastion can reach; otherwise one this
                process can reach (a public subnet with ``public_ip=True``,
                or a private one over a VPN).
            availability_domain: The AD; the compartment's first when omitted.
            image: The container image. It needs ``sshd`` or a package
                manager (``dnf``, ``microdnf``, ``apt-get`` or ``apk``) and
                egress to install it.
            shape: A Container Instances shape; flexible shapes take
                ``ocpus`` and ``memory_gb``.
            ocpus: Processor cores for a flexible shape.
            memory_gb: Memory in gigabytes for a flexible shape.
            display_name: How the instance is listed; ``locus-harness-…``
                when omitted.
            public_ip: Assign a public IP and connect to it.
            bastion_id: An OCI Bastion in the instance's VCN to connect
                through; a port-forwarding session is created for this
                backend and deleted on close.
            bastion_session_ttl: Seconds the bastion session lives. A
                restart after it has expired makes a new one.
            ssh_port: The port ``sshd`` listens on in the container.
            known_hosts: ``"strict"`` requires the host keys the container
                printed to its log, and the bastion's from the service, and
                fails without them; ``"accept-unknown"`` connects unverified
                when one of them is missing, with a warning.
            region: The region; the profile's when omitted.
            profile: The ``~/.oci/config`` profile.
            auth_type: ``api_key``, ``security_token``, ``instance_principal``
                or ``resource_principal``.
            root: The workspace directory, created by the bootstrap.
            env: Variables added to every harness command's environment.
            container_env: Environment variables set on the container itself.
            freeform_tags: Tags on the instance.
            ready_timeout: Seconds for the instance to run and ``sshd`` to
                answer. Installing ``sshd`` on the default image takes about
                a minute.
            poll_interval: Seconds between state checks.
            connect_timeout: Seconds for each SSH connection attempt.
            path: A ``PATH`` for every command in the container.
            options: Passed to the backend (``state_dir``, ``max_background``).

        Raises:
            RuntimeError: The instance failed, or ``sshd`` was not reachable
                within ``ready_timeout``; the instance is deleted first.
        """
        clients = _clients(profile, auth_type, region)
        pkey, public = _keypair()
        models = _models()
        if availability_domain is None:
            availability_domain = (
                clients.identity.list_availability_domains(compartment_id).data[0].name
            )
        variables = {
            **(container_env or {}),
            "LOCUS_AUTHORIZED_KEY": public,
            "LOCUS_SSH_PORT": str(ssh_port),
            "LOCUS_WORKSPACE": root,
        }
        name = display_name or f"locus-harness-{uuid.uuid4().hex[:8]}"
        details = models.CreateContainerInstanceDetails(
            compartment_id=compartment_id,
            availability_domain=availability_domain,
            display_name=name,
            shape=shape,
            shape_config=models.CreateContainerInstanceShapeConfigDetails(
                ocpus=ocpus, memory_in_gbs=memory_gb
            ),
            containers=[
                models.CreateContainerDetails(
                    display_name="sandbox",
                    image_url=image,
                    command=["/bin/sh", "-c", BOOTSTRAP],
                    environment_variables=variables,
                )
            ],
            vnics=[
                models.CreateContainerVnicDetails(
                    subnet_id=subnet_id, is_public_ip_assigned=public_ip
                )
            ],
            container_restart_policy="NEVER",
            graceful_shutdown_timeout_in_seconds=5,
            freeform_tags=dict(freeform_tags or {}),
        )
        instance_id = clients.instances.create_container_instance(details).data.id
        deadline = time.monotonic() + ready_timeout
        bastion_session: str | None = None
        try:
            reach = _reach(
                clients,
                instance_id,
                public_ip=public_ip,
                bastion_id=bastion_id,
                public=public,
                pkey=pkey,
                ssh_port=ssh_port,
                deadline=deadline,
                interval=poll_interval,
                known_hosts=known_hosts,
                bastion_session_ttl=bastion_session_ttl,
            )
            bastion_session = reach.bastion_session_id

            def reconnect(
                _previous: tuple[str, ...],
            ) -> tuple[SSHSession, tuple[str, ...], str | None]:
                nonlocal reach
                # The bastion session is reused while it lives; an expired
                # or deleted one is replaced, and the new id goes back to
                # the backend for its close.
                via, session_id = reach.via, reach.bastion_session_id
                if session_id is not None and not _bastion_session_alive(clients, session_id):
                    via = session_id = None
                fresh = _reach(
                    clients,
                    instance_id,
                    public_ip=public_ip,
                    bastion_id=bastion_id,
                    public=public,
                    pkey=pkey,
                    ssh_port=ssh_port,
                    deadline=time.monotonic() + ready_timeout,
                    interval=poll_interval,
                    via=via,
                    previous_boot=reach.boot,
                    known_hosts=known_hosts,
                    bastion_session_ttl=bastion_session_ttl,
                )
                if fresh.bastion_session_id is None:
                    fresh = replace(fresh, bastion_session_id=session_id)
                session = _connect_until(
                    fresh,
                    pkey,
                    deadline=time.monotonic() + ready_timeout,
                    interval=poll_interval,
                    connect_timeout=connect_timeout,
                    path=path,
                )
                reach = fresh
                return session, fresh.host_keys, fresh.bastion_session_id

            session = _connect_until(
                reach,
                pkey,
                deadline=deadline,
                interval=poll_interval,
                connect_timeout=connect_timeout,
                path=path,
            )
            return cls(
                session,
                instance_id,
                clients=clients,
                region=clients.region,
                owns_instance=True,
                reconnect=reconnect,
                host_keys=reach.host_keys,
                bastion_session_id=bastion_session,
                limits=ResourceLimits(cpus=ocpus, memory_mb=int(memory_gb * 1024)),
                root=root,
                env=env,
                **options,
            )
        except BaseException:
            # Whatever failed once the instance exists, it must not outlive us.
            if bastion_session is not None:
                try:
                    clients.bastion.delete_session(bastion_session)
                except Exception:  # noqa: BLE001 - best effort on the way out
                    logger.warning("could not delete bastion session %s", bastion_session)
            try:
                _delete_instance(clients, instance_id)
            except Exception:  # noqa: BLE001 - the original error is the one to raise
                logger.exception("could not delete container instance %s", instance_id)
            raise

    @classmethod
    def connect(  # type: ignore[override]
        cls,
        instance_id: str,
        *,
        pkey: Any = None,
        key_filename: str | None = None,
        username: str = "root",
        ssh_port: int = 22,
        host_key: str | None = None,
        known_hosts: KnownHosts = "strict",
        public_ip: bool = False,
        bastion_id: str | None = None,
        bastion_session_ttl: int = BASTION_SESSION_TTL,
        region: str | None = None,
        profile: str = "DEFAULT",
        auth_type: str = "api_key",
        root: str = SANDBOX_ROOT,
        env: Mapping[str, str] | None = None,
        connect_timeout: float = 30.0,
        path: str | None = None,
        **options: Any,
    ) -> OCIContainerInstanceBackend:
        """A backend over an instance that already runs ``sshd``. It is never deleted.

        The instance's IP comes from the service; the key is yours. With
        ``bastion_id``, a port-forwarding session is created with the public
        half of ``pkey`` or ``key_filename`` and deleted on close; a restart
        after ``bastion_session_ttl`` seconds makes a new one.
        """
        clients = _clients(profile, auth_type, region)
        if pkey is None and key_filename is not None:
            pkey = _paramiko().PKey.from_path(str(Path(key_filename).expanduser()))
        if pkey is None:
            raise ValueError("connect needs pkey= or key_filename= for the instance's sshd")
        public = f"{pkey.get_name()} {pkey.get_base64()}"
        instance = clients.instances.get_container_instance(instance_id).data
        ip = _ip(clients, instance, public_ip=public_ip)
        jump = bastion_id

        def bastion() -> tuple[SSHJump | None, str | None]:
            if jump is None:
                return None, None
            return _bastion_session(
                clients,
                jump,
                ip,
                ssh_port,
                public,
                pkey,
                time.monotonic() + connect_timeout * 4,
                known_hosts=known_hosts,
                ttl=bastion_session_ttl,
            )

        via, bastion_session = bastion()
        reach = _Reach(
            ip,
            ssh_port,
            (host_key,) if host_key else (),
            via,
            bastion_session,
            known_hosts="accept-unknown" if known_hosts == "accept-unknown" else "strict",
            username=username,
        )
        try:
            session = _open_ssh(reach, pkey, connect_timeout=connect_timeout, path=path)
        except BaseException:
            if bastion_session is not None:
                clients.bastion.delete_session(bastion_session)
            raise

        def reconnect(_previous: tuple[str, ...]) -> tuple[SSHSession, tuple[str, ...], str | None]:
            nonlocal reach
            session_id = reach.bastion_session_id
            if session_id is not None and not _bastion_session_alive(clients, session_id):
                via, session_id = bastion()
                reach = replace(reach, via=via, bastion_session_id=session_id)
            return (
                _open_ssh(reach, pkey, connect_timeout=connect_timeout, path=path),
                reach.host_keys,
                reach.bastion_session_id,
            )

        return cls(
            session,
            instance_id,
            clients=clients,
            region=clients.region,
            owns_instance=False,
            reconnect=reconnect,
            host_keys=reach.host_keys,
            bastion_session_id=bastion_session,
            root=root,
            env=env,
            **options,
        )


def _ip(clients: _Clients, instance: Any, *, public_ip: bool) -> str:
    vnic = clients.network.get_vnic(instance.vnics[0].vnic_id).data
    ip = vnic.public_ip if public_ip else vnic.private_ip
    if not ip:
        kind = "public" if public_ip else "private"
        raise RuntimeError(f"container instance {instance.id} has no {kind} IP")
    return str(ip)


def _reach(
    clients: _Clients,
    instance_id: str,
    *,
    public_ip: bool,
    bastion_id: str | None,
    public: str,
    pkey: Any,
    ssh_port: int,
    deadline: float,
    interval: float,
    via: SSHJump | None = None,
    previous_boot: str | None = None,
    known_hosts: KnownHosts = "strict",
    bastion_session_ttl: int = BASTION_SESSION_TTL,
) -> _Reach:
    """Wait for the instance to run and its ``sshd`` to announce itself; how to reach it.

    After a restart the log may still show the previous boot, so with
    ``previous_boot`` the wait goes on until the latest boot's id differs
    from it; its keys may well be the same. ``via`` is a jump to keep;
    without one, ``bastion_id`` gets a session.
    """
    instances = clients.instances
    _wait(
        lambda: str(instances.get_container_instance(instance_id).data.lifecycle_state),
        _ACTIVE,
        failed=_FAILED,
        timeout=max(1.0, deadline - time.monotonic()),
        interval=interval,
        what=f"container instance {instance_id}",
    )
    instance = instances.get_container_instance(instance_id).data
    ip = _ip(clients, instance, public_ip=public_ip)
    container_id = instance.containers[0].container_id
    keys: tuple[str, ...] = ()
    boot = ""
    while time.monotonic() < deadline:
        log = _text(instances.retrieve_logs(container_id).data)
        if FAILED_MARK in log:
            line = next(ln for ln in log.splitlines() if FAILED_MARK in ln)
            raise RuntimeError(f"the container could not start sshd: {line}")
        if READY_MARK in log:
            lines, boot = _latest_boot(log)
            if boot != previous_boot:
                keys = _host_keys(lines)
                break
        time.sleep(interval)
    else:
        raise RuntimeError(
            f"sshd in container instance {instance_id} did not start within the ready timeout"
        )
    if not keys:
        if known_hosts != "accept-unknown":
            raise RuntimeError(
                f"container instance {instance_id} printed no host key to its log; "
                'pass known_hosts="accept-unknown" to connect to its sshd unverified'
            )
        logger.warning("the container printed no host key; its sshd is accepted unverified")
    session_id: str | None = None
    if via is None and bastion_id is not None:
        via, session_id = _bastion_session(
            clients,
            bastion_id,
            ip,
            ssh_port,
            public,
            pkey,
            deadline,
            known_hosts=known_hosts,
            ttl=bastion_session_ttl,
        )
    return _Reach(
        ip,
        ssh_port,
        keys,
        via,
        session_id,
        known_hosts="strict" if keys else "accept-unknown",
        boot=boot,
    )


def _bastion_session(
    clients: _Clients,
    bastion_id: str,
    ip: str,
    port: int,
    public: str,
    pkey: Any,
    deadline: float,
    *,
    known_hosts: KnownHosts = "strict",
    ttl: int = BASTION_SESSION_TTL,
) -> tuple[SSHJump, str]:
    """A port-forwarding bastion session to ``ip:port``; the jump to use and the session id.

    The session is authorized for ``public``, and the jump logs in with its
    private half ``pkey``: the same key pair the container's ``sshd`` takes.
    The bastion's host key, as the service reports it, is pinned on the
    jump; a session that reports none is deleted and refused unless
    ``known_hosts`` is ``"accept-unknown"``.
    """
    models = _bastion_models()
    details = models.CreateSessionDetails(
        bastion_id=bastion_id,
        display_name=f"locus-harness-{uuid.uuid4().hex[:8]}",
        key_type="PUB",
        key_details=models.PublicKeyDetails(public_key_content=public),
        session_ttl_in_seconds=ttl,
        target_resource_details=models.CreatePortForwardingSessionTargetResourceDetails(
            target_resource_private_ip_address=ip, target_resource_port=port
        ),
    )
    session_id = clients.bastion.create_session(details).data.id
    try:
        _wait(
            lambda: str(clients.bastion.get_session(session_id).data.lifecycle_state),
            _ACTIVE,
            failed=_FAILED,
            timeout=max(1.0, deadline - time.monotonic()),
            interval=2.0,
            what=f"bastion session {session_id}",
        )
        session = clients.bastion.get_session(session_id).data
    except BaseException:
        clients.bastion.delete_session(session_id)
        raise
    command = str((getattr(session, "ssh_metadata", None) or {}).get("command", ""))
    match = _BASTION_HOST.search(command)
    region = clients.region or "unknown"
    host = match.group(1) if match else f"host.bastion.{region}.oci.oraclecloud.com"
    host_key = _public_key_line(getattr(session, "bastion_public_host_key_info", None))
    if host_key is None:
        if known_hosts != "accept-unknown":
            clients.bastion.delete_session(session_id)
            raise RuntimeError(
                f"bastion session {session_id} reports no host key for {host}; "
                'pass known_hosts="accept-unknown" to connect through it unverified'
            )
        logger.warning("the bastion reported no host key; %s is accepted unverified", host)
    via = SSHJump(
        host=host,
        username=str(getattr(session, "bastion_user_name", None) or session_id),
        pkey=pkey,
        host_key=host_key,
        known_hosts="strict" if host_key else "accept-unknown",
    )
    return via, session_id


def _bastion_session_alive(clients: _Clients, session_id: str) -> bool:
    """Whether the bastion session can still carry a connection: ``ACTIVE``, not gone."""
    try:
        state = str(clients.bastion.get_session(session_id).data.lifecycle_state).upper()
    except Exception as exc:
        if _status(exc) == 404:
            return False
        raise
    return state == _ACTIVE


def _connect_until(
    reach: _Reach,
    pkey: Any,
    *,
    deadline: float,
    interval: float,
    connect_timeout: float,
    path: str | None,
) -> SSHSession:
    """Connect to ``sshd``, retrying until ``deadline``: it answers a little after it logs.

    The bastion reports a session ACTIVE a little before it accepts the
    session's key, and ``sshd`` answers a little after it logs; an
    authentication or transport failure now is "not yet", not "no". A host
    key that does not match the pinned one is "no", at once.
    """
    last: BaseException | None = None
    while True:
        try:
            return _open_ssh(reach, pkey, connect_timeout=connect_timeout, path=path)
        except Exception as exc:
            if not _transient(exc):
                raise
            last = exc
        if time.monotonic() >= deadline:
            raise RuntimeError(f"could not reach sshd at {reach.host}:{reach.port}: {last}")
        time.sleep(interval)


def _delete_instance(clients: _Clients, instance_id: str, timeout: float = 600.0) -> None:
    """Delete the instance and wait until it is gone. Gone already is fine."""
    instances = clients.instances
    try:
        state = str(instances.get_container_instance(instance_id).data.lifecycle_state).upper()
        if state not in ("DELETING", _DELETED):
            instances.delete_container_instance(instance_id)
        deadline = time.monotonic() + timeout
        while state != _DELETED and time.monotonic() < deadline:
            time.sleep(5.0)
            state = str(instances.get_container_instance(instance_id).data.lifecycle_state).upper()
    except Exception as exc:
        if _status(exc) != 404:
            raise
