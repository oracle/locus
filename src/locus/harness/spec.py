# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Choose and open a sandbox backend from a spec.

A harness that is configured rather than coded — a gateway's run manifest,
a project's ``locus.yaml``, a CLI flag — names its sandbox the way it names
its model: by a short string and a few options. :class:`SandboxSpec` is
that description, and :func:`open_sandbox` turns it into a running
:class:`~locus.harness.sandbox.SandboxBackend`::

    from locus.harness import SandboxSpec, WarmPoolHint, open_sandbox

    spec = SandboxSpec(
        backend="oci-container",
        options={
            "compartment_id": "ocid1.compartment...",
            "subnet_id": "ocid1.subnet...",
        },
        warm_pool=WarmPoolHint(ready=2),
    )
    backend = open_sandbox(spec)

The built-in names are ``local``, ``memory``, ``ssh``, ``oci-container``,
``oci-sandbox``, ``openshell``, ``e2b`` and ``daytona``; each maps to the
backend's ``create`` or ``connect`` with ``options`` as its keyword
arguments. :func:`register_backend` adds a name for a backend of your own.

The spec also carries what every backend understands: a ``root``, an
``env`` for every command, :class:`~locus.harness.sandbox.ResourceLimits`,
and a :class:`~locus.harness.sandbox.WarmPoolHint`. The limits are applied
to the backend after it opens (a command-time limit is enforced by every
backend; CPUs and memory are passed to a backend that provisions compute).
The warm-pool hint is honoured two ways: ``name`` goes to a backend whose
service keeps a pool (OCI GenAI Sandboxes), and ``ready`` is what
:func:`open_pool` uses to keep that many sandboxes created ahead of need.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from typing import Any

from locus.harness.sandbox import ResourceLimits, SandboxBackend, SandboxPool, WarmPoolHint


__all__ = [
    "BUILTIN_BACKENDS",
    "SandboxSpec",
    "open_pool",
    "open_sandbox",
    "register_backend",
    "registered_backends",
]

#: Builds a backend from a spec.
BackendFactory = Callable[["SandboxSpec"], SandboxBackend]


@dataclass(frozen=True)
class SandboxSpec:
    """Which sandbox, and how.

    Attributes:
        backend: The backend's name: a built-in or one registered with
            :func:`register_backend`.
        root: The workspace directory; the backend's default when ``None``.
        env: Variables added to every command's environment.
        limits: What the sandbox gets, where the backend can give it, and
            the command-time limit every backend enforces.
        warm_pool: How many sandboxes to have ready, and the service-side
            pool to draw from where there is one.
        options: Keyword arguments for the backend's constructor — the
            host for ``ssh``, the compartment and subnet for
            ``oci-container``, the project for ``oci-sandbox``.
    """

    backend: str = "local"
    root: str | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    limits: ResourceLimits = field(default_factory=ResourceLimits)
    warm_pool: WarmPoolHint = field(default_factory=WarmPoolHint)
    options: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> SandboxSpec:
        """A spec from a plain mapping (a parsed YAML or JSON document).

        ``limits`` and ``warm_pool`` may be mappings themselves, and must
        be mappings when given. An unknown key is a ``ValueError`` that
        names it, so a typo in a config file does not silently fall back to
        the host.
        """
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown sandbox spec keys: {', '.join(unknown)}")
        values = dict(data)
        limits = values.get("limits")
        if isinstance(limits, Mapping):
            values["limits"] = _dataclass(ResourceLimits, limits, "limits")
        elif limits is not None and not isinstance(limits, ResourceLimits):
            raise ValueError(f"limits must be a mapping, got {type(limits).__name__}")
        pool = values.get("warm_pool")
        if isinstance(pool, Mapping):
            values["warm_pool"] = _dataclass(WarmPoolHint, pool, "warm_pool")
        elif pool is not None and not isinstance(pool, WarmPoolHint):
            raise ValueError(f"warm_pool must be a mapping, got {type(pool).__name__}")
        backend = values.get("backend", "local")
        if not isinstance(backend, str) or not backend:
            raise ValueError(f"backend must be a non-empty string, got {backend!r}")
        return cls(**values)


def _dataclass(kind: type[Any], data: Mapping[str, Any], where: str) -> Any:
    known = {f.name for f in fields(kind)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"unknown {where} keys: {', '.join(unknown)}")
    return kind(**data)


def _common(spec: SandboxSpec) -> dict[str, Any]:
    """The constructor arguments every backend takes, from the spec."""
    args: dict[str, Any] = dict(spec.options)
    if spec.root is not None:
        args.setdefault("root", spec.root)
    if spec.env:
        args.setdefault("env", dict(spec.env))
    return args


def _local(spec: SandboxSpec) -> SandboxBackend:
    from locus.harness.local import LocalBackend

    args = _common(spec)
    root = args.pop("root", ".")
    return LocalBackend(root, **args)


def _memory(spec: SandboxSpec) -> SandboxBackend:
    from locus.harness.state import MemoryBackend

    files = dict(spec.options).get("files")
    return MemoryBackend(files)


def _ssh(spec: SandboxSpec) -> SandboxBackend:
    from locus.harness.ssh import SSHSandboxBackend

    args = _common(spec)
    host = args.pop("host", None)
    if not host:
        raise ValueError("the ssh backend needs options={'host': ...}")
    return SSHSandboxBackend.connect(host, **args)


def _oci_container(spec: SandboxSpec) -> SandboxBackend:
    from locus.harness.oci_container import OCIContainerInstanceBackend

    args = _common(spec)
    if "instance_id" in args:
        return OCIContainerInstanceBackend.connect(args.pop("instance_id"), **args)
    compartment = args.pop("compartment_id", None)
    subnet = args.pop("subnet_id", None)
    if not compartment or not subnet:
        raise ValueError(
            "the oci-container backend needs options={'compartment_id': ..., 'subnet_id': ...}"
            " (or 'instance_id' to connect)"
        )
    if spec.limits.cpus is not None:
        args.setdefault("ocpus", spec.limits.cpus)
    if spec.limits.memory_mb is not None:
        args.setdefault("memory_gb", spec.limits.memory_mb / 1024)
    return OCIContainerInstanceBackend.create(compartment, subnet, **args)


def _oci_sandbox(spec: SandboxSpec) -> SandboxBackend:
    from locus.harness.oci_sandbox import OCISandboxBackend

    args = _common(spec)
    if "sandbox_id" in args:
        return OCISandboxBackend.connect(args.pop("sandbox_id"), **args)
    project = args.pop("project_id", None)
    if not project:
        raise ValueError(
            "the oci-sandbox backend needs options={'project_id': ...} (or 'sandbox_id' to connect)"
        )
    if spec.warm_pool.name:
        args.setdefault("warmpool_name", spec.warm_pool.name)
    return OCISandboxBackend.create(project, **args)


def _openshell(spec: SandboxSpec) -> SandboxBackend:
    from locus.harness.openshell import OpenShellBackend

    args = _common(spec)
    if "sandbox" in args:
        return OpenShellBackend.connect(args.pop("sandbox"), **args)
    return OpenShellBackend.create(**args)


def _e2b(spec: SandboxSpec) -> SandboxBackend:
    from locus.harness.e2b import E2BBackend

    args = _common(spec)
    if "sandbox_id" in args:
        return E2BBackend.connect(args.pop("sandbox_id"), **args)
    return E2BBackend.create(**args)


def _daytona(spec: SandboxSpec) -> SandboxBackend:
    from locus.harness.daytona import DaytonaBackend

    args = _common(spec)
    if "sandbox_id" in args:
        return DaytonaBackend.connect(args.pop("sandbox_id"), **args)
    return DaytonaBackend.create(**args)


#: The names :func:`open_sandbox` knows without registration.
BUILTIN_BACKENDS: Mapping[str, BackendFactory] = {
    "local": _local,
    "memory": _memory,
    "ssh": _ssh,
    "oci-container": _oci_container,
    "oci-sandbox": _oci_sandbox,
    "openshell": _openshell,
    "e2b": _e2b,
    "daytona": _daytona,
}

_REGISTRY: dict[str, BackendFactory] = dict(BUILTIN_BACKENDS)


def register_backend(name: str, factory: BackendFactory) -> None:
    """Make ``SandboxSpec(backend=name)`` build with ``factory``.

    ``factory`` receives the whole spec and returns an open backend. A
    built-in name can be overridden; register the original from
    :data:`BUILTIN_BACKENDS` to restore it.
    """
    if not name:
        raise ValueError("a backend name cannot be empty")
    _REGISTRY[name] = factory


def registered_backends() -> tuple[str, ...]:
    """Every name :func:`open_sandbox` accepts, sorted."""
    return tuple(sorted(_REGISTRY))


def open_sandbox(spec: SandboxSpec) -> SandboxBackend:
    """Build the backend ``spec`` names, apply the spec's limits, return it running.

    Raises:
        ValueError: The backend name is unknown, or a required option is
            missing.
        ImportError: The backend's extra is not installed; the message says
            which.
    """
    factory = _REGISTRY.get(spec.backend)
    if factory is None:
        raise ValueError(
            f"no sandbox backend named {spec.backend!r}; known: {', '.join(registered_backends())}"
        )
    backend = factory(spec)
    if spec.limits != ResourceLimits():
        try:
            backend.limits = spec.limits  # type: ignore[misc]
        except AttributeError:
            pass  # the backend fixes its own limits
    return backend


def open_pool(spec: SandboxSpec, *, ready: int | None = None) -> SandboxPool:
    """A :class:`~locus.harness.sandbox.SandboxPool` that opens ``spec`` ahead of need.

    ``ready`` defaults to the spec's ``warm_pool.ready``; zero makes the
    pool create inline on each ``acquire``.
    """
    count = spec.warm_pool.ready if ready is None else ready
    return SandboxPool(lambda: open_sandbox(spec), ready=count)
