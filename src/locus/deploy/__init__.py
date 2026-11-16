# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""``locus deploy``: the agent server on OCI, from a values file and Terraform.

``init`` writes a deployment directory: Terraform roots for the three targets
(OCI Container Instances, OKE with a Helm chart, OCI Functions), a Dockerfile
per image shape, the ``app/`` directory the image is built from and
``locus-deploy.yaml``, the one file a person edits. ``plan``, ``apply`` and
``destroy`` read that file and run Terraform with the chosen target; ``apply``
also builds the image and pushes it to OCIR between the registry root and the
target root::

    locus deploy init my-agent --target container-instances \\
        --compartment-id ocid1.compartment.oc1..example --region eu-frankfurt-1
    cd my-agent && locus deploy plan && locus deploy apply
    locus deploy destroy

Every OCI resource is a Terraform variable; nothing here names a tenancy. The
container authenticates with its own principal (resource principal on
Container Instances, Functions and OKE workload identity; instance principal
on OKE node identity) and secrets reach it as OCI Vault secret OCIDs that
:mod:`locus.deploy.entrypoint` resolves at start-up.
"""

from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cache
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = [
    "ACTIONS",
    "ARCHITECTURES",
    "TARGETS",
    "VALUES_FILE",
    "DeployError",
    "DeployToolError",
    "DeployValues",
    "Step",
    "declared_variables",
    "image_steps",
    "init",
    "load_values",
    "registry_steps",
    "render_values",
    "run",
    "sdk_requirement",
    "target_steps",
    "templates_root",
]

#: The deployment targets, each a Terraform root under ``terraform/``.
TARGETS: tuple[str, ...] = ("container-instances", "oke", "functions")

#: Image architectures; each target maps one to its compute shape family.
ARCHITECTURES: tuple[str, ...] = ("amd64", "arm64")

#: What ``plan``, ``apply`` and ``destroy`` are called in the API.
ACTIONS: tuple[str, ...] = ("plan", "apply", "destroy")

#: The values file ``init`` writes and the other commands read.
VALUES_FILE = "locus-deploy.yaml"

#: Files ``init`` never overwrites without ``force``: they hold the user's work.
_OWNED_BY_USER = (VALUES_FILE, "app/server.py", "app/requirements.txt")

#: Files ``init`` refuses to change when they exist with other content: an
#: existing project's build files are never replaced silently.
_NEVER_REPLACED = ("Dockerfile", "Dockerfile.functions", ".dockerignore")

#: Where ``--sdk-wheel`` puts the wheel the Dockerfiles install before requirements.txt.
_WHEELS_DIR = "app/wheels"

#: Variables a target cannot run without; ``load_values`` refuses early.
_REQUIRED_VARIABLES: dict[str, tuple[tuple[str, ...], ...]] = {
    "container-instances": (),
    "oke": (("cluster_id", "kubeconfig_path"),),
    "functions": (("subnet_id",),),
}

#: Starting ``variables:`` per target, with the knobs a first deployment touches.
_DEFAULT_VARIABLES: dict[str, dict[str, Any]] = {
    "container-instances": {
        "create_network": True,
        "ingress_cidr": "0.0.0.0/0",
        "ocpus": 1,
        "memory_gbs": 4,
        "server_api_key_secret_id": None,
        "allow_unauthenticated": False,
    },
    "oke": {
        "cluster_id": None,
        "namespace": "locus",
        "replicas": 2,
        "max_replicas": 2,
        "workload_identity": True,
        "server_api_key_secret_id": None,
    },
    "functions": {
        "subnet_id": None,
        "memory_mbs": 1024,
        "timeout_seconds": 300,
        "server_api_key_secret_id": None,
    },
}

#: The open-to-everyone ingress CIDR, the Container Instances root's default.
_ANY_CIDR = "0.0.0.0/0"

#: ``init`` appends what is missing of these under the marker; an existing ``.gitignore`` is kept.
_GITIGNORE_MARKER = "# locus deploy"
_GITIGNORE_LINES = (
    "# Terraform state and working directories; the state names your tenancy.",
    "terraform/**/.terraform/",
    "terraform/**/*.tfstate",
    "terraform/**/*.tfstate.backup",
    "terraform/**/locus.auto.tfvars.json",
)

#: The build context is the deployment directory; the Dockerfiles copy app/ and nothing else.
_DOCKERIGNORE = """# Written by `locus deploy init`. The Dockerfiles copy app/ and nothing else;
# Terraform working directories, state and variable files (they hold secret
# OCIDs) must never enter the build context or the image.
.terraform/
**/.terraform/
*.tfstate
*.tfstate.*
*.tfvars
*.tfvars.json
terraform/
helm/
.git/
.gitignore
locus-deploy.yaml
Dockerfile
Dockerfile.functions
**/__pycache__/
**/*.pyc
"""


class DeployError(Exception):
    """A problem with the values file, the directory or a tool, reported without a traceback."""


class DeployToolError(DeployError):
    """A tool ``run`` executed (terraform, docker) exited with an error; its own output says why."""


@dataclass(frozen=True)
class DeployValues:
    """What ``locus-deploy.yaml`` says.

    Attributes:
        target: One of :data:`TARGETS`.
        compartment_id: Compartment every resource lands in.
        region: OCI region identifier.
        name: Prefix for the resources, the repository and the Helm release.
        app: ``MODULE:ATTR`` of the ASGI app inside ``app/``.
        port: Port the server listens on inside the container.
        image_tag: Tag the image is pushed with.
        architecture: ``amd64`` or ``arm64``; the build platform and the shape family.
        registry_public: Create the OCIR repository public, so no pull secret is needed.
        repository: An existing repository path to push to; skips the registry root.
        oci_profile: Profile in ``~/.oci/config`` for Terraform and docker login hints.
        variables: Terraform variables of the target root, passed through unchanged.
        environment: Extra environment variables for the agent server.
    """

    target: str
    compartment_id: str
    region: str
    name: str = "locus-agent"
    app: str = "server:app"
    port: int = 8080
    image_tag: str = "0.1.0"
    architecture: str = "amd64"
    registry_public: bool = False
    repository: str | None = None
    oci_profile: str | None = None
    variables: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.target not in TARGETS:
            raise DeployError(f"target must be one of {', '.join(TARGETS)}, got {self.target!r}")
        if self.architecture not in ARCHITECTURES:
            raise DeployError(
                f"architecture must be one of {', '.join(ARCHITECTURES)}, got {self.architecture!r}"
            )
        for key in ("compartment_id", "region", "name", "app", "image_tag"):
            value = getattr(self, key)
            if not isinstance(value, str) or not value.strip():
                raise DeployError(f"{key} must be a non-empty string")
        if ":" not in self.app:
            raise DeployError(f"app must be MODULE:ATTR, got {self.app!r}")
        if (
            not isinstance(self.port, int)
            or isinstance(self.port, bool)
            or not 1 <= self.port <= 65535
        ):
            raise DeployError(f"port must be an integer between 1 and 65535, got {self.port!r}")
        for name, value in self.environment.items():
            if not isinstance(name, str) or not isinstance(value, str):
                raise DeployError(
                    f"environment values must be strings; {name!r} is {type(value).__name__}"
                )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> DeployValues:
        """The values in a parsed ``locus-deploy.yaml``."""
        if not isinstance(data, Mapping):
            raise DeployError(f"{VALUES_FILE} must be a mapping at the top level")
        known = {
            "target",
            "compartment_id",
            "region",
            "name",
            "app",
            "port",
            "image",
            "oci_profile",
            "variables",
            "environment",
        }
        unknown = sorted(set(data) - known)
        if unknown:
            raise DeployError(f"unknown key(s) in {VALUES_FILE}: {', '.join(unknown)}")
        for key in ("target", "compartment_id", "region"):
            if key not in data:
                raise DeployError(f"{VALUES_FILE} needs {key}")
        image = data.get("image") or {}
        if not isinstance(image, Mapping):
            raise DeployError(
                "image must be a mapping (tag, architecture, registry_public, repository)"
            )
        unknown_image = sorted(
            set(image) - {"tag", "architecture", "registry_public", "repository"}
        )
        if unknown_image:
            raise DeployError(f"unknown key(s) under image: {', '.join(unknown_image)}")
        variables = data.get("variables") or {}
        environment = data.get("environment") or {}
        if not isinstance(variables, Mapping) or not isinstance(environment, Mapping):
            raise DeployError("variables and environment must be mappings")
        return cls(
            target=str(data["target"]),
            compartment_id=str(data["compartment_id"]),
            region=str(data["region"]),
            name=str(data.get("name", "locus-agent")),
            app=str(data.get("app", "server:app")),
            port=data.get("port", 8080),
            image_tag=str(image.get("tag", "0.1.0")),
            architecture=str(image.get("architecture", "amd64")),
            registry_public=bool(image.get("registry_public", False)),
            repository=image.get("repository") or None,
            oci_profile=data.get("oci_profile") or None,
            variables=dict(variables),
            environment=dict(environment),
        )

    def to_mapping(self) -> dict[str, Any]:
        """The values as the mapping ``from_mapping`` reads."""
        return {
            "target": self.target,
            "name": self.name,
            "region": self.region,
            "compartment_id": self.compartment_id,
            "oci_profile": self.oci_profile,
            "app": self.app,
            "port": self.port,
            "image": {
                "tag": self.image_tag,
                "architecture": self.architecture,
                "registry_public": self.registry_public,
                "repository": self.repository,
            },
            "variables": dict(self.variables),
            "environment": dict(self.environment),
        }

    @property
    def dockerfile(self) -> str:
        """The Dockerfile the target's image is built from."""
        return "Dockerfile.functions" if self.target == "functions" else "Dockerfile"

    @property
    def terraform_root(self) -> str:
        """The target's Terraform root, relative to the deployment directory."""
        return f"terraform/{self.target}"

    @property
    def placeholder_repository(self) -> str:
        """The repository path before the registry root has been applied."""
        return f"{self.region}.ocir.io/<namespace>/{self.name}"

    def image_reference(self, repository: str | None) -> str:
        """The full image reference for ``repository`` (or the placeholder)."""
        return f"{repository or self.repository or self.placeholder_repository}:{self.image_tag}"

    def registry_tfvars(self) -> dict[str, Any]:
        """Variables of the registry root."""
        return {
            "compartment_id": self.compartment_id,
            "region": self.region,
            "oci_profile": self.oci_profile,
            "name": self.name,
            "is_public": self.registry_public,
        }

    def tfvars(self, image: str) -> dict[str, Any]:
        """Variables of the target root, with ``image`` and the user's ``variables`` folded in.

        Only variables the root declares are written (``port`` has no meaning
        for a function); the user's ``variables`` pass through unchanged.
        """
        declared = declared_variables(self.target)
        base: dict[str, Any] = {
            "compartment_id": self.compartment_id,
            "region": self.region,
            "oci_profile": self.oci_profile,
            "name": self.name,
            "image": image,
            "architecture": self.architecture,
            "environment": dict(self.environment),
            "port": self.port,
        }
        base = {key: value for key, value in base.items() if key in declared}
        for key, value in self.variables.items():
            if value is None:
                continue
            base[key] = value
        return base

    def check_required_variables(self) -> None:
        """Refuse a target whose required variables are still unset.

        A Container Instance with a public IP open to ``0.0.0.0/0`` is an
        agent anyone can call: it needs ``server_api_key_secret_id`` (the
        bearer key in OCI Vault) or an explicit ``allow_unauthenticated``.
        """
        for alternatives in _REQUIRED_VARIABLES[self.target]:
            if not any(self.variables.get(name) for name in alternatives):
                names = " or ".join(f"variables.{name}" for name in alternatives)
                raise DeployError(f"the {self.target} target needs {names} in {VALUES_FILE}")
        if self.target == "container-instances" and self._public_and_open():
            variables = self.variables
            if not variables.get("server_api_key_secret_id") and (
                variables.get("allow_unauthenticated") is not True
            ):
                raise DeployError(
                    "the container-instances target gets a public IP open to 0.0.0.0/0; set "
                    "variables.server_api_key_secret_id (the Vault secret holding the bearer "
                    f"key) or variables.allow_unauthenticated: true in {VALUES_FILE}"
                )

    def _public_and_open(self) -> bool:
        assign_public_ip = self.variables.get("assign_public_ip")
        ingress_cidr = self.variables.get("ingress_cidr")
        public = assign_public_ip is None or assign_public_ip is True
        return public and ingress_cidr in (None, _ANY_CIDR)


@dataclass(frozen=True)
class Step:
    """One command ``run`` executes, relative to the deployment directory."""

    argv: tuple[str, ...]
    cwd: str = "."

    def render(self) -> str:
        """The command as a person would type it."""
        command = shlex.join(self.argv)
        return command if self.cwd == "." else f"(cd {shlex.quote(self.cwd)} && {command})"


def _yaml() -> Any:
    try:
        import yaml  # type: ignore[import-untyped]  # PyYAML ships no inline types
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise DeployError(
            f"{VALUES_FILE} is read with PyYAML: pip install 'locus-sdk[deploy]'"
        ) from exc
    return yaml


def templates_root() -> Path:
    """Where the bundled Terraform roots, Helm chart and Dockerfiles are."""
    return Path(str(resources.files(__name__) / "templates"))


@cache
def declared_variables(target: str) -> frozenset[str]:
    """The variables the bundled Terraform root of ``target`` declares."""
    text = (templates_root() / "terraform" / target / "variables.tf").read_text(encoding="utf-8")
    return frozenset(re.findall(r'^variable\s+"([^"]+)"', text, flags=re.MULTILINE))


_WHEEL_NAME = re.compile(r"^locus_sdk-(?P<version>[^-]+)(-[^-]+)?-py3-none-any\.whl$")


def sdk_requirement(sdk_wheel: Path | None = None) -> str:
    """The ``locus-sdk`` line of ``app/requirements.txt``.

    Pinned to the running SDK's version, or to the version in ``sdk_wheel``'s
    file name when a wheel is given: the Dockerfiles install the wheel first,
    so pip finds the pin satisfied and only adds the extras.
    """
    if sdk_wheel is None:
        from locus import __version__

        return f"locus-sdk[server,oci]=={__version__}"
    match = _WHEEL_NAME.match(sdk_wheel.name)
    if match is None:
        raise DeployError(
            f"{sdk_wheel} is not a locus-sdk wheel (expected locus_sdk-<version>-py3-none-any.whl)"
        )
    return f"locus-sdk[server,oci]=={match.group('version')}"


def render_values(values: DeployValues) -> str:
    """``locus-deploy.yaml`` for ``values``, with a comment on every key."""
    yaml = _yaml()

    def scalar(value: Any) -> str:
        return str(
            yaml.safe_dump(value, default_flow_style=True).strip().removesuffix("\n...")
        ).strip()

    def block(mapping: Mapping[str, Any]) -> str:
        if not mapping:
            return " {}\n"
        dumped = yaml.safe_dump(dict(mapping), default_flow_style=False, sort_keys=False)
        return "\n" + "".join(f"  {line}\n" for line in dumped.rstrip("\n").splitlines())

    return (
        f"# Written by `locus deploy init`; read by `locus deploy plan|apply|destroy`.\n"
        f"# Terraform variables of the target live under `variables:`; see\n"
        f"# terraform/<target>/variables.tf for every name and its default.\n"
        f"\n"
        f"target: {scalar(values.target)}          # {' | '.join(TARGETS)}\n"
        f"name: {scalar(values.name)}              # prefix for every resource, the repository and the release\n"
        f"region: {scalar(values.region)}\n"
        f"compartment_id: {scalar(values.compartment_id)}\n"
        f"oci_profile: {scalar(values.oci_profile)}   # profile in ~/.oci/config for Terraform; null uses DEFAULT\n"
        f"\n"
        f"app: {scalar(values.app)}                # MODULE:ATTR of the ASGI app in app/\n"
        f"port: {values.port}\n"
        f"\n"
        f"image:\n"
        f"  tag: {scalar(values.image_tag)}\n"
        f"  architecture: {scalar(values.architecture)}   # {' | '.join(ARCHITECTURES)}: build platform and shape family\n"
        f"  registry_public: {scalar(values.registry_public)}   # a private repository needs image_pull_secret_id (or a pull Secret on OKE)\n"
        f"  repository: {scalar(values.repository)}   # an existing <registry>/<namespace>/<repo> to push to; skips terraform/registry\n"
        f"\n"
        f"# Terraform variables of terraform/{values.target}; a null value is left at the module default.\n"
        f"variables:{block(values.variables)}"
        f"\n"
        f"# Extra environment variables for the agent server. Put a Vault secret OCID in\n"
        f"# <VAR>_SECRET_OCID and the container resolves it into <VAR> at start-up.\n"
        f"environment:{block(values.environment)}"
    )


def load_values(path: Path) -> DeployValues:
    """The values in ``path`` (a ``locus-deploy.yaml``)."""
    if not path.is_file():
        raise DeployError(f"no {path.name} in {path.parent}; run `locus deploy init` first")
    try:
        data = _yaml().safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 — any parse failure is a usage error with the same remedy
        raise DeployError(f"{path}: {exc}") from exc
    return DeployValues.from_mapping(data or {})


def _render_template(text: str, values: DeployValues, sdk_spec: str) -> str:
    return (
        text.replace("__LOCUS_APP__", values.app)
        .replace("__LOCUS_PORT__", str(values.port))
        .replace("__LOCUS_SDK_SPEC__", sdk_spec)
    )


def _merge_gitignore(path: Path) -> bool:
    """Add the missing ignore lines under the marker; return whether the file changed."""
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    present = {line.strip() for line in existing.splitlines()}
    missing = [line for line in _GITIGNORE_LINES if line not in present]
    if not missing:
        return False
    block = "\n".join((_GITIGNORE_MARKER, *missing)) + "\n"
    if existing and not existing.endswith("\n"):
        existing += "\n"
    path.write_text(f"{existing}\n{block}" if existing else block, encoding="utf-8")
    return True


def init(
    dest: Path, values: DeployValues, *, force: bool = False, sdk_wheel: Path | None = None
) -> list[Path]:
    """Write the deployment directory for ``values`` and return the files written.

    Terraform roots and the Helm chart are managed files and are refreshed on
    every call. ``locus-deploy.yaml``, ``app/server.py`` and
    ``app/requirements.txt`` are yours and are only written when missing or
    with ``force``. A ``Dockerfile``, ``Dockerfile.functions`` or
    ``.dockerignore`` that exists with other content is never replaced without
    ``force``, and an existing ``.gitignore`` only gains the missing lines.
    ``sdk_wheel`` is copied into ``app/wheels/`` and installed before
    ``requirements.txt``, for an image built from an unreleased SDK.
    """
    if dest.exists() and not dest.is_dir():
        raise DeployError(f"{dest} is a file; the deployment directory must be a directory")
    if sdk_wheel is not None and not sdk_wheel.is_file():
        raise DeployError(f"--sdk-wheel {sdk_wheel} is not a file")
    sdk_spec = sdk_requirement(sdk_wheel)
    source = templates_root()
    rendered = {
        name: _render_template(
            (source / "app" / name).read_text(encoding="utf-8"), values, sdk_spec
        )
        for name in (
            "Dockerfile",
            "Dockerfile.functions",
            "server.py",
            "func.py",
            "requirements.txt",
        )
    }
    rendered[".dockerignore"] = _DOCKERIGNORE
    if not force:
        for name in _NEVER_REPLACED:
            target = dest / name
            if target.exists() and target.read_text(encoding="utf-8") != rendered[name]:
                raise DeployError(
                    f"{target} exists and differs from the template; pass --force to replace it"
                )
    written: list[Path] = []

    def write(relative: str, content: str) -> None:
        target = dest / relative
        if relative in _OWNED_BY_USER and target.exists() and not force:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or target.read_text(encoding="utf-8") != content:
            target.write_text(content, encoding="utf-8")
            written.append(target)

    for folder in ("terraform", "helm"):
        for file in sorted((source / folder).rglob("*")):
            if file.is_file():
                relative = str(file.relative_to(source))
                write(relative, file.read_text(encoding="utf-8"))
    for name in _NEVER_REPLACED:
        write(name, rendered[name])
    for name in ("server.py", "func.py", "requirements.txt"):
        write(f"app/{name}", rendered[name])
    if sdk_wheel is not None:
        wheels = dest / _WHEELS_DIR
        wheels.mkdir(parents=True, exist_ok=True)
        for stale in wheels.glob("locus_sdk-*.whl"):
            if stale.name != sdk_wheel.name:
                stale.unlink()
        copied = wheels / sdk_wheel.name
        if not copied.exists() or copied.read_bytes() != sdk_wheel.read_bytes():
            shutil.copyfile(sdk_wheel, copied)
            written.append(copied)
    dest.mkdir(parents=True, exist_ok=True)
    if _merge_gitignore(dest / ".gitignore"):
        written.append(dest / ".gitignore")
    write(VALUES_FILE, render_values(values))
    return written


def _terraform(root: str, *args: str) -> Step:
    return Step(("terraform", f"-chdir={root}", *args))


def registry_steps(values: DeployValues, action: str) -> list[Step]:
    """Terraform on the registry root; none when ``repository`` names an existing one."""
    if values.repository is not None:
        return []
    steps = [_terraform("terraform/registry", "init", "-input=false")]
    if action == "plan":
        steps.append(_terraform("terraform/registry", "plan", "-input=false"))
    else:
        steps.append(_terraform("terraform/registry", action, "-input=false", "-auto-approve"))
    return steps


def image_steps(values: DeployValues, image: str) -> list[Step]:
    """Build the target's image for its architecture and push it."""
    return [
        Step(
            (
                "docker",
                "build",
                "--platform",
                f"linux/{values.architecture}",
                "-f",
                values.dockerfile,
                "-t",
                image,
                ".",
            )
        ),
        Step(("docker", "push", image)),
    ]


def target_steps(values: DeployValues, action: str) -> list[Step]:
    """Terraform on the target root."""
    root = values.terraform_root
    steps = [_terraform(root, "init", "-input=false")]
    if action == "plan":
        steps.append(_terraform(root, "plan", "-input=false"))
    else:
        steps.append(_terraform(root, action, "-input=false", "-auto-approve"))
    if action == "apply":
        steps.append(_terraform(root, "output"))
    return steps


def _write_tfvars(dest: Path, root: str, variables: Mapping[str, Any]) -> Path:
    path = dest / root / "locus.auto.tfvars.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(variables), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _execute(step: Step, dest: Path, *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    executable = shutil.which(step.argv[0])
    if executable is None:
        raise DeployError(f"{step.argv[0]} is not on PATH; install it and retry")
    return subprocess.run(  # noqa: S603 — argv is built here from the values file, never from a shell string
        (executable, *step.argv[1:]),
        cwd=dest / step.cwd,
        check=False,
        text=True,
        capture_output=capture,
    )


def run(
    dest: Path,
    action: str,
    *,
    dry_run: bool = False,
    skip_image: bool = False,
    out: Any = None,
    execute: Callable[..., subprocess.CompletedProcess[str]] = _execute,
) -> int:
    """``plan``, ``apply`` or ``destroy`` the deployment in ``dest``.

    With ``dry_run`` the Terraform variable files are written and every
    command is printed, none is run. Returns the exit code.
    """
    if action not in ACTIONS:
        raise DeployError(f"action must be one of {', '.join(ACTIONS)}, got {action!r}")
    out = sys.stdout if out is None else out
    values = load_values(dest / VALUES_FILE)
    values.check_required_variables()
    if not (dest / values.terraform_root).is_dir():
        raise DeployError(
            f"{dest / values.terraform_root} is missing; run `locus deploy init {dest}`"
        )

    def perform(steps: Sequence[Step]) -> None:
        for step in steps:
            print(f"$ {step.render()}", file=out)
            if dry_run:
                continue
            completed = execute(step, dest)
            if completed.returncode != 0:
                raise DeployToolError(
                    f"{step.render()} exited with {completed.returncode}; see its output above"
                )

    repository = values.repository
    if repository is None:
        _write_tfvars(dest, "terraform/registry", values.registry_tfvars())
    if action in ("plan", "apply"):
        perform(registry_steps(values, action))
        if repository is None and not dry_run:
            # After an apply the repository is known; a first plan shows the placeholder.
            repository = _registry_output(dest, execute, required=action == "apply")
    image = values.image_reference(repository)
    _write_tfvars(dest, values.terraform_root, values.tfvars(image))
    if action == "apply" and not skip_image:
        perform(image_steps(values, image))
    perform(target_steps(values, action))
    if action == "destroy":
        perform(registry_steps(values, action))
    return 0


def _registry_output(
    dest: Path, execute: Callable[..., subprocess.CompletedProcess[str]], *, required: bool
) -> str | None:
    step = _terraform("terraform/registry", "output", "-raw", "image_repository")
    completed = execute(step, dest, capture=True)
    if completed.returncode != 0 or not completed.stdout.strip():
        if required:
            detail = (completed.stderr or "").strip()
            raise DeployToolError(
                "could not read image_repository from the registry root's outputs"
                + (f": {detail}" if detail else "")
            )
        return None
    return str(completed.stdout.strip())
