# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""``locus deploy``: the values file, the directory ``init`` writes, and the commands the actions run."""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from locus import __version__, deploy
from locus.cli import main
from locus.deploy import DeployError, DeployValues, Step, entrypoint, functions


pytest.importorskip("yaml")

COMPARTMENT = "ocid1.compartment.oc1..aaaaaaaaexample"
REGION = "eu-frankfurt-1"
SERVER_KEY_SECRET = "ocid1.vaultsecret.oc1..serverkey"  # noqa: S105 — a secret OCID placeholder


def _values(**overrides: Any) -> DeployValues:
    fields: dict[str, Any] = {
        "target": "container-instances",
        "compartment_id": COMPARTMENT,
        "region": REGION,
    }
    fields.update(overrides)
    return DeployValues(**fields)


def _init(tmp_path: Path, *args: str) -> Path:
    """``locus deploy init`` with the bearer key in Vault, as the quickstart does."""
    dest = tmp_path / "deploy"
    assert (
        main(
            [
                "deploy",
                "init",
                str(dest),
                "--compartment-id",
                COMPARTMENT,
                "--region",
                REGION,
                "--server-api-key-secret-id",
                SERVER_KEY_SECRET,
                *args,
            ]
        )
        == 0
    )
    return dest


# --------------------------------------------------------------------------
# Values
# --------------------------------------------------------------------------


def test_values_round_trip_through_yaml() -> None:
    import yaml  # type: ignore[import-untyped]

    values = _values(
        target="oke",
        name="concierge",
        app="concierge:app",
        port=9000,
        image_tag="1.2.3",
        architecture="arm64",
        registry_public=True,
        oci_profile="PROD",
        variables={"cluster_id": "ocid1.cluster.oc1..x", "replicas": 3},
        environment={"LOCUS_SERVER_API_KEY_SECRET_OCID": "ocid1.vaultsecret.oc1..x"},
    )
    text = deploy.render_values(values)
    assert text.startswith("# Written by `locus deploy init`")
    assert DeployValues.from_mapping(yaml.safe_load(text)) == values


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"target": "lambda"}, "target must be one of"),
        ({"architecture": "x86"}, "architecture must be one of"),
        ({"app": "server"}, "MODULE:ATTR"),
        ({"port": 70000}, "port must be"),
        ({"compartment_id": " "}, "non-empty"),
        ({"environment": {"A": 1}}, "must be strings"),
    ],
)
def test_bad_values_are_usage_errors(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(DeployError, match=message):
        _values(**overrides)


def test_unknown_keys_in_the_values_file_are_reported() -> None:
    with pytest.raises(DeployError, match="unknown key"):
        DeployValues.from_mapping(
            {"target": "oke", "compartment_id": "c", "region": "r", "shape": "x"}
        )
    with pytest.raises(DeployError, match="under image"):
        DeployValues.from_mapping(
            {"target": "oke", "compartment_id": "c", "region": "r", "image": {"name": "x"}}
        )
    with pytest.raises(DeployError, match="needs region"):
        DeployValues.from_mapping({"target": "oke", "compartment_id": "c"})


def test_tfvars_fold_in_the_users_variables_and_drop_nulls() -> None:
    values = _values(
        variables={"create_network": False, "subnet_id": "ocid1.subnet.oc1..x", "shape": None}
    )
    tfvars = values.tfvars("eu-frankfurt-1.ocir.io/ns/locus-agent:0.1.0")
    assert tfvars["image"] == "eu-frankfurt-1.ocir.io/ns/locus-agent:0.1.0"
    assert tfvars["create_network"] is False
    assert tfvars["subnet_id"] == "ocid1.subnet.oc1..x"
    assert "shape" not in tfvars
    assert tfvars["compartment_id"] == COMPARTMENT
    assert tfvars["architecture"] == "amd64"
    assert tfvars["port"] == 8080
    assert values.registry_tfvars() == {
        "compartment_id": COMPARTMENT,
        "region": REGION,
        "oci_profile": None,
        "name": "locus-agent",
        "is_public": False,
    }


def test_required_variables_per_target() -> None:
    _values(variables={"server_api_key_secret_id": SERVER_KEY_SECRET}).check_required_variables()
    with pytest.raises(DeployError, match=r"variables\.cluster_id or variables\.kubeconfig_path"):
        _values(target="oke").check_required_variables()
    _values(
        target="oke", variables={"kubeconfig_path": "~/.kube/config"}
    ).check_required_variables()
    with pytest.raises(DeployError, match=r"variables\.subnet_id"):
        _values(target="functions", variables={"subnet_id": None}).check_required_variables()


def test_a_public_open_container_instance_needs_a_key_or_an_explicit_opt_out() -> None:
    """The default Container Instance is public on 0.0.0.0/0: not without a bearer key."""
    message = r"variables\.server_api_key_secret_id .* or variables\.allow_unauthenticated: true"
    with pytest.raises(DeployError, match=message):
        _values().check_required_variables()
    with pytest.raises(DeployError, match=message):
        _values(
            variables={"assign_public_ip": True, "ingress_cidr": "0.0.0.0/0"}
        ).check_required_variables()
    with pytest.raises(DeployError, match=message):
        _values(variables={"allow_unauthenticated": False}).check_required_variables()
    # Any one of these makes it acceptable.
    _values(variables={"server_api_key_secret_id": SERVER_KEY_SECRET}).check_required_variables()
    _values(variables={"allow_unauthenticated": True}).check_required_variables()
    _values(variables={"assign_public_ip": False}).check_required_variables()
    _values(variables={"ingress_cidr": "10.0.0.0/8"}).check_required_variables()
    # Only Container Instances get a public IP from the root.
    _values(
        target="functions", variables={"subnet_id": "ocid1.subnet.oc1..x"}
    ).check_required_variables()


def test_tfvars_write_only_the_variables_each_root_declares() -> None:
    """``port`` has no meaning for a function; Terraform rejects undeclared variables."""
    assert "port" in deploy.declared_variables("container-instances")
    assert "port" in deploy.declared_variables("oke")
    assert "port" not in deploy.declared_variables("functions")
    functions = _values(target="functions", variables={"subnet_id": "ocid1.subnet.oc1..x"})
    tfvars = functions.tfvars("x.ocir.io/ns/agent:1")
    assert "port" not in tfvars
    assert tfvars["subnet_id"] == "ocid1.subnet.oc1..x"
    assert set(tfvars) <= deploy.declared_variables("functions")
    for target in deploy.TARGETS:
        declared = deploy.declared_variables(target)
        assert {"compartment_id", "region", "image", "architecture", "environment"} <= declared
        assert "allow_unauthenticated" in declared


def test_image_reference_and_dockerfile_follow_the_target() -> None:
    values = _values(image_tag="2.0")
    assert values.image_reference(None) == "eu-frankfurt-1.ocir.io/<namespace>/locus-agent:2.0"
    assert values.image_reference("x.ocir.io/ns/repo") == "x.ocir.io/ns/repo:2.0"
    assert values.dockerfile == "Dockerfile"
    assert _values(target="functions").dockerfile == "Dockerfile.functions"
    existing = _values(repository="ghcr.io/example/agent")
    assert existing.image_reference(None) == "ghcr.io/example/agent:0.1.0"


# --------------------------------------------------------------------------
# init
# --------------------------------------------------------------------------


def test_init_writes_every_target_and_the_values_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dest = _init(
        tmp_path,
        "--target",
        "functions",
        "--architecture",
        "arm64",
        "--app",
        "svc:application",
        "--port",
        "9000",
    )
    for target in deploy.TARGETS:
        assert (dest / "terraform" / target / "main.tf").is_file()
        assert (dest / "terraform" / target / "variables.tf").is_file()
    assert (dest / "terraform" / "registry" / "main.tf").is_file()
    assert (dest / "terraform" / "modules" / "agent-environment" / "main.tf").is_file()
    assert (dest / "helm" / "locus-agent" / "Chart.yaml").is_file()
    assert (dest / "helm" / "locus-agent" / "templates" / "deployment.yaml").is_file()
    dockerfile = (dest / "Dockerfile").read_text()
    assert '"uvicorn", "svc:application", "--host", "0.0.0.0", "--port", "9000"' in dockerfile
    assert "EXPOSE 9000" in dockerfile
    assert "__LOCUS_" not in dockerfile
    assert 'asgi_handler("svc:application")' in (dest / "app" / "func.py").read_text()
    requirements = (dest / "app" / "requirements.txt").read_text()
    assert f"locus-sdk[server,oci]=={__version__}" in requirements
    assert "server.py" in {p.name for p in (dest / "app").iterdir()}
    values = deploy.load_values(dest / deploy.VALUES_FILE)
    assert values.target == "functions"
    assert values.architecture == "arm64"
    assert values.port == 9000
    assert values.variables["memory_mbs"] == 1024
    assert "subnet_id" in values.variables
    assert values.variables["server_api_key_secret_id"] == SERVER_KEY_SECRET
    assert "*.tfstate" in (dest / ".gitignore").read_text()
    assert (dest / ".dockerignore").is_file()
    out = capsys.readouterr().out
    assert "written" in out
    assert "locus deploy plan" in out
    assert f"the image installs locus-sdk[server,oci]=={__version__} from PyPI" in out
    assert "--sdk-wheel" in out


def test_init_writes_a_dockerignore_and_the_dockerfiles_copy_only_app(tmp_path: Path) -> None:
    """The build context is the whole directory: Terraform state and tfvars must stay out."""
    dest = _init(tmp_path)
    ignored = (dest / ".dockerignore").read_text().splitlines()
    for pattern in (
        ".terraform/",
        "**/.terraform/",
        "*.tfstate",
        "*.tfstate.*",
        "*.tfvars",
        "*.tfvars.json",
        "terraform/",
        "helm/",
        ".git/",
        "locus-deploy.yaml",
    ):
        assert pattern in ignored, pattern
    for name in ("Dockerfile", "Dockerfile.functions"):
        copies = [
            line.split()
            for line in (dest / name).read_text().splitlines()
            if line.startswith("COPY ") and "--from=" not in line
        ]
        assert copies, name
        assert all(words[1] == "app/" for words in copies), (name, copies)
        assert "wheels/*.whl" in (dest / name).read_text()


def test_init_installs_a_local_sdk_wheel_when_asked(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    wheel = tmp_path / "locus_sdk-9.9.9-py3-none-any.whl"
    wheel.write_bytes(b"not really a wheel")
    dest = _init(tmp_path, "--sdk-wheel", str(wheel))
    out = capsys.readouterr().out
    copied = dest / "app" / "wheels" / wheel.name
    assert copied.read_bytes() == b"not really a wheel"
    assert f"written {copied}" in out
    assert "the image installs locus_sdk-9.9.9-py3-none-any.whl from app/wheels/" in out
    requirements = (dest / "app" / "requirements.txt").read_text()
    assert "locus-sdk[server,oci]==9.9.9" in requirements

    # A newer wheel replaces the old one; two locus-sdk wheels would conflict in pip.
    newer = tmp_path / "locus_sdk-9.9.10-py3-none-any.whl"
    newer.write_bytes(b"newer")
    assert (
        main(
            [
                "deploy",
                "init",
                str(dest),
                "--compartment-id",
                COMPARTMENT,
                "--region",
                REGION,
                "--sdk-wheel",
                str(newer),
            ]
        )
        == 0
    )
    assert [p.name for p in (dest / "app" / "wheels").iterdir()] == [newer.name]


def test_sdk_wheel_must_be_a_locus_sdk_wheel_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "locus_sdk-1.0.0-py3-none-any.whl"
    assert (
        main(
            [
                "deploy",
                "init",
                str(tmp_path / "d"),
                "--compartment-id",
                COMPARTMENT,
                "--region",
                REGION,
                "--sdk-wheel",
                str(missing),
            ]
        )
        == 2
    )
    assert "is not a file" in capsys.readouterr().err
    other = tmp_path / "requests-2.0-py3-none-any.whl"
    other.write_bytes(b"x")
    assert (
        main(
            [
                "deploy",
                "init",
                str(tmp_path / "d"),
                "--compartment-id",
                COMPARTMENT,
                "--region",
                REGION,
                "--sdk-wheel",
                str(other),
            ]
        )
        == 2
    )
    assert "is not a locus-sdk wheel" in capsys.readouterr().err
    assert deploy.sdk_requirement(Path("locus_sdk-5.6.0.dev3-py3-none-any.whl")) == (
        "locus-sdk[server,oci]==5.6.0.dev3"
    )


def test_the_sdk_note_flags_a_development_version(monkeypatch: pytest.MonkeyPatch) -> None:
    import locus
    from locus.deploy.cli import _sdk_note

    monkeypatch.setattr(locus, "__version__", "5.6.0.dev0")
    note = _sdk_note(None)
    assert "locus-sdk[server,oci]==5.6.0.dev0 from PyPI" in note
    assert "development version" in note
    assert "--sdk-wheel" in note
    monkeypatch.setattr(locus, "__version__", "5.6.0")
    assert "development version" not in _sdk_note(None)


def test_init_never_replaces_an_existing_dockerfile_without_force(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dest = tmp_path / "project"
    dest.mkdir()
    (dest / "Dockerfile").write_text("FROM scratch\n")
    (dest / ".dockerignore").write_text("node_modules/\n")
    args = ["deploy", "init", str(dest), "--compartment-id", COMPARTMENT, "--region", REGION]
    assert main(args) == 2
    err = capsys.readouterr().err
    assert "Dockerfile exists and differs from the template; pass --force" in err
    # Nothing was written before the refusal.
    assert (dest / "Dockerfile").read_text() == "FROM scratch\n"
    assert not (dest / "terraform").exists()
    assert not (dest / deploy.VALUES_FILE).exists()

    (dest / "Dockerfile").unlink()
    assert main(args) == 2
    assert ".dockerignore exists and differs" in capsys.readouterr().err

    assert main([*args, "--force"]) == 0
    assert "FROM ${BASE_IMAGE}" in (dest / "Dockerfile").read_text()
    assert "terraform/" in (dest / ".dockerignore").read_text()
    # Re-running on what init itself wrote is fine: identical content is not a conflict.
    assert main(args) == 0


def test_init_appends_to_an_existing_gitignore_under_a_marker(tmp_path: Path) -> None:
    dest = tmp_path / "project"
    dest.mkdir()
    (dest / ".gitignore").write_text("node_modules/\nterraform/**/*.tfstate")
    args = ["deploy", "init", str(dest), "--compartment-id", COMPARTMENT, "--region", REGION]
    assert main(args) == 0
    text = (dest / ".gitignore").read_text()
    assert text.startswith("node_modules/\nterraform/**/*.tfstate\n\n# locus deploy\n")
    assert text.count("terraform/**/*.tfstate\n") == 1  # already there; not repeated
    assert "terraform/**/.terraform/" in text
    assert "terraform/**/locus.auto.tfvars.json" in text
    assert main(args) == 0
    assert (dest / ".gitignore").read_text() == text  # idempotent


def test_init_refuses_a_destination_that_is_a_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    file = tmp_path / "notes.txt"
    file.write_text("x")
    assert (
        main(["deploy", "init", str(file), "--compartment-id", COMPARTMENT, "--region", REGION])
        == 2
    )
    assert "is a file" in capsys.readouterr().err
    assert file.read_text() == "x"


def test_init_refreshes_managed_files_but_keeps_the_users(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dest = _init(tmp_path)
    capsys.readouterr()
    (dest / "app" / "server.py").write_text("# mine\n")
    (dest / "app" / "requirements.txt").write_text("locus-sdk\nmy-tools\n")
    (dest / "terraform" / "registry" / "main.tf").write_text("# stale\n")
    values_before = (dest / deploy.VALUES_FILE).read_text()

    assert (
        main(["deploy", "init", str(dest), "--compartment-id", "other", "--region", "other"]) == 0
    )
    assert (dest / "app" / "server.py").read_text() == "# mine\n"
    assert (dest / "app" / "requirements.txt").read_text() == "locus-sdk\nmy-tools\n"
    assert (dest / "terraform" / "registry" / "main.tf").read_text() != "# stale\n"
    assert (dest / deploy.VALUES_FILE).read_text() == values_before
    written = [line for line in capsys.readouterr().out.splitlines() if line.startswith("written ")]
    assert any(line.endswith("registry/main.tf") for line in written)
    assert not any(line.endswith("server.py") for line in written)

    assert (
        main(
            [
                "deploy",
                "init",
                str(dest),
                "--compartment-id",
                "other",
                "--region",
                "other",
                "--force",
            ]
        )
        == 0
    )
    assert "other" in (dest / deploy.VALUES_FILE).read_text()
    assert (dest / "app" / "server.py").read_text() != "# mine\n"


def test_init_is_idempotent(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dest = _init(tmp_path)
    capsys.readouterr()
    assert (
        main(["deploy", "init", str(dest), "--compartment-id", COMPARTMENT, "--region", REGION])
        == 0
    )
    assert "up to date" in capsys.readouterr().out


def test_init_without_a_compartment_is_a_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("OCI_COMPARTMENT_ID", raising=False)
    monkeypatch.delenv("OCI_REGION", raising=False)
    assert main(["deploy", "init", str(tmp_path / "d")]) == 2
    assert "--compartment-id and --region required" in capsys.readouterr().err


def test_init_reads_the_environment_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OCI_COMPARTMENT_ID", COMPARTMENT)
    monkeypatch.setenv("OCI_REGION", REGION)
    monkeypatch.setenv("OCI_PROFILE", "TEAM")
    dest = tmp_path / "d"
    assert main(["deploy", "init", str(dest)]) == 0
    values = deploy.load_values(dest / deploy.VALUES_FILE)
    assert (values.compartment_id, values.region, values.oci_profile) == (
        COMPARTMENT,
        REGION,
        "TEAM",
    )


# --------------------------------------------------------------------------
# plan / apply / destroy
# --------------------------------------------------------------------------


def _commands(out: str) -> list[str]:
    return [line[2:] for line in out.splitlines() if line.startswith("$ ")]


def test_dry_run_prints_the_terraform_and_docker_commands(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dest = _init(tmp_path, "--architecture", "arm64", "--tag", "7")
    capsys.readouterr()

    assert main(["deploy", "plan", str(dest), "--dry-run"]) == 0
    assert _commands(capsys.readouterr().out) == [
        "terraform -chdir=terraform/registry init -input=false",
        "terraform -chdir=terraform/registry plan -input=false",
        "terraform -chdir=terraform/container-instances init -input=false",
        "terraform -chdir=terraform/container-instances plan -input=false",
    ]

    assert main(["deploy", "apply", str(dest), "--dry-run"]) == 0
    image = "eu-frankfurt-1.ocir.io/<namespace>/locus-agent:7"
    assert _commands(capsys.readouterr().out) == [
        "terraform -chdir=terraform/registry init -input=false",
        "terraform -chdir=terraform/registry apply -input=false -auto-approve",
        f"docker build --platform linux/arm64 -f Dockerfile -t '{image}' .",
        f"docker push '{image}'",
        "terraform -chdir=terraform/container-instances init -input=false",
        "terraform -chdir=terraform/container-instances apply -input=false -auto-approve",
        "terraform -chdir=terraform/container-instances output",
    ]
    tfvars = json.loads(
        (dest / "terraform" / "container-instances" / "locus.auto.tfvars.json").read_text()
    )
    assert tfvars["image"] == image
    assert tfvars["compartment_id"] == COMPARTMENT
    registry = json.loads((dest / "terraform" / "registry" / "locus.auto.tfvars.json").read_text())
    assert registry["name"] == "locus-agent"

    assert main(["deploy", "destroy", str(dest), "--dry-run"]) == 0
    assert _commands(capsys.readouterr().out) == [
        "terraform -chdir=terraform/container-instances init -input=false",
        "terraform -chdir=terraform/container-instances destroy -input=false -auto-approve",
        "terraform -chdir=terraform/registry init -input=false",
        "terraform -chdir=terraform/registry destroy -input=false -auto-approve",
    ]


def test_an_existing_repository_skips_the_registry_root(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dest = _init(tmp_path, "--repository", "ghcr.io/example/agent", "--target", "functions")
    text = (
        (dest / deploy.VALUES_FILE)
        .read_text()
        .replace("subnet_id: null", "subnet_id: ocid1.subnet.oc1..x")
    )
    (dest / deploy.VALUES_FILE).write_text(text)
    capsys.readouterr()
    assert main(["deploy", "apply", str(dest), "--dry-run", "--skip-image"]) == 0
    assert _commands(capsys.readouterr().out) == [
        "terraform -chdir=terraform/functions init -input=false",
        "terraform -chdir=terraform/functions apply -input=false -auto-approve",
        "terraform -chdir=terraform/functions output",
    ]
    tfvars = json.loads((dest / "terraform" / "functions" / "locus.auto.tfvars.json").read_text())
    assert tfvars["image"] == "ghcr.io/example/agent:0.1.0"
    assert tfvars["subnet_id"] == "ocid1.subnet.oc1..x"
    assert not (dest / "terraform" / "registry" / "locus.auto.tfvars.json").exists()


def test_a_missing_required_variable_stops_before_terraform(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dest = _init(tmp_path, "--target", "oke")
    capsys.readouterr()
    assert main(["deploy", "plan", str(dest), "--dry-run"]) == 2
    err = capsys.readouterr().err
    assert "variables.cluster_id or variables.kubeconfig_path" in err


def test_plan_without_init_is_a_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["deploy", "plan", str(tmp_path)]) == 2
    assert "run `locus deploy init` first" in capsys.readouterr().err


def test_run_executes_steps_and_reads_the_repository_from_terraform(tmp_path: Path) -> None:
    dest = _init(tmp_path)
    calls: list[tuple[tuple[str, ...], str]] = []

    def execute(
        step: Step, cwd: Path, *, capture: bool = False
    ) -> subprocess.CompletedProcess[str]:
        assert cwd == dest
        calls.append((step.argv, step.cwd))
        stdout = "eu-frankfurt-1.ocir.io/realns/locus-agent\n" if capture else ""
        return subprocess.CompletedProcess(step.argv, 0, stdout=stdout, stderr="")

    out = io.StringIO()
    assert deploy.run(dest, "apply", out=out, execute=execute) == 0
    argvs = [argv for argv, _ in calls]
    assert ("terraform", "-chdir=terraform/registry", "output", "-raw", "image_repository") in argvs
    assert ("docker", "push", "eu-frankfurt-1.ocir.io/realns/locus-agent:0.1.0") in argvs
    tfvars = json.loads(
        (dest / "terraform" / "container-instances" / "locus.auto.tfvars.json").read_text()
    )
    assert tfvars["image"] == "eu-frankfurt-1.ocir.io/realns/locus-agent:0.1.0"
    assert "$ docker push eu-frankfurt-1.ocir.io/realns/locus-agent:0.1.0" in out.getvalue()


def test_plan_uses_the_repository_once_the_registry_exists(tmp_path: Path) -> None:
    dest = _init(tmp_path)
    outputs = iter(["", "eu-frankfurt-1.ocir.io/realns/locus-agent\n"])

    def execute(
        step: Step, cwd: Path, *, capture: bool = False
    ) -> subprocess.CompletedProcess[str]:
        del cwd
        if capture:
            text = next(outputs)
            return subprocess.CompletedProcess(step.argv, 0 if text else 1, stdout=text, stderr="")
        return subprocess.CompletedProcess(step.argv, 0, stdout="", stderr="")

    tfvars = dest / "terraform" / "container-instances" / "locus.auto.tfvars.json"
    assert deploy.run(dest, "plan", out=io.StringIO(), execute=execute) == 0
    assert (
        json.loads(tfvars.read_text())["image"]
        == "eu-frankfurt-1.ocir.io/<namespace>/locus-agent:0.1.0"
    )
    assert deploy.run(dest, "plan", out=io.StringIO(), execute=execute) == 0
    assert (
        json.loads(tfvars.read_text())["image"] == "eu-frankfurt-1.ocir.io/realns/locus-agent:0.1.0"
    )


def test_a_failing_step_is_reported_with_its_exit_code(tmp_path: Path) -> None:
    dest = _init(tmp_path)

    def execute(
        step: Step, cwd: Path, *, capture: bool = False
    ) -> subprocess.CompletedProcess[str]:
        del cwd, capture
        return subprocess.CompletedProcess(step.argv, 3, stdout="", stderr="")

    with pytest.raises(
        deploy.DeployToolError,
        match="terraform -chdir=terraform/registry init -input=false exited with 3",
    ):
        deploy.run(dest, "plan", out=io.StringIO(), execute=execute)


def test_a_failing_tool_exits_1_not_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Exit 2 is for usage errors; a tool that ran and failed exits 1 with what failed."""
    dest = _init(tmp_path)
    capsys.readouterr()
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda argv, **kwargs: subprocess.CompletedProcess(argv, 1, stdout="", stderr=""),
    )
    assert main(["deploy", "plan", str(dest)]) == 1
    err = capsys.readouterr().err
    assert (
        "locus deploy: terraform -chdir=terraform/registry init -input=false exited with 1" in err
    )
    assert "usage" not in err


def test_a_missing_registry_output_after_apply_is_a_tool_error(tmp_path: Path) -> None:
    dest = _init(tmp_path)

    def execute(
        step: Step, cwd: Path, *, capture: bool = False
    ) -> subprocess.CompletedProcess[str]:
        del cwd
        stderr = "No outputs found" if capture else ""
        return subprocess.CompletedProcess(step.argv, 1 if capture else 0, stdout="", stderr=stderr)

    with pytest.raises(deploy.DeployToolError, match=r"image_repository.*No outputs found"):
        deploy.run(dest, "apply", out=io.StringIO(), execute=execute)


def test_a_missing_tool_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dest = _init(tmp_path)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(DeployError, match="terraform is not on PATH"):
        deploy.run(dest, "plan", out=io.StringIO())


def test_step_renders_like_a_shell_command() -> None:
    assert Step(("docker", "push", "a b")).render() == "docker push 'a b'"
    assert (
        Step(("terraform", "plan"), cwd="terraform/oke").render()
        == "(cd terraform/oke && terraform plan)"
    )


# --------------------------------------------------------------------------
# The container entrypoint
# --------------------------------------------------------------------------


def test_entrypoint_resolves_secret_ocids_into_their_variables() -> None:
    environ = {
        "LOCUS_SERVER_API_KEY_SECRET_OCID": "ocid1.vaultsecret.oc1..key",
        "ORACLE_PASSWORD_SECRET_OCID": "ocid1.vaultsecret.oc1..pw",
        "ORACLE_PASSWORD": "already-set",  # noqa: S105 — a test value
        "EMPTY_SECRET_OCID": "",
        "_SECRET_OCID": "ocid1.vaultsecret.oc1..nameless",
        "LOCUS_APM_DATA_KEY_SECRET_OCID": "ocid1.vaultsecret.oc1..apm",
    }
    fetched: list[str] = []

    def fetch(secret_id: str) -> str:
        fetched.append(secret_id)
        return f"value-of-{secret_id.rsplit('.', 1)[-1]}"

    resolved = entrypoint.resolve_secrets(environ, fetch)
    assert resolved == ["LOCUS_APM_DATA_KEY", "LOCUS_SERVER_API_KEY"]
    assert environ["LOCUS_SERVER_API_KEY"] == "value-of-key"
    assert environ["ORACLE_PASSWORD"] == "already-set"  # noqa: S105 — a test value
    assert "EMPTY" not in environ
    assert "" not in environ
    assert fetched == ["ocid1.vaultsecret.oc1..apm", "ocid1.vaultsecret.oc1..key"]
    assert environ["OTEL_EXPORTER_OTLP_HEADERS"] == "Authorization=dataKey%20value-of-apm"


def test_entrypoint_fetches_nothing_when_nothing_asks() -> None:
    environ = {"LOCUS_SERVER_API_KEY": "plain"}

    def fetch(secret_id: str) -> str:
        raise AssertionError(secret_id)

    assert entrypoint.resolve_secrets(environ, fetch) == []
    assert environ == {"LOCUS_SERVER_API_KEY": "plain"}


def test_entrypoint_hands_over_to_the_command(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[tuple[str, list[str]]] = []
    # The entrypoint writes into the process environment; keep that to this test.
    monkeypatch.setattr(os, "environ", dict(os.environ))
    monkeypatch.setattr(os, "execvp", lambda file, args: calls.append((file, args)))
    monkeypatch.setenv("GREETING_SECRET_OCID", "ocid1.vaultsecret.oc1..hello")
    monkeypatch.delenv("GREETING", raising=False)
    monkeypatch.setattr(entrypoint, "vault_fetcher", lambda environ: lambda secret_id: "hello")
    assert entrypoint.main(["uvicorn", "server:app", "--port", "8080"]) == 0
    assert calls == [("uvicorn", ["uvicorn", "server:app", "--port", "8080"])]
    assert os.environ["GREETING"] == "hello"
    assert "resolved GREETING" in capsys.readouterr().err
    assert entrypoint.main([]) == 2


# --------------------------------------------------------------------------
# The Functions bridge
# --------------------------------------------------------------------------


class _Ctx:
    def __init__(
        self,
        method: str | None = None,
        url: str | None = None,
        headers: dict[str, Any] | None = None,
    ) -> None:
        self._method, self._url, self._headers = method, url, headers or {}

    def Method(self) -> str | None:  # noqa: N802 — the FDK's spelling
        return self._method

    def RequestURL(self) -> str | None:  # noqa: N802 — the FDK's spelling
        return self._url

    def HTTPHeaders(self) -> dict[str, Any]:  # noqa: N802 — the FDK's spelling
        return self._headers


def test_direct_invocations_become_post_invoke_with_the_servers_key() -> None:
    method, path, headers = functions.request_for(_Ctx(), b"{}", {"LOCUS_SERVER_API_KEY": "k"})
    assert (method, path) == ("POST", "/invoke")
    assert headers == {"content-type": "application/json", "authorization": "Bearer k"}
    _, _, headers = functions.request_for(_Ctx(), b"{}", {})
    assert "authorization" not in headers


def test_http_invocations_keep_the_callers_request() -> None:
    ctx = _Ctx(
        "get",
        "https://gw.example/agent/threads/t1?x=1",
        {"Authorization": ["Bearer caller"], "X-Trace": "1"},
    )
    method, path, headers = functions.request_for(ctx, b"", {"LOCUS_SERVER_API_KEY": "k"})
    assert (method, path) == ("GET", "/agent/threads/t1?x=1")
    method, path, headers = functions.request_for(
        ctx, b"", {"LOCUS_SERVER_API_KEY": "k", "LOCUS_FN_PATH_PREFIX": "/agent"}
    )
    assert (method, path) == ("GET", "/threads/t1?x=1")
    assert headers["Authorization"] == "Bearer caller"
    assert headers["X-Trace"] == "1"
    assert headers["content-type"] == "application/json"


@pytest.fixture
def fake_fdk(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    responses: list[dict[str, Any]] = []

    class Response:
        def __init__(
            self, ctx: Any, response_data: bytes, headers: dict[str, str], status_code: int
        ) -> None:
            responses.append(
                {"ctx": ctx, "data": response_data, "headers": headers, "status": status_code}
            )

    module = types.ModuleType("fdk")
    response_module = types.ModuleType("fdk.response")
    response_module.Response = Response  # type: ignore[attr-defined]
    module.response = response_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fdk", module)
    monkeypatch.setitem(sys.modules, "fdk.response", response_module)
    return responses


async def test_the_handler_runs_the_asgi_app_in_process(
    fake_fdk: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    from locus.agent import Agent
    from locus.server import AgentServer
    from locus.testing import FunctionModel

    monkeypatch.setenv("LOCUS_SERVER_API_KEY", "k")
    agent = Agent(
        model=FunctionModel(lambda messages, tools: "pong"),
        tools=[],
        reflexion=False,
        grounding=False,
    )
    app = AgentServer(agent=agent, api_key="k").app
    handler = functions.asgi_handler(app)

    await handler(_Ctx(), io.BytesIO(b'{"prompt": "ping"}'))
    assert fake_fdk[-1]["status"] == 200
    assert json.loads(fake_fdk[-1]["data"])["message"] == "pong"

    await handler(_Ctx("GET", "https://gw.example/health"), None)
    assert fake_fdk[-1]["status"] == 200
    assert json.loads(fake_fdk[-1]["data"])["status"] == "ok"

    await handler(
        _Ctx("POST", "https://gw.example/invoke", {"Authorization": "Bearer wrong"}),
        io.BytesIO(b'{"prompt": "x"}'),
    )
    assert fake_fdk[-1]["status"] == 401


async def test_the_first_direct_invocation_after_a_cold_start_is_authenticated(
    fake_fdk: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Secrets resolve before the request is built, so the bearer is the key, not a Vault OCID."""
    from locus.agent import Agent
    from locus.server import AgentServer
    from locus.testing import FunctionModel

    monkeypatch.delenv("LOCUS_SERVER_API_KEY", raising=False)
    monkeypatch.setenv("LOCUS_SERVER_API_KEY_SECRET_OCID", "ocid1.vaultsecret.oc1..serverkey")
    fetched: list[str] = []

    def fetch(secret_id: str) -> str:
        fetched.append(secret_id)
        return "k"

    monkeypatch.setattr(entrypoint, "vault_fetcher", lambda environ: fetch)
    # prepare() resolves the key into the process environment; keep that to this test.
    monkeypatch.setattr(os, "environ", dict(os.environ))
    agent = Agent(
        model=FunctionModel(lambda messages, tools: "pong"),
        tools=[],
        reflexion=False,
        grounding=False,
    )
    handler = functions.asgi_handler(AgentServer(agent=agent, api_key="k").app)

    await handler(_Ctx(), io.BytesIO(b'{"prompt": "ping"}'))
    assert fetched == ["ocid1.vaultsecret.oc1..serverkey"]
    assert fake_fdk[-1]["status"] == 200
    assert os.environ["LOCUS_SERVER_API_KEY"] == "k"
    await handler(_Ctx(), io.BytesIO(b'{"prompt": "again"}'))
    assert fetched == ["ocid1.vaultsecret.oc1..serverkey"]  # resolved once
    assert fake_fdk[-1]["status"] == 200


async def test_the_handler_loads_the_app_by_name(
    fake_fdk: list[dict[str, Any]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "fnapp.py").write_text(
        "from locus.agent import Agent\n"
        "from locus.server import AgentServer\n"
        "from locus.testing import FunctionModel\n"
        "agent = Agent(model=FunctionModel(lambda m, t: 'named'), tools=[], reflexion=False, grounding=False)\n"
        "app = AgentServer(agent=agent, allow_unauthenticated=True).app\n"
    )
    monkeypatch.delenv("LOCUS_SERVER_API_KEY", raising=False)
    handler = functions.asgi_handler(f"{tmp_path / 'fnapp.py'}:app")
    await handler(_Ctx(), io.BytesIO(b'{"prompt": "hi"}'))
    assert json.loads(fake_fdk[-1]["data"])["message"] == "named"


# --------------------------------------------------------------------------
# The generated server
# --------------------------------------------------------------------------


def _run_server_template(tmp_path: Path, **environ: str) -> subprocess.CompletedProcess[str]:
    pytest.importorskip("fastapi")
    dest = _init(tmp_path)
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("LOCUS_SERVER_API_KEY", "LOCUS_SERVER_ALLOW_UNAUTHENTICATED")
    }
    env["PYTHONPATH"] = os.pathsep.join(
        [str(Path(deploy.__file__).resolve().parents[2]), env.get("PYTHONPATH", "")]
    )
    env["OCI_COMPARTMENT_ID"] = COMPARTMENT
    env.update(environ)
    return subprocess.run(  # noqa: S603 — the generated server module, imported as a script
        [sys.executable, "-c", "import server; print(type(server.app).__name__)"],
        cwd=dest / "app",
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


def test_the_server_template_refuses_to_start_without_a_bearer_key(tmp_path: Path) -> None:
    refused = _run_server_template(tmp_path, LOCUS_MODEL="mock")
    assert refused.returncode == 1, refused.stderr
    assert "LOCUS_SERVER_API_KEY is not set" in refused.stderr
    assert "allow_unauthenticated" in refused.stderr
    assert "LOCUS_SERVER_ALLOW_UNAUTHENTICATED=1" in refused.stderr


@pytest.mark.parametrize(
    "environ",
    [
        {"LOCUS_SERVER_API_KEY": "k"},
        {"LOCUS_SERVER_ALLOW_UNAUTHENTICATED": "1"},
    ],
)
def test_the_server_template_starts_with_a_key_or_an_explicit_opt_out(
    tmp_path: Path, environ: dict[str, str]
) -> None:
    started = _run_server_template(tmp_path, LOCUS_MODEL="mock", **environ)
    assert started.returncode == 0, started.stderr
    assert started.stdout.strip() == "FastAPI"


# --------------------------------------------------------------------------
# The bundled Terraform and Helm, with the tools when they are installed
# --------------------------------------------------------------------------


@pytest.mark.parametrize("root", ["registry", *deploy.TARGETS])
def test_every_root_requires_terraform_1_9(root: str) -> None:
    """Cross-variable validation (the oke root, the public-instance check) needs 1.9."""
    text = (deploy.templates_root() / "terraform" / root / "main.tf").read_text()
    assert 'required_version = ">= 1.9"' in text


def test_the_oke_outputs_follow_the_charts_fullname_rule() -> None:
    main_tf = (deploy.templates_root() / "terraform" / "oke" / "main.tf").read_text()
    helpers = (
        deploy.templates_root() / "helm" / "locus-agent" / "templates" / "_helpers.tpl"
    ).read_text()
    assert "contains .Chart.Name .Release.Name" in helpers
    assert 'strcontains(var.name, "locus-agent") ? var.name : "${var.name}-locus-agent"' in main_tf
    assert "value       = local.fullname" in main_tf
    assert "${var.name}-locus-agent'" not in main_tf
    assert (
        'auth_type   = var.workload_identity ? "oke_workload_identity" : "instance_principal"'
        in main_tf
    )


@pytest.mark.skipif(shutil.which("terraform") is None, reason="terraform is not installed")
@pytest.mark.parametrize("root", ["registry", *deploy.TARGETS])
def test_terraform_validate(root: str, tmp_path: Path) -> None:
    source = deploy.templates_root() / "terraform"
    shutil.copytree(source, tmp_path / "terraform")
    chdir = f"-chdir={tmp_path / 'terraform' / root}"
    init = subprocess.run(  # noqa: S603 — terraform with fixed arguments on a copy of the bundled roots
        ["terraform", chdir, "init", "-backend=false", "-input=false", "-no-color"],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    if init.returncode != 0:
        pytest.skip(f"terraform init could not fetch providers: {init.stderr.strip()[:200]}")
    validate = subprocess.run(  # noqa: S603
        ["terraform", chdir, "validate", "-no-color"],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert validate.returncode == 0, validate.stdout + validate.stderr


@pytest.mark.skipif(shutil.which("helm") is None, reason="helm is not installed")
def test_helm_lint_and_template() -> None:
    chart = str(deploy.templates_root() / "helm" / "locus-agent")
    sets = [
        "--set",
        "image.repository=example.ocir.io/ns/locus-agent",
        "--set",
        "image.tag=0.1.0",
        "--set",
        "oci.region=eu-frankfurt-1",
        "--set",
        "autoscaling.enabled=true",
        "--set",
        "env.OCI_COMPARTMENT_ID=ocid1.compartment.oc1..x",
    ]
    lint = subprocess.run(  # noqa: S603 — helm with fixed arguments on the bundled chart
        ["helm", "lint", chart, *sets],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert lint.returncode == 0, lint.stdout + lint.stderr
    template = subprocess.run(  # noqa: S603
        ["helm", "template", "agent", chart, *sets],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert template.returncode == 0, template.stderr
    kinds = [
        line.split(": ", 1)[1] for line in template.stdout.splitlines() if line.startswith("kind: ")
    ]
    assert sorted(kinds) == [
        "Deployment",
        "HorizontalPodAutoscaler",
        "PodDisruptionBudget",
        "Service",
        "ServiceAccount",
    ]
    assert "OCI_RESOURCE_PRINCIPAL_VERSION" in template.stdout
    assert "readOnlyRootFilesystem: true" in template.stdout
    assert 'value: "oke_workload_identity"' in template.stdout
    names = {
        line.split(": ", 1)[1]
        for line in template.stdout.splitlines()
        if line.startswith("  name: ")
    }
    assert names == {"agent-locus-agent"}

    # The Terraform outputs apply the same rule: a release name that already
    # contains the chart name is used as is.
    for release, expected in (("locus-agent", "locus-agent"), ("my-locus-agent", "my-locus-agent")):
        rendered = subprocess.run(  # noqa: S603
            ["helm", "template", release, chart, *sets],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        assert rendered.returncode == 0, rendered.stderr
        assert {
            line.split(": ", 1)[1]
            for line in rendered.stdout.splitlines()
            if line.startswith("  name: ")
        } == {expected}

    node_identity = subprocess.run(  # noqa: S603
        ["helm", "template", "agent", chart, *sets, "--set", "oci.workloadIdentity=false"],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert node_identity.returncode == 0, node_identity.stderr
    assert 'value: "instance_principal"' in node_identity.stdout
    assert "OCI_RESOURCE_PRINCIPAL_VERSION" not in node_identity.stdout
