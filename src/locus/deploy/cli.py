# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""The ``locus deploy`` subcommand: argument handling for :mod:`locus.deploy`."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from locus.deploy import (
    ACTIONS,
    ARCHITECTURES,
    TARGETS,
    VALUES_FILE,
    DeployError,
    DeployValues,
    init,
    run,
)


__all__ = ["add_deploy_command", "cmd_deploy"]


def add_deploy_command(commands: Any) -> None:
    """Register ``deploy`` and its actions on the ``locus`` subparsers."""
    deploy = commands.add_parser(
        "deploy",
        help="deploy the agent server to OCI with Terraform: Container Instances, OKE or Functions",
    )
    actions = deploy.add_subparsers(dest="action", required=True)

    initialise = actions.add_parser(
        "init",
        help=f"write a deployment directory: Terraform roots, Helm chart, Dockerfiles, {VALUES_FILE}",
    )
    initialise.add_argument(
        "dir", nargs="?", default="deploy", help="the deployment directory (default: deploy)"
    )
    initialise.add_argument("--target", choices=TARGETS, default="container-instances")
    initialise.add_argument(
        "--compartment-id",
        default=os.environ.get("OCI_COMPARTMENT_ID"),
        help="compartment OCID (default: $OCI_COMPARTMENT_ID)",
    )
    initialise.add_argument(
        "--region", default=os.environ.get("OCI_REGION"), help="region (default: $OCI_REGION)"
    )
    initialise.add_argument(
        "--name", default="locus-agent", help="resource prefix (default: locus-agent)"
    )
    initialise.add_argument(
        "--app", default="server:app", help="MODULE:ATTR of the ASGI app in app/"
    )
    initialise.add_argument("--port", type=int, default=8080)
    initialise.add_argument("--architecture", choices=ARCHITECTURES, default="amd64")
    initialise.add_argument("--tag", default="0.1.0", help="image tag (default: 0.1.0)")
    initialise.add_argument(
        "--public-repository",
        action="store_true",
        help="create the OCIR repository public (no pull secret)",
    )
    initialise.add_argument(
        "--repository", help="push to this existing repository path instead of creating one"
    )
    initialise.add_argument(
        "--profile",
        default=os.environ.get("OCI_PROFILE"),
        help="~/.oci/config profile (default: $OCI_PROFILE)",
    )
    initialise.add_argument(
        "--force",
        action="store_true",
        help=f"overwrite {VALUES_FILE}, app/server.py and app/requirements.txt",
    )

    for action in ACTIONS:
        parser = actions.add_parser(action, help=f"terraform {action} the target in {VALUES_FILE}")
        parser.add_argument(
            "dir", nargs="?", default=".", help="the deployment directory (default: here)"
        )
        parser.add_argument("--dry-run", action="store_true", help="print the commands; run none")
        if action == "apply":
            parser.add_argument(
                "--skip-image",
                action="store_true",
                help="do not build or push the image (already pushed)",
            )
    deploy.set_defaults(handler=cmd_deploy)


def cmd_deploy(args: argparse.Namespace) -> int:
    """Run the ``deploy`` action in ``args``; usage problems become ``CLIError``."""
    from locus.cli import CLIError

    try:
        if args.action == "init":
            return _init(args)
        return run(
            Path(args.dir),
            args.action,
            dry_run=args.dry_run,
            skip_image=getattr(args, "skip_image", False),
        )
    except DeployError as exc:
        raise CLIError(str(exc)) from exc


def _init(args: argparse.Namespace) -> int:
    missing = [
        flag
        for flag, value in (("--compartment-id", args.compartment_id), ("--region", args.region))
        if not value
    ]
    if missing:
        raise DeployError(
            f"{' and '.join(missing)} required (or OCI_COMPARTMENT_ID / OCI_REGION in the environment)"
        )
    from locus.deploy import _DEFAULT_VARIABLES

    values = DeployValues(
        target=args.target,
        compartment_id=args.compartment_id,
        region=args.region,
        name=args.name,
        app=args.app,
        port=args.port,
        image_tag=args.tag,
        architecture=args.architecture,
        registry_public=args.public_repository,
        repository=args.repository,
        oci_profile=args.profile,
        variables=dict(_DEFAULT_VARIABLES[args.target]),
    )
    dest = Path(args.dir)
    written = init(dest, values, force=args.force)
    for path in written:
        print(f"written {path}")
    if not written:
        print(f"{dest} is up to date")
    print(
        f"\nNext: edit {dest / VALUES_FILE} and app/server.py, then\n"
        f"  locus deploy plan {dest}\n"
        f"  locus deploy apply {dest}",
        file=sys.stdout,
    )
    return 0
