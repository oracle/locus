# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""``locus a2a card AGENT``: write an agent's A2A v1.0 card, or serve it.

::

    locus a2a card app.py:agent --url https://research.example.com -o agent-card.json
    locus a2a card app.py:agent --url https://research.example.com \\
        --transport jsonrpc --transport http+json --transport grpc --grpc-port 7422 --push
    locus a2a card app.py:agent --url http://127.0.0.1:7421 --serve --port 7421

The card is built by :func:`locus.a2a.card.card_from_agent` from the agent's
tools, skills, streaming capability and the transports named. ``--serve``
starts an :class:`~locus.a2a.A2AServer` that publishes that card.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any


def register_a2a_commands(commands: Any, load_agent: Any) -> None:
    """Add the ``a2a`` command to the ``locus`` parser's subcommands."""
    a2a = commands.add_parser("a2a", help="agent-to-agent: write or serve an agent's card")
    actions = a2a.add_subparsers(dest="action", required=True)
    card = actions.add_parser(
        "card", help="write the A2A v1.0 agent card for an agent, or serve the agent with it"
    )
    card.add_argument("agent", help="MODULE:ATTR or FILE.py:ATTR naming an Agent or a factory")
    card.add_argument(
        "--url", default="http://127.0.0.1:7421", help="public base URL of the A2A server"
    )
    card.add_argument("--name", help="card name (default: the agent's name)")
    card.add_argument(
        "--description", help="card description (default: the system prompt's first line)"
    )
    card.add_argument("--version", dest="card_version", default="0.1.0", help="agent version")
    card.add_argument(
        "--transport",
        action="append",
        dest="transports",
        choices=["jsonrpc", "http+json", "grpc"],
        help="a protocol binding to advertise; repeat (default: jsonrpc and http+json)",
    )
    card.add_argument(
        "--grpc-port", type=int, help="port of the gRPC interface, with --transport grpc"
    )
    card.add_argument(
        "--grpc-url", help="address of the gRPC interface instead of host:--grpc-port"
    )
    card.add_argument(
        "--push",
        action="store_true",
        help="advertise (and with --serve, deliver) push notifications",
    )
    card.add_argument("--provider", help="publishing organisation")
    card.add_argument("--docs-url", help="documentationUrl for the card")
    card.add_argument("-o", "--output", help="write the card here instead of stdout")
    card.add_argument("--serve", action="store_true", help="run the A2A server with this card")
    card.add_argument("--host", default="127.0.0.1", help="with --serve: bind address")
    card.add_argument("--port", type=int, default=7421, help="with --serve: HTTP port")
    card.add_argument("--api-key", help="with --serve: bearer key; or set LOCUS_A2A_API_KEY")
    card.add_argument(
        "--allow-private-push",
        action="store_true",
        help="with --serve --push: deliver to loopback and private addresses",
    )
    card.set_defaults(handler=lambda args: _cmd_a2a_card(args, load_agent))


def _cmd_a2a_card(args: argparse.Namespace, load_agent: Any) -> int:
    import os

    from locus.a2a.card import DEFAULT_TRANSPORTS, card_from_agent, card_to_json, write_card
    from locus.a2a.spec import AgentProvider
    from locus.a2a.spec_v1 import BINDING_GRPC
    from locus.a2a.transports import normalise_binding

    agent = load_agent(args.agent)
    transports = [normalise_binding(t) for t in (args.transports or DEFAULT_TRANSPORTS)]
    api_key = args.api_key or os.environ.get("LOCUS_A2A_API_KEY") or None
    if BINDING_GRPC in transports and args.grpc_port is None and not args.grpc_url:
        print("locus: error: --transport grpc needs --grpc-port or --grpc-url", file=sys.stderr)
        return 2
    card = card_from_agent(
        agent,
        url=args.url,
        name=args.name,
        description=args.description,
        version=args.card_version,
        transports=transports,
        grpc_port=args.grpc_port,
        grpc_url=args.grpc_url,
        push_notifications=args.push,
        provider=AgentProvider(organization=args.provider) if args.provider else None,
        documentation_url=args.docs_url,
        bearer_auth=api_key is not None,
    )
    if args.output:
        path = write_card(card, args.output)
        print(f"agent card written to {path}")
    else:
        sys.stdout.write(card_to_json(card))

    if not args.serve:
        return 0

    from locus.a2a import A2AServer
    from locus.a2a.push import PushNotificationSettings

    server = A2AServer(
        agent=agent,
        name=card.name,
        description=card.description,
        skills=card.skills,
        url=card.url,
        provider=card.provider,
        version=card.version,
        api_key=api_key,
        transports=transports,
        grpc_port=args.grpc_port,
        grpc_url=args.grpc_url,
        push_notifications=(
            PushNotificationSettings(allow_private_urls=args.allow_private_push)
            if args.push
            else None
        ),
    )
    try:
        server.run(host=args.host, port=args.port)
    except RuntimeError as exc:
        print(f"locus: {exc}", file=sys.stderr)
        return 1
    return 0


__all__ = ["register_a2a_commands"]
