# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Notebook 42: durable agent threads on Oracle Database 26ai.

In-memory and on-disk checkpointers are fine for local development —
when the process restarts, the conversation is gone. This notebook
wires Locus's ``oracle_checkpointer`` into an ``Agent`` so the agent
loop itself writes ``AgentState`` (messages, metrics, all of it) to
Oracle 26ai after every iteration. A second ``Agent`` instance — built
fresh, simulating a different process — loads the same ``thread_id``
and the conversation continues uninterrupted.

The point: it isn't a state-blob primitive demo. The *Locus agent
runtime* is the protagonist; Oracle 26ai is the durable substrate it
checkpoints into. The user never calls ``cp.save`` — the agent does
that for them.

Key concepts:

- ``oracle_checkpointer(...)`` returns a ``StorageBackendAdapter``
  wrapped around ``OracleBackend``. Pass it to ``Agent(checkpointer=...)``
  and the agent loop persists state on every iteration.
- ``agent.run_sync(prompt, thread_id="...")`` loads state for that
  thread before the first model call and saves after each iteration.
- ``checkpoint_every_n_iterations=1`` saves after every iteration
  (already the default for this adapter on a thread), so a process
  killed mid-turn loses at most the iteration in flight.
- ``cp.list_threads()`` on the agent's own checkpointer — the
  enumeration an admin dashboard would use (see the note in
  ``first_session``).

The demo simulates two sessions. Session 1 starts a conversation and
the agent persists the state to Oracle automatically. Session 2 spins
up a brand-new ``Agent`` instance — different Python object, same
``thread_id``, same Oracle pool — and the agent recalls turn 1 without
being told what was said.

Run it:
    export ORACLE_DSN=mydb_low                   # tnsnames alias
    export ORACLE_USER=locus_app                 # least-privileged app schema
    export ORACLE_PASSWORD='<app-password>'
    export ORACLE_WALLET=~/.oci/wallets/mydb     # mTLS only; a TLS-only database has none
    export ORACLE_WALLET_PASSWORD='<wallet-pw>'  # if the wallet is encrypted
    # TLS-only (no wallet): a connect string in ORACLE_DSN, or an alias + ORACLE_CONFIG_DIR

    python examples/notebook_42_oracle_26ai_checkpointer.py

If any of ``ORACLE_DSN``, ``ORACLE_USER`` or ``ORACLE_PASSWORD`` is unset,
or a TNS alias has no ``ORACLE_WALLET`` / ``ORACLE_CONFIG_DIR`` to resolve
it, the notebook prints the missing names and
the wiring snippet, then exits cleanly — no traceback, no
half-initialised state.

Difficulty: Intermediate. Self-contained — no prior notebook required.
"""

from __future__ import annotations

import asyncio
import os
import sys

from config import get_model, print_config

from locus.agent import Agent
from locus.memory.backends import oracle_checkpointer


_REQUIRED_ENV = (
    "ORACLE_DSN",
    "ORACLE_USER",
    "ORACLE_PASSWORD",
)


THREAD_ID = "notebook_42_thread"
TABLE_NAME = "locus_notebook_42"


def _missing_env() -> list[str]:
    missing = [name for name in _REQUIRED_ENV if not os.environ.get(name)]
    # A tnsnames alias in ORACLE_DSN needs a directory to resolve it: the
    # mTLS wallet, or ORACLE_CONFIG_DIR / TNS_ADMIN for a TLS-only database.
    # A connect descriptor or an Easy Connect address (host:port/service)
    # needs neither, and an unset DSN is already reported above.
    dsn = os.environ.get("ORACLE_DSN", "")
    if (
        dsn
        and "(" not in dsn
        and "/" not in dsn
        and ":" not in dsn
        and not any(
            os.environ.get(name) for name in ("ORACLE_WALLET", "ORACLE_CONFIG_DIR", "TNS_ADMIN")
        )
    ):
        missing.append("ORACLE_WALLET (mTLS) or ORACLE_CONFIG_DIR (TLS-only alias)")
    return missing


def _oracle_connection() -> dict[str, str]:
    """dsn / user / password, plus the TNS directory or wallet the environment names."""
    kwargs = {
        "dsn": os.environ["ORACLE_DSN"],
        "user": os.environ["ORACLE_USER"],
        "password": os.environ["ORACLE_PASSWORD"],
    }
    if os.environ.get("ORACLE_CONFIG_DIR"):
        kwargs["config_dir"] = os.path.expanduser(os.environ["ORACLE_CONFIG_DIR"])
    if os.environ.get("ORACLE_WALLET"):  # mTLS only; a TLS-only database has no wallet
        kwargs["wallet_location"] = os.path.expanduser(os.environ["ORACLE_WALLET"])
        kwargs["wallet_password"] = os.environ.get("ORACLE_WALLET_PASSWORD", "")
    return kwargs


def _print_skip_banner(missing: list[str]) -> None:
    print("\n--- Notebook 42: Oracle 26ai durable agent thread ---")
    print(
        "Required environment variables not set; skipping the live demo so "
        "this file still runs cleanly in CI.\n"
    )
    print("Missing:")
    for name in missing:
        print(f"  - {name}")
    print(
        "\nProvision an Autonomous Database. With mTLS, drop its wallet under "
        "$ORACLE_WALLET; without it (TLS-only), put the connect string in "
        "ORACLE_DSN or an alias in a tnsnames.ora under $ORACLE_CONFIG_DIR. "
        "Then set the variables above and re-run."
    )
    print("\nMinimal wiring (what the live path below builds):")
    print(
        """
    from locus.agent import Agent
    from locus.memory.backends import oracle_checkpointer

    cp = oracle_checkpointer(
        dsn=os.environ["ORACLE_DSN"],
        user=os.environ["ORACLE_USER"],
        password=os.environ["ORACLE_PASSWORD"],
        wallet_location=os.environ["ORACLE_WALLET"],
        wallet_password=os.environ.get("ORACLE_WALLET_PASSWORD", ""),
        table_name="locus_checkpoints",
    )
    agent = Agent(model=model, checkpointer=cp, checkpoint_every_n_iterations=1)
    agent.run_sync("My name is Alice.", thread_id="t1")
    agent.run_sync("What's my name?", thread_id="t1")  # recalls turn 1
        """.rstrip()
    )


def _build_checkpointer():
    return oracle_checkpointer(
        **_oracle_connection(),
        table_name=TABLE_NAME,
    )


def _build_agent():
    """Fresh ``Agent`` and checkpointer pool per call — the two sessions share
    nothing in memory, so continuity can only come from Oracle 26ai."""
    return Agent(
        model=get_model(max_tokens=120),
        system_prompt=(
            "You are an on-call SRE assistant. Be concise — one or two "
            "sentences per turn. Remember everything the user has told "
            "you earlier in this thread."
        ),
        checkpointer=_build_checkpointer(),
        # Save after every iteration, so a crash mid-turn loses at most the
        # iteration in flight. On a thread this adapter already saves that
        # often by default (it can delete a single save); the setting makes
        # it explicit.
        checkpoint_every_n_iterations=1,
    )


async def first_session() -> None:
    """Session 1 — operator starts a conversation; agent persists turn 1."""
    print("\n--- Session 1: agent persists turn 1 to Oracle 26ai ---")
    agent = _build_agent()
    cp = agent.config.checkpointer
    try:
        prompt = "p99 on checkout-api spiked at 14:02. The deploy at 13:58 looks suspicious."
        result = agent.run_sync(prompt, thread_id=THREAD_ID)
        print(f"User : {prompt}")
        print(f"Agent: {result.message.strip()}")
        print(f"  state size after this turn: {len(result.state.messages)} messages")

        # The agent's own checkpointer is the admin surface a dashboard would
        # use. Note: the adapter forwards its default pattern "*" unchanged to
        # the Oracle backend's LIKE filter, which matches no row, so this
        # prints []; cp.list_threads(pattern="%") lists the stored keys.
        threads = await cp.list_threads()
        print(f"  threads currently in {TABLE_NAME}: {threads}")
    finally:
        await cp.close()


async def second_session() -> None:
    """Session 2 — different ``Agent`` object, same ``thread_id``.

    The new agent loads the thread's newest checkpoint from Oracle on
    the first ``run_sync`` call. The application passes only the new
    prompt; Locus loads the saved state for ``thread_id`` from the
    checkpointer and starts the new turn on it, so the model sees turn 1
    as prior messages.
    """
    print("\n--- Session 2: a brand-new Agent resumes the thread ---")
    agent = _build_agent()
    cp = agent.config.checkpointer
    try:
        prompt = "Has anything like that happened in the last fortnight?"
        result = agent.run_sync(prompt, thread_id=THREAD_ID)
        print(f"User : {prompt}")
        print(f"Agent: {result.message.strip()}")
        print(f"  state size after this turn: {len(result.state.messages)} messages")
    finally:
        await cp.close()

    print(
        "\nThe second Agent loaded prior turns from Oracle 26ai before "
        "the first model call — no AgentState was hand-built, no save/load "
        "was called manually. The Locus agent loop did it on every iteration."
    )


async def main() -> None:
    missing = _missing_env()
    if missing:
        _print_skip_banner(missing)
        return

    print_config()
    await first_session()
    await second_session()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
