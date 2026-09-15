# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Durable execution for agent runs.

:mod:`tulip.durable.temporal` runs an agent on Temporal: a run is a workflow
that survives worker restarts, and a held approval is a wait of any length for
a signal. Needs ``pip install "tulip-agents[temporal]"``.

:mod:`tulip.durable.dbos` does the same on DBOS, in your own process with the
state in Postgres (or SQLite). Needs ``pip install "tulip-agents[dbos]"``.

Both run the agent in segments (:mod:`tulip.durable.segments`), each at most
once. Nothing is imported here, so this package costs nothing unless you use it.
"""
