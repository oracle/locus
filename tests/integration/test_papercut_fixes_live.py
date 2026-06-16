# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""End-to-end coverage for the three papercut fixes that ship together.

Each test exercises a fix against a real OCI GenAI inference endpoint
so the contract is verified all the way down to the wire, not just at
the Python API surface.

Skips automatically when OCI credentials / endpoint env vars aren't
set. The same envelope as ``tests/integration/test_oci_graph_integration.py``:

- ``OCI_MODEL_ID`` — e.g. ``"openai.gpt-oss-20b"``
- ``OCI_PROFILE`` — profile name in ``~/.oci/config``
- ``OCI_AUTH_TYPE`` — ``api_key`` or ``security_token``
- ``OCI_COMPARTMENT`` — compartment OCID for the GenAI tenancy
- ``OCI_ENDPOINT`` *or* ``OCI_GENAI_REGION`` — the inference service.
  When only ``OCI_GENAI_REGION`` is set, the new ``region=`` constructor
  param is exercised; when ``OCI_ENDPOINT`` is set, it wins.

To run locally::

    OCI_MODEL_ID=openai.gpt-oss-20b \
    OCI_PROFILE=<your-profile> \
    OCI_AUTH_TYPE=api_key \
    OCI_COMPARTMENT=ocid1.compartment.oc1..xxx \
    OCI_GENAI_REGION=us-chicago-1 \
    uv run pytest tests/integration/test_papercut_fixes_live.py -v
"""

from __future__ import annotations

import os

import pytest


pytest.importorskip("oci")


from locus.agent import Agent, AgentConfig
from locus.core import Command, Message
from locus.memory.backends.memory import MemoryCheckpointer
from locus.models.providers.oci import OCIAuthType, OCIModel
from locus.multiagent import END, START, GraphConfig, StateGraph
from locus.tools.decorator import tool


pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_oci,
]


# =============================================================================
# Env helpers
# =============================================================================


def _env() -> dict[str, str | None]:
    return {
        "model_id": os.environ.get("OCI_MODEL_ID"),
        "profile": os.environ.get("OCI_PROFILE"),
        "auth_type": os.environ.get("OCI_AUTH_TYPE", "api_key"),
        "endpoint": os.environ.get("OCI_ENDPOINT"),
        "region": os.environ.get("OCI_GENAI_REGION"),
        "compartment": os.environ.get("OCI_COMPARTMENT"),
    }


def _skip_unless_ready() -> dict[str, str]:
    env = _env()
    missing = [k for k in ("model_id", "profile", "compartment") if not env[k]]
    if missing:
        pytest.skip(f"OCI integration env not set: missing {missing}")
    if not (env["endpoint"] or env["region"]):
        pytest.skip("Set OCI_ENDPOINT or OCI_GENAI_REGION to target an inference endpoint")
    return {k: v for k, v in env.items() if v is not None}  # type: ignore[misc]


def _auth_from(s: str) -> OCIAuthType:
    return OCIAuthType(s)


# =============================================================================
# #259 — OCIModel region= and profile= aliases (real call against the endpoint)
# =============================================================================


class TestOCIModelRegionLive:
    """``region=`` builds the inference endpoint and the resulting model
    actually reaches OCI GenAI. Without the fix, ``region=`` was
    silently swallowed by ``**kwargs`` and the model went to the
    profile's home region (404 if the model lived elsewhere)."""

    @pytest.mark.asyncio
    async def test_region_alone_targets_that_region(self) -> None:
        env = _skip_unless_ready()
        if not env.get("region"):
            pytest.skip("OCI_GENAI_REGION not set; cannot exercise region= path")

        model = OCIModel(
            model_id=env["model_id"],
            profile=env["profile"],
            auth_type=_auth_from(env["auth_type"]),
            compartment_id=env["compartment"],
            region=env["region"],
            max_tokens=256,
            temperature=0.0,
        )
        expected_endpoint = f"https://inference.generativeai.{env['region']}.oci.oraclecloud.com"
        assert model.config.service_endpoint == expected_endpoint

        # The endpoint round-trip is the contract we care about here.
        # Don't assert on visible content — some models (gpt-oss-*,
        # other reasoning models) can produce only hidden reasoning
        # tokens at small budgets. The presence of a ``usage`` count
        # proves the wire call actually reached the regional endpoint.
        resp = await model.complete([Message.user("Reply with the word 'ok'.")])
        assert resp.message is not None
        prompt_tokens = (resp.usage or {}).get("prompt_tokens", 0)
        assert prompt_tokens > 0, "no token usage returned — the model never billed the request"

    @pytest.mark.asyncio
    async def test_explicit_endpoint_wins_over_region(self) -> None:
        """``service_endpoint=`` is authoritative when both are set."""
        env = _skip_unless_ready()
        if not env.get("endpoint"):
            pytest.skip("OCI_ENDPOINT not set; cannot exercise explicit-endpoint path")

        model = OCIModel(
            model_id=env["model_id"],
            profile=env["profile"],
            auth_type=_auth_from(env["auth_type"]),
            compartment_id=env["compartment"],
            service_endpoint=env["endpoint"],
            region="us-ashburn-1",  # mismatched on purpose
            max_tokens=256,
            temperature=0.0,
        )
        assert model.config.service_endpoint == env["endpoint"]

        resp = await model.complete([Message.user("Reply with the word 'ok'.")])
        assert resp.message is not None
        assert (resp.usage or {}).get("prompt_tokens", 0) > 0


class TestOCIModelProfileAliasLive:
    """``profile=`` is the natural name; before the fix it fell through
    to ``**kwargs`` and the model silently used the ``DEFAULT`` profile.
    Now it aliases ``profile_name=`` and a real call goes through with
    the chosen profile's credentials."""

    @pytest.mark.asyncio
    async def test_profile_alias_round_trip(self) -> None:
        env = _skip_unless_ready()

        kwargs: dict[str, object] = {
            "model_id": env["model_id"],
            "auth_type": _auth_from(env["auth_type"]),
            "compartment_id": env["compartment"],
            "max_tokens": 256,
            "temperature": 0.0,
        }
        if env.get("endpoint"):
            kwargs["service_endpoint"] = env["endpoint"]
        elif env.get("region"):
            kwargs["region"] = env["region"]

        # Build with ``profile=`` (the alias).
        m1 = OCIModel(profile=env["profile"], **kwargs)  # type: ignore[arg-type]
        assert m1.config.profile_name == env["profile"]
        r1 = await m1.complete([Message.user("Reply with the word 'ok'.")])
        assert r1.message is not None
        # Token-usage presence proves the call reached the endpoint
        # under the aliased profile's credentials (visible content is
        # optional for reasoning models at small token budgets).
        assert (r1.usage or {}).get("prompt_tokens", 0) > 0

        # And the same profile via ``profile_name=`` should behave
        # identically — proves the alias didn't change semantics.
        m2 = OCIModel(profile_name=env["profile"], **kwargs)  # type: ignore[arg-type]
        r2 = await m2.complete([Message.user("Reply with the word 'ok'.")])
        assert r2.message is not None
        assert (r2.usage or {}).get("prompt_tokens", 0) > 0


# =============================================================================
# #258 — StateGraph interrupt_before with checkpointer + resume past gate
# =============================================================================


class TestInterruptBeforeResumeLive:
    """Pause a graph at an ``interrupt_before`` gate, let the
    checkpointer persist the paused state, then resume — the post-gate
    node makes a real LLM call and the graph completes.

    Before the fix this path was broken in two ways: (1) the pause
    boundary didn't write to the checkpointer (so cross-process resume
    lost the state), and (2) resume re-paused at the same gate (so even
    in-process resume never advanced)."""

    @pytest.mark.asyncio
    async def test_pause_then_resume_runs_llm_node(self) -> None:
        env = _skip_unless_ready()

        # Build the model from env. Use ``region=`` if no explicit
        # endpoint — exercises the #259 fix alongside #258.
        model_kwargs: dict[str, object] = {
            "model_id": env["model_id"],
            "profile": env["profile"],
            "auth_type": _auth_from(env["auth_type"]),
            "compartment_id": env["compartment"],
            "max_tokens": 32,
            "temperature": 0.0,
        }
        if env.get("endpoint"):
            model_kwargs["service_endpoint"] = env["endpoint"]
        else:
            model_kwargs["region"] = env["region"]
        oci_model = OCIModel(**model_kwargs)  # type: ignore[arg-type]

        graph = StateGraph()

        async def gate(_inputs: dict) -> dict:
            return {"approved": True}

        async def llm_node(_inputs: dict) -> dict:
            resp = await oci_model.complete([Message.user("Reply with the word 'done'.")])
            text = (resp.message.content or "").strip().lower() if resp.message else ""
            return {"llm_text": text, "done": True}

        graph.add_node("gate", gate)
        graph.add_node("llm_node", llm_node)
        graph.add_edge(START, "gate")
        graph.add_edge("gate", "llm_node")
        graph.add_edge("llm_node", END)

        cp = MemoryCheckpointer()
        cfg = GraphConfig(
            interrupt_before=["gate"],
            checkpointer=cp,
            thread_id="papercut-258-live",
        )

        # First execute — should pause at the gate without ever invoking the LLM.
        paused = await graph.execute({}, config=cfg)
        assert paused.interrupt is not None
        assert paused.interrupt.node_id == "gate"

        # The fix saves through the checkpointer at the pause boundary.
        saved = await cp.load("papercut-258-live")
        assert saved is not None, "interrupt_before must persist when a checkpointer is wired"
        assert saved.metadata.get("interrupted_node") == "gate"

        # Resume — the LLM node fires and the graph completes.
        resumed = await graph.execute(Command(resume=True), config=cfg)
        assert resumed.interrupt is None, "resume should not re-pause at the gate"
        assert resumed.final_state.get("done") is True
        assert (resumed.final_state.get("llm_text") or "").strip() != ""


# =============================================================================
# #258 — interrupt_before save shape across non-memory checkpointer backends
# =============================================================================
#
# The fix changes the save shape from ``save(state=None, metadata={...})`` to
# ``save(state=AgentState(metadata={...}))`` at both the ``interrupt_before``
# and inline ``interrupt()`` sites in ``StateGraph.execute()``. That shape
# round-trips correctly through ``MemoryCheckpointer`` and
# ``FileCheckpointer`` (covered by unit tests). These integration tests
# prove it also round-trips through the network-backed protocol backends:
# Redis (in-memory KV) and OpenSearch (JSON document store), with real
# serialization, network I/O, and indexing semantics in play.
#
# Each test skips cleanly when its backend env vars aren't set.


def _has_redis() -> bool:
    return bool(os.environ.get("REDIS_URL"))


def _has_opensearch() -> bool:
    return bool(os.environ.get("OPENSEARCH_HOSTS") or os.environ.get("OPENSEARCH_HOST"))


@pytest.mark.skipif(not _has_redis(), reason="REDIS_URL not set")
class TestInterruptBeforeRedisLive:
    """Pause→save→reload→resume through a real Redis backend.

    Verifies ``AgentState.metadata`` (which is where the fix packs the
    graph-level state) survives Redis's JSON serialization round-trip.
    """

    @pytest.mark.asyncio
    async def test_redis_backend_persists_and_resumes(self) -> None:
        from locus.memory.backends import redis_checkpointer

        cp = redis_checkpointer(
            url=os.environ["REDIS_URL"],
            prefix="locus:papercut258:",
        )

        graph = StateGraph()

        async def gate(_inputs: dict) -> dict:
            return {"approved": True}

        async def after(_inputs: dict) -> dict:
            return {"done": True}

        graph.add_node("gate", gate)
        graph.add_node("after", after)
        graph.add_edge(START, "gate")
        graph.add_edge("gate", "after")
        graph.add_edge("after", END)

        thread_id = "papercut258-redis"
        cfg = GraphConfig(
            interrupt_before=["gate"],
            checkpointer=cp,
            thread_id=thread_id,
        )

        try:
            paused = await graph.execute({"seed": 7}, config=cfg)
            assert paused.interrupt is not None
            assert paused.interrupt.node_id == "gate"

            saved = await cp.load(thread_id)
            assert saved is not None, "Redis must hold the paused state"
            assert saved.metadata.get("interrupted_node") == "gate"
            assert saved.metadata.get("graph_state", {}).get("seed") == 7

            resumed = await graph.execute(Command(resume=True), config=cfg)
            assert resumed.interrupt is None
            assert resumed.final_state.get("done") is True
        finally:
            # Idempotent cleanup — tests own their thread namespace.
            try:
                await cp.delete(thread_id)
            except Exception:
                pass
            await cp.close()


@pytest.mark.skipif(not _has_opensearch(), reason="OPENSEARCH_HOSTS not set")
class TestInterruptBeforeOpenSearchLive:
    """Pause→save→reload→resume through a real OpenSearch backend.

    OpenSearch stores the checkpoint as a JSON document with index-level
    semantics. This test verifies the fix's save shape survives that
    indexing pass and that resume continues past the gate."""

    @pytest.mark.asyncio
    async def test_opensearch_backend_persists_and_resumes(self) -> None:
        from locus.memory.backends import opensearch_checkpointer

        hosts = os.environ.get("OPENSEARCH_HOSTS") or os.environ["OPENSEARCH_HOST"]
        cp = opensearch_checkpointer(
            hosts=[hosts],
            index_name="locus-papercut258",
            username=os.environ.get("OPENSEARCH_USER"),
            password=os.environ.get("OPENSEARCH_PASSWORD"),
            use_ssl=os.environ.get("OPENSEARCH_USE_SSL", "false").lower() == "true",
            verify_certs=os.environ.get("OPENSEARCH_VERIFY_CERTS", "true").lower() == "true",
        )

        graph = StateGraph()

        async def gate(_inputs: dict) -> dict:
            return {"approved": True}

        async def after(_inputs: dict) -> dict:
            return {"done": True}

        graph.add_node("gate", gate)
        graph.add_node("after", after)
        graph.add_edge(START, "gate")
        graph.add_edge("gate", "after")
        graph.add_edge("after", END)

        thread_id = "papercut258-opensearch"
        cfg = GraphConfig(
            interrupt_before=["gate"],
            checkpointer=cp,
            thread_id=thread_id,
        )

        try:
            paused = await graph.execute({"seed": 11}, config=cfg)
            assert paused.interrupt is not None
            assert paused.interrupt.node_id == "gate"

            saved = await cp.load(thread_id)
            assert saved is not None, "OpenSearch must hold the paused state"
            assert saved.metadata.get("interrupted_node") == "gate"
            assert saved.metadata.get("graph_state", {}).get("seed") == 11

            resumed = await graph.execute(Command(resume=True), config=cfg)
            assert resumed.interrupt is None
            assert resumed.final_state.get("done") is True
        finally:
            try:
                await cp.delete(thread_id)
            except Exception:
                pass
            await cp.close()


# =============================================================================
# #260 — AgentConfig.name + Tool.func alias (real Agent run, real LLM call)
# =============================================================================


class TestAgentNameAndToolFuncLive:
    """End-to-end: build an ``Agent`` with ``name=`` set, attach a tool
    via ``@tool``, run a prompt that should drive the tool. Verifies
    the new ``name`` field flows through the config, the agent runs
    cleanly against a real LLM, and ``.func`` returns the same callable
    as ``.fn`` after decoration."""

    @pytest.mark.asyncio
    async def test_named_agent_runs_with_real_llm_and_tool(self) -> None:
        env = _skip_unless_ready()

        model_kwargs: dict[str, object] = {
            "model_id": env["model_id"],
            "profile": env["profile"],
            "auth_type": _auth_from(env["auth_type"]),
            "compartment_id": env["compartment"],
            "max_tokens": 96,
            "temperature": 0.0,
        }
        if env.get("endpoint"):
            model_kwargs["service_endpoint"] = env["endpoint"]
        else:
            model_kwargs["region"] = env["region"]
        oci_model = OCIModel(**model_kwargs)  # type: ignore[arg-type]

        @tool
        def add(a: int, b: int) -> int:
            """Return a + b."""
            return a + b

        # ``.func`` is the alias for ``.fn`` — both must point at the
        # same wrapped callable after decoration.
        assert add.func is add.fn
        assert add.func(2, 3) == 5

        config = AgentConfig(
            model=oci_model,
            name="papercut-260-live",
            tools=[add],
            system_prompt=(
                "You are a calculator. Use the `add` tool to compute sums; "
                "do not compute the answer yourself."
            ),
            max_iterations=4,
        )
        assert config.name == "papercut-260-live"

        agent = Agent(config=config)
        assert agent.name == "papercut-260-live"

        result = agent.run_sync("What is 17 + 25?")
        # Either the model returns the tool result inline or it answers
        # in plain text — both are acceptable for this smoke test. The
        # important assertion is that the full Agent + Tool path runs
        # against a real LLM without exceptions.
        assert result is not None
