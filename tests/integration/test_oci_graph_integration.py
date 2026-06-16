# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Integration tests for StateGraph with OCI GenAI (Luigi) API.

These tests require:
1. Valid OCI credentials in ~/.oci/config (DEFAULT profile)
2. Active OCI session (run `oci session authenticate` if expired)
3. Network access to OCI GenAI service

To run:
    pytest tests/integration/test_oci_graph_integration.py -v

To skip these tests:
    pytest -m "not integration"
"""

import os
from typing import Annotated

import pytest
from pydantic import BaseModel


# Skip all tests if OCI SDK is not installed
pytest.importorskip("oci")


# Check if OCI config exists
OCI_CONFIG_EXISTS = os.path.exists(os.path.expanduser("~/.oci/config"))

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_oci,
]


from locus.core import (
    Command,
    Message,
    add_messages,
    scatter,
)
from locus.memory import InMemoryStore
from locus.models.providers.oci import OCIAuthType, OCIModel
from locus.multiagent import END, START, StateGraph


# =============================================================================
# Fixtures
# =============================================================================


def _get_oci_env_vars():
    """Get required OCI environment variables."""
    model_id = os.environ.get("OCI_MODEL_ID")
    profile = os.environ.get("OCI_PROFILE")
    auth_type_str = os.environ.get("OCI_AUTH_TYPE")
    endpoint = os.environ.get("OCI_ENDPOINT")
    compartment = os.environ.get("OCI_COMPARTMENT")
    return model_id, profile, auth_type_str, endpoint, compartment


def _has_oci_config():
    """Check if required OCI env vars are set."""
    model_id, profile, auth_type, endpoint, _ = _get_oci_env_vars()
    return all([model_id, profile, auth_type, endpoint])


@pytest.fixture
def oci_model():
    """Create OCI model from environment variables."""
    model_id, profile, auth_type_str, endpoint, compartment = _get_oci_env_vars()

    if not all([model_id, profile, auth_type_str, endpoint]):
        pytest.skip(
            "OCI environment variables not set (OCI_MODEL_ID, OCI_PROFILE, OCI_AUTH_TYPE, OCI_ENDPOINT)"
        )

    auth_type_map = {
        "api_key": OCIAuthType.API_KEY,
        "security_token": OCIAuthType.SECURITY_TOKEN,
    }
    auth_type = auth_type_map.get(auth_type_str, OCIAuthType.API_KEY)

    return OCIModel(
        model_id=model_id,
        profile_name=profile,
        auth_type=auth_type,
        service_endpoint=endpoint,
        compartment_id=compartment,
        max_tokens=256,
        temperature=0.3,
    )


@pytest.fixture
def store():
    """Create in-memory store."""
    return InMemoryStore()


# =============================================================================
# Basic LLM Integration Tests
# =============================================================================


class TestOCIModelBasic:
    """Basic OCI model tests."""

    @pytest.mark.asyncio
    async def test_simple_completion(self, oci_model):
        """Test simple completion with OCI model."""
        messages = [Message.user("Say 'hello' in exactly one word.")]
        response = await oci_model.complete(messages)

        assert response.message is not None
        assert response.message.content is not None
        assert len(response.message.content) > 0

    @pytest.mark.asyncio
    async def test_tool_calling(self, oci_model):
        """Test tool calling with OCI model."""
        messages = [Message.user("What's the weather in Paris?")]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"},
                        },
                        "required": ["city"],
                    },
                },
            }
        ]

        response = await oci_model.complete(messages, tools=tools)

        # Model should either call the tool or respond with text
        assert response.message is not None


# =============================================================================
# Graph with LLM Integration Tests
# =============================================================================


class TestGraphWithOCI:
    """Test StateGraph with OCI model integration."""

    @pytest.mark.asyncio
    async def test_simple_llm_node(self, oci_model):
        """Test graph with single LLM node."""
        graph = StateGraph()

        async def llm_node(inputs):
            prompt = inputs.get("prompt", "Hello")
            messages = [Message.user(prompt)]
            response = await oci_model.complete(messages)
            return {"response": response.message.content}

        graph.add_node("llm", llm_node)
        graph.add_edge(START, "llm")
        graph.add_edge("llm", END)

        result = await graph.execute({"prompt": "Say 'test' in one word"})

        assert result.success
        assert result.final_state.get("response") is not None

    @pytest.mark.asyncio
    async def test_chain_of_llm_nodes(self, oci_model):
        """Test chain of LLM processing nodes."""
        graph = StateGraph()

        async def generate(inputs):
            topic = inputs.get("topic", "AI")
            messages = [Message.user(f"Write one sentence about {topic}")]
            response = await oci_model.complete(messages)
            return {"sentence": response.message.content}

        async def summarize(inputs):
            sentence = inputs.get("sentence", "")
            messages = [Message.user(f"Summarize in 3 words: {sentence}")]
            response = await oci_model.complete(messages)
            return {"summary": response.message.content}

        graph.add_node("generate", generate)
        graph.add_node("summarize", summarize)
        graph.add_edge(START, "generate")
        graph.add_edge("generate", "summarize")
        graph.add_edge("summarize", END)

        result = await graph.execute({"topic": "machine learning"})

        assert result.success
        assert "sentence" in result.final_state
        assert "summary" in result.final_state

    @pytest.mark.asyncio
    async def test_conditional_llm_routing(self, oci_model):
        """Test conditional routing with LLM classification."""
        graph = StateGraph()

        async def classify(inputs):
            text = inputs.get("text", "")
            messages = [
                Message.user(f"Classify this as 'positive' or 'negative' (one word only): {text}")
            ]
            response = await oci_model.complete(messages)
            sentiment = response.message.content.lower().strip()
            return {"sentiment": sentiment, "text": text}

        async def handle_positive(inputs):
            return {"action": "celebrate", "original": inputs.get("text")}

        async def handle_negative(inputs):
            return {"action": "investigate", "original": inputs.get("text")}

        graph.add_node("classify", classify)
        graph.add_node("positive", handle_positive)
        graph.add_node("negative", handle_negative)

        graph.add_edge(START, "classify")
        graph.add_conditional_edges(
            "classify",
            lambda s: "positive" if "positive" in s.get("sentiment", "") else "negative",
            {"positive": "positive", "negative": "negative"},
        )
        graph.add_edge("positive", END)
        graph.add_edge("negative", END)

        result = await graph.execute({"text": "I love this product!"})
        assert result.success
        assert result.final_state.get("action") is not None


# =============================================================================
# Command Integration Tests
# =============================================================================


class TestCommandWithOCI:
    """Test Command primitive with OCI model."""

    @pytest.mark.asyncio
    async def test_llm_driven_routing(self, oci_model):
        """Test LLM-driven routing with Command."""
        graph = StateGraph()

        async def router(inputs):
            query = inputs.get("query", "")
            messages = [
                Message.user(
                    f"Is this a question about 'code' or 'general'? "
                    f"Reply with one word only: {query}"
                )
            ]
            response = await oci_model.complete(messages)
            category = response.message.content.lower().strip()

            if "code" in category:
                return Command(update={"category": "code"}, goto="code_expert")
            return Command(update={"category": "general"}, goto="general_expert")

        async def code_expert(inputs):
            return {"expert": "code", "response": "Code help here"}

        async def general_expert(inputs):
            return {"expert": "general", "response": "General help here"}

        graph.add_node("router", router)
        graph.add_node("code_expert", code_expert)
        graph.add_node("general_expert", general_expert)

        graph.add_edge(START, "router")
        graph.add_edge("code_expert", END)
        graph.add_edge("general_expert", END)

        result = await graph.execute({"query": "How do I write a Python function?"})
        assert result.success


# =============================================================================
# Store Integration Tests
# =============================================================================


class TestStoreWithOCI:
    """Test Store with OCI model integration."""

    @pytest.mark.asyncio
    async def test_memory_across_calls(self, oci_model, store):
        """Test persistent memory across graph calls."""
        graph = StateGraph()

        async def remember_fact(inputs):
            fact = inputs.get("fact", "")
            user_id = inputs.get("user_id", "default")

            # Store the fact
            await store.put(("users", user_id, "facts"), "last_fact", fact)

            return {"stored": True, "fact": fact}

        async def recall_fact(inputs):
            user_id = inputs.get("user_id", "default")

            # Recall the fact
            fact = await store.get(("users", user_id, "facts"), "last_fact")

            return {"recalled": fact}

        graph.add_node("remember", remember_fact)
        graph.add_node("recall", recall_fact)

        graph.add_edge(START, "remember")
        graph.add_edge("remember", "recall")
        graph.add_edge("recall", END)

        result = await graph.execute(
            {
                "fact": "The sky is blue",
                "user_id": "test_user",
            }
        )

        assert result.success
        assert result.final_state.get("recalled") == "The sky is blue"

    @pytest.mark.asyncio
    async def test_llm_with_memory_context(self, oci_model, store):
        """Test LLM using memory context."""
        # First, store some context
        await store.put(("context",), "user_name", "Alice")
        await store.put(("context",), "preference", "brief responses")

        graph = StateGraph()

        async def personalized_response(inputs):
            # Get context from store
            name = await store.get(("context",), "user_name") or "User"
            pref = await store.get(("context",), "preference") or "detailed"

            query = inputs.get("query", "Hello")
            messages = [
                Message.system(f"The user is {name}. They prefer {pref}."),
                Message.user(query),
            ]
            response = await oci_model.complete(messages)
            return {"response": response.message.content, "personalized_for": name}

        graph.add_node("respond", personalized_response)
        graph.add_edge(START, "respond")
        graph.add_edge("respond", END)

        result = await graph.execute({"query": "What's your name?"})

        assert result.success
        assert result.final_state.get("personalized_for") == "Alice"


# =============================================================================
# State Reducers Integration Tests
# =============================================================================


class TestReducersWithOCI:
    """Test state reducers with OCI model."""

    @pytest.mark.asyncio
    async def test_message_accumulation(self, oci_model):
        """Test message accumulation with add_messages reducer."""

        class ConversationState(BaseModel):
            messages: Annotated[list, add_messages] = []
            turn: int = 0

        graph = StateGraph(state_schema=ConversationState)

        async def user_turn(inputs):
            messages = inputs.get("messages", [])
            return {
                "messages": [Message.user(f"Turn {inputs.get('turn', 0)}: Hello")],
                "turn": inputs.get("turn", 0) + 1,
            }

        async def assistant_turn(inputs):
            messages = inputs.get("messages", [])
            response = await oci_model.complete(messages)
            return {
                "messages": [response.message],
            }

        graph.add_node("user", user_turn)
        graph.add_node("assistant", assistant_turn)

        graph.add_edge(START, "user")
        graph.add_edge("user", "assistant")
        graph.add_edge("assistant", END)

        result = await graph.execute({"turn": 1})

        assert result.success
        # Should have accumulated messages from both turns


# =============================================================================
# Send (Map-Reduce) Integration Tests
# =============================================================================


class TestSendWithOCI:
    """Test Send pattern with OCI model."""

    @pytest.mark.asyncio
    async def test_parallel_llm_processing(self, oci_model):
        """Test parallel LLM processing with Send."""
        graph = StateGraph()

        async def splitter(inputs):
            topics = inputs.get("topics", ["AI", "ML", "DL"])
            return scatter("processor", topics, key="topic")

        async def processor(inputs):
            topic = inputs.get("topic", "technology")
            messages = [Message.user(f"Define {topic} in 5 words or less")]
            response = await oci_model.complete(messages)
            return {
                "topic": topic,
                "definition": response.message.content,
            }

        async def aggregator(inputs):
            # Collect all send results
            definitions = {}
            for key, value in inputs.items():
                if key.startswith("send_") and isinstance(value, dict):
                    if "topic" in value and "definition" in value:
                        definitions[value["topic"]] = value["definition"]
            return {"all_definitions": definitions}

        graph.add_node("splitter", splitter)
        graph.add_node("processor", processor)
        graph.add_node("aggregator", aggregator)

        graph.add_edge(START, "splitter")
        graph.add_edge("splitter", "aggregator")
        graph.add_edge("aggregator", END)

        result = await graph.execute({"topics": ["Python", "Java"]})

        assert result.success


# =============================================================================
# End-to-End Workflow Tests
# =============================================================================


class TestE2EWorkflows:
    """End-to-end workflow tests with OCI."""

    @pytest.mark.asyncio
    async def test_research_assistant_workflow(self, oci_model, store):
        """Test a complete research assistant workflow."""
        graph = StateGraph()

        async def understand_query(inputs):
            query = inputs.get("query", "")
            messages = [
                Message.user(f"What is the main topic of this query? Reply in 1-2 words: {query}")
            ]
            response = await oci_model.complete(messages)
            topic = response.message.content.strip()

            # Store the topic
            await store.put(("research",), "current_topic", topic)

            return {"topic": topic, "query": query}

        async def research_topic(inputs):
            topic = inputs.get("topic", "")
            messages = [Message.user(f"Give one key fact about {topic} in one sentence.")]
            response = await oci_model.complete(messages)
            return {"fact": response.message.content}

        async def synthesize(inputs):
            topic = inputs.get("topic", "")
            fact = inputs.get("fact", "")
            return {
                "summary": f"Topic: {topic}. Key fact: {fact}",
                "completed": True,
            }

        graph.add_node("understand", understand_query)
        graph.add_node("research", research_topic)
        graph.add_node("synthesize", synthesize)

        graph.add_edge(START, "understand")
        graph.add_edge("understand", "research")
        graph.add_edge("research", "synthesize")
        graph.add_edge("synthesize", END)

        result = await graph.execute(
            {
                "query": "Tell me about quantum computing",
            }
        )

        assert result.success
        assert result.final_state.get("completed")
        assert "topic" in result.final_state
        assert "summary" in result.final_state

        # Verify store was used
        stored_topic = await store.get(("research",), "current_topic")
        assert stored_topic is not None
