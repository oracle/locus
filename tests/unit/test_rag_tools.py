# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for RAG tools."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from locus.rag.stores.base import Document, SearchResult
from locus.rag.tools import RAGToolkit, create_rag_context_tool, create_rag_tool


class TestCreateRagTool:
    """Tests for create_rag_tool function."""

    @pytest.fixture
    def mock_retriever(self):
        """Create a mock retriever."""
        retriever = MagicMock()
        retriever.retrieve = AsyncMock()
        return retriever

    @pytest.fixture
    def mock_retrieval_result(self):
        """Create a mock retrieval result."""
        result = MagicMock()
        result.documents = [
            SearchResult(
                document=Document(
                    id="doc1",
                    content="First document content",
                    metadata={"source": "test"},
                ),
                score=0.95,
            ),
            SearchResult(
                document=Document(
                    id="doc2",
                    content="Second document content",
                    metadata={"source": "test"},
                ),
                score=0.80,
            ),
        ]
        result.total_results = 2
        return result

    def test_create_tool_with_defaults(self, mock_retriever):
        """Test creating tool with default settings."""
        tool = create_rag_tool(mock_retriever)
        assert tool.name == "search_knowledge"

    def test_create_tool_with_custom_name(self, mock_retriever):
        """Test creating tool with custom name."""
        tool = create_rag_tool(mock_retriever, name="my_search")
        assert tool.name == "my_search"

    def test_create_tool_with_custom_description(self, mock_retriever):
        """Test creating tool with custom description."""
        tool = create_rag_tool(mock_retriever, description="Custom description")
        assert tool.description == "Custom description"

    @pytest.mark.asyncio
    async def test_tool_calls_retriever(self, mock_retriever, mock_retrieval_result):
        """Test that tool calls retriever correctly."""
        import json

        mock_retriever.retrieve.return_value = mock_retrieval_result
        tool = create_rag_tool(mock_retriever)

        result_str = await tool.execute(query="test query")
        result = json.loads(result_str)

        mock_retriever.retrieve.assert_called_once_with(
            query="test query",
            limit=5,
            threshold=0.5,
        )
        assert result["total"] == 2
        assert len(result["results"]) == 2
        assert result["query"] == "test query"

    @pytest.mark.asyncio
    async def test_tool_returns_formatted_results(self, mock_retriever, mock_retrieval_result):
        """Test that tool returns properly formatted results."""
        import json

        mock_retriever.retrieve.return_value = mock_retrieval_result
        tool = create_rag_tool(mock_retriever)

        result_str = await tool.execute(query="test query")
        result = json.loads(result_str)

        assert result["results"][0]["id"] == "doc1"
        assert result["results"][0]["content"] == "First document content"
        assert result["results"][0]["score"] == 0.95
        assert result["results"][0]["metadata"]["source"] == "test"

    @pytest.mark.asyncio
    async def test_tool_with_custom_params(self, mock_retriever, mock_retrieval_result):
        """Test that tool respects custom parameters."""
        mock_retriever.retrieve.return_value = mock_retrieval_result
        tool = create_rag_tool(mock_retriever, limit=10, threshold=0.7)

        await tool.execute(query="query", max_results=3, min_score=0.8)

        mock_retriever.retrieve.assert_called_once_with(
            query="query",
            limit=3,
            threshold=0.8,
        )


class TestCreateRagContextTool:
    """Tests for create_rag_context_tool function."""

    @pytest.fixture
    def mock_retriever(self):
        """Create a mock retriever."""
        retriever = MagicMock()
        retriever.retrieve_text = AsyncMock()
        return retriever

    def test_create_context_tool_with_defaults(self, mock_retriever):
        """Test creating context tool with defaults."""
        tool = create_rag_context_tool(mock_retriever)
        assert tool.name == "get_context"

    def test_create_context_tool_with_custom_name(self, mock_retriever):
        """Test creating context tool with custom name."""
        tool = create_rag_context_tool(mock_retriever, name="my_context")
        assert tool.name == "my_context"

    @pytest.mark.asyncio
    async def test_context_tool_calls_retriever(self, mock_retriever):
        """Test that context tool calls retriever correctly."""
        mock_retriever.retrieve_text.return_value = "Some relevant context"
        tool = create_rag_context_tool(mock_retriever)

        result = await tool.execute(query="test query")

        mock_retriever.retrieve_text.assert_called_once_with(
            query="test query",
            limit=3,
            separator="\n\n---\n\n",
            spotlight=True,
        )
        assert "Relevant context" in result
        assert "Some relevant context" in result

    @pytest.mark.asyncio
    async def test_context_tool_handles_empty_results(self, mock_retriever):
        """Test that context tool handles empty results."""
        mock_retriever.retrieve_text.return_value = ""
        tool = create_rag_context_tool(mock_retriever)

        result = await tool.execute(query="test query")

        assert result == "No relevant context found."


class TestRAGToolkit:
    """Tests for RAGToolkit class."""

    @pytest.fixture
    def mock_retriever(self):
        """Create a mock retriever."""
        retriever = MagicMock()
        retriever.retrieve = AsyncMock()
        retriever.retrieve_text = AsyncMock()
        retriever.store = MagicMock()
        retriever.store.get = AsyncMock()
        return retriever

    def test_create_toolkit(self, mock_retriever):
        """Test creating toolkit."""
        toolkit = RAGToolkit(mock_retriever)
        assert toolkit.retriever is mock_retriever
        assert toolkit.prefix == "kb"

    def test_create_toolkit_with_custom_prefix(self, mock_retriever):
        """Test creating toolkit with custom prefix."""
        toolkit = RAGToolkit(mock_retriever, prefix="docs")
        assert toolkit.prefix == "docs"

    def test_get_tools(self, mock_retriever):
        """Test getting all tools."""
        toolkit = RAGToolkit(mock_retriever)
        tools = toolkit.get_tools()

        assert len(tools) == 3
        assert tools[0].name == "kb_search"
        assert tools[1].name == "kb_context"
        assert tools[2].name == "kb_lookup"

    def test_search_tool(self, mock_retriever):
        """Test getting search tool."""
        toolkit = RAGToolkit(mock_retriever)
        tool = toolkit.search_tool()
        assert tool.name == "kb_search"

    def test_context_tool(self, mock_retriever):
        """Test getting context tool."""
        toolkit = RAGToolkit(mock_retriever)
        tool = toolkit.context_tool()
        assert tool.name == "kb_context"

    def test_lookup_tool(self, mock_retriever):
        """Test getting lookup tool."""
        toolkit = RAGToolkit(mock_retriever)
        tool = toolkit.lookup_tool()
        assert tool.name == "kb_lookup"

    @pytest.mark.asyncio
    async def test_lookup_tool_found(self, mock_retriever):
        """Test lookup tool when document is found."""
        import json

        doc = Document(
            id="test_doc",
            content="Test content",
            metadata={"key": "value"},
            created_at=datetime.now(UTC),
        )
        mock_retriever.store.get.return_value = doc

        toolkit = RAGToolkit(mock_retriever)
        tool = toolkit.lookup_tool()

        result_str = await tool.execute(doc_id="test_doc")
        result = json.loads(result_str)

        assert result["id"] == "test_doc"
        assert result["content"] == "Test content"
        assert result["metadata"]["key"] == "value"

    @pytest.mark.asyncio
    async def test_lookup_tool_not_found(self, mock_retriever):
        """Test lookup tool when document is not found."""
        import json

        mock_retriever.store.get.return_value = None

        toolkit = RAGToolkit(mock_retriever)
        tool = toolkit.lookup_tool()

        result_str = await tool.execute(doc_id="missing_doc")
        result = json.loads(result_str)

        assert "error" in result
        assert "not found" in result["error"]
