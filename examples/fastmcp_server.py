# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""Simple FastMCP server for integration testing."""

import json
from datetime import datetime

from fastmcp import FastMCP


mcp = FastMCP("locus-test-server")


@mcp.tool()
def get_current_time() -> str:
    """Get the current time."""
    return datetime.now().isoformat()


@mcp.tool()
def add_numbers(a: int, b: int) -> str:
    """Add two numbers together."""
    return str(a + b)


@mcp.tool()
def search_database(query: str, limit: int = 10) -> str:
    """Search a mock database."""
    results = [
        {"id": 1, "name": "Alice", "score": 95},
        {"id": 2, "name": "Bob", "score": 87},
        {"id": 3, "name": "Charlie", "score": 92},
    ]
    filtered = [r for r in results if query.lower() in r["name"].lower()]
    return json.dumps(filtered[:limit])


@mcp.tool()
def analyze_text(text: str) -> str:
    """Analyze text and return stats."""
    words = text.split()
    return json.dumps(
        {
            "word_count": len(words),
            "char_count": len(text),
            "avg_word_length": sum(len(w) for w in words) / len(words) if words else 0,
        }
    )


if __name__ == "__main__":
    mcp.run()
