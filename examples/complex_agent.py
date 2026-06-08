# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL
"""Complex Agent with FastMCP-style tools - Full demonstration.

This example shows:
- Complex system prompt
- Multiple sophisticated tools
- Reflexion (self-reflection)
- Structured outputs
- Async streaming
"""

import ast
import asyncio
import json
import math
import operator as _op
import os
import random
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from tulip.agent import Agent
from tulip.core.structured import create_schema_prompt, parse_structured
from tulip.models import OCIChatCompletionsModel
from tulip.tools import tool


_SAFE_MATH_BIN_OPS = {
    ast.Add: _op.add,
    ast.Sub: _op.sub,
    ast.Mult: _op.mul,
    ast.Div: _op.truediv,
    ast.FloorDiv: _op.floordiv,
    ast.Mod: _op.mod,
    ast.Pow: _op.pow,
}
_SAFE_MATH_UNARY_OPS = {ast.USub: _op.neg, ast.UAdd: _op.pos}

# Allowed function calls + constants inside `execute_calculation` expressions.
_SAFE_MATH_FUNCTIONS: dict[str, Any] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "pow": pow,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
}
_SAFE_MATH_NAMES: dict[str, float] = {"pi": math.pi, "e": math.e}


def _safe_math_eval(
    expression: str,
    *,
    functions: dict[str, Any] | None = None,
    names: dict[str, float] | None = None,
) -> float:
    """AST-based arithmetic evaluator with optional function/constant whitelist.

    Disallows attribute access, imports, subscripts, comprehensions, and any
    callable not in `functions`. Safe to run on LLM-generated expressions.
    """
    functions = functions or {}
    names = names or {}
    tree = ast.parse(expression, mode="eval")

    def _eval(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in names:
                return names[node.id]
            raise ValueError(f"Name not allowed: {node.id!r}")
        if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_MATH_BIN_OPS:
            return _SAFE_MATH_BIN_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_MATH_UNARY_OPS:
            return _SAFE_MATH_UNARY_OPS[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            fn = functions.get(node.func.id)
            if fn is None:
                raise ValueError(f"Function not allowed: {node.func.id!r}")
            return fn(*[_eval(arg) for arg in node.args])
        raise ValueError("Unsupported expression")

    return _eval(tree)


# =============================================================================
# Structured Output Schemas
# =============================================================================


class AnalysisReport(BaseModel):
    """Structured analysis report."""

    title: str = Field(description="Report title")
    summary: str = Field(description="Executive summary")
    findings: list[str] = Field(description="Key findings")
    recommendations: list[str] = Field(description="Action recommendations")
    confidence: float = Field(ge=0, le=1, description="Confidence score 0-1")
    data_sources: list[str] = Field(description="Sources used")


class TaskPlan(BaseModel):
    """Structured task execution plan."""

    goal: str
    steps: list[str]
    dependencies: list[str] = Field(default_factory=list)
    estimated_complexity: str = Field(description="low, medium, high")
    risks: list[str] = Field(default_factory=list)


# =============================================================================
# Simulated Database
# =============================================================================

MOCK_DATABASE = {
    "users": [
        {"id": 1, "name": "Alice", "role": "admin", "department": "Engineering"},
        {"id": 2, "name": "Bob", "role": "developer", "department": "Engineering"},
        {"id": 3, "name": "Charlie", "role": "analyst", "department": "Data"},
        {"id": 4, "name": "Diana", "role": "manager", "department": "Product"},
    ],
    "projects": [
        {"id": 1, "name": "Tulip SDK", "status": "active", "budget": 150000},
        {"id": 2, "name": "Data Pipeline", "status": "active", "budget": 80000},
        {"id": 3, "name": "ML Platform", "status": "planning", "budget": 200000},
    ],
    "metrics": [
        {"date": "2024-01", "revenue": 50000, "users": 1200, "churn": 0.05},
        {"date": "2024-02", "revenue": 55000, "users": 1350, "churn": 0.04},
        {"date": "2024-03", "revenue": 62000, "users": 1500, "churn": 0.03},
    ],
}


# =============================================================================
# Complex Tools
# =============================================================================


@tool
async def query_database(
    table: str,
    filters: dict[str, Any] | None = None,
    limit: int = 10,
) -> str:
    """
    Query the internal database.

    Args:
        table: Table name (users, projects, metrics)
        filters: Optional filters as key-value pairs
        limit: Maximum results to return

    Returns:
        JSON string of matching records
    """
    await asyncio.sleep(0.1)  # Simulate query time

    if table not in MOCK_DATABASE:
        return json.dumps({"error": f"Table '{table}' not found"})

    data = MOCK_DATABASE[table]

    # Apply filters
    if filters:
        filtered = []
        for record in data:
            match = all(record.get(k) == v for k, v in filters.items())
            if match:
                filtered.append(record)
        data = filtered

    return json.dumps(data[:limit], indent=2)


@tool
async def call_external_api(
    endpoint: str,
    method: str = "GET",
    body: dict[str, Any] | None = None,
) -> str:
    """
    Make an external API call.

    Args:
        endpoint: API endpoint (e.g., /users, /analytics)
        method: HTTP method (GET, POST, PUT, DELETE)
        body: Request body for POST/PUT

    Returns:
        API response as JSON string
    """
    await asyncio.sleep(0.2)  # Simulate API latency

    # Simulate various API endpoints
    if endpoint == "/weather":
        return json.dumps(
            {
                "location": "San Francisco",
                "temperature": 68,
                "conditions": "Partly cloudy",
                "forecast": ["Sunny tomorrow", "Rain expected Thursday"],
            }
        )

    if endpoint == "/stock":
        symbol = body.get("symbol", "AAPL") if body else "AAPL"
        return json.dumps(
            {
                "symbol": symbol,
                "price": round(random.uniform(100, 200), 2),
                "change": round(random.uniform(-5, 5), 2),
                "volume": random.randint(1000000, 5000000),
            }
        )

    if endpoint == "/analytics":
        return json.dumps(
            {
                "daily_active_users": 12500,
                "session_duration_avg": 8.5,
                "conversion_rate": 0.032,
                "top_features": ["dashboard", "reports", "exports"],
            }
        )

    return json.dumps({"error": f"Unknown endpoint: {endpoint}"})


@tool
async def analyze_data(
    data: list[dict[str, Any]],
    analysis_type: str = "summary",
) -> str:
    """
    Perform statistical analysis on data.

    Args:
        data: List of records to analyze
        analysis_type: Type of analysis (summary, trends, anomalies)

    Returns:
        Analysis results as formatted string
    """
    await asyncio.sleep(0.15)

    if not data:
        return "Error: No data provided for analysis"

    if analysis_type == "summary":
        # Calculate basic stats
        numeric_keys = []
        for key in data[0]:
            if isinstance(data[0][key], (int, float)):
                numeric_keys.append(key)

        stats = {}
        for key in numeric_keys:
            values = [r[key] for r in data if key in r]
            stats[key] = {
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "count": len(values),
            }

        return f"Summary Statistics:\n{json.dumps(stats, indent=2)}"

    if analysis_type == "trends":
        return "Trend Analysis: Upward trend detected in key metrics. Growth rate: +15% MoM"

    if analysis_type == "anomalies":
        return "Anomaly Detection: No significant anomalies detected in the dataset."

    return f"Unknown analysis type: {analysis_type}"


@tool
async def generate_report(
    title: str,
    sections: list[str],
    format: str = "markdown",
) -> str:
    """
    Generate a formatted report.

    Args:
        title: Report title
        sections: List of section contents
        format: Output format (markdown, html, text)

    Returns:
        Formatted report content
    """
    await asyncio.sleep(0.1)

    if format == "markdown":
        lines = [f"# {title}", "", f"*Generated: {datetime.now().isoformat()}*", ""]
        for i, section in enumerate(sections, 1):
            lines.extend([f"## Section {i}", "", section, ""])
        return "\n".join(lines)

    if format == "html":
        sections_html = "".join(f"<section><p>{s}</p></section>" for s in sections)
        return f"<html><head><title>{title}</title></head><body><h1>{title}</h1>{sections_html}</body></html>"

    return f"{title}\n{'=' * len(title)}\n\n" + "\n\n".join(sections)


@tool
async def execute_calculation(expression: str) -> str:
    """
    Safely evaluate a mathematical expression.

    Args:
        expression: Math expression (e.g., "2 + 2", "sqrt(16)", "100 * 0.15")

    Returns:
        Calculation result
    """
    try:
        result = _safe_math_eval(
            expression,
            functions=_SAFE_MATH_FUNCTIONS,
            names=_SAFE_MATH_NAMES,
        )
        return f"Result: {result}"
    except (ValueError, SyntaxError, ZeroDivisionError, ArithmeticError) as e:
        return f"Calculation error: {e}"


@tool
async def search_knowledge_base(
    query: str,
    max_results: int = 5,
) -> str:
    """
    Search the internal knowledge base.

    Args:
        query: Search query
        max_results: Maximum results to return

    Returns:
        Relevant knowledge base entries
    """
    await asyncio.sleep(0.1)

    # Simulated knowledge base
    kb = [
        {
            "topic": "Agent Architecture",
            "content": "Agents use ReAct loop with optional Reflexion for self-correction.",
        },
        {
            "topic": "Tool Calling",
            "content": "Tools are defined with @tool decorator and auto-generate JSON schemas.",
        },
        {
            "topic": "Streaming",
            "content": "Events are streamed via AsyncIterator for real-time updates.",
        },
        {
            "topic": "Multi-Agent",
            "content": "Swarm pattern allows multiple agents to collaborate on tasks.",
        },
        {
            "topic": "Checkpointing",
            "content": "State can be persisted to Redis, PostgreSQL, MySQL, OpenSearch, S3 object storage, or custom backends.",
        },
    ]

    # Simple keyword matching
    query_lower = query.lower()
    matches = [
        entry
        for entry in kb
        if query_lower in entry["topic"].lower() or query_lower in entry["content"].lower()
    ]

    if not matches:
        matches = kb[:max_results]  # Return some defaults

    return json.dumps(matches[:max_results], indent=2)


# =============================================================================
# Complex System Prompt
# =============================================================================

COMPLEX_SYSTEM_PROMPT = """You are an advanced AI assistant with access to multiple tools and capabilities.

## Your Identity
- You are a senior analyst with expertise in data analysis, research, and report generation
- You think step-by-step and explain your reasoning
- You verify information before presenting conclusions
- You acknowledge uncertainty and limitations

## Available Capabilities
1. **Database Queries**: Access internal databases (users, projects, metrics)
2. **API Integration**: Call external APIs for real-time data
3. **Data Analysis**: Perform statistical analysis and trend detection
4. **Report Generation**: Create formatted reports in markdown/HTML
5. **Calculations**: Execute mathematical computations
6. **Knowledge Search**: Query the internal knowledge base

## Working Process
1. **Understand**: Carefully analyze the user's request
2. **Plan**: Break down complex tasks into steps
3. **Execute**: Use appropriate tools for each step
4. **Verify**: Check results for accuracy and completeness
5. **Synthesize**: Combine findings into coherent response
6. **Reflect**: Consider if the answer is complete and accurate

## Guidelines
- Always explain your reasoning process
- Use tools when you need factual information
- Cross-reference multiple sources when possible
- Provide confidence levels for your conclusions
- Suggest follow-up actions when appropriate
- Format responses for clarity (use headers, lists, etc.)

## Response Format
Structure your responses with:
1. Brief acknowledgment of the request
2. Your analysis/reasoning
3. Tool-derived findings (if applicable)
4. Conclusions and recommendations
5. Confidence assessment"""


# =============================================================================
# Main Execution
# =============================================================================


async def run_complex_agent():
    """Run the complex agent demonstration."""
    print("=" * 60)
    print("TULIP Complex Agent Demo")
    print("=" * 60)
    print()

    # Create model
    model = OCIChatCompletionsModel(
        model="openai.gpt-5",
        profile=os.environ.get("OCI_PROFILE", "DEFAULT"),
        region=os.environ.get("OCI_REGION", "us-chicago-1"),
        max_tokens=2048,
    )

    # Create agent with all tools
    agent = Agent(
        model=model,
        tools=[
            query_database,
            call_external_api,
            analyze_data,
            generate_report,
            execute_calculation,
            search_knowledge_base,
        ],
        system_prompt=COMPLEX_SYSTEM_PROMPT,
        max_iterations=10,
    )

    # Complex multi-step task
    task = """
    I need a comprehensive analysis of our engineering team and projects.

    Please:
    1. Query the database for all engineering team members
    2. Get the list of active projects
    3. Analyze the project budgets
    4. Search the knowledge base for information about our architecture
    5. Generate a brief executive report with your findings

    Include specific numbers and actionable recommendations.
    """

    print(f"Task: {task.strip()}")
    print()
    print("-" * 60)
    print("Agent Execution:")
    print("-" * 60)
    print()

    # Stream the execution
    async for event in agent.run(task):
        if event.event_type == "think":
            print(f"💭 THINKING (iter {event.iteration}):")
            if event.reasoning:
                # Truncate long reasoning for display
                reasoning = (
                    event.reasoning[:500] + "..." if len(event.reasoning) > 500 else event.reasoning
                )
                print(f"   {reasoning}")
            if event.tool_calls:
                for tc in event.tool_calls:
                    print(f"   🔧 Calling: {tc.name}({json.dumps(tc.arguments)[:100]}...)")
            print()

        elif event.event_type == "tool_start":
            print(f"⚙️  TOOL START: {event.tool_name}")

        elif event.event_type == "tool_complete":
            status = "✅" if event.success else "❌"
            result_preview = (event.result or "")[:100]
            print(f"{status} TOOL COMPLETE: {event.tool_name}")
            if result_preview:
                print(f"   Result: {result_preview}...")
            print()

        elif event.event_type == "terminate":
            print("-" * 60)
            print(f"🏁 COMPLETED: {event.reason}")
            print(f"   Iterations: {event.iterations_used}")
            print(f"   Tool calls: {event.total_tool_calls}")
            if event.final_message:
                print()
                print("📋 FINAL RESPONSE:")
                print("-" * 40)
                print(event.final_message)


async def run_structured_output_demo():
    """Demonstrate structured output parsing."""
    print("\n" + "=" * 60)
    print("Structured Output Demo")
    print("=" * 60)
    print()

    model = OCIChatCompletionsModel(
        model="openai.gpt-5",
        profile=os.environ.get("OCI_PROFILE", "DEFAULT"),
        region=os.environ.get("OCI_REGION", "us-chicago-1"),
    )

    # Create prompt with schema instructions
    schema_prompt = create_schema_prompt(AnalysisReport)

    agent = Agent(
        model=model,
        system_prompt=f"You are a data analyst. {schema_prompt}",
    )

    result = agent.run_sync("Analyze the trends in AI adoption for enterprise companies in 2024.")

    print(f"Raw response:\n{result.message[:300]}...")
    print()

    # Parse structured output
    try:
        structured = parse_structured(result.message, AnalysisReport, strict=False)
        if structured.success:
            report = structured.parsed
            print("✅ Parsed successfully!")
            print(f"   Title: {report.title}")
            print(f"   Confidence: {report.confidence}")
            print(f"   Findings: {len(report.findings)} items")
        else:
            print(f"⚠️ Parse warning: {structured.error}")
    except Exception as e:
        print(f"❌ Parse error: {e}")


if __name__ == "__main__":
    asyncio.run(run_complex_agent())
    asyncio.run(run_structured_output_demo())
