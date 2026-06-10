# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""Full SOC analyst agent — the complete tool-using triage workflow.

A senior-analyst agent that works a shift-handover request end to end:
queries the case database, enriches indicators against intel APIs,
crunches alert metrics, searches the runbook, and writes the handover
report — verifying each conclusion against tool output before it ships.

This example shows:
- A detailed analyst system prompt with an explicit verify step
- Multiple deterministic SOC tools (case DB, intel APIs, analysis, runbook)
- Reflexion (self-reflection) through the agent run loop
- Structured incident-report output
- Async streaming of the agent's reasoning

It runs offline against the bundled mock model by default via
``config.get_model`` and upgrades to a live provider when
``TULIP_MODEL_PROVIDER`` (plus the matching API key) is set.

All security data below (IPs, hashes, incidents) is invented and uses
documentation-safe placeholders (RFC 5737 addresses, EICAR-style test
entries).
"""

import ast
import asyncio
import json
import math
import operator as _op
from datetime import datetime
from typing import Any

from config import check_structured_output_capable, get_model
from pydantic import BaseModel, Field

from tulip.agent import Agent
from tulip.core.structured import create_schema_prompt, parse_structured
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


class IncidentReport(BaseModel):
    """Structured incident analysis report."""

    title: str = Field(description="Report title")
    summary: str = Field(description="Executive summary")
    findings: list[str] = Field(description="Key findings, each traceable to evidence")
    recommendations: list[str] = Field(description="Defensive action recommendations")
    confidence: float = Field(ge=0, le=1, description="Confidence score 0-1")
    data_sources: list[str] = Field(description="Telemetry and intel sources used")


class ContainmentPlan(BaseModel):
    """Structured containment execution plan."""

    goal: str
    steps: list[str]
    dependencies: list[str] = Field(default_factory=list)
    estimated_complexity: str = Field(description="low, medium, high")
    risks: list[str] = Field(default_factory=list)


# =============================================================================
# Simulated SOC Database
# =============================================================================

MOCK_DATABASE = {
    "analysts": [
        {"id": 1, "name": "Alice", "role": "soc_lead", "shift": "day"},
        {"id": 2, "name": "Bob", "role": "tier1_analyst", "shift": "day"},
        {"id": 3, "name": "Charlie", "role": "threat_hunter", "shift": "night"},
        {"id": 4, "name": "Diana", "role": "ir_manager", "shift": "day"},
    ],
    "incidents": [
        {"id": 1, "name": "Phishing wave vs. finance", "status": "open", "severity": "high"},
        {"id": 2, "name": "Impossible travel — jdoe", "status": "open", "severity": "medium"},
        {"id": 3, "name": "EICAR test-file detection", "status": "closed", "severity": "low"},
    ],
    "alert_metrics": [
        {"date": "2026-01", "alerts": 1250, "true_positives": 85, "mttr_hours": 6.5},
        {"date": "2026-02", "alerts": 1310, "true_positives": 92, "mttr_hours": 5.8},
        {"date": "2026-03", "alerts": 1190, "true_positives": 78, "mttr_hours": 5.1},
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
    Query the SOC case database.

    Args:
        table: Table name (analysts, incidents, alert_metrics)
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
    Call a (simulated) security intel API.

    Args:
        endpoint: API endpoint (e.g., /ip_reputation, /hash_lookup, /alert_stats)
        method: HTTP method (GET, POST, PUT, DELETE)
        body: Request body for POST/PUT

    Returns:
        API response as JSON string (all data is invented, clearly fake)
    """
    await asyncio.sleep(0.2)  # Simulate API latency

    # Simulate various intel endpoints with deterministic fake data
    if endpoint == "/ip_reputation":
        ip = body.get("ip", "198.51.100.23") if body else "198.51.100.23"
        return json.dumps(
            {
                "ip": ip,
                "reputation": "suspicious",
                "last_seen": "2026-03-02T14:11:00Z",
                "reports": ["port scanning", "brute-force attempts"],
            }
        )

    if endpoint == "/hash_lookup":
        file_hash = body.get("hash", "aa11bb22cc33dd44") if body else "aa11bb22cc33dd44"
        return json.dumps(
            {
                "hash": file_hash,
                "verdict": "known-test-file",
                "family": "EICAR test signature",
                "first_seen": "2026-01-15",
            }
        )

    if endpoint == "/alert_stats":
        return json.dumps(
            {
                "alerts_today": 412,
                "auto_closed": 358,
                "escalated": 9,
                "top_alert_types": ["phishing", "impossible travel", "test-file detection"],
            }
        )

    return json.dumps({"error": f"Unknown endpoint: {endpoint}"})


@tool
async def analyze_data(
    data: list[dict[str, Any]],
    analysis_type: str = "summary",
) -> str:
    """
    Perform statistical analysis on alert/incident data.

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
        return (
            "Trend Analysis: Alert volume steady while MTTR improved 22% over the "
            "quarter — detection tuning is working."
        )

    if analysis_type == "anomalies":
        return "Anomaly Detection: No significant anomalies detected in the alert dataset."

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
        expression: Math expression (e.g., "412 - 358", "sqrt(16)", "85 / 1250")

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
    Search the internal security runbook knowledge base.

    Args:
        query: Search query
        max_results: Maximum results to return

    Returns:
        Relevant runbook entries
    """
    await asyncio.sleep(0.1)

    # Simulated runbook knowledge base
    kb = [
        {
            "topic": "Phishing Triage",
            "content": "Quarantine the message, extract indicators, check who else received it.",
        },
        {
            "topic": "Containment",
            "content": "Isolate the host at the EDR layer before rotating credentials.",
        },
        {
            "topic": "IOC Enrichment",
            "content": "Cross-check indicators against internal telemetry before external feeds.",
        },
        {
            "topic": "Escalation",
            "content": "Escalate to IR when scope crosses one host or credentials are stolen.",
        },
        {
            "topic": "Audit Trail",
            "content": "Every agent action is recorded as a typed event for the case audit trail.",
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

COMPLEX_SYSTEM_PROMPT = """You are an advanced SOC analyst assistant with access to multiple tools and capabilities.

## Your Identity
- You are a senior security analyst with expertise in alert triage, investigation, and reporting
- You think step-by-step and explain your reasoning
- You verify claims against tool output before presenting conclusions
- You acknowledge uncertainty — an unverified finding is a false positive waiting to happen

## Available Capabilities
1. **Case Database**: Access SOC data (analysts, incidents, alert_metrics)
2. **Intel APIs**: Look up IP reputation, file hashes, and alert statistics
3. **Data Analysis**: Perform statistical analysis and trend detection on alert data
4. **Report Generation**: Create formatted reports in markdown/HTML
5. **Calculations**: Execute mathematical computations (rates, deltas, ratios)
6. **Runbook Search**: Query the internal security runbook knowledge base

## Working Process
1. **Understand**: Carefully analyze the request
2. **Plan**: Break down complex investigations into steps
3. **Execute**: Use appropriate tools for each step
4. **Verify**: Check results for accuracy and completeness
5. **Synthesize**: Combine findings into a coherent assessment
6. **Reflect**: Consider if the answer is complete and accurate

## Guidelines
- Always explain your reasoning process
- Use tools when you need factual information — never invent indicators
- Cross-reference multiple sources when possible
- Provide confidence levels for your conclusions
- Recommend defensive follow-up actions when appropriate
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
    """Run the SOC agent demonstration."""
    print("=" * 60)
    print("TULIP SOC Analyst Agent Demo")
    print("=" * 60)
    print()

    # Pick up the configured provider (mock by default; live when
    # TULIP_MODEL_PROVIDER + credentials are set).
    model = get_model(max_tokens=2048)

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
    I need a comprehensive shift-handover analysis for the SOC.

    Please:
    1. Query the database for all day-shift analysts
    2. Get the list of open incidents
    3. Analyze the alert metrics for trends
    4. Search the knowledge base for our escalation runbook
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
    """Demonstrate structured incident-report parsing.

    Constrained JSON output needs a model that can honor a schema. The
    bundled mock returns plain text, so under it this part prints guidance
    and exits cleanly rather than fabricating a report.
    """
    print("\n" + "=" * 60)
    print("Structured Output Demo")
    print("=" * 60)
    print()

    check_structured_output_capable()

    model = get_model()

    # Create prompt with schema instructions
    schema_prompt = create_schema_prompt(IncidentReport)

    agent = Agent(
        model=model,
        system_prompt=f"You are a SOC analyst. {schema_prompt}",
    )

    result = agent.run_sync(
        "Analyze the trends in phishing alert volume for enterprise SOCs in Q1 2026."
    )

    print(f"Raw response:\n{result.message[:300]}...")
    print()

    # Parse structured output
    try:
        structured = parse_structured(result.message, IncidentReport, strict=False)
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
