#!/usr/bin/env python3
# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""Interactive Secure-Coding Assistant powered by Tulip.

An application-security reviewer that hunts for vulnerabilities in a
codebase, applies fixes, and verifies them — and only reports findings
it can point to in the code. It looks for the usual suspects: injection
(CWE-89 SQL, CWE-78 OS command), hard-coded secrets (CWE-798), unsafe
deserialization (CWE-502), and path traversal (CWE-22).

Demonstrates the full interactive agent loop:
- completion_mode="explicit" — agent keeps going until task_complete
- ask_user — agent asks clarifying questions mid-execution
- verification reminders — agent reminded to test after fixing
- reflexion — agent self-assesses progress

Usage:
    python examples/coding_assistant.py "Audit /tmp/myapp for injection bugs, then fix them"

Requires:
    OPENAI_API_KEY or ANTHROPIC_API_KEY set for a live model.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from tulip.agent import Agent, ReflexionConfig
from tulip.core.events import (
    InterruptEvent,
    ReflectEvent,
    TerminateEvent,
    ThinkEvent,
    ToolCompleteEvent,
    ToolStartEvent,
)
from tulip.tools.decorator import tool


# =============================================================================
# Code-Review Tools (user-land, not SDK)
# =============================================================================


@tool
def read_file(path: str) -> str:
    """Read the contents of a file."""
    try:
        return Path(path).read_text()
    except FileNotFoundError:
        return f"Error: File '{path}' not found."
    except Exception as e:
        return f"Error reading '{path}': {e}"


@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file. Creates parent directories if needed."""
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Written {len(content)} chars to {path}"
    except Exception as e:
        return f"Error writing '{path}': {e}"


@tool
def list_directory(path: str) -> str:
    """List files and directories recursively."""
    try:
        entries = []
        for item in sorted(Path(path).rglob("*")):
            if item.is_file() and "__pycache__" not in str(item) and ".venv" not in str(item):
                entries.append(f"{item.relative_to(path)} ({item.stat().st_size}B)")
        return "\n".join(entries) if entries else "Empty or does not exist."
    except Exception as e:
        return f"Error: {e}"


@tool
def run_command(command: str, working_dir: str) -> str:
    """Run a shell command in a directory. Returns stdout+stderr."""
    import subprocess

    try:
        r = subprocess.run(  # noqa: S602 — example code; user-controlled command in their own dir
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=working_dir,
            check=False,
        )
        output = (r.stdout + r.stderr).strip()
        return output[:4000] if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 60 seconds."
    except Exception as e:  # noqa: BLE001 — example: surface any error string to the model
        return f"Error: {e}"


# =============================================================================
# Main
# =============================================================================


def get_model():
    """Build model from environment variables.

    Picks OpenAI or Anthropic from whichever API key is present. See
    ``docs/concepts/models.md``.
    """
    if os.getenv("OPENAI_API_KEY"):
        from tulip.models import OpenAIModel

        return OpenAIModel(model="gpt-4o-mini", max_tokens=4096)

    if os.getenv("ANTHROPIC_API_KEY"):
        from tulip.models.native.anthropic import AnthropicModel

        return AnthropicModel(model="claude-sonnet-4-6", max_tokens=4096)

    print("Error: Set OPENAI_API_KEY or ANTHROPIC_API_KEY")
    sys.exit(1)


async def main():
    task = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
    if not task:
        print('Usage: python examples/coding_assistant.py "<review task>"')
        print(
            "Example: python examples/coding_assistant.py "
            '"Audit /tmp/myapp for SQL injection and hard-coded secrets, then fix them"'
        )
        sys.exit(1)

    model = get_model()

    agent = Agent(
        model=model,
        tools=[read_file, write_file, list_directory, run_command],
        system_prompt=(
            "You are a senior application-security engineer doing a secure code review.\n\n"
            "Available tools:\n"
            "- list_directory(path): Map the project structure\n"
            "- read_file(path): Read file contents\n"
            "- write_file(path, content): Apply fixes to files\n"
            "- run_command(command, working_dir): Run tests / linters\n"
            "- ask_user(question, options): Ask the user a question\n"
            "- task_complete(summary, status): Signal you're done\n\n"
            "Workflow:\n"
            "1. If the review scope is ambiguous, use ask_user to clarify\n"
            "2. Map the codebase, then read the files in scope\n"
            "3. Hunt for: SQL/command injection (CWE-89, CWE-78), missing\n"
            "   authz checks, hard-coded secrets (CWE-798), unsafe\n"
            "   deserialization (CWE-502), path traversal (CWE-22)\n"
            "4. For each finding, cite the exact file and line — never report\n"
            "   a vulnerability you cannot point to in the code\n"
            "5. Apply minimal fixes with write_file, then run the test suite\n"
            "6. If tests fail, read errors, fix code, rerun\n"
            "7. Only call task_complete after fixes are verified\n\n"
            "In your summary, list each finding with severity (low/medium/high/critical).\n"
            "Always verify your fixes work before completing.\n"
            "Use python3 for running commands."
        ),
        completion_mode="explicit",
        reflexion=ReflexionConfig(enabled=True, include_guidance=True),
        max_iterations=20,
        max_tool_result_length=4000,
        time_budget_seconds=300,
    )

    print(f"\n{'=' * 60}")
    print(f"  TULIP SECURE-CODING ASSISTANT")
    print(f"{'=' * 60}")
    print(f"  Task: {task}")
    print(f"  Mode: explicit (agent runs until task_complete)")
    print(f"  Max iterations: 20 | Time budget: 5min")
    print(f"{'=' * 60}\n")

    events_iter = agent.run(task)

    while True:
        try:
            event = await events_iter.__anext__()
        except StopAsyncIteration:
            break

        if isinstance(event, ThinkEvent):
            reasoning = event.reasoning or ""
            lines = [ln for ln in reasoning.split("\n") if ln.strip()][:3]
            print(f"\n💭 Thinking...")
            for line in lines:
                print(f"   {line.strip()[:80]}")
            if event.tool_calls:
                print(f"   → Calling {len(event.tool_calls)} tool(s):")
                for tc in event.tool_calls:
                    if tc.name == "write_file":
                        path = tc.arguments.get("path", "?")
                        print(f"     ✏️  fix {path}")
                    elif tc.name == "run_command":
                        cmd = tc.arguments.get("command", "?")[:50]
                        print(f"     ⚡ run: {cmd}")
                    elif tc.name == "ask_user":
                        print(f"     ❓ asking user...")
                    elif tc.name == "task_complete":
                        print(f"     ✅ signaling done")
                    else:
                        print(f"     🔧 {tc.name}")

        elif isinstance(event, ToolCompleteEvent):
            if event.error:
                print(f"     ✗ {event.tool_name}: {event.error[:60]}")
            else:
                preview = (event.result or "")[:60].replace("\n", " ")
                print(f"     ✓ {event.tool_name} → {preview}")

        elif isinstance(event, ReflectEvent):
            emoji = {"on_track": "📊", "new_findings": "🔍", "stuck": "⚠️", "loop_detected": "🔄"}
            e = emoji.get(event.assessment, "📊")
            print(
                f"\n   {e} Reflection: {event.assessment} (confidence: {event.new_confidence:.0%})"
            )

        elif isinstance(event, InterruptEvent):
            print(f"\n{'=' * 60}")
            print(f"  ❓ AGENT ASKS: {event.question}")
            if event.options:
                print(f"     Options: {', '.join(event.options)}")
            print(f"{'=' * 60}")
            answer = input("  Your answer: ").strip()  # noqa: ASYNC250 — interactive demo, blocking is intentional
            print()

            # Resume with user's answer
            events_iter = agent.resume(answer)
            continue

        elif isinstance(event, TerminateEvent):
            print(f"\n{'=' * 60}")
            reason_emoji = {
                "terminal_tool": "✅",
                "complete": "✅",
                "max_iterations": "⏰",
                "time_budget": "⏱️",
                "token_budget": "💰",
                "error": "❌",
            }
            e = reason_emoji.get(event.reason, "🏁")
            print(f"  {e} Done: {event.reason}")
            print(f"     Iterations: {event.iterations_used}")
            print(f"     Tool calls: {event.total_tool_calls}")
            if event.final_message:
                print(f"\n  Final report:")
                for line in event.final_message.split("\n")[:10]:
                    print(f"     {line}")
            print(f"{'=' * 60}")
            break


if __name__ == "__main__":
    asyncio.run(main())
