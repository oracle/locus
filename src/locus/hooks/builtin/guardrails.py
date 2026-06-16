# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Guardrails hook provider for input/output filtering and safety checks."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from locus.hooks.provider import (
    AfterToolCallEvent,
    BeforeToolCallEvent,
    HookPriority,
    HookProvider,
)


if TYPE_CHECKING:
    from locus.core.state import AgentState


class GuardrailAction(Enum):
    """Action to take when a guardrail is triggered."""

    BLOCK = "block"  # Block the request entirely
    WARN = "warn"  # Log warning but allow
    REDACT = "redact"  # Redact the sensitive content
    ALLOW = "allow"  # Allow without modification


@dataclass
class GuardrailViolation:
    """Record of a guardrail violation."""

    rule_name: str
    description: str
    action: GuardrailAction
    matched_content: str | None = None
    location: str | None = None  # "input", "output", "tool_args", "tool_result"


@dataclass
class GuardrailConfig:
    """Configuration for guardrails.

    Attributes:
        block_dangerous_tools: Tools that should never be called
        allow_only_tools: If set, only these tools are allowed
        pii_patterns: Regex patterns for PII detection
        blocked_content_patterns: Patterns that should block content
        max_prompt_length: Maximum allowed prompt length
        max_tool_result_length: Maximum tool result length
        default_action: Default action for violations
    """

    block_dangerous_tools: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                "eval",
                "exec",
                "system",
                "shell",
                "rm",
                "delete",
                "drop",
                "truncate",
            }
        )
    )
    allow_only_tools: frozenset[str] | None = None

    # PII patterns (basic examples - production should use more comprehensive patterns)
    pii_patterns: dict[str, str] = field(
        default_factory=lambda: {
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "phone_us": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
            "ip_address": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
        }
    )

    # Content patterns to block.
    # Patterns use \S+ (not .+) to avoid catastrophic backtracking (ReDoS) on
    # crafted inputs; overlapping greedy quantifiers with . (which matches
    # whitespace) create exponential worst-case behavior.
    blocked_content_patterns: dict[str, str] = field(
        default_factory=lambda: {
            "sql_injection": (
                r"(?i)"
                r"(DROP\s+TABLE(\s+IF\s+EXISTS)?)"
                r"|(DELETE\s+FROM)"
                r"|(TRUNCATE\s+TABLE)"
                r"|(INSERT\s+INTO\s+\S+\s+VALUES)"
                r"|(UPDATE\s+\S+\s+SET\s+\S+\s+WHERE)"
                r"|(UNION\s+(ALL\s+)?SELECT)"
                r"|(SELECT\s+\S+\s+FROM\s+\S+\s+WHERE\s+\S+\s*(OR|AND)\s+['\"]?\d+['\"]?\s*=\s*['\"]?\d+)"
                r"|(\b(ALTER|CREATE|RENAME)\s+TABLE)"
                r"|(--\s)"
                r"|(/\*[^*]*\*/)"
                r"|(;\s*(DROP|DELETE|TRUNCATE|ALTER|CREATE|INSERT|UPDATE|EXEC))"
            ),
            "path_traversal": (
                r"\.\./|\.\.\\|"
                r"%2e%2e[/\\%]|"
                r"%252e%252e|"
                r"\.%2e[/\\]|%2e\.[/\\]"
            ),
            "command_injection": (
                r"[;&|`]|"
                r"\$\(|"
                r"\$\{|"
                r"\n\s*(cat|ls|rm|wget|curl|bash|sh|python|perl|ruby|nc|ncat)\b|"
                r">\s*/|"
                r"\|\s*(bash|sh|zsh|cmd)"
            ),
        }
    )

    max_prompt_length: int = 100000  # 100k characters
    max_tool_result_length: int = 50000  # 50k characters

    default_action: GuardrailAction = GuardrailAction.BLOCK

    # Action overrides per pattern type
    action_overrides: dict[str, GuardrailAction] = field(default_factory=dict)


class GuardrailsHook(HookProvider):
    """Hook provider for security guardrails.

    Provides:
    - Input validation and filtering
    - Output sanitization
    - PII detection and redaction
    - Dangerous content blocking
    - Tool allowlist/blocklist enforcement

    Example:
        config = GuardrailConfig(
            block_dangerous_tools=frozenset({"shell", "exec"}),
            default_action=GuardrailAction.BLOCK,
        )
        registry.add_provider(GuardrailsHook(config))
    """

    # Upper bound on bytes scanned by regex-based blocked-content patterns.
    # Bounds worst-case regex runtime to protect against ReDoS.
    _REGEX_SCAN_LIMIT: int = 8 * 1024

    def __init__(
        self,
        config: GuardrailConfig | None = None,
        on_violation: Callable[[GuardrailViolation], None] | None = None,
        priority: int = HookPriority.SECURITY_DEFAULT,
    ) -> None:
        """Initialize guardrails hook.

        Args:
            config: Guardrail configuration
            on_violation: Callback for violations (receives GuardrailViolation)
            priority: Hook priority (default: middle of security range)
        """
        self._config = config or GuardrailConfig()
        self._on_violation = on_violation
        self._priority = priority
        self._violations: list[GuardrailViolation] = []

        # Compile patterns for efficiency
        self._compiled_pii: dict[str, re.Pattern[str]] = {
            name: re.compile(pattern) for name, pattern in self._config.pii_patterns.items()
        }
        self._compiled_blocked: dict[str, re.Pattern[str]] = {
            name: re.compile(pattern)
            for name, pattern in self._config.blocked_content_patterns.items()
        }

    @property
    def priority(self) -> int:
        """Return hook priority."""
        return self._priority

    @property
    def name(self) -> str:
        """Return hook name."""
        return "GuardrailsHook"

    @property
    def violations(self) -> list[GuardrailViolation]:
        """Get recorded violations."""
        return list(self._violations)

    def clear_violations(self) -> None:
        """Clear recorded violations."""
        self._violations.clear()

    def _get_action(self, rule_name: str) -> GuardrailAction:
        """Get action for a rule.

        Args:
            rule_name: Name of the rule

        Returns:
            Action to take
        """
        return self._config.action_overrides.get(rule_name, self._config.default_action)

    def _record_violation(self, violation: GuardrailViolation) -> None:
        """Record a violation.

        Args:
            violation: The violation to record
        """
        self._violations.append(violation)
        if self._on_violation:
            self._on_violation(violation)
        from locus.observability.emit import (  # noqa: PLC0415
            EV_HOOK_GUARDRAIL_TRIGGERED,
            emit_sync,
        )

        emit_sync(
            EV_HOOK_GUARDRAIL_TRIGGERED,
            rule_name=violation.rule_name,
            action=str(violation.action),
            location=violation.location,
            description=violation.description,
        )

    def _check_pii(self, text: str, location: str) -> list[GuardrailViolation]:
        """Check text for PII patterns.

        Args:
            text: Text to check
            location: Where the text came from

        Returns:
            List of violations found
        """
        violations = []
        for name, pattern in self._compiled_pii.items():
            matches = pattern.findall(text)
            if matches:
                action = self._get_action(f"pii_{name}")
                violation = GuardrailViolation(
                    rule_name=f"pii_{name}",
                    description=f"Detected {name} PII pattern",
                    action=action,
                    matched_content=matches[0] if len(matches) == 1 else f"{len(matches)} matches",
                    location=location,
                )
                violations.append(violation)
                self._record_violation(violation)
        return violations

    def _check_blocked_content(self, text: str, location: str) -> list[GuardrailViolation]:
        """Check text for blocked content patterns.

        Args:
            text: Text to check
            location: Where the text came from

        Returns:
            List of violations found
        """
        violations = []
        # Cap regex input to bound worst-case runtime (ReDoS defense-in-depth).
        # Any SQL/path/command-injection signature fits comfortably inside 8 KiB.
        scan_text = text[: self._REGEX_SCAN_LIMIT]
        for name, pattern in self._compiled_blocked.items():
            if pattern.search(scan_text):
                action = self._get_action(f"blocked_{name}")
                violation = GuardrailViolation(
                    rule_name=f"blocked_{name}",
                    description=f"Detected blocked pattern: {name}",
                    action=action,
                    matched_content=None,  # Don't expose matched content for security
                    location=location,
                )
                violations.append(violation)
                self._record_violation(violation)
        return violations

    def _redact_pii(self, text: str) -> str:
        """Redact PII from text.

        Args:
            text: Text to redact

        Returns:
            Text with PII redacted
        """
        result = text
        for name, pattern in self._compiled_pii.items():
            result = pattern.sub(f"[REDACTED_{name.upper()}]", result)
        return result

    def _should_block(self, violations: list[GuardrailViolation]) -> bool:
        """Check if any violation requires blocking.

        Args:
            violations: List of violations

        Returns:
            True if request should be blocked
        """
        return any(v.action == GuardrailAction.BLOCK for v in violations)

    async def on_before_invocation(
        self,
        prompt: str,
        state: AgentState,
    ) -> AgentState:
        """Validate input prompt.

        Args:
            prompt: User prompt
            state: Agent state

        Returns:
            State, potentially with metadata about violations

        Raises:
            ValueError: If prompt is blocked
        """
        violations: list[GuardrailViolation] = []

        # Check prompt length
        if len(prompt) > self._config.max_prompt_length:
            violation = GuardrailViolation(
                rule_name="max_prompt_length",
                description=f"Prompt exceeds maximum length ({len(prompt)} > {self._config.max_prompt_length})",
                action=self._get_action("max_prompt_length"),
                location="input",
            )
            violations.append(violation)
            self._record_violation(violation)

        # Check for blocked content
        violations.extend(self._check_blocked_content(prompt, "input"))

        # Check for PII
        pii_violations = self._check_pii(prompt, "input")
        violations.extend(pii_violations)

        # Handle blocking
        if self._should_block(violations):
            msg = f"Input blocked by guardrails: {violations[0].description}"
            raise ValueError(msg)

        # Store violations in metadata
        if violations:
            state = state.with_metadata(
                "guardrail_violations",
                [
                    {
                        "rule_name": v.rule_name,
                        "description": v.description,
                        "action": v.action.value,
                        "location": v.location,
                    }
                    for v in violations
                ],
            )

        return state

    async def on_before_tool_call(self, event: BeforeToolCallEvent) -> None:
        """Validate tool call.

        Args:
            event: Write-protected event. The hook may mutate
                ``event.arguments`` (PII redaction) or set
                ``event.cancel`` to short-circuit a blocked tool.

        Raises:
            ValueError: If tool is blocked
        """
        tool_name = event.tool_name
        arguments = event.arguments

        # Check tool blocklist
        if tool_name in self._config.block_dangerous_tools:
            violation = GuardrailViolation(
                rule_name="blocked_tool",
                description=f"Tool '{tool_name}' is blocked",
                action=GuardrailAction.BLOCK,
                location="tool_args",
            )
            self._record_violation(violation)
            msg = f"Tool '{tool_name}' is blocked by guardrails"
            raise ValueError(msg)

        # Check tool allowlist
        if (
            self._config.allow_only_tools is not None
            and tool_name not in self._config.allow_only_tools
        ):
            violation = GuardrailViolation(
                rule_name="tool_not_allowed",
                description=f"Tool '{tool_name}' is not in allowlist",
                action=GuardrailAction.BLOCK,
                location="tool_args",
            )
            self._record_violation(violation)
            msg = f"Tool '{tool_name}' is not allowed"
            raise ValueError(msg)

        # Check arguments for dangerous content
        args_str = str(arguments)
        violations = self._check_blocked_content(args_str, "tool_args")

        if self._should_block(violations):
            msg = f"Tool arguments blocked: {violations[0].description}"
            raise ValueError(msg)

        # Check for and optionally redact PII in arguments
        pii_violations = self._check_pii(args_str, "tool_args")
        if pii_violations and any(v.action == GuardrailAction.REDACT for v in pii_violations):
            # Redact PII from string arguments — write back to the event
            # so downstream hooks and the executor see the redacted form.
            redacted_args: dict[str, Any] = {}
            for key, value in arguments.items():
                if isinstance(value, str):
                    redacted_args[key] = self._redact_pii(value)
                else:
                    redacted_args[key] = value
            event.arguments = redacted_args

    async def on_after_tool_call(self, event: AfterToolCallEvent) -> None:
        """Validate tool result.

        Args:
            event: Write-protected event carrying ``tool_name``,
                ``result``, and ``error``.
        """
        result = event.result
        if result is None:
            return

        result_str = str(result)

        # Check result length
        if len(result_str) > self._config.max_tool_result_length:
            violation = GuardrailViolation(
                rule_name="max_tool_result_length",
                description=(
                    f"Tool result exceeds maximum length "
                    f"({len(result_str)} > {self._config.max_tool_result_length})"
                ),
                action=self._get_action("max_tool_result_length"),
                location="tool_result",
            )
            self._record_violation(violation)

        # Check for PII in results
        self._check_pii(result_str, "tool_result")


@dataclass
class TopicPolicy:
    """Policy for blocking specific conversation topics.

    Example:
        policy = TopicPolicy(
            blocked_topics={"weapons", "drugs", "hacking"},
            keywords={"weapons": ["gun", "rifle", "ammunition", "firearm"],
                      "drugs": ["cocaine", "heroin", "meth"],
                      "hacking": ["exploit", "zero-day", "rootkit"]},
        )
    """

    blocked_topics: set[str] = field(default_factory=set)
    keywords: dict[str, list[str]] = field(default_factory=dict)

    def check(self, text: str) -> str | None:
        """Check text against topic policies. Returns topic name if blocked."""
        text_lower = text.lower()
        for topic in self.blocked_topics:
            topic_keywords = self.keywords.get(topic, [topic])
            for keyword in topic_keywords:
                if keyword.lower() in text_lower:
                    return topic
        return None


@dataclass
class ContentPolicy:
    """Policy for content safety categories.

    Detects harmful content categories using keyword patterns.
    For production, integrate with a proper content moderation API.

    Example:
        policy = ContentPolicy(enabled_categories={"hate_speech", "self_harm"})
    """

    enabled_categories: set[str] = field(
        default_factory=lambda: {"hate_speech", "violence", "self_harm", "illegal_activity"}
    )

    # Keyword patterns per category (basic detection — not a replacement for ML classifiers)
    _patterns: dict[str, list[str]] = field(
        default_factory=lambda: {
            "hate_speech": [
                "kill all",
                "exterminate",
                "inferior race",
                "ethnic cleansing",
            ],
            "violence": [
                "how to make a bomb",
                "how to poison",
                "how to murder",
                "instructions to harm",
                "ways to hurt",
            ],
            "self_harm": [
                "how to kill myself",
                "suicide methods",
                "self-harm techniques",
                "ways to end my life",
            ],
            "illegal_activity": [
                "how to hack into",
                "how to steal",
                "money laundering",
                "counterfeit",
                "forge documents",
            ],
        }
    )

    def check(self, text: str) -> str | None:
        """Check text for harmful content. Returns category if detected."""
        text_lower = text.lower()
        for category in self.enabled_categories:
            for keyword in self._patterns.get(category, []):
                if keyword in text_lower:
                    return category
        return None


class OutputFilterHook(HookProvider):
    """Filter agent output for safety.

    Scans agent responses for PII, blocked content, and policy violations.
    Redacts or blocks unsafe output before it reaches the user.

    Example:
        hook = OutputFilterHook(
            redact_pii=True,
            content_policy=ContentPolicy(),
        )
    """

    def __init__(
        self,
        redact_pii: bool = True,
        pii_patterns: dict[str, str] | None = None,
        content_policy: ContentPolicy | None = None,
        topic_policy: TopicPolicy | None = None,
        priority: int = HookPriority.SECURITY_DEFAULT + 5,
    ) -> None:
        self._redact_pii = redact_pii
        self._pii_patterns = {
            name: re.compile(pattern)
            for name, pattern in (
                pii_patterns
                or {
                    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                    "phone_us": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
                    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
                    "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
                }
            ).items()
        }
        self._content_policy = content_policy
        self._topic_policy = topic_policy
        self._priority = priority
        self.violations: list[str] = []

    @property
    def priority(self) -> int:
        return self._priority

    @property
    def name(self) -> str:
        return "OutputFilterHook"

    async def on_after_model_call(self, event: Any) -> None:
        """Filter model response for safety."""
        content = event.response.message.content or ""
        if not content:
            return

        # Check content policy
        if self._content_policy:
            category = self._content_policy.check(content)
            if category:
                self.violations.append(f"content_policy:{category}")
                # Replace unsafe content
                from locus.core.messages import Message
                from locus.models.base import ModelResponse

                event.response = ModelResponse(
                    message=Message.assistant(
                        f"I can't provide that information as it relates to {category.replace('_', ' ')}."
                    ),
                )
                return

        # Check topic policy
        if self._topic_policy:
            topic = self._topic_policy.check(content)
            if topic:
                self.violations.append(f"topic_policy:{topic}")
                from locus.core.messages import Message
                from locus.models.base import ModelResponse

                event.response = ModelResponse(
                    message=Message.assistant(
                        f"I can't discuss {topic} as it's outside my allowed topics."
                    ),
                )
                return

        # Redact PII from output
        if self._redact_pii:
            redacted = content
            for pii_name, pattern in self._pii_patterns.items():
                redacted = pattern.sub(f"[REDACTED_{pii_name.upper()}]", redacted)
            if redacted != content:
                self.violations.append("pii_redacted")
                from locus.core.messages import Message
                from locus.models.base import ModelResponse

                event.response = ModelResponse(
                    message=Message.assistant(redacted),
                    usage=event.response.usage if hasattr(event.response, "usage") else {},
                )


class ContentFilterHook(HookProvider):
    """Simplified content filter for common use cases.

    Provides basic input/output content filtering without the full
    guardrails configuration. Useful for quick safety checks.

    Example:
        registry.add_provider(ContentFilterHook(
            blocked_words=["password", "secret"],
            max_input_length=10000,
        ))
    """

    def __init__(
        self,
        blocked_words: list[str] | None = None,
        blocked_patterns: list[str] | None = None,
        max_input_length: int = 50000,
        max_output_length: int = 100000,
        case_sensitive: bool = False,
        priority: int = HookPriority.SECURITY_DEFAULT + 10,
    ) -> None:
        """Initialize content filter.

        Args:
            blocked_words: Words to block
            blocked_patterns: Regex patterns to block
            max_input_length: Maximum input length
            max_output_length: Maximum output length
            case_sensitive: Whether matching is case-sensitive
            priority: Hook priority
        """
        self._blocked_words = set(blocked_words or [])
        self._blocked_patterns = [
            re.compile(p, 0 if case_sensitive else re.IGNORECASE) for p in (blocked_patterns or [])
        ]
        self._max_input_length = max_input_length
        self._max_output_length = max_output_length
        self._case_sensitive = case_sensitive
        self._priority = priority

    @property
    def priority(self) -> int:
        """Return hook priority."""
        return self._priority

    @property
    def name(self) -> str:
        """Return hook name."""
        return "ContentFilterHook"

    def _check_content(self, text: str) -> str | None:
        """Check content for blocked terms.

        Args:
            text: Text to check

        Returns:
            Error message if blocked, None otherwise
        """
        check_text = text if self._case_sensitive else text.lower()

        # Check blocked words
        for word in self._blocked_words:
            check_word = word if self._case_sensitive else word.lower()
            if check_word in check_text:
                return f"Blocked word detected: {word}"

        # Check blocked patterns
        for pattern in self._blocked_patterns:
            if pattern.search(text):
                return "Blocked pattern detected"

        return None

    async def on_before_invocation(
        self,
        prompt: str,
        state: AgentState,
    ) -> AgentState:
        """Filter input prompt.

        Args:
            prompt: User prompt
            state: Agent state

        Returns:
            Unchanged state

        Raises:
            ValueError: If content is blocked
        """
        if len(prompt) > self._max_input_length:
            msg = f"Input too long: {len(prompt)} > {self._max_input_length}"
            raise ValueError(msg)

        error = self._check_content(prompt)
        if error:
            raise ValueError(error)

        return state

    async def on_before_tool_call(self, event: BeforeToolCallEvent) -> None:
        """Filter tool arguments.

        Args:
            event: Write-protected event carrying ``tool_name`` and
                ``arguments``.

        Raises:
            ValueError: If content is blocked
        """
        args_str = str(event.arguments)
        error = self._check_content(args_str)
        if error:
            msg = f"Tool arguments blocked: {error}"
            raise ValueError(msg)
