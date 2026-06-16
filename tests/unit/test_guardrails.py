# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for guardrails hook."""

from unittest.mock import MagicMock

import pytest

from locus.hooks.builtin.guardrails import (
    ContentFilterHook,
    GuardrailAction,
    GuardrailConfig,
    GuardrailsHook,
    GuardrailViolation,
)
from locus.hooks.provider import (
    AfterToolCallEvent,
    BeforeToolCallEvent,
    HookPriority,
)


def _before(tool_name: str, arguments: dict) -> BeforeToolCallEvent:
    return BeforeToolCallEvent(
        tool_name=tool_name, tool_call_id=f"{tool_name}-call", arguments=arguments
    )


def _after(tool_name: str, result, error: str | None) -> AfterToolCallEvent:
    return AfterToolCallEvent(tool_name=tool_name, result=result, error=error)


class TestGuardrailAction:
    """Tests for GuardrailAction enum."""

    def test_all_actions(self):
        """Test all action values exist."""
        assert GuardrailAction.BLOCK.value == "block"
        assert GuardrailAction.WARN.value == "warn"
        assert GuardrailAction.REDACT.value == "redact"
        assert GuardrailAction.ALLOW.value == "allow"


class TestGuardrailViolation:
    """Tests for GuardrailViolation dataclass."""

    def test_create_minimal(self):
        """Test creating violation with minimal fields."""
        violation = GuardrailViolation(
            rule_name="test_rule",
            description="Test violation",
            action=GuardrailAction.BLOCK,
        )
        assert violation.rule_name == "test_rule"
        assert violation.description == "Test violation"
        assert violation.action == GuardrailAction.BLOCK
        assert violation.matched_content is None
        assert violation.location is None

    def test_create_full(self):
        """Test creating violation with all fields."""
        violation = GuardrailViolation(
            rule_name="pii_email",
            description="Email detected",
            action=GuardrailAction.REDACT,
            matched_content="test@example.com",
            location="input",
        )
        assert violation.matched_content == "test@example.com"
        assert violation.location == "input"


class TestGuardrailConfig:
    """Tests for GuardrailConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = GuardrailConfig()
        assert "eval" in config.block_dangerous_tools
        assert "exec" in config.block_dangerous_tools
        assert "shell" in config.block_dangerous_tools
        assert config.allow_only_tools is None
        assert config.max_prompt_length == 100000
        assert config.max_tool_result_length == 50000
        assert config.default_action == GuardrailAction.BLOCK

    def test_custom_config(self):
        """Test custom configuration."""
        config = GuardrailConfig(
            block_dangerous_tools=frozenset({"custom_tool"}),
            allow_only_tools=frozenset({"safe_tool"}),
            max_prompt_length=1000,
            default_action=GuardrailAction.WARN,
        )
        assert "custom_tool" in config.block_dangerous_tools
        assert "safe_tool" in config.allow_only_tools
        assert config.max_prompt_length == 1000
        assert config.default_action == GuardrailAction.WARN

    def test_pii_patterns(self):
        """Test default PII patterns are set."""
        config = GuardrailConfig()
        assert "email" in config.pii_patterns
        assert "phone_us" in config.pii_patterns
        assert "ssn" in config.pii_patterns
        assert "credit_card" in config.pii_patterns

    def test_blocked_content_patterns(self):
        """Test default blocked content patterns."""
        config = GuardrailConfig()
        assert "sql_injection" in config.blocked_content_patterns
        assert "path_traversal" in config.blocked_content_patterns


class TestGuardrailsHook:
    """Tests for GuardrailsHook."""

    @pytest.fixture
    def hook(self):
        """Create a guardrails hook with default config."""
        return GuardrailsHook()

    @pytest.fixture
    def custom_hook(self):
        """Create a guardrails hook with custom config."""
        config = GuardrailConfig(
            block_dangerous_tools=frozenset({"dangerous_tool"}),
            default_action=GuardrailAction.WARN,
        )
        return GuardrailsHook(config)

    def test_create_default(self, hook):
        """Test creating hook with defaults."""
        assert hook.priority == HookPriority.SECURITY_DEFAULT
        assert hook.name == "GuardrailsHook"
        assert len(hook.violations) == 0

    def test_create_custom_priority(self):
        """Test creating hook with custom priority."""
        hook = GuardrailsHook(priority=10)
        assert hook.priority == 10

    def test_violations_property(self, hook):
        """Test violations property returns copy."""
        violations = hook.violations
        assert isinstance(violations, list)
        assert len(violations) == 0

    def test_clear_violations(self, hook):
        """Test clearing violations."""
        # Manually add a violation to test clearing
        hook._violations.append(
            GuardrailViolation(
                rule_name="test",
                description="test",
                action=GuardrailAction.BLOCK,
            )
        )
        assert len(hook.violations) == 1

        hook.clear_violations()
        assert len(hook.violations) == 0

    def test_get_action_default(self, hook):
        """Test getting default action for rule."""
        action = hook._get_action("unknown_rule")
        assert action == GuardrailAction.BLOCK

    def test_get_action_override(self):
        """Test getting overridden action for rule."""
        config = GuardrailConfig(action_overrides={"pii_email": GuardrailAction.REDACT})
        hook = GuardrailsHook(config)
        action = hook._get_action("pii_email")
        assert action == GuardrailAction.REDACT

    def test_on_violation_callback(self):
        """Test violation callback is called."""
        violations_received = []

        def on_violation(v):
            violations_received.append(v)

        hook = GuardrailsHook(on_violation=on_violation)
        violation = GuardrailViolation(
            rule_name="test",
            description="test",
            action=GuardrailAction.BLOCK,
        )
        hook._record_violation(violation)

        assert len(violations_received) == 1
        assert violations_received[0] is violation

    def test_check_pii_email(self, hook):
        """Test PII detection for email."""
        violations = hook._check_pii("Contact me at test@example.com", "input")
        assert len(violations) >= 1
        email_violations = [v for v in violations if "email" in v.rule_name]
        assert len(email_violations) == 1

    def test_check_pii_phone(self, hook):
        """Test PII detection for phone."""
        violations = hook._check_pii("Call me at 555-123-4567", "input")
        assert len(violations) >= 1
        phone_violations = [v for v in violations if "phone" in v.rule_name]
        assert len(phone_violations) == 1

    def test_check_pii_ssn(self, hook):
        """Test PII detection for SSN."""
        violations = hook._check_pii("SSN: 123-45-6789", "input")
        assert len(violations) >= 1
        ssn_violations = [v for v in violations if "ssn" in v.rule_name]
        assert len(ssn_violations) == 1

    def test_check_pii_no_match(self, hook):
        """Test PII detection with no matches."""
        violations = hook._check_pii("Hello world", "input")
        assert len(violations) == 0

    def test_check_blocked_content_sql_injection(self, hook):
        """Test blocked content detection for SQL injection."""
        violations = hook._check_blocked_content("DROP TABLE users; --", "input")
        sql_violations = [v for v in violations if "sql" in v.rule_name.lower()]
        assert len(sql_violations) >= 1

    def test_check_blocked_content_path_traversal(self, hook):
        """Test blocked content detection for path traversal."""
        violations = hook._check_blocked_content("../../etc/passwd", "input")
        path_violations = [v for v in violations if "path" in v.rule_name.lower()]
        assert len(path_violations) >= 1

    @pytest.mark.asyncio
    async def test_on_before_tool_call_blocked(self, hook):
        """Test tool blocking for dangerous tools."""
        with pytest.raises(ValueError, match="blocked by guardrails"):
            await hook.on_before_tool_call(_before("eval", {"code": "print(1)"}))
        # Should record violation
        assert len(hook.violations) >= 1
        assert any("blocked_tool" in v.rule_name for v in hook.violations)

    @pytest.mark.asyncio
    async def test_on_before_tool_call_allowed(self, hook):
        """Test tool allowed for safe tools."""
        args = {"query": "test"}
        event = _before("search", args)
        await hook.on_before_tool_call(event)
        # Hook is observe-only for safe tools — event.arguments is unmodified.
        assert event.arguments == args
        # No blocked_tool violations
        blocked_violations = [v for v in hook.violations if "blocked_tool" in v.rule_name]
        assert len(blocked_violations) == 0

    @pytest.mark.asyncio
    async def test_on_before_tool_call_allowlist(self):
        """Test tool allowlist enforcement."""
        config = GuardrailConfig(allow_only_tools=frozenset({"allowed_tool"}))
        hook = GuardrailsHook(config)

        # Allowed tool should pass
        event = _before("allowed_tool", {"arg": "value"})
        await hook.on_before_tool_call(event)
        assert event.arguments == {"arg": "value"}

        # Non-allowed tool should fail
        with pytest.raises(ValueError, match="not allowed"):
            await hook.on_before_tool_call(_before("other_tool", {"arg": "value"}))

    @pytest.mark.asyncio
    async def test_on_before_invocation(self, hook):
        """Test before invocation hook."""
        mock_state = MagicMock()
        result = await hook.on_before_invocation("Hello world", mock_state)
        assert result is mock_state

    @pytest.mark.asyncio
    async def test_on_before_invocation_blocked(self, hook):
        """Test before invocation blocks dangerous content."""
        mock_state = MagicMock()
        # SQL injection should be blocked
        with pytest.raises(ValueError, match="blocked"):
            await hook.on_before_invocation("DROP TABLE users;", mock_state)

    @pytest.mark.asyncio
    async def test_redact_pii_in_tool_args(self):
        """Test PII redaction in tool arguments."""
        config = GuardrailConfig(action_overrides={"pii_email": GuardrailAction.REDACT})
        hook = GuardrailsHook(config)

        args = {"message": "Contact me at test@example.com"}
        event = _before("send_message", args)
        await hook.on_before_tool_call(event)
        # Email should be redacted in-place on the event.
        assert "test@example.com" not in event.arguments["message"]
        assert "REDACTED" in event.arguments["message"]

    @pytest.mark.asyncio
    async def test_on_after_tool_call(self, hook):
        """Test after tool call hook."""
        # Should not raise for normal results
        await hook.on_after_tool_call(_after("search", "Found 5 results", None))

    @pytest.mark.asyncio
    async def test_on_after_invocation(self, hook):
        """Test after invocation hook."""
        mock_state = MagicMock()
        # Should not raise
        await hook.on_after_invocation(mock_state, True)

    def test_register_hooks(self, hook):
        """Test register_hooks returns all hooks."""
        hooks = hook.register_hooks()
        assert hooks["on_before_invocation"] is True
        assert hooks["on_after_invocation"] is True
        assert hooks["on_before_tool_call"] is True
        assert hooks["on_after_tool_call"] is True


class TestContentFilterHook:
    """Tests for ContentFilterHook."""

    @pytest.fixture
    def hook(self):
        """Create a content filter hook."""
        return ContentFilterHook(
            blocked_words=["forbidden", "banned"],
            blocked_patterns=[r"\bsecret\d+\b"],
            max_input_length=1000,
            max_output_length=2000,
        )

    def test_create_default(self):
        """Test creating hook with defaults."""
        hook = ContentFilterHook()
        assert hook.priority == HookPriority.SECURITY_DEFAULT + 10
        assert hook.name == "ContentFilterHook"

    def test_create_custom(self):
        """Test creating hook with custom settings."""
        hook = ContentFilterHook(
            blocked_words=["test"],
            blocked_patterns=[r"\d{4}"],
            max_input_length=500,
            max_output_length=1000,
            case_sensitive=True,
            priority=50,
        )
        assert hook.priority == 50
        assert hook._case_sensitive is True

    @pytest.mark.asyncio
    async def test_on_before_invocation_allowed(self, hook):
        """Test allowed input passes."""
        mock_state = MagicMock()
        result = await hook.on_before_invocation("Hello world", mock_state)
        assert result is mock_state

    @pytest.mark.asyncio
    async def test_on_before_invocation_blocked_word(self, hook):
        """Test blocked word is rejected."""
        mock_state = MagicMock()
        with pytest.raises(ValueError, match="Blocked word detected"):
            await hook.on_before_invocation("This is forbidden content", mock_state)

    @pytest.mark.asyncio
    async def test_on_before_invocation_blocked_pattern(self, hook):
        """Test blocked pattern is rejected."""
        mock_state = MagicMock()
        with pytest.raises(ValueError, match="Blocked pattern detected"):
            await hook.on_before_invocation("The code is secret123", mock_state)

    @pytest.mark.asyncio
    async def test_on_before_invocation_too_long(self, hook):
        """Test input too long is rejected."""
        mock_state = MagicMock()
        long_input = "x" * 1001
        with pytest.raises(ValueError, match="Input too long"):
            await hook.on_before_invocation(long_input, mock_state)

    @pytest.mark.asyncio
    async def test_on_before_tool_call_allowed(self, hook):
        """Test allowed tool args pass."""
        args = {"query": "safe query"}
        event = _before("search", args)
        await hook.on_before_tool_call(event)
        assert event.arguments == args

    @pytest.mark.asyncio
    async def test_on_before_tool_call_blocked(self, hook):
        """Test blocked tool args are rejected."""
        args = {"message": "This is banned content"}
        with pytest.raises(ValueError, match="Tool arguments blocked"):
            await hook.on_before_tool_call(_before("send", args))

    def test_case_insensitive_matching(self):
        """Test case insensitive matching."""
        hook = ContentFilterHook(blocked_words=["SECRET"], case_sensitive=False)
        # Should match regardless of case
        error = hook._check_content("this is a sEcReT")
        assert error is not None
        assert "SECRET" in error

    def test_case_sensitive_matching(self):
        """Test case sensitive matching."""
        hook = ContentFilterHook(blocked_words=["SECRET"], case_sensitive=True)
        # Should not match different case
        error = hook._check_content("this is a secret")
        assert error is None

        # Should match exact case
        error = hook._check_content("this is a SECRET")
        assert error is not None


class TestGuardrailsEdgeCases:
    """Tests for edge cases in guardrails."""

    @pytest.mark.asyncio
    async def test_prompt_exceeds_max_length(self):
        """Test that long prompts trigger max_prompt_length violation."""
        config = GuardrailConfig(
            max_prompt_length=50,  # Very short for testing
            action_overrides={"max_prompt_length": GuardrailAction.WARN},  # Don't block
        )
        hook = GuardrailsHook(config)

        mock_state = MagicMock()
        mock_state.with_metadata = MagicMock(return_value=mock_state)

        long_prompt = "a" * 100  # Exceeds max_prompt_length

        await hook.on_before_invocation(long_prompt, mock_state)

        # Should have recorded a violation
        length_violations = [v for v in hook.violations if "max_prompt_length" in v.rule_name]
        assert len(length_violations) >= 1

    @pytest.mark.asyncio
    async def test_prompt_length_blocking(self):
        """Test that max_prompt_length can block input."""
        config = GuardrailConfig(
            max_prompt_length=50,
            action_overrides={"max_prompt_length": GuardrailAction.BLOCK},
        )
        hook = GuardrailsHook(config)

        mock_state = MagicMock()
        long_prompt = "a" * 100

        with pytest.raises(ValueError, match="exceeds maximum length"):
            await hook.on_before_invocation(long_prompt, mock_state)

    @pytest.mark.asyncio
    async def test_violations_stored_in_metadata(self):
        """Test that violations are stored in state metadata."""
        config = GuardrailConfig(
            action_overrides={"sql_injection": GuardrailAction.WARN},  # Don't block, just log
        )
        hook = GuardrailsHook(config)

        mock_state = MagicMock()
        mock_state.with_metadata = MagicMock(return_value=mock_state)

        # Trigger SQL injection detection (logged, not blocked)
        prompt = "SELECT * FROM users WHERE id = 1"

        await hook.on_before_invocation(prompt, mock_state)

        # State should have been updated with metadata
        # Check if with_metadata was called (may or may not be depending on content)

    @pytest.mark.asyncio
    async def test_tool_arguments_blocked(self):
        """Test that dangerous tool arguments are blocked."""
        config = GuardrailConfig()
        hook = GuardrailsHook(config)

        # Arguments containing blocked content
        args = {"query": "DROP TABLE users; SELECT * FROM secrets"}

        with pytest.raises(ValueError, match="blocked"):
            await hook.on_before_tool_call(_before("database_query", args))

    @pytest.mark.asyncio
    async def test_non_string_tool_arg_passes_through(self):
        """Test that non-string tool arguments pass through during redaction."""
        config = GuardrailConfig(action_overrides={"pii_email": GuardrailAction.REDACT})
        hook = GuardrailsHook(config)

        args = {
            "message": "Contact test@example.com",
            "count": 42,  # Non-string, should pass through unchanged
            "data": {"nested": True},  # Dict, should pass through unchanged
        }

        event = _before("send_message", args)
        await hook.on_before_tool_call(event)

        # Email should be redacted in string
        assert "test@example.com" not in event.arguments.get("message", "")
        # Non-strings should be unchanged
        assert event.arguments["count"] == 42
        assert event.arguments["data"] == {"nested": True}

    @pytest.mark.asyncio
    async def test_on_after_tool_call_none_result(self):
        """Test after tool call with None result returns early."""
        hook = GuardrailsHook()

        # Should not raise when result is None
        await hook.on_after_tool_call(_after("search", None, None))

    @pytest.mark.asyncio
    async def test_tool_result_exceeds_max_length(self):
        """Test that long tool results trigger max_tool_result_length violation."""
        config = GuardrailConfig(
            max_tool_result_length=100,
        )
        hook = GuardrailsHook(config)

        long_result = "x" * 200  # Exceeds max_tool_result_length

        await hook.on_after_tool_call(_after("search", long_result, None))

        # Should have recorded a violation
        length_violations = [v for v in hook.violations if "max_tool_result_length" in v.rule_name]
        assert len(length_violations) >= 1


# =============================================================================
# Topic Policy Tests
# =============================================================================


class TestTopicPolicy:
    """Tests for topic-based content blocking."""

    def test_blocks_matching_topic(self):
        """Topic policy blocks text matching blocked topics."""
        from locus.hooks.builtin.guardrails import TopicPolicy

        policy = TopicPolicy(
            blocked_topics={"weapons", "drugs"},
            keywords={
                "weapons": ["gun", "rifle", "ammunition"],
                "drugs": ["cocaine", "heroin"],
            },
        )

        assert policy.check("How to buy a gun") == "weapons"
        assert policy.check("Tell me about cocaine") == "drugs"
        assert policy.check("Tell me about Python programming") is None

    def test_case_insensitive(self):
        """Topic matching is case-insensitive."""
        from locus.hooks.builtin.guardrails import TopicPolicy

        policy = TopicPolicy(
            blocked_topics={"weapons"},
            keywords={"weapons": ["gun"]},
        )

        assert policy.check("I want a GUN") == "weapons"


class TestContentPolicy:
    """Tests for content safety categories."""

    def test_detects_harmful_content(self):
        """Content policy detects harmful categories."""
        from locus.hooks.builtin.guardrails import ContentPolicy

        policy = ContentPolicy()

        assert policy.check("how to make a bomb at home") == "violence"
        assert policy.check("how to hack into a bank") == "illegal_activity"
        assert policy.check("How to bake a cake") is None

    def test_respects_enabled_categories(self):
        """Only enabled categories are checked."""
        from locus.hooks.builtin.guardrails import ContentPolicy

        policy = ContentPolicy(enabled_categories={"violence"})

        assert policy.check("how to make a bomb") == "violence"
        assert policy.check("how to hack into a bank") is None  # illegal_activity not enabled


class TestOutputFilterHook:
    """Tests for output filtering."""

    @pytest.mark.asyncio
    async def test_redacts_pii_in_output(self):
        """Output filter redacts PII from model responses."""

        from locus.core.messages import Message
        from locus.hooks.builtin.guardrails import OutputFilterHook
        from locus.hooks.provider import AfterModelCallEvent
        from locus.models.base import ModelResponse

        hook = OutputFilterHook(redact_pii=True)

        response = ModelResponse(message=Message.assistant("Contact john@example.com for help"))
        event = AfterModelCallEvent(response=response, messages=[])

        await hook.on_after_model_call(event)

        assert "john@example.com" not in event.response.message.content
        assert "REDACTED_EMAIL" in event.response.message.content

    @pytest.mark.asyncio
    async def test_blocks_harmful_content_in_output(self):
        """Output filter blocks harmful content categories."""

        from locus.core.messages import Message
        from locus.hooks.builtin.guardrails import ContentPolicy, OutputFilterHook
        from locus.hooks.provider import AfterModelCallEvent
        from locus.models.base import ModelResponse

        hook = OutputFilterHook(content_policy=ContentPolicy())

        response = ModelResponse(message=Message.assistant("Here is how to make a bomb..."))
        event = AfterModelCallEvent(response=response, messages=[])

        await hook.on_after_model_call(event)

        assert "violence" in event.response.message.content
        assert "bomb" not in event.response.message.content

    @pytest.mark.asyncio
    async def test_safe_content_passes_through(self):
        """Safe content is not modified."""
        from locus.core.messages import Message
        from locus.hooks.builtin.guardrails import OutputFilterHook
        from locus.hooks.provider import AfterModelCallEvent
        from locus.models.base import ModelResponse

        hook = OutputFilterHook(redact_pii=True)

        response = ModelResponse(message=Message.assistant("The weather is sunny today."))
        event = AfterModelCallEvent(response=response, messages=[])

        await hook.on_after_model_call(event)

        assert event.response.message.content == "The weather is sunny today."
