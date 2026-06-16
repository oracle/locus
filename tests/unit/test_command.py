# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for Command primitive."""

import pytest

from locus.core.command import (
    Command,
    Continue,
    End,
    end,
    goto,
    is_command,
    normalize_node_output,
    resume_with,
)


class TestCommand:
    """Tests for Command class."""

    def test_basic_creation(self):
        """Test basic Command creation."""
        cmd = Command(update={"x": 1}, goto="next")
        assert cmd.update == {"x": 1}
        assert cmd.goto == "next"
        assert cmd.resume is None

    def test_frozen(self):
        """Test Command is immutable."""
        from pydantic import ValidationError

        cmd = Command(goto="next")
        with pytest.raises(ValidationError, match="frozen"):
            cmd.goto = "other"

    def test_has_update(self):
        """Test has_update property."""
        assert Command(update={"x": 1}).has_update
        assert not Command(update={}).has_update
        assert not Command().has_update

    def test_has_goto(self):
        """Test has_goto property."""
        assert Command(goto="next").has_goto
        assert not Command(goto=None).has_goto
        assert not Command().has_goto

    def test_has_resume(self):
        """Test has_resume property."""
        assert Command(resume="value").has_resume
        assert not Command(resume=None).has_resume
        assert not Command().has_resume

    def test_is_parallel_goto(self):
        """Test is_parallel_goto property."""
        assert Command(goto=["a", "b"]).is_parallel_goto
        assert not Command(goto="single").is_parallel_goto
        assert not Command(goto=None).is_parallel_goto

    def test_goto_nodes(self):
        """Test goto_nodes normalization."""
        assert Command(goto="single").goto_nodes == ["single"]
        assert Command(goto=["a", "b"]).goto_nodes == ["a", "b"]
        assert Command(goto=None).goto_nodes == []

    def test_with_update(self):
        """Test with_update method."""
        cmd = Command(update={"a": 1})
        new_cmd = cmd.with_update(b=2)
        assert new_cmd.update == {"a": 1, "b": 2}
        assert cmd.update == {"a": 1}  # Original unchanged

    def test_with_goto(self):
        """Test with_goto method."""
        cmd = Command(goto="original")
        new_cmd = cmd.with_goto("new")
        assert new_cmd.goto == "new"
        assert cmd.goto == "original"  # Original unchanged


class TestEnd:
    """Tests for End command."""

    def test_end_goto(self):
        """Test End has __END__ goto."""
        e = End()
        assert e.goto == "__END__"

    def test_end_with_update(self):
        """Test End with state update."""
        e = End(update={"result": "done"})
        assert e.update == {"result": "done"}
        assert e.goto == "__END__"


class TestContinue:
    """Tests for Continue command."""

    def test_continue_no_goto(self):
        """Test Continue has no goto."""
        c = Continue()
        assert c.goto is None

    def test_continue_with_update(self):
        """Test Continue with state update."""
        c = Continue(update={"processed": True})
        assert c.update == {"processed": True}
        assert not c.has_goto


class TestIsCommand:
    """Tests for is_command function."""

    def test_detects_command(self):
        """Test is_command with Command instance."""
        assert is_command(Command())
        assert is_command(End())
        assert is_command(Continue())

    def test_rejects_non_command(self):
        """Test is_command with non-Command values."""
        assert not is_command({})
        assert not is_command(None)
        assert not is_command("string")
        assert not is_command({"goto": "next"})  # Dict is not Command


class TestNormalizeNodeOutput:
    """Tests for normalize_node_output function."""

    def test_normalize_none(self):
        """Test normalizing None output."""
        update, cmd = normalize_node_output(None)
        assert update == {}
        assert cmd is None

    def test_normalize_dict(self):
        """Test normalizing dict output."""
        update, cmd = normalize_node_output({"x": 1})
        assert update == {"x": 1}
        assert cmd is None

    def test_normalize_command(self):
        """Test normalizing Command output."""
        command = Command(update={"x": 1}, goto="next")
        update, cmd = normalize_node_output(command)
        assert update == {"x": 1}
        assert cmd is command

    def test_normalize_other(self):
        """Test normalizing other values (wrapped in result key)."""
        update, cmd = normalize_node_output("string value")
        assert update == {"result": "string value"}
        assert cmd is None

        update, cmd = normalize_node_output(42)
        assert update == {"result": 42}

    def test_normalize_pydantic_basemodel(self):
        """Pydantic BaseModel output is converted to a dict state update."""
        from pydantic import BaseModel

        class MyOutput(BaseModel):
            answer: str
            score: float

        output = MyOutput(answer="yes", score=0.9)
        update, cmd = normalize_node_output(output)
        assert update == {"answer": "yes", "score": 0.9}
        assert cmd is None


class TestConvenienceConstructors:
    """Tests for convenience constructor functions."""

    def test_goto_simple(self):
        """Test goto function."""
        cmd = goto("next")
        assert cmd.goto == "next"
        assert cmd.update == {}

    def test_goto_with_updates(self):
        """Test goto with keyword updates."""
        cmd = goto("next", processed=True, count=5)
        assert cmd.goto == "next"
        assert cmd.update == {"processed": True, "count": 5}

    def test_goto_parallel(self):
        """Test goto with multiple targets."""
        cmd = goto(["a", "b", "c"])
        assert cmd.goto == ["a", "b", "c"]
        assert cmd.is_parallel_goto

    def test_end_simple(self):
        """Test end function."""
        cmd = end()
        assert isinstance(cmd, End)
        assert cmd.update == {}

    def test_end_with_updates(self):
        """Test end with updates."""
        cmd = end(result="success", data={"x": 1})
        assert cmd.update == {"result": "success", "data": {"x": 1}}

    def test_resume_with_value(self):
        """Test resume_with function."""
        cmd = resume_with("approved")
        assert cmd.resume == "approved"
        assert cmd.update == {}

    def test_resume_with_updates(self):
        """Test resume_with with updates."""
        cmd = resume_with("approved", reviewed_by="user123")
        assert cmd.resume == "approved"
        assert cmd.update == {"reviewed_by": "user123"}
