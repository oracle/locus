# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for playbooks module."""

import json
import tempfile
from pathlib import Path

import pytest

from locus.playbooks.loader import PlaybookLoader, PlaybookLoadError
from locus.playbooks.models import (
    Playbook,
    PlaybookStep,
    StepStatus,
)


class TestStepStatus:
    """Tests for StepStatus enum."""

    def test_all_statuses(self):
        """Test all status values exist."""
        assert StepStatus.PENDING == "pending"
        assert StepStatus.IN_PROGRESS == "in_progress"
        assert StepStatus.COMPLETED == "completed"
        assert StepStatus.SKIPPED == "skipped"
        assert StepStatus.FAILED == "failed"


class TestPlaybookStep:
    """Tests for PlaybookStep model."""

    def test_create_minimal_step(self):
        """Create step with minimal fields."""
        step = PlaybookStep(id="step1", description="Test step")
        assert step.id == "step1"
        assert step.description == "Test step"
        assert step.expected_tools == []
        assert step.hints == []
        assert step.required is True

    def test_create_full_step(self):
        """Create step with all fields."""
        step = PlaybookStep(
            id="step1",
            description="Full step",
            expected_tools=["tool1", "tool2"],
            hints=["hint1", "hint2"],
            required=False,
            validation={"field": "value"},
            max_tool_calls=5,
            timeout_seconds=30.0,
            metadata={"key": "value"},
        )
        assert step.expected_tools == ["tool1", "tool2"]
        assert step.hints == ["hint1", "hint2"]
        assert step.required is False
        assert step.max_tool_calls == 5
        assert step.timeout_seconds == 30.0

    def test_step_is_frozen(self):
        """Test step is immutable."""
        from pydantic import ValidationError

        step = PlaybookStep(id="step1", description="Test")
        with pytest.raises(ValidationError, match="frozen"):
            step.id = "new_id"


class TestPlaybook:
    """Tests for Playbook model."""

    def test_create_minimal_playbook(self):
        """Create playbook with minimal fields."""
        playbook = Playbook(id="rb1", name="Test Playbook")
        assert playbook.id == "rb1"
        assert playbook.name == "Test Playbook"
        assert playbook.steps == []
        assert playbook.version == "1.0.0"
        assert playbook.strict_sequence is True

    def test_create_full_playbook(self):
        """Create playbook with all fields."""
        step = PlaybookStep(id="s1", description="Step 1")
        playbook = Playbook(
            id="rb1",
            name="Full Playbook",
            description="A full playbook",
            version="2.0.0",
            steps=[step],
            strict_sequence=False,
            allow_extra_tools=True,
            max_iterations=10,
            metadata={"key": "value"},
            tags=["test", "demo"],
        )
        assert len(playbook.steps) == 1
        assert playbook.version == "2.0.0"
        assert playbook.strict_sequence is False
        assert playbook.allow_extra_tools is True
        assert playbook.max_iterations == 10
        assert playbook.tags == ["test", "demo"]

    def test_playbook_is_frozen(self):
        """Test playbook is immutable."""
        from pydantic import ValidationError

        playbook = Playbook(id="rb1", name="Test")
        with pytest.raises(ValidationError, match="frozen"):
            playbook.name = "New Name"


class TestPlaybookLoadError:
    """Tests for PlaybookLoadError."""

    def test_error_with_path(self):
        """Test error with path."""
        path = Path("/test/path.json")
        error = PlaybookLoadError("Test error", path=path)
        assert error.path == path
        assert str(error) == "Test error"

    def test_error_with_errors_list(self):
        """Test error with errors list."""
        errors = ["error1", "error2"]
        error = PlaybookLoadError("Multiple errors", errors=errors)
        assert error.errors == errors

    def test_error_defaults(self):
        """Test error default values."""
        error = PlaybookLoadError("Simple error")
        assert error.path is None
        assert error.errors == []


class TestPlaybookLoader:
    """Tests for PlaybookLoader."""

    @pytest.fixture
    def loader(self):
        """Create a PlaybookLoader instance."""
        return PlaybookLoader()

    def test_load_dict_valid(self, loader):
        """Load valid playbook from dict."""
        data = {
            "id": "test_rb",
            "name": "Test Playbook",
            "steps": [
                {"id": "s1", "description": "Step 1"},
            ],
        }
        playbook = loader.load_dict(data)
        assert playbook.id == "test_rb"
        assert playbook.name == "Test Playbook"
        assert len(playbook.steps) == 1

    def test_load_json_file(self, loader):
        """Load playbook from JSON file."""
        data = {
            "id": "json_rb",
            "name": "JSON Playbook",
            "steps": [],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            playbook = loader.load_file(f.name)

        assert playbook.id == "json_rb"
        assert playbook.name == "JSON Playbook"

    def test_load_file_not_found(self, loader):
        """Load file that doesn't exist raises error."""
        with pytest.raises(PlaybookLoadError) as exc_info:
            loader.load_file("/nonexistent/path.json")
        assert "not found" in str(exc_info.value).lower()

    def test_load_unsupported_format(self, loader):
        """Load unsupported format raises error."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"test")
            f.flush()
            with pytest.raises(PlaybookLoadError) as exc_info:
                loader.load_file(f.name)
            assert "unsupported" in str(exc_info.value).lower()

    def test_load_json_string_valid(self, loader):
        """Load playbook from JSON string."""
        json_str = '{"id": "str_rb", "name": "String Playbook"}'
        playbook = loader.load_json_string(json_str)
        assert playbook.id == "str_rb"

    def test_load_json_string_invalid(self, loader):
        """Load invalid JSON string raises error."""
        with pytest.raises(PlaybookLoadError):
            loader.load_json_string("not valid json")

    def test_load_dict_missing_required(self, loader):
        """Load dict missing required field raises error."""
        data = {"name": "No ID"}  # missing 'id'
        with pytest.raises(PlaybookLoadError):
            loader.load_dict(data)
