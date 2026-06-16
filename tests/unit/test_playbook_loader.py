# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for playbook loader module."""

import json
import tempfile
from pathlib import Path

import pytest

from locus.playbooks.loader import (
    PlaybookLoader,
    PlaybookLoadError,
    load_playbook,
)
from locus.playbooks.models import Playbook


class TestPlaybookLoadError:
    """Tests for PlaybookLoadError exception."""

    def test_basic_error(self):
        """Test basic error creation."""
        error = PlaybookLoadError("Test error")
        assert str(error) == "Test error"
        assert error.path is None
        assert error.errors == []

    def test_error_with_path(self):
        """Test error with path."""
        path = Path("test_path") / "test.json"  # Use relative path
        error = PlaybookLoadError("File not found", path=path)
        assert error.path == path

    def test_error_with_errors(self):
        """Test error with validation errors."""
        errors = ["Error 1", "Error 2"]
        error = PlaybookLoadError("Validation failed", errors=errors)
        assert error.errors == errors


class TestPlaybookLoader:
    """Tests for PlaybookLoader class."""

    @pytest.fixture
    def loader(self):
        return PlaybookLoader()

    @pytest.fixture
    def valid_playbook_dict(self):
        return {
            "id": "test-playbook",
            "name": "Test Playbook",
            "description": "A test playbook",
            "steps": [
                {
                    "id": "step1",
                    "description": "First step",
                    "expected_tools": ["tool1"],
                    "hints": ["Hint 1"],
                }
            ],
        }

    def test_load_dict_valid(self, loader, valid_playbook_dict):
        """Test loading a valid dictionary."""
        playbook = loader.load_dict(valid_playbook_dict)

        assert isinstance(playbook, Playbook)
        assert playbook.id == "test-playbook"
        assert playbook.name == "Test Playbook"
        assert len(playbook.steps) == 1

    def test_load_dict_missing_id(self, loader):
        """Test loading a dictionary missing id."""
        data = {"name": "Test"}

        with pytest.raises(PlaybookLoadError) as exc_info:
            loader.load_dict(data)

        assert any("id" in e.lower() for e in exc_info.value.errors)

    def test_load_dict_missing_name(self, loader):
        """Test loading a dictionary missing name."""
        data = {"id": "test"}

        with pytest.raises(PlaybookLoadError) as exc_info:
            loader.load_dict(data)

        assert any("name" in e.lower() for e in exc_info.value.errors)

    def test_load_dict_invalid_steps(self, loader):
        """Test loading with invalid steps type."""
        data = {"id": "test", "name": "Test", "steps": "not a list"}

        with pytest.raises(PlaybookLoadError) as exc_info:
            loader.load_dict(data)

        assert "list" in str(exc_info.value).lower() or len(exc_info.value.errors) > 0

    def test_load_json_string_valid(self, loader, valid_playbook_dict):
        """Test loading from valid JSON string."""
        json_str = json.dumps(valid_playbook_dict)
        playbook = loader.load_json_string(json_str)

        assert playbook.id == "test-playbook"

    def test_load_json_string_invalid(self, loader):
        """Test loading from invalid JSON string."""
        with pytest.raises(PlaybookLoadError) as exc_info:
            loader.load_json_string("not valid json")

        assert "JSON" in str(exc_info.value)

    def test_load_file_json(self, loader, valid_playbook_dict):
        """Test loading from JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(valid_playbook_dict, f)
            temp_path = Path(f.name)

        try:
            playbook = loader.load_file(temp_path)
            assert playbook.id == "test-playbook"
        finally:
            temp_path.unlink()

    def test_load_file_not_found(self, loader):
        """Test loading from non-existent file."""
        with pytest.raises(PlaybookLoadError) as exc_info:
            loader.load_file("/nonexistent/path/file.json")

        assert "not found" in str(exc_info.value).lower()

    def test_load_file_unsupported_format(self, loader):
        """Test loading from unsupported file format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("some text")
            temp_path = Path(f.name)

        try:
            with pytest.raises(PlaybookLoadError) as exc_info:
                loader.load_file(temp_path)

            assert "Unsupported" in str(exc_info.value)
        finally:
            temp_path.unlink()

    def test_load_file_invalid_json(self, loader):
        """Test loading from file with invalid JSON."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json {")
            temp_path = Path(f.name)

        try:
            with pytest.raises(PlaybookLoadError) as exc_info:
                loader.load_file(temp_path)

            assert "JSON" in str(exc_info.value)
        finally:
            temp_path.unlink()

    def test_load_yaml_string_valid(self, loader):
        """Test loading from valid YAML string."""
        yaml_str = """
id: test-playbook
name: Test Playbook
steps:
  - id: step1
    description: First step
"""
        playbook = loader.load_yaml_string(yaml_str)
        assert playbook.id == "test-playbook"

    def test_load_file_yaml(self, loader):
        """Test loading from YAML file."""
        yaml_content = """
id: yaml-playbook
name: YAML Playbook
steps:
  - id: step1
    description: First step
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = Path(f.name)

        try:
            playbook = loader.load_file(temp_path)
            assert playbook.id == "yaml-playbook"
        finally:
            temp_path.unlink()

    def test_load_file_yml_extension(self, loader):
        """Test loading from .yml file."""
        yaml_content = """
id: yml-playbook
name: YML Playbook
steps: []
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(yaml_content)
            temp_path = Path(f.name)

        try:
            playbook = loader.load_file(temp_path)
            assert playbook.id == "yml-playbook"
        finally:
            temp_path.unlink()


class TestValidateStructure:
    """Tests for structure validation."""

    @pytest.fixture
    def loader(self):
        return PlaybookLoader()

    def test_validate_non_dict(self, loader):
        """Test validation of non-dict data."""
        errors = loader._validate_structure([])
        assert "dictionary" in errors[0].lower()

    def test_validate_duplicate_step_ids(self, loader):
        """Test validation catches duplicate step IDs."""
        data = {
            "id": "test",
            "name": "Test",
            "steps": [
                {"id": "step1", "description": "First"},
                {"id": "step1", "description": "Duplicate"},
            ],
        }
        errors = loader._validate_structure(data)
        assert any("duplicate" in e.lower() for e in errors)

    def test_validate_step_not_dict(self, loader):
        """Test validation of non-dict step."""
        data = {
            "id": "test",
            "name": "Test",
            "steps": ["not a dict"],
        }
        errors = loader._validate_structure(data)
        assert any("dictionary" in e.lower() for e in errors)

    def test_validate_step_missing_id(self, loader):
        """Test validation of step missing id."""
        data = {
            "id": "test",
            "name": "Test",
            "steps": [{"description": "Step without id"}],
        }
        errors = loader._validate_structure(data)
        assert any("id" in e.lower() for e in errors)

    def test_validate_step_missing_description(self, loader):
        """Test validation of step missing description."""
        data = {
            "id": "test",
            "name": "Test",
            "steps": [{"id": "step1"}],
        }
        errors = loader._validate_structure(data)
        assert any("description" in e.lower() for e in errors)

    def test_validate_expected_tools_not_list(self, loader):
        """Test validation of expected_tools not being a list."""
        data = {
            "id": "test",
            "name": "Test",
            "steps": [{"id": "step1", "description": "Test", "expected_tools": "not a list"}],
        }
        errors = loader._validate_structure(data)
        assert any("expected_tools" in e.lower() for e in errors)

    def test_validate_expected_tools_not_strings(self, loader):
        """Test validation of expected_tools containing non-strings."""
        data = {
            "id": "test",
            "name": "Test",
            "steps": [{"id": "step1", "description": "Test", "expected_tools": [123]}],
        }
        errors = loader._validate_structure(data)
        assert any("expected_tools" in e.lower() for e in errors)

    def test_validate_hints_not_list(self, loader):
        """Test validation of hints not being a list."""
        data = {
            "id": "test",
            "name": "Test",
            "steps": [{"id": "step1", "description": "Test", "hints": "not a list"}],
        }
        errors = loader._validate_structure(data)
        assert any("hints" in e.lower() for e in errors)

    def test_validate_hints_not_strings(self, loader):
        """Test validation of hints containing non-strings."""
        data = {
            "id": "test",
            "name": "Test",
            "steps": [{"id": "step1", "description": "Test", "hints": [123]}],
        }
        errors = loader._validate_structure(data)
        assert any("hints" in e.lower() for e in errors)


class TestLoadPlaybookConvenience:
    """Tests for load_playbook convenience function."""

    @pytest.fixture
    def valid_playbook_dict(self):
        return {
            "id": "test-playbook",
            "name": "Test Playbook",
            "steps": [],
        }

    def test_load_from_dict(self, valid_playbook_dict):
        """Test loading from dictionary."""
        playbook = load_playbook(valid_playbook_dict)
        assert playbook.id == "test-playbook"

    def test_load_from_path(self, valid_playbook_dict):
        """Test loading from Path object."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(valid_playbook_dict, f)
            temp_path = Path(f.name)

        try:
            playbook = load_playbook(temp_path)
            assert playbook.id == "test-playbook"
        finally:
            temp_path.unlink()

    def test_load_from_string_path(self, valid_playbook_dict):
        """Test loading from string path."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(valid_playbook_dict, f)
            temp_path = Path(f.name)

        try:
            playbook = load_playbook(str(temp_path))
            assert playbook.id == "test-playbook"
        finally:
            temp_path.unlink()

    def test_load_from_json_string(self, valid_playbook_dict):
        """Test loading from JSON string."""
        json_str = json.dumps(valid_playbook_dict)
        playbook = load_playbook(json_str)
        assert playbook.id == "test-playbook"

    def test_load_from_nonexistent_path(self):
        """Test loading from non-existent path."""
        with pytest.raises(PlaybookLoadError):
            load_playbook("/definitely/not/a/real/path.json")
