# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Playbook loading from JSON and YAML files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from locus.playbooks.models import Playbook


class PlaybookLoadError(Exception):
    """Error loading a playbook."""

    def __init__(self, message: str, path: Path | None = None, errors: list[str] | None = None):
        self.path = path
        self.errors = errors or []
        super().__init__(message)


class PlaybookLoader:
    """Load playbooks from JSON and YAML files.

    Supports loading from:
    - JSON files (.json)
    - YAML files (.yaml, .yml)
    - Dictionaries (for programmatic use)
    """

    def load_file(self, path: str | Path) -> Playbook:
        """Load a playbook from a file.

        Args:
            path: Path to the playbook file (.json, .yaml, or .yml)

        Returns:
            Loaded and validated Playbook

        Raises:
            PlaybookLoadError: If file cannot be loaded or validated
        """
        path = Path(path)

        if not path.exists():
            raise PlaybookLoadError(f"File not found: {path}", path=path)

        suffix = path.suffix.lower()

        try:
            if suffix == ".json":
                return self._load_json(path)
            if suffix in (".yaml", ".yml"):
                return self._load_yaml(path)
            raise PlaybookLoadError(
                f"Unsupported file format: {suffix}. Use .json, .yaml, or .yml",
                path=path,
            )
        except PlaybookLoadError:
            raise
        except Exception as e:
            raise PlaybookLoadError(f"Failed to load {path}: {e}", path=path) from e

    def load_dict(self, data: dict[str, Any]) -> Playbook:
        """Load a playbook from a dictionary.

        Args:
            data: Dictionary containing playbook definition

        Returns:
            Loaded and validated Playbook

        Raises:
            PlaybookLoadError: If data is invalid
        """
        errors = self._validate_structure(data)
        if errors:
            raise PlaybookLoadError(
                f"Invalid playbook structure: {len(errors)} errors",
                errors=errors,
            )

        try:
            return Playbook(**data)
        except ValidationError as e:
            errors = [str(err) for err in e.errors()]
            raise PlaybookLoadError(
                f"Playbook validation failed: {len(errors)} errors",
                errors=errors,
            ) from e

    def load_json_string(self, json_string: str) -> Playbook:
        """Load a playbook from a JSON string.

        Args:
            json_string: JSON string containing playbook definition

        Returns:
            Loaded and validated Playbook

        Raises:
            PlaybookLoadError: If JSON is invalid or playbook validation fails
        """
        try:
            data = json.loads(json_string)
        except json.JSONDecodeError as e:
            raise PlaybookLoadError(f"Invalid JSON: {e}") from e

        return self.load_dict(data)

    def load_yaml_string(self, yaml_string: str) -> Playbook:
        """Load a playbook from a YAML string.

        Args:
            yaml_string: YAML string containing playbook definition

        Returns:
            Loaded and validated Playbook

        Raises:
            PlaybookLoadError: If YAML is invalid or playbook validation fails
        """
        try:
            import yaml  # type: ignore[import-untyped]  # PyYAML ships no inline types
        except ImportError as e:
            raise PlaybookLoadError(
                "PyYAML is required for YAML support. Install with: pip install pyyaml"
            ) from e

        try:
            data = yaml.safe_load(yaml_string)
        except yaml.YAMLError as e:
            raise PlaybookLoadError(f"Invalid YAML: {e}") from e

        return self.load_dict(data)

    def _load_json(self, path: Path) -> Playbook:
        """Load playbook from JSON file."""
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise PlaybookLoadError(f"Invalid JSON in {path}: {e}", path=path) from e

        return self.load_dict(data)

    def _load_yaml(self, path: Path) -> Playbook:
        """Load playbook from YAML file."""
        try:
            import yaml
        except ImportError as e:
            raise PlaybookLoadError(
                "PyYAML is required for YAML support. Install with: pip install pyyaml",
                path=path,
            ) from e

        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise PlaybookLoadError(f"Invalid YAML in {path}: {e}", path=path) from e

        return self.load_dict(data)

    def _validate_structure(self, data: dict[str, Any]) -> list[str]:
        """Validate the basic structure of playbook data.

        Returns list of validation errors, empty if valid.
        """
        errors: list[str] = []

        if not isinstance(data, dict):
            errors.append("Playbook must be a dictionary")
            return errors

        # Required fields
        if "id" not in data:
            errors.append("Missing required field: id")
        if "name" not in data:
            errors.append("Missing required field: name")

        # Validate steps structure
        steps = data.get("steps", [])
        if not isinstance(steps, list):
            errors.append("'steps' must be a list")
        else:
            step_ids = set()
            for i, step in enumerate(steps):
                step_errors = self._validate_step(step, i)
                errors.extend(step_errors)

                # Check for duplicate step IDs
                step_id = step.get("id") if isinstance(step, dict) else None
                if step_id:
                    if step_id in step_ids:
                        errors.append(f"Duplicate step id: {step_id}")
                    step_ids.add(step_id)

        return errors

    def _validate_step(self, step: Any, index: int) -> list[str]:
        """Validate a single step structure."""
        errors: list[str] = []

        if not isinstance(step, dict):
            errors.append(f"Step {index} must be a dictionary")
            return errors

        if "id" not in step:
            errors.append(f"Step {index} missing required field: id")
        if "description" not in step:
            errors.append(f"Step {index} missing required field: description")

        # Validate expected_tools is a list of strings
        expected_tools = step.get("expected_tools", [])
        if not isinstance(expected_tools, list):
            errors.append(f"Step {index}: 'expected_tools' must be a list")
        elif not all(isinstance(t, str) for t in expected_tools):
            errors.append(f"Step {index}: all expected_tools must be strings")

        # Validate hints is a list of strings
        hints = step.get("hints", [])
        if not isinstance(hints, list):
            errors.append(f"Step {index}: 'hints' must be a list")
        elif not all(isinstance(h, str) for h in hints):
            errors.append(f"Step {index}: all hints must be strings")

        return errors


# Convenience function
def load_playbook(source: str | Path | dict[str, Any]) -> Playbook:
    """Load a playbook from various sources.

    Args:
        source: Path to file, JSON string, or dictionary

    Returns:
        Loaded and validated Playbook

    Examples:
        >>> playbook = load_playbook("./playbooks/deploy.yaml")
        >>> playbook = load_playbook({"id": "test", "name": "Test", "steps": []})
    """
    loader = PlaybookLoader()

    if isinstance(source, dict):
        return loader.load_dict(source)

    if isinstance(source, Path):
        return loader.load_file(source)

    # String - could be path or JSON
    source_str = str(source)

    # Check if it's a file path
    path = Path(source_str)
    if path.exists():
        return loader.load_file(path)

    # Try as JSON string
    if source_str.strip().startswith("{"):
        return loader.load_json_string(source_str)

    # Assume it's a non-existent file path
    raise PlaybookLoadError(f"File not found: {source_str}", path=path)
