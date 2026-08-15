# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Playbook loading from JSON and YAML files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from tulip.playbooks.models import Playbook


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
        data = _flatten_step_groups(data)
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


def _flatten_step_groups(data: dict[str, Any]) -> dict[str, Any]:
    """Accept the grouped playbook shape as well as the flat one.

    optic writes procedures as ``step_groups`` — an ordered list of groups, each
    with its own goal, each holding ordered steps — and names the capability a
    step is carried out with under ``guidance.skill_refs``. That grouping is how
    a real runbook reads: "establish the blast radius" is a phase containing
    several checks, not a single step.

    The enforcer tracks one flat ordered sequence, so groups are flattened in
    order. Nothing is invented: a group's title/goal is folded into the first
    step's description so the phase it belonged to survives into the record and
    the prompt, and step ids are qualified with the group id where they would
    otherwise collide across groups.

    A dict that already has ``steps`` is returned untouched, so this is additive
    and no existing playbook changes meaning.
    """
    groups = data.get("step_groups")
    if "steps" in data or not isinstance(groups, list):
        return data

    steps: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group in groups:
        if not isinstance(group, dict):
            continue
        group_id = str(group.get("id", "") or "")
        heading = str(group.get("goal") or group.get("title") or "").strip()
        for index, step in enumerate(group.get("steps") or []):
            if not isinstance(step, dict):
                continue
            flat = dict(step)
            step_id = str(flat.get("id", "") or "")
            if not step_id or step_id in seen:
                step_id = f"{group_id}.{step_id}" if group_id else f"step-{len(steps)}"
            seen.add(step_id)
            flat["id"] = step_id

            # `goal` is the step's real instruction; `title` is a label. Prefer
            # the goal, fall back to the title, and keep the group's heading in
            # front of the first step so the phase is not lost.
            description = str(flat.pop("goal", "") or "").strip()
            title = str(flat.pop("title", "") or "").strip()
            body = description or title or str(flat.get("description", "") or "")
            if index == 0 and heading and heading not in body:
                body = f"{heading}\n\n{body}".strip()
            flat["description"] = body

            guidance = flat.pop("guidance", None)
            if isinstance(guidance, dict):
                refs = guidance.get("skill_refs") or []
                if isinstance(refs, list):
                    flat.setdefault("uses", [str(ref) for ref in refs if ref])

            # Fields the grouped shape carries that the flat model does not
            # model as first-class; kept rather than dropped so nothing an
            # author wrote disappears silently.
            for extra in ("priority", "product", "service_type"):
                if extra in flat:
                    flat.setdefault("metadata", {})[extra] = flat.pop(extra)

            steps.append(flat)

    out = {key: value for key, value in data.items() if key != "step_groups"}
    out["steps"] = steps
    # optic titles a playbook and summarises it; this model names and describes
    # one. Same fields, different words — map rather than make an author rename
    # a corpus.
    out.setdefault("name", data.get("title") or data.get("playbook_id") or data.get("id", ""))
    if "description" not in out and data.get("summary"):
        out["description"] = data["summary"]
    # Kept, not modelled: `completion.conclusion_requirements`, `decision_policy`,
    # `mode`, `product`, `service_type`. They travel in metadata so nothing an
    # author wrote is lost, and so the day they ARE modelled the data is already
    # there. `completion` in particular is close to `unresolved_required_steps()`.
    carried = {
        key: data[key]
        for key in ("completion", "decision_policy", "mode", "product", "service_type")
        if key in data
    }
    if carried:
        metadata = dict(out.get("metadata") or {})
        metadata.update(carried)
        out["metadata"] = metadata
    for key in carried:
        out.pop(key, None)
    return out


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
