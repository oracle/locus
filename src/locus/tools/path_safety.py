# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Filesystem path-traversal guard for tools that open user-supplied paths.

Locus does not ship built-in filesystem tools by default, but users
frequently author their own (``@tool def read_file(path: str) -> str:
...``). Any such tool that joins a model-supplied path to a trusted
base directory is vulnerable to path-traversal via ``../../etc/passwd``,
URL-encoded variants (``%2e%2e``), symlink indirection, or absolute
paths. This helper collapses all of those into one canonical
"resolve-and-contain" check so user tools can stay focused on I/O.

Typical use::

    from pathlib import Path
    from locus.tools.path_safety import safe_resolve
    from locus.tools.decorator import tool

    ALLOWED_ROOT = Path("/srv/workspace").resolve()


    @tool
    def read_file(relative_path: str) -> str:
        target = safe_resolve(ALLOWED_ROOT, relative_path)
        return target.read_text()

Guarantees:

* The returned path is fully resolved (``Path.resolve(strict=False)``),
  so symlinks inside it are followed and normalised. If the resolved
  target escapes ``base``, :class:`locus.core.errors.ValidationError`
  is raised.
* Absolute user paths that happen to coincide with ``base`` are
  accepted; any other absolute path is rejected.
* Missing intermediate components are tolerated — the caller is
  responsible for asserting the target exists (``.exists()``) if that
  matters. This matches ``open()``'s semantics and avoids a
  second-roundtrip race condition.

Not handled here (out of scope):

* TOCTOU between the ``safe_resolve`` call and the actual ``open``;
  a concurrent symlink swap can still redirect I/O. Mitigate with
  ``O_NOFOLLOW`` or a chroot / namespace.
* Windows reserved names (``CON``, ``PRN``, …) — not a path-traversal
  concern, but worth knowing.
"""

from __future__ import annotations

from pathlib import Path

from locus.core.errors import ValidationError


__all__ = ["safe_resolve"]


def safe_resolve(base: Path | str, user_path: str) -> Path:
    """Resolve ``user_path`` under ``base`` and confirm containment.

    Args:
        base: The trusted root directory. Must be a real, absolute
            path. If a relative ``Path`` or ``str`` is passed it is
            resolved first — the caller is expected to pass a location
            they control.
        user_path: The untrusted, model- or user-supplied path. May be
            relative, contain ``..`` components, or be absolute. Will
            be rejected if the final resolved location lies outside
            ``base``.

    Returns:
        A fully resolved :class:`~pathlib.Path` inside ``base``.

    Raises:
        ValidationError: The resolved target is outside ``base``, or
            the input is not a string (tool schemas may pass through
            bytes or ``None``).
    """
    if not isinstance(user_path, str):
        raise ValidationError(f"path must be a string, got {type(user_path).__name__}")

    base_resolved = Path(base).resolve(strict=False)
    # ``Path("/abs") / "/other"`` silently drops ``/abs`` on POSIX and
    # ``Path / "abs"`` on Windows. Guard explicitly so absolute user
    # paths either equal ``base`` or are rejected.
    candidate = (base_resolved / user_path).resolve(strict=False)

    if candidate == base_resolved:
        return candidate
    try:
        candidate.relative_to(base_resolved)
    except ValueError as exc:
        raise ValidationError(
            f"path {user_path!r} resolves outside the allowed base directory"
        ) from exc
    return candidate
