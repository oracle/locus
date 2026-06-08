# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL

"""Deepagent built-in tools — currently the filesystem-as-memory ops.

Use :func:`make_filesystem_tools` to attach the 6 FS ops to any
agent built with :func:`tulip.create_deepagent`. The factory's
``enable_filesystem=True`` knob is just a convenience that calls
this with a default ``StateBackend()``.
"""

from tulip.deepagent.tools.filesystem import make_filesystem_tools


__all__ = ["make_filesystem_tools"]
