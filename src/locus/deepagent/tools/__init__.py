# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Deepagent built-in tools — currently the filesystem-as-memory ops.

Use :func:`make_filesystem_tools` to attach the 6 FS ops to any
agent built with :func:`locus.create_deepagent`. The factory's
``enable_filesystem=True`` knob is just a convenience that calls
this with a default ``StateBackend()``.
"""

from locus.deepagent.tools.filesystem import make_filesystem_tools


__all__ = ["make_filesystem_tools"]
