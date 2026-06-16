# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Custom warning hierarchy for Locus.

Consumers can opt into treating deprecations as errors during their
own test runs::

    import warnings
    from locus.core.warnings import LocusDeprecationWarning

    warnings.simplefilter("error", LocusDeprecationWarning)

See :doc:`/DEPRECATION` for the deprecation policy.
"""

from __future__ import annotations


class LocusWarning(UserWarning):
    """Root of the Locus warning hierarchy.

    All Locus-originated warnings subclass this so consumers can filter
    or elevate them collectively::

        warnings.simplefilter("error", LocusWarning)
    """


class LocusDeprecationWarning(LocusWarning, DeprecationWarning):
    """API marked for removal.

    Inherits from :class:`DeprecationWarning` so standard warning
    filters (``python -W error::DeprecationWarning``) continue to work,
    and from :class:`LocusWarning` so Locus-specific filters still
    pick it up.
    """


__all__ = ["LocusDeprecationWarning", "LocusWarning"]
