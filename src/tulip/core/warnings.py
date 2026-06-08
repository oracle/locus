# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL

"""Custom warning hierarchy for Tulip.

Consumers can opt into treating deprecations as errors during their
own test runs::

    import warnings
    from tulip.core.warnings import TulipDeprecationWarning

    warnings.simplefilter("error", TulipDeprecationWarning)

See :doc:`/DEPRECATION` for the deprecation policy.
"""

from __future__ import annotations


class TulipWarning(UserWarning):
    """Root of the Tulip warning hierarchy.

    All Tulip-originated warnings subclass this so consumers can filter
    or elevate them collectively::

        warnings.simplefilter("error", TulipWarning)
    """


class TulipDeprecationWarning(TulipWarning, DeprecationWarning):
    """API marked for removal.

    Inherits from :class:`DeprecationWarning` so standard warning
    filters (``python -W error::DeprecationWarning``) continue to work,
    and from :class:`TulipWarning` so Tulip-specific filters still
    pick it up.
    """


__all__ = ["TulipDeprecationWarning", "TulipWarning"]
