# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Say what to install when a package of the ``oci`` extra is missing.

The request signer builds on ``requests`` when this package is imported, so
that is checked on import: without it the import would fail on
``No module named 'requests'``, which does not say what to do.

The OCI SDK is imported only when a model signs or sends a request, and the
package imports without it (documentation builds and type checks do that).
:func:`require_oci_sdk` is called where the SDK is first needed.
"""

from __future__ import annotations

from locus.models._extras import require_packages


_WHAT = "The OCI provider"

require_packages(_WHAT, "oci", ("requests",))


def require_oci_sdk() -> None:
    """Raise the install hint when the OCI SDK is not installed."""
    require_packages(_WHAT, "oci", ("oci",))
