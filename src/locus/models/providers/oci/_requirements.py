# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Check, on import, that the packages of the ``oci`` extra are installed.

Every OCI transport signs its requests with the OCI SDK, and the HTTP ones
sign through ``requests``. Without them nothing in this package can make a
call, so importing it says what to install instead of failing on whichever
import comes first (``No module named 'requests'``).
"""

from __future__ import annotations

from locus.models._extras import require_packages


require_packages("The OCI provider", "oci", ("oci", "requests"))
