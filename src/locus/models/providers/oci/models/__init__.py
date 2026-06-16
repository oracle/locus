# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""OCI GenAI model-specific providers."""

from locus.models.providers.oci.models.cohere import CohereProvider
from locus.models.providers.oci.models.generic import GenericProvider


__all__ = [
    "CohereProvider",
    "GenericProvider",
]
