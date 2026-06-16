# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Hosted model providers for Locus.

Providers are platforms that host multiple model families (e.g., OCI GenAI, AWS Bedrock).
Each provider may support different models with different API formats.
"""

from locus.models.providers.oci import OCIAuthType, OCIConfig, OCIModel


__all__ = [
    "OCIModel",
    "OCIConfig",
    "OCIAuthType",
]
