# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL

"""Common Pydantic types used across multi-modal providers."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """One result from a web-search provider."""

    title: str = Field(description="Page title")
    url: str = Field(description="Canonical URL")
    snippet: str = Field(default="", description="Short excerpt")
    score: float | None = Field(
        default=None,
        description="Provider relevance score (0..1) when available",
    )

    model_config = {"frozen": True}


class WebPage(BaseModel):
    """A fetched web page, normalized to text."""

    url: str = Field(description="Final URL after any redirects")
    status: int = Field(description="HTTP status code")
    title: str = Field(default="", description="HTML <title>, when present")
    text: str = Field(default="", description="Cleaned page text")
    html: str | None = Field(
        default=None,
        description="Raw HTML (only kept when explicitly requested)",
    )
    truncated: bool = Field(
        default=False,
        description="True when text was capped at the requested max_chars",
    )

    model_config = {"frozen": True}
