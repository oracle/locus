# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Image-generation provider protocol + OpenAI implementation."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class ImageResult(BaseModel):
    """One generated image, exposed as either a URL or base64 PNG."""

    prompt: str = Field(description="The prompt the image was generated from")
    url: str | None = Field(default=None, description="Hosted image URL")
    b64_png: str | None = Field(
        default=None,
        description="Base64-encoded PNG when the provider returned bytes",
    )
    revised_prompt: str | None = Field(
        default=None,
        description="The provider's rewritten prompt (DALL-E does this)",
    )

    model_config = {"frozen": True}


@runtime_checkable
class BaseImageGenerationProvider(Protocol):
    """Protocol every image-generation provider must implement."""

    async def generate(
        self,
        prompt: str,
        *,
        size: str = "1024x1024",
        n: int = 1,
        **kwargs: Any,
    ) -> list[ImageResult]:
        """Generate ``n`` images for ``prompt`` and return their refs."""
        ...


class OpenAIImageProvider:
    """Image generation via OpenAI ``images.generate`` (DALL-E / gpt-image).

    Args:
        model: Model id — ``"gpt-image-1"`` (default; OpenAI's current
            image model), or any OCI-hosted equivalent the user wants to
            call through an OpenAI-compatible client. ``"dall-e-3"`` and
            ``"dall-e-2"`` are deprecated by OpenAI; pass them explicitly
            if you still need them.
        api_key: Optional explicit key. Defaults to ``OPENAI_API_KEY``.
        base_url: Optional base URL for OpenAI-compatible endpoints
            (e.g. OCI's ``/openai/v1``).

    The provider lazily imports ``openai`` to keep the package extra
    optional.
    """

    def __init__(
        self,
        model: str = "gpt-image-1",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            import openai

            self._client = openai.AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
            )
        return self._client

    async def generate(
        self,
        prompt: str,
        *,
        size: str = "1024x1024",
        n: int = 1,
        **kwargs: Any,
    ) -> list[ImageResult]:
        client = self._get_client()
        resp = await client.images.generate(
            model=self._model,
            prompt=prompt,
            size=size,
            n=n,
            **kwargs,
        )
        out: list[ImageResult] = []
        for entry in resp.data:
            out.append(
                ImageResult(
                    prompt=prompt,
                    url=getattr(entry, "url", None),
                    b64_png=getattr(entry, "b64_json", None),
                    revised_prompt=getattr(entry, "revised_prompt", None),
                )
            )
        return out


__all__ = ["BaseImageGenerationProvider", "ImageResult", "OpenAIImageProvider"]
