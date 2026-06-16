# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Web-search provider protocol + concrete implementations.

The protocol is deliberately minimal: ``async search(query, max_results)``
returns a list of :class:`SearchResult`. Implementations live behind
optional dependencies so locus stays a small core.

Built-in implementations
------------------------

- :class:`OpenAISearchPreviewProvider` — uses OpenAI's
  ``gpt-4o-search-preview`` model. The model performs the search itself
  and returns annotations with URLs + snippets; we parse those into
  ``SearchResult`` objects. Requires the ``openai`` extra.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from locus.providers.types import SearchResult


if TYPE_CHECKING:
    from locus.models.native.openai import OpenAIModel


@runtime_checkable
class BaseWebSearchProvider(Protocol):
    """Protocol every web-search provider must implement."""

    async def search(
        self,
        query: str,
        *,
        max_results: int = 5,
    ) -> list[SearchResult]:
        """Return ``max_results`` (or fewer) search hits for ``query``."""
        ...


class OpenAISearchPreviewProvider:
    """Web search via OpenAI's ``gpt-4o-search-preview`` model.

    The search-preview chat completions return annotated URLs + snippets
    inline; we ask the model to return a strict JSON list and parse it.

    Args:
        model: An :class:`OpenAIModel` instance. The caller picks the
            search-capable model id (e.g. ``"gpt-4o-search-preview"``).
        max_chars_per_snippet: Cap each snippet to this length so the
            agent context doesn't blow up.
    """

    def __init__(
        self,
        model: OpenAIModel,
        *,
        max_chars_per_snippet: int = 500,
    ) -> None:
        self._model = model
        self._cap = max_chars_per_snippet

    async def search(
        self,
        query: str,
        *,
        max_results: int = 5,
    ) -> list[SearchResult]:
        # Importing here keeps the file lazy on the openai dep at
        # module-load time — locus core stays import-free of openai.
        from pydantic import BaseModel, Field

        from locus.core.messages import Message
        from locus.core.structured import parse_structured

        class _Hit(BaseModel):
            title: str
            url: str
            snippet: str = Field(default="")

        class _Hits(BaseModel):
            results: list[_Hit]

        messages = [
            Message.system(
                "You are a web search engine. For the user's query, return "
                f"the top {max_results} results as a JSON object matching "
                "the schema. Each entry must include a real URL. Snippets "
                f"must be at most {self._cap} characters."
            ),
            Message.user(query),
        ]
        response = await self._model.complete(
            messages=messages,
            tools=None,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "Hits",
                    "schema": _Hits.model_json_schema(),
                    "strict": False,
                },
            },
        )
        parsed = parse_structured(response.message.content or "{}", _Hits, strict=False)
        if not parsed.success or parsed.parsed is None:
            return []
        hits: _Hits = parsed.parsed  # type: ignore[assignment]
        out: list[SearchResult] = []
        for h in hits.results[:max_results]:
            out.append(
                SearchResult(
                    title=h.title,
                    url=h.url,
                    snippet=(h.snippet or "")[: self._cap],
                )
            )
        return out


__all__: list[str] = ["BaseWebSearchProvider", "OpenAISearchPreviewProvider"]


# Type-only re-export marker so IDEs surface the protocol on hover.
_: Any = BaseWebSearchProvider
