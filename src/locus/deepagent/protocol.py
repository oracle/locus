# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Provider protocol + supporting types for the deep-research surface.

A ``KnowledgeProvider`` is the contract a research domain implements
(Mimir metrics, Oracle DB schemas, Loki streams, …). The deepagent
runtime drives it through:

    discover  →  ground  →  agent.research (with tools)  →  merge_to_row

Anything that can produce a list of items, fetch live evidence per
item, expose research tools, and turn the agent's structured output
into a typed catalog row can be a provider.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class ItemRef(BaseModel):
    """A discoverable research target.

    The ``key`` uniquely identifies the item across runs (used as a
    checkpointer thread id and as a catalog primary key). The
    ``provider`` field tags which provider produced the ref so a
    cross-provider catalog can keep them straight.
    """

    model_config = ConfigDict(extra="ignore")

    name: str = Field(..., description="Human-readable item name (e.g. metric name).")
    provider: str = Field(..., description="Provider id ('mimir', 'database', …).")
    key: str = Field(
        default="",
        description=("Stable cross-run identifier. Defaults to ``f'{provider}:{name}'``."),
    )
    tags: dict[str, str] = Field(default_factory=dict)

    def model_post_init(self, _ctx: Any) -> None:
        if not self.key:
            object.__setattr__(self, "key", f"{self.provider}:{self.name}")


class Grounding(BaseModel):
    """Live evidence the provider gathered about an item.

    Free-form ``payload`` so each provider can stash whatever shape
    its research prompt expects (label values, sample rows, current
    metric values, etc.). ``summary`` is a short string the deepagent
    can splice into the per-item prompt.
    """

    model_config = ConfigDict(extra="ignore")

    summary: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)


class KnowledgeRow(BaseModel):
    """One catalog row — what the agent ultimately produces per item.

    Concrete providers subclass this to add typed fields. The base
    keeps only the cross-provider invariants so a generic catalog
    layer can search / persist across providers without coupling to
    any one schema.
    """

    model_config = ConfigDict(extra="allow")

    name: str
    provider: str
    short_description: str = ""
    long_description: str = ""
    domains: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    embedding: list[float] | None = None
    search_text: str = ""


@runtime_checkable
class KnowledgeProvider(Protocol):
    """The contract a research domain implements.

    The deepagent runtime calls these methods in order:

    1. ``open()`` once per scan
    2. ``discover(query)`` to enumerate ``ItemRef``s
    3. for each item:
       a. ``ground(item)`` for live evidence
       b. agent runs with ``tools_for_agent()`` and submits structured output
       c. ``merge_to_row(item, grounding, research, model_id, prompt_hash)``
    4. ``close()`` at end of scan

    All methods are async. The runtime calls them from its event loop.
    """

    async def open(self) -> None: ...

    async def close(self) -> None: ...

    async def discover(self, query: str | None = None) -> list[ItemRef]: ...

    async def ground(self, item: ItemRef) -> Grounding: ...

    def tools_for_agent(self) -> list[Any]:
        """Locus tools the deepagent should attach for this provider."""
        ...

    def output_schema(self) -> type[BaseModel]:
        """Pydantic schema the deepagent's ``submit_research`` expects."""
        ...

    def merge_to_row(
        self,
        item: ItemRef,
        grounding: Grounding,
        research: dict[str, Any],
        *,
        model_id: str,
        prompt_hash: str,
    ) -> KnowledgeRow: ...
