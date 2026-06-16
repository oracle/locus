# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Cross-thread persistent store for long-term memory.

The Store provides key-value storage that persists across
different conversation threads, enabling:
- User preferences that persist across sessions
- Learned facts about users/topics
- Cross-conversation context sharing
- Semantic memory with search capabilities

Example:
    from locus.memory.store import InMemoryStore

    store = InMemoryStore()
    graph = builder.compile(store=store)

    async def my_node(inputs, *, store):
        # Get user preferences
        prefs = await store.get(("users", user_id), "preferences")

        # Save new memory
        await store.put(
            ("users", user_id, "memories"),
            "last_topic",
            {"topic": "python", "discussed_at": now}
        )

        # Search for related memories
        related = await store.search(
            ("users", user_id, "memories"),
            query="python programming"
        )
"""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


# =============================================================================
# Capabilities + typed capability error
# =============================================================================


@dataclass(frozen=True)
class StoreCapabilities:
    """
    Capabilities supported by a store implementation.

    Use this to discover what features a store supports.
    """

    search: bool = False  # Full-text search
    semantic_search: bool = False  # Vector/embedding similarity search
    embedding_dimension: int | None = None  # Size of embedding vectors (e.g., 1536 for OpenAI)
    ttl: bool = False  # Time-to-live / auto-expiration
    list_namespaces: bool = False  # List all namespaces
    batch_operations: bool = False  # Batch put/get
    transactions: bool = False  # Atomic transactions


class StoreCapabilityError(NotImplementedError):
    """Raised when a store operation is requested that the backend does not support.

    Carries structured context (``capability``, ``store_class``, ``hint``)
    so callers can branch on the capability or surface a clean message.

    Subclasses :class:`NotImplementedError` so legacy ``except
    NotImplementedError`` blocks continue to catch it — this is strictly
    a richer payload, not a behaviour change.

    Example::

        try:
            results = await store.search(("users", uid), query="...")
        except StoreCapabilityError as exc:
            log.info(
                "store %s does not support %s; falling back",
                exc.store_class,
                exc.capability,
            )
            results = await store.list_keys(("users", uid))
    """

    def __init__(
        self,
        capability: str,
        store_class: str,
        hint: str | None = None,
    ) -> None:
        self.capability = capability
        self.store_class = store_class
        self.hint = hint
        msg = f"{store_class} does not support {capability!r}"
        if hint:
            msg = f"{msg}. {hint}"
        super().__init__(msg)


# =============================================================================
# Store Item
# =============================================================================


@dataclass
class StoreItem:
    """
    An item stored in the store.

    Attributes:
        namespace: The namespace tuple
        key: The key within namespace
        value: The stored value
        metadata: Optional metadata
        created_at: Creation timestamp
        updated_at: Last update timestamp
        version: Version counter for optimistic locking
    """

    namespace: tuple[str, ...]
    key: str
    value: Any
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "namespace": list(self.namespace),
            "key": self.key,
            "value": self.value,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StoreItem:
        """Create from dictionary."""
        return cls(
            namespace=tuple(d["namespace"]),
            key=d["key"],
            value=d["value"],
            metadata=d.get("metadata", {}),
            created_at=datetime.fromisoformat(d["created_at"]),
            updated_at=datetime.fromisoformat(d["updated_at"]),
            version=d.get("version", 1),
        )


@dataclass
class SemanticSearchResult:
    """
    Result from semantic (vector similarity) search.

    Attributes:
        item: The matching store item
        score: Similarity score (0.0 to 1.0, higher is more similar)
        distance: Raw distance metric (interpretation depends on distance type)
    """

    item: StoreItem
    score: float  # Normalized similarity (0-1)
    distance: float | None = None  # Raw distance (cosine, L2, etc.)


# =============================================================================
# Base Store
# =============================================================================


class BaseStore(ABC):
    """
    Abstract base class for store implementations.

    Provides common functionality and defines the interface.
    """

    @property
    def capabilities(self) -> StoreCapabilities:
        """Return capabilities. Override in subclasses."""
        return StoreCapabilities()

    @abstractmethod
    async def put(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store a value."""
        ...

    @abstractmethod
    async def get(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> Any | None:
        """Retrieve a value."""
        ...

    @abstractmethod
    async def delete(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> bool:
        """Delete a value."""
        ...

    @abstractmethod
    async def list_keys(
        self,
        namespace: tuple[str, ...],
        limit: int = 100,
    ) -> list[str]:
        """List keys in namespace."""
        ...

    # Optional methods with default implementations

    async def exists(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> bool:
        """Check if a key exists."""
        value = await self.get(namespace, key)
        return value is not None

    async def get_item(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> StoreItem | None:
        """Get full item with metadata."""
        # Default implementation - subclasses can override
        value = await self.get(namespace, key)
        if value is None:
            return None
        now = datetime.now(UTC)
        return StoreItem(
            namespace=namespace,
            key=key,
            value=value,
            metadata={},
            created_at=now,
            updated_at=now,
        )

    async def search(
        self,
        namespace: tuple[str, ...],
        query: str | None = None,
        limit: int = 10,
    ) -> list[StoreItem]:
        """
        Search for items in namespace.

        Optional method — backends advertise support via
        ``capabilities.search``. Subclasses that support search must
        override this method; the default raises
        :class:`StoreCapabilityError`.

        Args:
            namespace: Namespace to search in
            query: Search query (implementation-specific)
            limit: Maximum results

        Returns:
            List of matching items

        Raises:
            StoreCapabilityError: If the backend does not support search.
        """
        raise StoreCapabilityError(
            capability="search",
            store_class=type(self).__name__,
            hint=(
                "Choose a backend that advertises capabilities.search "
                "(e.g., InMemoryStore, OpenSearch, Oracle 26ai)."
            ),
        )

    # -------------------------------------------------------------------------
    # Semantic Search Methods (Vector/Embedding-based)
    # -------------------------------------------------------------------------

    async def put_with_embedding(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: Any,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Store a value with its embedding vector for semantic search.

        Optional method — backends advertise support via
        ``capabilities.semantic_search``. Subclasses that support
        semantic search must override this method.

        Args:
            namespace: Hierarchical namespace tuple
            key: Key within the namespace
            value: Value to store (must be JSON-serializable)
            embedding: Vector embedding (e.g., from OpenAI, Cohere, etc.)
            metadata: Optional metadata for filtering

        Raises:
            StoreCapabilityError: If the backend does not support
                semantic search.

        Example:
            # Get embedding from your provider
            embedding = await embedder.embed("User prefers dark theme")

            await store.put_with_embedding(
                ("users", user_id, "memories"),
                "theme_preference",
                {"theme": "dark", "reason": "easier on eyes"},
                embedding=embedding,
                metadata={"category": "preferences"}
            )
        """
        raise StoreCapabilityError(
            capability="semantic_search",
            store_class=type(self).__name__,
            hint="Use put() for plain storage, or pick a vector-capable backend.",
        )

    async def search_by_embedding(
        self,
        namespace: tuple[str, ...],
        query_embedding: list[float],
        limit: int = 10,
        threshold: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SemanticSearchResult]:
        """
        Search for similar items using vector similarity.

        Optional method — see ``capabilities.semantic_search``.

        Args:
            namespace: Namespace to search in
            query_embedding: Vector to compare against stored embeddings
            limit: Maximum results to return
            threshold: Minimum similarity score (0.0-1.0), or None for no threshold
            metadata_filter: Optional metadata constraints

        Returns:
            List of SemanticSearchResult, sorted by similarity (highest first)

        Raises:
            StoreCapabilityError: If the backend does not support
                semantic search.
        """
        raise StoreCapabilityError(
            capability="semantic_search",
            store_class=type(self).__name__,
            hint="Pick a vector-capable backend (e.g., Oracle 26ai, pgvector).",
        )

    async def get_embedding(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> list[float] | None:
        """
        Get the embedding vector for a stored item.

        Optional method — see ``capabilities.semantic_search``.

        Args:
            namespace: Hierarchical namespace tuple
            key: Key within the namespace

        Returns:
            Embedding vector or None if not found / no embedding stored.

        Raises:
            StoreCapabilityError: If the backend does not support
                semantic search.
        """
        raise StoreCapabilityError(
            capability="semantic_search",
            store_class=type(self).__name__,
        )

    async def put_batch(
        self,
        items: list[tuple[tuple[str, ...], str, Any, dict[str, Any] | None]],
    ) -> None:
        """
        Store multiple items.

        Backends that advertise ``capabilities.batch_operations`` should
        override this method to dispatch atomically. The default falls
        back to sequential ``put()`` calls so callers always get
        functional batching, just without atomicity guarantees.

        Args:
            items: List of (namespace, key, value, metadata) tuples
        """
        # Sequential fallback — works for every backend, even those that
        # don't advertise batch_operations. Backends that *do* advertise
        # the capability override this method to dispatch in one
        # transaction / round-trip.
        for namespace, key, value, metadata in items:
            await self.put(namespace, key, value, metadata)

    async def get_batch(
        self,
        keys: list[tuple[tuple[str, ...], str]],
    ) -> dict[tuple[tuple[str, ...], str], Any]:
        """
        Retrieve multiple items.

        Args:
            keys: List of (namespace, key) tuples

        Returns:
            Dict mapping (namespace, key) to value
        """
        results = {}
        for namespace, key in keys:
            value = await self.get(namespace, key)
            if value is not None:
                results[(namespace, key)] = value
        return results

    async def list_namespaces(
        self,
        prefix: tuple[str, ...] | None = None,
        limit: int = 100,
    ) -> list[tuple[str, ...]]:
        """
        List all namespaces.

        Optional method — see ``capabilities.list_namespaces``.

        Args:
            prefix: Optional prefix to filter namespaces
            limit: Maximum namespaces to return

        Returns:
            List of namespace tuples

        Raises:
            StoreCapabilityError: If the backend does not support
                namespace listing.
        """
        raise StoreCapabilityError(
            capability="list_namespaces",
            store_class=type(self).__name__,
        )

    async def clear_namespace(
        self,
        namespace: tuple[str, ...],
    ) -> int:
        """
        Delete all items in a namespace.

        Args:
            namespace: Namespace to clear

        Returns:
            Number of items deleted
        """
        keys = await self.list_keys(namespace, limit=10000)
        count = 0
        for key in keys:
            if await self.delete(namespace, key):
                count += 1
        return count

    async def close(self) -> None:
        """Close any resources."""

    def _make_storage_key(self, namespace: tuple[str, ...], key: str) -> str:
        """Create internal storage key from namespace and key."""
        ns_str = "/".join(namespace)
        return f"{ns_str}:{key}"


# =============================================================================
# In-Memory Store
# =============================================================================


class InMemoryStore(BaseStore):
    """
    In-memory store implementation.

    Fast but not persistent - data is lost when process exits.
    Useful for testing and development.
    """

    def __init__(self) -> None:
        self._data: dict[str, StoreItem] = {}
        self._namespaces: set[tuple[str, ...]] = set()
        self._lock = asyncio.Lock()

    @property
    def capabilities(self) -> StoreCapabilities:
        return StoreCapabilities(
            search=True,  # Basic substring search
            list_namespaces=True,
            batch_operations=True,
        )

    async def put(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        async with self._lock:
            storage_key = self._make_storage_key(namespace, key)
            now = datetime.now(UTC)

            existing = self._data.get(storage_key)
            if existing:
                # Update existing
                self._data[storage_key] = StoreItem(
                    namespace=namespace,
                    key=key,
                    value=value,
                    metadata=metadata or {},
                    created_at=existing.created_at,
                    updated_at=now,
                    version=existing.version + 1,
                )
            else:
                # Create new
                self._data[storage_key] = StoreItem(
                    namespace=namespace,
                    key=key,
                    value=value,
                    metadata=metadata or {},
                    created_at=now,
                    updated_at=now,
                    version=1,
                )

            self._namespaces.add(namespace)

    async def get(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> Any | None:
        storage_key = self._make_storage_key(namespace, key)
        item = self._data.get(storage_key)
        return item.value if item else None

    async def get_item(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> StoreItem | None:
        storage_key = self._make_storage_key(namespace, key)
        return self._data.get(storage_key)

    async def delete(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> bool:
        async with self._lock:
            storage_key = self._make_storage_key(namespace, key)
            if storage_key in self._data:
                del self._data[storage_key]
                return True
            return False

    async def list_keys(
        self,
        namespace: tuple[str, ...],
        limit: int = 100,
    ) -> list[str]:
        prefix = self._make_storage_key(namespace, "")
        keys = []
        for storage_key, item in self._data.items():
            if storage_key.startswith(prefix):
                keys.append(item.key)
                if len(keys) >= limit:
                    break
        return keys

    async def search(
        self,
        namespace: tuple[str, ...],
        query: str | None = None,
        limit: int = 10,
    ) -> list[StoreItem]:
        prefix = self._make_storage_key(namespace, "")
        results = []

        for storage_key, item in self._data.items():
            if not storage_key.startswith(prefix):
                continue

            if query:
                # Simple substring search in value and metadata
                value_str = json.dumps(item.value) if item.value else ""
                meta_str = json.dumps(item.metadata) if item.metadata else ""
                if query.lower() not in (value_str + meta_str).lower():
                    continue

            results.append(item)
            if len(results) >= limit:
                break

        return results

    async def list_namespaces(
        self,
        prefix: tuple[str, ...] | None = None,
        limit: int = 100,
    ) -> list[tuple[str, ...]]:
        results = []
        for ns in self._namespaces:
            if prefix is None or ns[: len(prefix)] == prefix:
                results.append(ns)
                if len(results) >= limit:
                    break
        return results

    async def put_batch(
        self,
        items: list[tuple[tuple[str, ...], str, Any, dict[str, Any] | None]],
    ) -> None:
        async with self._lock:
            now = datetime.now(UTC)
            for namespace, key, value, metadata in items:
                storage_key = self._make_storage_key(namespace, key)
                existing = self._data.get(storage_key)
                if existing:
                    self._data[storage_key] = StoreItem(
                        namespace=namespace,
                        key=key,
                        value=value,
                        metadata=metadata or {},
                        created_at=existing.created_at,
                        updated_at=now,
                        version=existing.version + 1,
                    )
                else:
                    self._data[storage_key] = StoreItem(
                        namespace=namespace,
                        key=key,
                        value=value,
                        metadata=metadata or {},
                        created_at=now,
                        updated_at=now,
                        version=1,
                    )
                self._namespaces.add(namespace)


# =============================================================================
# Namespaced Store Wrapper
# =============================================================================


class NamespacedStore:
    """
    Store wrapper with a fixed namespace prefix.

    Makes it easier to work with a specific scope.

    Example:
        user_store = NamespacedStore(store, ("users", user_id))
        await user_store.put("preferences", {"theme": "dark"})
        prefs = await user_store.get("preferences")
    """

    def __init__(
        self,
        store: BaseStore,
        namespace: tuple[str, ...],
    ):
        self._store = store
        self._namespace = namespace

    @property
    def namespace(self) -> tuple[str, ...]:
        return self._namespace

    def scoped(self, *suffix: str) -> NamespacedStore:
        """Create new wrapper with extended namespace."""
        return NamespacedStore(self._store, (*self._namespace, *suffix))

    async def put(
        self,
        key: str,
        value: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        await self._store.put(self._namespace, key, value, metadata)

    async def get(self, key: str) -> Any | None:
        return await self._store.get(self._namespace, key)

    async def delete(self, key: str) -> bool:
        return await self._store.delete(self._namespace, key)

    async def list_keys(self, limit: int = 100) -> list[str]:
        return await self._store.list_keys(self._namespace, limit)

    async def exists(self, key: str) -> bool:
        return await self._store.exists(self._namespace, key)

    async def search(
        self,
        query: str | None = None,
        limit: int = 10,
    ) -> list[StoreItem]:
        return await self._store.search(self._namespace, query, limit)

    async def clear(self) -> int:
        return await self._store.clear_namespace(self._namespace)


# =============================================================================
# Store Context for Node Injection
# =============================================================================


class StoreContext:
    """
    Context object passed to nodes for store access.

    Provides convenient methods for common memory operations.

    Example:
        async def my_node(inputs, *, store: StoreContext):
            # Get user memory
            user_prefs = await store.get_user_memory("preferences")

            # Remember something
            await store.remember(
                "discussed_topic",
                {"topic": "python", "timestamp": now}
            )

            # Search memories
            related = await store.search_memories("python")
    """

    def __init__(
        self,
        store: BaseStore,
        user_id: str | None = None,
        session_id: str | None = None,
    ):
        self._store = store
        self._user_id = user_id
        self._session_id = session_id

    @property
    def store(self) -> BaseStore:
        """Access the underlying store."""
        return self._store

    def for_user(self, user_id: str | None = None) -> NamespacedStore:
        """Get namespaced store for a user.

        If user_id is not provided, uses the user_id from the context.
        """
        uid = user_id or self._user_id
        if not uid:
            raise ValueError("user_id must be provided or set in context")
        return NamespacedStore(self._store, ("users", uid))

    def for_session(self, session_id: str | None = None) -> NamespacedStore:
        """Get namespaced store for a session.

        If session_id is not provided, uses the session_id from the context.
        """
        sid = session_id or self._session_id
        if not sid:
            raise ValueError("session_id must be provided or set in context")
        return NamespacedStore(self._store, ("sessions", sid))

    async def get_user_memory(self, key: str) -> Any | None:
        """Get a memory for the current user."""
        if not self._user_id:
            return None
        return await self._store.get(("users", self._user_id, "memories"), key)

    async def remember(
        self,
        key: str,
        value: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store a memory for the current user."""
        if not self._user_id:
            raise ValueError("No user_id set in store context")
        await self._store.put(
            ("users", self._user_id, "memories"),
            key,
            value,
            metadata,
        )

    async def forget(self, key: str) -> bool:
        """Delete a memory for the current user."""
        if not self._user_id:
            return False
        return await self._store.delete(("users", self._user_id, "memories"), key)

    async def search_memories(
        self,
        query: str,
        limit: int = 10,
    ) -> list[StoreItem]:
        """Search user memories (full-text)."""
        if not self._user_id:
            return []
        return await self._store.search(
            ("users", self._user_id, "memories"),
            query,
            limit,
        )

    # -------------------------------------------------------------------------
    # Semantic Memory (Vector/Embedding-based)
    # -------------------------------------------------------------------------

    async def remember_with_embedding(
        self,
        key: str,
        value: Any,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Store a memory with its embedding for semantic search.

        Requires: capabilities.semantic_search = True

        Args:
            key: Memory key
            value: Value to store
            embedding: Vector embedding for semantic search
            metadata: Optional metadata

        Example:
            embedding = await embedder.embed("User likes dark theme")
            await store.remember_with_embedding(
                "theme_pref",
                {"theme": "dark"},
                embedding=embedding,
            )
        """
        if not self._user_id:
            raise ValueError("No user_id set in store context")
        await self._store.put_with_embedding(
            ("users", self._user_id, "memories"),
            key,
            value,
            embedding,
            metadata,
        )

    async def search_memories_semantic(
        self,
        query_embedding: list[float],
        limit: int = 10,
        threshold: float | None = None,
    ) -> list[SemanticSearchResult]:
        """
        Search user memories by semantic similarity.

        Requires: capabilities.semantic_search = True

        Args:
            query_embedding: Vector to search with
            limit: Maximum results
            threshold: Minimum similarity score (0.0-1.0)

        Returns:
            List of SemanticSearchResult sorted by similarity

        Example:
            query_vec = await embedder.embed("user preferences")
            results = await store.search_memories_semantic(
                query_embedding=query_vec,
                limit=5,
                threshold=0.7,
            )
            for r in results:
                print(f"{r.item.key}: {r.score:.2f} similar")
        """
        if not self._user_id:
            return []
        return await self._store.search_by_embedding(
            ("users", self._user_id, "memories"),
            query_embedding,
            limit,
            threshold,
        )

    async def get_global(self, key: str) -> Any | None:
        """Get a global value."""
        return await self._store.get(("global",), key)

    async def set_global(
        self,
        key: str,
        value: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Set a global value."""
        await self._store.put(("global",), key, value, metadata)
