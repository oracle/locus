# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Memory and state persistence for Tulip.

This module provides conversation management, checkpointing, cross-thread storage,
and long-term memory management:

Conversation Management:
- ConversationManager: Base class for conversation strategies
- NullManager: Keep all messages unchanged
- SlidingWindowManager: Keep last N messages
- SummarizingManager: Summarize older messages

Checkpointing:
- BaseCheckpointer: Abstract base for checkpointer implementations
- DeltaCheckpointer: Efficient delta-based checkpointing (~77% storage savings)
- get_checkpointer: Get a checkpointer by string identifier
- register_checkpointer: Register a custom checkpointer provider
- list_checkpointers: List available checkpointer providers

Cross-Thread Store (Long-term Memory):
- BaseStore: Abstract base for store implementations
- InMemoryStore: In-memory store (testing/development)
- NamespacedStore: Scoped store wrapper
- StoreContext: Convenient store access for nodes

Long-term Memory Manager:
- BaseMemoryManager: Abstract base for memory managers
- NoopMemoryManager: Pass-through (no storage) for testing
- LLMMemoryManager: LLM-backed extraction, persists via any BaseStore backend
- Memory: A single durable memory entry
- MemoryType: Semantic category (user / feedback / project / reference)

Backends (in tulip.memory.backends):
- MemoryCheckpointer: In-memory storage (testing/development)
- FileCheckpointer: Local file storage
- HTTPCheckpointer: Remote HTTP API storage
- RedisBackend, PostgreSQLBackend, MySQLBackend, OpenSearchBackend, S3Backend
"""

from tulip.core.protocols import CheckpointerCapabilities
from tulip.memory.checkpointer import BaseCheckpointer
from tulip.memory.conversation import (
    ConversationManager,
    NullManager,
    SlidingWindowManager,
    SummarizingManager,
)
from tulip.memory.delta import (
    CheckpointMetadata,
    DeltaCheckpoint,
    DeltaCheckpointer,
    DeltaStorage,
    InMemoryDeltaStorage,
)
from tulip.memory.manager import (
    BaseMemoryManager,
    LLMMemoryManager,
    Memory,
    MemoryType,
    NoopMemoryManager,
)
from tulip.memory.registry import (
    get_checkpointer,
    list_checkpointers,
    register_checkpointer,
)
from tulip.memory.store import (
    BaseStore,
    InMemoryStore,
    NamespacedStore,
    SemanticSearchResult,
    StoreCapabilities,
    StoreCapabilityError,
    StoreContext,
    StoreItem,
)


__all__ = [
    # Conversation management
    "ConversationManager",
    "NullManager",
    "SlidingWindowManager",
    "SummarizingManager",
    # Checkpointing
    "BaseCheckpointer",
    "CheckpointerCapabilities",
    "DeltaCheckpointer",
    "DeltaStorage",
    "InMemoryDeltaStorage",
    "DeltaCheckpoint",
    "CheckpointMetadata",
    # Registry
    "get_checkpointer",
    "register_checkpointer",
    "list_checkpointers",
    # Cross-Thread Store
    "BaseStore",
    "InMemoryStore",
    "NamespacedStore",
    "SemanticSearchResult",
    "StoreCapabilities",
    "StoreCapabilityError",
    "StoreContext",
    "StoreItem",
    # Long-term Memory Manager
    "BaseMemoryManager",
    "LLMMemoryManager",
    "Memory",
    "MemoryType",
    "NoopMemoryManager",
]
