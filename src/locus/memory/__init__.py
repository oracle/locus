# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Memory and state persistence for Locus.

This module provides conversation management, checkpointing, cross-thread storage,
and long-term memory management:

Conversation Management:
- ConversationManager: Base class for conversation strategies
- NullManager: Keep all messages unchanged
- SlidingWindowManager: Keep last N messages
- SummarizingManager: Summarize older messages
- ContextCompactor: Clear old tool output, then summarise, under a token threshold
- TieredCompaction: Tiers in order, each at its own fraction of the window
  (ClearToolResults, SlidingWindow, Summarize, Native, Relevance); pin() keeps
  a message through every tier
- ToolResultOffload: Large tool results to storage by handle; fetch_tool_result
  reads them back

Checkpointing:
- BaseCheckpointer: Abstract base for checkpointer implementations
- DeltaCheckpointer: Efficient delta-based checkpointing (~77% storage savings)
- ToolEffectLedger: Every executed tool call, recorded in the thread's checkpointer
  as it returns, so a run continued after a crash replays instead of re-running
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
- MemoryScope: A run's write namespace plus read-only recall namespaces
- MemoryType: Semantic category (user / feedback / project / reference)

Backends (in locus.memory.backends):
- MemoryCheckpointer: In-memory storage (testing/development)
- FileCheckpointer: Local file storage
- HTTPCheckpointer: Remote HTTP API storage
- RedisBackend, PostgreSQLBackend, MySQLBackend, OpenSearchBackend, S3Backend
"""

from locus.core.protocols import CheckpointerCapabilities
from locus.memory.checkpointer import BaseCheckpointer
from locus.memory.compaction import ContextCompactor, is_pinned, pin
from locus.memory.conversation import (
    ConversationManager,
    NullManager,
    SlidingWindowManager,
    SummarizingManager,
)
from locus.memory.delta import (
    CheckpointMetadata,
    DeltaCheckpoint,
    DeltaCheckpointer,
    DeltaStorage,
    InMemoryDeltaStorage,
)
from locus.memory.ledger import LedgerEntry, LedgerManifest, ToolEffectLedger
from locus.memory.manager import (
    BaseMemoryManager,
    LLMMemoryManager,
    Memory,
    MemoryScope,
    MemoryType,
    NoopMemoryManager,
)
from locus.memory.offload import ToolResultOffload, offload_handle
from locus.memory.registry import (
    get_checkpointer,
    list_checkpointers,
    register_checkpointer,
)
from locus.memory.store import (
    BaseStore,
    InMemoryStore,
    NamespacedStore,
    SemanticSearchResult,
    StoreCapabilities,
    StoreCapabilityError,
    StoreContext,
    StoreItem,
)
from locus.memory.tiers import (
    ClearToolResults,
    CompactionTier,
    Native,
    Relevance,
    SlidingWindow,
    Summarize,
    TieredCompaction,
)


__all__ = [
    # Conversation management
    "ConversationManager",
    "NullManager",
    "SlidingWindowManager",
    "SummarizingManager",
    "ContextCompactor",
    # Tiered compaction
    "TieredCompaction",
    "CompactionTier",
    "ClearToolResults",
    "SlidingWindow",
    "Summarize",
    "Native",
    "Relevance",
    "pin",
    "is_pinned",
    # Tool-result offload
    "ToolResultOffload",
    "offload_handle",
    # Checkpointing
    "BaseCheckpointer",
    "CheckpointerCapabilities",
    "DeltaCheckpointer",
    "DeltaStorage",
    "InMemoryDeltaStorage",
    "DeltaCheckpoint",
    "CheckpointMetadata",
    # Tool-effect ledger
    "ToolEffectLedger",
    "LedgerEntry",
    "LedgerManifest",
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
    "MemoryScope",
    "MemoryType",
    "NoopMemoryManager",
]
