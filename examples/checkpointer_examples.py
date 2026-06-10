# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""
Checkpointer Examples - Durable investigation state across backends.

Security investigations run for days and outlive any single process:
shift changes, restarts, crashes. This file shows how to persist an
investigation agent's state with each checkpoint backend so a case
can be picked up exactly where the last analyst (or process) left it.

Run with: uv run python examples/checkpointer_examples.py
"""

import asyncio


# =============================================================================
# Helper Functions
# =============================================================================


def create_sample_state():
    """Create a sample investigation state for testing."""
    from tulip.core.messages import Message, Role
    from tulip.core.state import AgentState

    state = AgentState(
        agent_id="ir-copilot",
        max_iterations=20,
        confidence=0.75,
        metadata={"case_id": "IR-2026-0042", "analyst": "analyst-7"},
    )
    state = state.with_message(
        Message(
            role=Role.USER,
            content="New alert: impossible travel for jdoe — Toronto and Warsaw 40 minutes apart.",
        )
    )
    state = state.with_message(
        Message(
            role=Role.ASSISTANT,
            content="Logged. Pulling jdoe's sign-in history before judging the alert.",
        )
    )
    state = state.with_message(
        Message(
            role=Role.USER,
            content="Also check whether source IP 198.51.100.23 has prior reports.",
        )
    )

    return state


def print_state_summary(state):
    """Print a summary of the investigation state."""
    print(f"  Agent ID: {state.agent_id}")
    print(f"  Messages: {len(state.messages)}")
    print(f"  Confidence: {state.confidence}")
    print(f"  Iteration: {state.iteration}")
    print(f"  Metadata: {state.metadata}")


# =============================================================================
# 1. MemoryCheckpointer - For testing and development
# =============================================================================


async def example_memory_checkpointer():
    """
    MemoryCheckpointer stores state in memory (dictionary).

    Use cases:
    - Unit testing
    - Development/prototyping
    - Short-lived triage sessions
    - Caching layer on top of persistent storage
    """
    print("\n" + "=" * 60)
    print("1. MemoryCheckpointer Example")
    print("=" * 60)

    from tulip.memory.backends import MemoryCheckpointer

    # Create backend
    backend = MemoryCheckpointer()
    print(f"\nBackend: {backend}")

    # Save state
    state = create_sample_state()
    checkpoint_id = await backend.save(state, "case-0042")
    print(f"\nSaved checkpoint: {checkpoint_id}")

    # Load state
    loaded = await backend.load("case-0042")
    print("\nLoaded state:")
    print_state_summary(loaded)

    # Create multiple checkpoints as the investigation firms up
    state = state.with_confidence(0.85)
    await backend.save(state, "case-0042", "checkpoint-v2")

    state = state.with_confidence(0.95)
    await backend.save(state, "case-0042", "checkpoint-v3")

    # List checkpoints
    checkpoints = await backend.list_checkpoints("case-0042")
    print(f"\nAll checkpoints: {checkpoints}")

    # Get thread count
    print(f"Thread IDs: {backend.get_thread_ids()}")
    print(f"Total checkpoints: {backend.get_checkpoint_count()}")


# =============================================================================
# 2. RedisBackend - For distributed/production use
# =============================================================================


async def example_redis_backend():
    """
    RedisBackend stores state in Redis.

    Use cases:
    - Distributed SOC deployments
    - High-performance requirements
    - Session caching
    - Multi-instance triage services

    Requires: redis-py and running Redis server
    """
    print("\n" + "=" * 60)
    print("2. RedisBackend Example")
    print("=" * 60)

    try:
        from tulip.memory.backends import RedisBackend

        # Create backend
        backend = RedisBackend(
            url="redis://localhost:6379",
            prefix="tulip:demo:",
            ttl_seconds=3600,  # Optional: expire after 1 hour
        )
        print("\nConnecting to Redis...")

        # Save state
        state = create_sample_state()
        data = state.to_checkpoint()
        await backend.save("redis-case-0042", data)
        print("Saved checkpoint to Redis")

        # Load state
        loaded = await backend.load("redis-case-0042")
        if loaded:
            print(f"Loaded: {loaded.get('agent_id')}")

        # Check existence
        exists = await backend.exists("redis-case-0042")
        print(f"Exists: {exists}")

        # List threads
        threads = await backend.list_threads()
        print(f"Threads: {threads}")

        # Cleanup
        await backend.delete("redis-case-0042")
        await backend.close()

    except ImportError:
        print("\nSkipping: redis package not installed")
        print("Install with: pip install redis")
    except Exception as e:
        print(f"\nSkipping: {e}")
        print("Ensure Redis is running on localhost:6379")


# =============================================================================
# 3. PostgreSQLBackend - For enterprise/production use
# =============================================================================


async def example_postgresql_backend():
    """
    PostgreSQLBackend stores state in PostgreSQL with JSONB.

    Use cases:
    - Enterprise SOC platforms
    - Complex querying needs (find every case touching one analyst)
    - ACID guarantees required
    - Integration with existing PostgreSQL infrastructure

    Features:
    - JSONB for efficient querying
    - Connection pooling
    - Metadata indexing
    - Full SQL power

    Requires: asyncpg and running PostgreSQL server
    """
    print("\n" + "=" * 60)
    print("3. PostgreSQLBackend Example")
    print("=" * 60)

    try:
        from tulip.memory.backends import PostgreSQLBackend

        # Create backend
        backend = PostgreSQLBackend(
            host="localhost",
            port=5432,
            database="tulip_demo",
            user="postgres",
            password="",
            table_name="agent_checkpoints",
        )
        print("\nConnecting to PostgreSQL...")

        # Or use DSN
        # backend = PostgreSQLBackend(
        #     dsn="postgresql://user:pass@localhost:5432/mydb"
        # )

        # Save with metadata
        state = create_sample_state()
        data = state.to_checkpoint()
        checkpoint_id = await backend.save(
            "pg-case-0042",
            data,
            metadata={"analyst": "analyst-7", "session_type": "investigation"},
        )
        print(f"Saved checkpoint: {checkpoint_id}")

        # Query by metadata
        results = await backend.query_by_metadata("analyst", "analyst-7")
        print(f"Found {len(results)} cases for analyst-7")

        # Search by data field
        results = await backend.search_data("agent_id", "ir-copilot")
        print(f"Found {len(results)} threads with ir-copilot")

        # Get count
        count = await backend.count()
        print(f"Total checkpoints: {count}")

        # Cleanup
        await backend.delete("pg-case-0042")
        await backend.close()

    except ImportError:
        print("\nSkipping: asyncpg package not installed")
        print("Install with: pip install asyncpg")
    except Exception as e:
        print(f"\nSkipping: {e}")
        print("Ensure PostgreSQL is running")


# =============================================================================
# 4. MySQLBackend - For official MySQL deployments
# =============================================================================


async def example_mysql_backend():
    """
    MySQLBackend stores state in MySQL with JSON columns.

    Use cases:
    - Existing MySQL infrastructure
    - Official Connector/Python requirement
    - ACID guarantees required
    - Metadata queries over JSON

    Features:
    - Official mysql-connector-python asyncio driver
    - JSON columns for checkpoint data and metadata
    - Connection pooling
    - MySQL JSON_CONTAINS metadata queries

    Requires: mysql-connector-python and running MySQL server
    """
    print("\n" + "=" * 60)
    print("4. MySQLBackend Example")
    print("=" * 60)

    try:
        from tulip.memory.backends import MySQLBackend

        # Create backend
        backend = MySQLBackend(
            host="localhost",
            port=3306,
            database="tulip_demo",
            user="root",
            password="",
            table_name="agent_checkpoints",
        )
        print("\nConnecting to MySQL...")

        # Or use DSN
        # backend = MySQLBackend(
        #     dsn="mysql://user:pass@localhost:3306/mydb"
        # )

        # Save with metadata
        state = create_sample_state()
        data = state.to_checkpoint()
        checkpoint_id = await backend.save(
            "mysql-case-0042",
            data,
            metadata={"analyst": "analyst-7", "session_type": "investigation"},
        )
        print(f"Saved checkpoint: {checkpoint_id}")

        # Query by metadata
        results = await backend.query_by_metadata("analyst", "analyst-7")
        print(f"Found {len(results)} cases for analyst-7")

        # Search by data field
        results = await backend.search_data("agent_id", "ir-copilot")
        print(f"Found {len(results)} threads with ir-copilot")

        # Get count
        count = await backend.count()
        print(f"Total checkpoints: {count}")

        # Cleanup
        await backend.delete("mysql-case-0042")
        await backend.close()

    except ImportError:
        print("\nSkipping: mysql-connector-python package not installed")
        print("Install with: pip install mysql-connector-python")
    except Exception as e:
        print(f"\nSkipping: {e}")
        print("Ensure MySQL is running")


# =============================================================================
# 5. OpenSearchBackend - For search and analytics
# =============================================================================


async def example_opensearch_backend():
    """
    OpenSearchBackend stores state in OpenSearch.

    Use cases:
    - Full-text search across past investigations
    - Analytics and reporting ("how many cases mentioned this IP?")
    - Log aggregation
    - Complex queries

    Features:
    - Full-text search
    - Metadata filtering
    - Scalable storage
    - Analytics capabilities

    Requires: opensearch-py and running OpenSearch
    """
    print("\n" + "=" * 60)
    print("5. OpenSearchBackend Example")
    print("=" * 60)

    try:
        from tulip.memory.backends import OpenSearchBackend

        # Create backend
        backend = OpenSearchBackend(
            hosts=["localhost:9200"],
            index_name="tulip-demo-checkpoints",
        )
        print("\nConnecting to OpenSearch...")

        # Save with metadata
        state = create_sample_state()
        data = state.to_checkpoint()
        await backend.save(
            "os-case-0042",
            data,
            metadata={"category": "intrusion", "priority": "high"},
        )
        print("Saved checkpoint to OpenSearch")

        # Wait for indexing
        await asyncio.sleep(1)

        # Full-text search across investigation transcripts
        results = await backend.search("impossible travel")
        print(f"Search results: {len(results)}")

        # Query by metadata
        results = await backend.get_by_metadata("category", "intrusion")
        print(f"Category 'intrusion' results: {len(results)}")

        # List threads
        threads = await backend.list_threads()
        print(f"All threads: {threads}")

        # Cleanup
        await backend.delete("os-case-0042")
        await backend.close()

    except ImportError:
        print("\nSkipping: opensearch-py package not installed")
        print("Install with: pip install opensearch-py")
    except Exception as e:
        print(f"\nSkipping: {e}")
        print("Ensure OpenSearch is running on localhost:9200")


# =============================================================================
# 6. S3Backend - For S3 / MinIO / R2 cloud deployments
# =============================================================================


async def example_s3_backend():
    """
    S3Backend stores state in S3-compatible object storage (S3, MinIO, R2).

    Use cases:
    - Cloud deployments
    - Serverless applications
    - Cross-region replication
    - Cost-effective long-term case retention (compliance archives)

    Features:
    - Scalable cloud storage
    - Lifecycle policies
    - Versioning support

    Requires: boto3 and S3 credentials
    """
    print("\n" + "=" * 60)
    print("6. S3Backend Example")
    print("=" * 60)

    try:
        import os

        from tulip.memory.backends import S3Backend

        # Check for S3 credentials
        if not os.environ.get("S3_BUCKET"):
            print("\nSkipping: S3_BUCKET not set")
            return

        # Create backend
        backend = S3Backend(
            bucket=os.environ["S3_BUCKET"],
            endpoint_url=os.environ.get("S3_ENDPOINT_URL") or None,
            prefix="demo/checkpoints/",
        )
        print(f"\nBackend: {backend}")

        # Save with metadata
        state = create_sample_state()
        data = state.to_checkpoint()
        await backend.save(
            "s3-case-0042",
            data,
            metadata={"environment": "ir-demo"},
        )
        print("Saved checkpoint to S3 Object Storage")

        # Load state
        loaded = await backend.load("s3-case-0042")
        if loaded:
            print(f"Loaded agent: {loaded.get('agent_id')}")

        # List with metadata
        results = await backend.list_with_metadata()
        print(f"Threads with metadata: {len(results)}")

        # Cleanup
        await backend.delete("s3-case-0042")

    except ImportError:
        print("\nSkipping: boto3 not installed")
        print("Install with: pip install 'tulip-agents[s3]'")
    except Exception as e:
        print(f"\nSkipping: {e}")


# =============================================================================
# 7. Agent with Checkpointing Example
# =============================================================================


async def example_agent_with_checkpointing():
    """
    Complete example: investigation agent with checkpoint persistence.

    This shows how to integrate checkpointing with an agent so a case
    survives restarts and hand-offs between analysts.
    """
    print("\n" + "=" * 60)
    print("7. Agent with Checkpointing (Full Integration)")
    print("=" * 60)

    from tulip.core.messages import Message, Role
    from tulip.core.state import AgentState
    from tulip.memory.backends import FileCheckpointer, MemoryCheckpointer

    # ==========================================================================
    # Option 1: Using MemoryCheckpointer (for testing)
    # ==========================================================================
    print("\nOption 1: MemoryCheckpointer")
    print("-" * 40)

    memory_checkpointer = MemoryCheckpointer()

    # This checkpointer can be passed directly to Agent:
    # agent = Agent(
    #     model="openai:gpt-4o",
    #     checkpointer=memory_checkpointer,
    #     checkpoint_every_n_iterations=1,
    # )
    # result = agent.run_sync("Open a case for this alert.", thread_id="case-0042")

    # Manual state management for demo
    state = AgentState(agent_id="ir-copilot")
    state = state.with_message(
        Message(role=Role.USER, content="Open a case for the jdoe impossible-travel alert.")
    )
    state = state.with_message(
        Message(role=Role.ASSISTANT, content="Case opened. Gathering sign-in telemetry.")
    )

    await memory_checkpointer.save(state, "case-0042")
    loaded = await memory_checkpointer.load("case-0042")
    print(f"  Saved and loaded state: {len(loaded.messages)} messages")

    # ==========================================================================
    # Option 2: Using FileCheckpointer (simple local persistence)
    # ==========================================================================
    print("\nOption 2: FileCheckpointer")
    print("-" * 40)

    checkpointer = FileCheckpointer(base_dir="/tmp/agent_sessions")

    # Save a state
    checkpoint_id = await checkpointer.save(state, "file-case-0042")
    print(f"  Checkpoint saved: {checkpoint_id[:8]}...")

    # Load it back
    loaded = await checkpointer.load("file-case-0042")
    print(f"  Loaded: {len(loaded.messages)} messages, agent_id={loaded.agent_id}")

    # ==========================================================================
    # Option 3: Production backends (Redis, PostgreSQL, MySQL, S3, etc.)
    # ==========================================================================
    print("\nOption 3: Production Backends")
    print("-" * 40)

    print("  Available factory functions:")
    print("    - redis_checkpointer(url='redis://localhost:6379')")
    print("    - postgresql_checkpointer(host='localhost', database='myapp')")
    print("    - mysql_checkpointer(host='localhost', database='myapp')")
    print("    - opensearch_checkpointer(hosts=['localhost:9200'])")
    print("    - s3_checkpointer(bucket='...', endpoint_url='...')")

    print("\n  Example with Redis:")
    print("    from tulip.memory.backends import redis_checkpointer")
    print("    checkpointer = redis_checkpointer('redis://localhost:6379')")
    print("    agent = Agent(model=model, checkpointer=checkpointer)")

    # ==========================================================================
    # Full Agent Example (with mock model for demo)
    # ==========================================================================
    print("\nFull Agent + Checkpointer Pattern:")
    print("-" * 40)
    print("""
    from tulip.agent import Agent
    from tulip.memory.backends import redis_checkpointer

    # Create checkpointer
    checkpointer = redis_checkpointer("redis://localhost:6379")

    # Create agent with checkpointing
    agent = Agent(
        model="openai:gpt-4o",
        checkpointer=checkpointer,
        checkpoint_every_n_iterations=1,  # Auto-save after each iteration
    )

    # First shift opens the case
    result = agent.run_sync("Triage the jdoe alert.", thread_id="case-0042")

    # Next shift resumes it (different process, same thread_id)
    result = agent.run_sync("Where did we leave this case?", thread_id="case-0042")
    # Agent will load previous state and continue the investigation
    """)


# =============================================================================
# Main
# =============================================================================


async def main():
    """Run all examples."""
    print("=" * 60)
    print("Tulip Checkpointer Examples — durable investigation state")
    print("=" * 60)

    # Run examples
    await example_memory_checkpointer()
    await example_redis_backend()
    await example_postgresql_backend()
    await example_mysql_backend()
    await example_opensearch_backend()
    await example_s3_backend()
    await example_agent_with_checkpointing()

    print("\n" + "=" * 60)
    print("Examples Complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
