# Checkpoint Backends

The checkpointer contract is backend-agnostic. This notebook drives it
against `S3Backend` — S3 / MinIO / Cloudflare R2 via boto3 — a durable
store with the full capability set (`list_threads`, `search`, `vacuum`)
over a single bucket. Portable SQL deployments can use PostgreSQL or
MySQL through the same adapter shape; key/value deployments can use
Redis. Notebook 06 covers the checkpointer contract itself.

- Save and load `AgentState` via `S3Backend`.
- Inspect the reported capabilities.
- Walk thread history with `list_threads` / `list_checkpoints`.
- Vacuum old checkpoints with `S3Backend.vacuum`.
- Full-text search across stored conversations.

Run it (requires an S3-compatible endpoint + a bucket):

    export S3_ENDPOINT_URL=http://localhost:9000   # MinIO / R2 endpoint
    export S3_BUCKET=tulip-checkpoints
    export AWS_ACCESS_KEY_ID=minioadmin
    export AWS_SECRET_ACCESS_KEY=minioadmin
    python examples/notebook_52_checkpoint_backends.py

Without the env vars the notebook prints what's missing and exits
cleanly so CI stays green. The in-memory checkpointer covered in
[notebook 08](notebook_08_agent_memory.md) is the developer default;
S3 / object storage is a production default.

## Source

```python
--8<-- "examples/notebook_52_checkpoint_backends.py"
```
