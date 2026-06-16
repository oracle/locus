#!/usr/bin/env python3
# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Locus port of the Deep Research + OCI Object Storage gist.

Mirrors `gist 00cb5682/deep_research_oci_storage.py`:
- defines three `@tool` functions wrapping the OCI Object Storage SDK
  (`list_bucket_objects`, `read_bucket_object`, `search_bucket_data`)
- wires them into `create_deepagent(...)` — no vector datastore, the
  tools themselves are the retrieval path
- runs a research query that requires listing + reading + (optionally)
  searching for snippets across pre-populated buckets in OCI Object
  Storage.

Locus equivalents:
- `from langchain_core.tools import tool`         -> `from locus.tools import tool`
- `from langchain_oci.agents
    import create_deep_research_agent`            -> `from locus.deepagent import create_deepagent`
- `from langchain_core.messages import HumanMessage` -> plain str via `agent.run_sync(...)`
- `from oci.object_storage import ObjectStorageClient`  -> same (this is the raw OCI SDK)

Buckets read (from the DEFAULT tenancy, namespace <your-os-namespace>):
- deep-research-medical : medmcqa_sample.json, pubmedqa_sample.json
- deep-research-legal   : cuad_contracts_sample.json
- deep-research-large   : (optional) Wikipedia / C4 / ArXiv samples

Run:
    export OCI_PROFILE=DEFAULT         # api_key, ca-toronto-1
    export OCI_GENAI_PROFILE=DEFAULT   # api_key, has Gemini/OpenAI in us-chicago-1
    export OCI_GENAI_COMPARTMENT=ocid1.tenancy.oc1..<your-tenancy>
    export OCI_OS_NAMESPACE=<your-os-namespace>
    .venv/bin/python examples/projects/deep-research/demo_object_storage.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from oci.config import from_file
from oci.exceptions import ServiceError
from oci.object_storage import ObjectStorageClient

from locus.deepagent import create_deepagent
from locus.models import get_model
from locus.tools import tool


def _format_storage_error(error: ServiceError) -> str:
    code = getattr(error, "code", "UnknownError")
    status = getattr(error, "status", "unknown")
    message = str(getattr(error, "message", "")).strip()
    if code == "ObjectNotFound":
        return (
            "Object not found in bucket. Verify object name and prefix with "
            "list_bucket_objects before reading."
        )
    if code == "BucketNotFound":
        return "Bucket not found. Verify bucket name and OCI namespace/region."
    return f"OCI Object Storage error ({status}/{code}): {message}"


def build_storage_tools(
    *,
    namespace: str,
    buckets: list[str],
    region: str,
    auth_profile: str,
    auth_file_location: str = "~/.oci/config",
) -> list:
    """Return three @tool callables for list/read/search of bucket objects.

    Same contract as the langchain gist; only the `tool` decorator's
    origin changes (locus vs langchain_core).
    """
    config = from_file(
        file_location=str(Path(auth_file_location).expanduser()),
        profile_name=auth_profile,
    )
    config["region"] = region
    client = ObjectStorageClient(config)
    allowed = set(buckets)

    @tool
    def list_bucket_objects(bucket: str, prefix: str = "", limit: int = 50) -> str:
        """List object names in a bucket, optionally filtered by prefix.

        Args:
            bucket: The OCI Object Storage bucket name.
            prefix: Optional prefix filter for object names.
            limit: Max objects to return (1-100).
        """
        if bucket not in allowed:
            return f"Bucket '{bucket}' not allowed. Allowed: {sorted(allowed)}"
        try:
            response = client.list_objects(
                namespace_name=namespace,
                bucket_name=bucket,
                prefix=prefix or None,
                limit=min(max(limit, 1), 100),
            )
        except ServiceError as error:
            return _format_storage_error(error)
        objects = response.data.objects or []
        if not objects:
            return f"No objects found in bucket '{bucket}' (prefix='{prefix}')."
        return "\n".join(f"{obj.name} ({obj.size} bytes)" for obj in objects)

    @tool
    def read_bucket_object(bucket: str, object_name: str, max_chars: int = 8000) -> str:
        """Read and return text content from a bucket object.

        Args:
            bucket: The OCI Object Storage bucket name.
            object_name: Exact object name (use list_bucket_objects first).
            max_chars: Truncate the returned content to this length.
        """
        if bucket not in allowed:
            return f"Bucket '{bucket}' not allowed. Allowed: {sorted(allowed)}"
        try:
            response = client.get_object(
                namespace_name=namespace,
                bucket_name=bucket,
                object_name=object_name,
            )
        except ServiceError as error:
            return _format_storage_error(error)
        content = response.data.content.decode("utf-8", errors="ignore")
        if len(content) > max_chars:
            return content[:max_chars] + "\n\n[truncated]"
        return content

    @tool
    def search_bucket_data(
        query: str, bucket: str = "", prefix: str = "", max_objects: int = 20
    ) -> str:
        """Search text objects in one or all allowed buckets for a query string.

        Returns up to 10 snippets, each ~400 chars wide around the match.

        Args:
            query: Substring to look for (case-insensitive).
            bucket: Optional single bucket; empty searches all allowed.
            prefix: Optional prefix filter.
            max_objects: Max objects to inspect per bucket (1-50).
        """
        selected = [bucket] if bucket else sorted(allowed)
        results: list[str] = []
        q = query.lower()
        for bkt in selected:
            if bkt not in allowed:
                continue
            try:
                listing = client.list_objects(
                    namespace_name=namespace,
                    bucket_name=bkt,
                    prefix=prefix or None,
                    limit=min(max(max_objects, 1), 50),
                )
            except ServiceError:
                continue
            for obj in listing.data.objects or []:
                name = obj.name or ""
                if not name.lower().endswith((".json", ".txt", ".md")):
                    continue
                try:
                    payload = client.get_object(
                        namespace_name=namespace,
                        bucket_name=bkt,
                        object_name=name,
                    )
                    text = payload.data.content.decode("utf-8", errors="ignore")
                except ServiceError:
                    continue
                idx = text.lower().find(q)
                if idx < 0:
                    continue
                start = max(0, idx - 200)
                end = min(len(text), idx + len(query) + 200)
                snippet = text[start:end].replace("\n", " ")
                results.append(f"[{bkt}/{name}] ...{snippet}...")
                if len(results) >= 10:
                    break
            if len(results) >= 10:
                break
        return "\n\n".join(results) if results else f"No matches for '{query}'."

    return [list_bucket_objects, read_bucket_object, search_bucket_data]


MEDICAL_BUCKET = "deep-research-medical"
LEGAL_BUCKET = "deep-research-legal"
LARGE_BUCKET = "deep-research-large"


def main() -> int:
    os_profile = os.environ.get("OCI_PROFILE", "DEFAULT")
    os_region = os.environ.get("OCI_OS_REGION", "ca-toronto-1")
    namespace = os.environ.get("OCI_OS_NAMESPACE", "<your-os-namespace>")

    genai_profile = os.environ.get("OCI_GENAI_PROFILE", "DEFAULT")
    genai_compartment = os.environ.get(
        "OCI_GENAI_COMPARTMENT",
        "ocid1.tenancy.oc1..<your-tenancy>",
    )
    genai_region = os.environ.get("OCI_GENAI_REGION", "us-chicago-1")
    model_id = os.environ.get("OCI_RESEARCH_MODEL", "oci:openai.gpt-4o-mini")

    print("=" * 70)
    print("DEEP RESEARCH + OCI OBJECT STORAGE — LOCUS PORT")
    print("=" * 70)
    print(f"  Storage profile : {os_profile} ({os_region}) ns={namespace}")
    print(f"  GenAI profile   : {genai_profile} ({genai_region})")
    print(f"  Model           : {model_id}")
    print(f"  Buckets         : {MEDICAL_BUCKET}, {LEGAL_BUCKET}, {LARGE_BUCKET}")
    print()

    tools = build_storage_tools(
        namespace=namespace,
        buckets=[MEDICAL_BUCKET, LEGAL_BUCKET, LARGE_BUCKET],
        region=os_region,
        auth_profile=os_profile,
    )
    print(f"Created {len(tools)} storage tools: {[t.name for t in tools]}")

    chat = get_model(
        model_id, profile=genai_profile, compartment_id=genai_compartment, region=genai_region
    )
    agent = create_deepagent(
        model=chat,
        tools=tools,
        system_prompt=(
            "You are a deep research analyst with access to massive datasets "
            "stored in OCI Object Storage. "
            "Available buckets:\n"
            f"- {MEDICAL_BUCKET}: MedMCQA + PubMedQA (biomedical research)\n"
            f"- {LEGAL_BUCKET}: CUAD (contract clauses)\n"
            f"- {LARGE_BUCKET}: Wikipedia / C4 / ArXiv (large corpora; may be empty)\n\n"
            "Use the tools to:\n"
            "1. list_bucket_objects: discover available files\n"
            "2. read_bucket_object: read a specific file\n"
            "3. search_bucket_data: find snippets across objects\n\n"
            "Cite the bucket/object name for every claim. Do not invent files."
        ),
        max_output_tokens=2048,
        max_iterations=10,
        reflexion=False,
        grounding=False,
    )

    queries = [
        (
            "medical-overview",
            "Discover what's in the deep-research-medical bucket. List the "
            "files, read the smallest one, and summarize what kind of data it "
            "contains. Cite the file name.",
        ),
        (
            "legal-overview",
            "What's in the deep-research-legal bucket? List the files, read "
            "the contents, and describe the type of legal data present. "
            "Cite the file name.",
        ),
    ]

    for label, q in queries:
        print(f"\n{'-' * 70}\n[{label}] {q}\n{'-' * 70}")
        result = agent.run_sync(q)
        text = getattr(result, "text", "") or ""
        execs = list(result.tool_executions or ())  # type: ignore[arg-type]
        metrics = getattr(result, "metrics", None)
        print(f"\nTool calls: {len(execs)}")
        for e in execs:
            args_short = str(e.arguments)[:120]
            print(f"  - {e.tool_name}({args_short}) -> {len(e.result or '')} chars")
        if metrics:
            print(f"Iterations: {metrics.iterations}, tokens={metrics.total_tokens}")
        print(f"\nResponse:\n{text}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
