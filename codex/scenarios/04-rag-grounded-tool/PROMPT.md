Build a retrieval tool with Locus's RAG layer that works offline.

1. Write `kb_agent.py`. Implement a deterministic embedding double `HashEmbedding` that
   satisfies `locus.rag.embeddings.base.BaseEmbedding` (64 dimensions, a bag-of-words hash).
   Build a `RAGRetriever` over `InMemoryVectorStore(dimension=64)` and index two documents:
   a refund policy saying orders over 200 need manager approval, and a deploy runbook about
   canary health. Expose `make_retriever()` and `build_agent(model)` where the agent's only
   tool is `create_rag_tool(retriever, name="search_knowledge")`.
2. Write `test_kb_agent.py`: `retrieve("refund approval threshold", limit=1)` returns the
   refund policy first; an agent on a `ScriptedModel` that calls `search_knowledge` and then
   answers is traced with `AgentTestClient`, the tool call is asserted, and the tool's result
   text contains `manager approval`.
3. Run the tests; all must pass.
