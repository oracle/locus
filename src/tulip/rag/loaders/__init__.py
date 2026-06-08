"""Document loaders for RAG ingestion.

Each loader yields :class:`tulip.rag.stores.base.Document` instances from
some external source (filesystem, HTTP, object storage, etc.) so the
downstream chunker / embedder / vector store pipeline can stay agnostic
of where the raw rows came from.

No loaders ship in the core package today — implement one by returning
``Document`` instances and feed them to ``RAGRetriever.add_documents``.
"""

__all__: list[str] = []
