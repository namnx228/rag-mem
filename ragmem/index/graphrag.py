"""GraphRAG retrieval via llama-index ``PropertyGraphIndex`` over embedded Kuzu.

llama-index is confined to this module and imported lazily, so the rest of
``ragmem`` (and the semantic/bm25 retrievers) never depends on it. Our chunks are
handed to llama-index as ``TextNode``s carrying ``chunk_id`` in metadata; on
retrieval we map the returned source nodes back to our ``Chunk`` objects.

``search`` only reads duck-typed attributes off the retrieved ``NodeWithScore``
objects, so it is unit-tested with a fake retriever. ``build`` (which runs LLM
extraction and embeddings) is covered by the manual live smoke, not pytest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ragmem.types import Chunk, SearchResult

DEFAULT_SIMILARITY_TOP_K = 20


class GraphRagIndex:
    def __init__(self, chunk_by_id: dict[str, Chunk], retriever: Any | None):
        self._chunk_by_id = chunk_by_id
        self._retriever = retriever

    @classmethod
    def from_retriever(cls, chunks: list[Chunk], retriever: Any | None) -> "GraphRagIndex":
        """Wrap an existing llama-index retriever (the seam used by tests)."""
        return cls({c.id: c for c in chunks}, retriever)

    @classmethod
    def build(
        cls,
        chunks: list[Chunk],
        *,
        llm: Any | None = None,
        embed_model: Any | None = None,
        graph_store: Any | None = None,
        persist_dir: str | Path | None = None,
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        show_progress: bool = False,
    ) -> "GraphRagIndex":
        """Extract a property graph from *chunks* and return a queryable index.

        Runs real LLM extraction + embeddings — only used outside tests. Every
        external client is injectable; defaults are built from the environment.
        """
        chunks = list(chunks)
        chunk_by_id = {c.id: c for c in chunks}
        if not chunks:
            return cls(chunk_by_id, None)

        from llama_index.core import PropertyGraphIndex
        from llama_index.core.indices.property_graph import SimpleLLMPathExtractor

        from ragmem.llm import make_extraction_llm

        llm = llm or make_extraction_llm()
        embed_model = embed_model or _default_embed_model()
        graph_store = graph_store or _default_kuzu_store(persist_dir)

        index = PropertyGraphIndex(
            nodes=_chunks_to_nodes(chunks),
            llm=llm,
            embed_model=embed_model,
            kg_extractors=[SimpleLLMPathExtractor(llm=llm)],
            property_graph_store=graph_store,
            embed_kg_nodes=True,
            use_async=False,
            show_progress=show_progress,
        )
        retriever = index.as_retriever(include_text=True, similarity_top_k=similarity_top_k)
        return cls(chunk_by_id, retriever)

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        if self._retriever is None or not self._chunk_by_id or k <= 0:
            return []

        best: dict[str, SearchResult] = {}
        for node in self._retriever.retrieve(query):
            chunk_id = _node_chunk_id(node)
            chunk = self._chunk_by_id.get(chunk_id) if chunk_id else None
            if chunk is None:  # entity/community node or chunk from another KB
                continue
            score = float(node.score) if getattr(node, "score", None) is not None else 0.0
            existing = best.get(chunk_id)
            if existing is None or score > existing.score:
                best[chunk_id] = SearchResult(chunk=chunk, score=score, source="graphrag")

        results = sorted(best.values(), key=lambda r: r.score, reverse=True)
        return results[:k]


def _chunks_to_nodes(chunks: list[Chunk]) -> list[Any]:
    from llama_index.core.schema import TextNode

    nodes = []
    for chunk in chunks:
        node = TextNode(text=chunk.embedding_text, id_=chunk.id, metadata={"chunk_id": chunk.id})
        # Keep the bookkeeping id out of the text fed to the LLM / embeddings.
        node.excluded_embed_metadata_keys = ["chunk_id"]
        node.excluded_llm_metadata_keys = ["chunk_id"]
        nodes.append(node)
    return nodes


def _node_chunk_id(node: Any) -> str | None:
    inner = getattr(node, "node", None)
    if inner is None:
        return None
    metadata = getattr(inner, "metadata", None) or {}
    return metadata.get("chunk_id") or getattr(inner, "node_id", None)


def _default_embed_model() -> Any:
    from llama_index.embeddings.openai import OpenAIEmbedding

    return OpenAIEmbedding(model="text-embedding-3-small")


def _default_kuzu_store(persist_dir: str | Path | None) -> Any:
    import kuzu
    from llama_index.graph_stores.kuzu import KuzuPropertyGraphStore

    if persist_dir is None:
        db = kuzu.Database()  # in-memory
    else:
        db = kuzu.Database(str(Path(persist_dir) / "graph.kuzu"))
    return KuzuPropertyGraphStore(db)
