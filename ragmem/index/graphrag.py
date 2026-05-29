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

import re
from pathlib import Path
from typing import Any

from ragmem.types import Chunk, SearchResult

DEFAULT_SIMILARITY_TOP_K = 20

_STOPWORDS = {
    "the", "and", "who", "what", "when", "where", "why", "how", "does", "did",
    "for", "with", "that", "this", "from", "into", "are", "was", "were",
}


class GraphRagIndex:
    def __init__(self, chunk_by_id: dict[str, Chunk], retriever: Any | None):
        self._chunk_by_id = chunk_by_id
        self._retriever = retriever
        self._chunk_texts_lower = {
            cid: chunk.embedding_text.lower() for cid, chunk in chunk_by_id.items()
        }

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
        use_cache: bool = True,
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        show_progress: bool = False,
    ) -> "GraphRagIndex":
        """Extract a property graph from *chunks* and return a queryable index.

        Runs real LLM extraction + embeddings — only used outside tests; covered
        by the manual live smoke. Every external client is injectable. When
        ``persist_dir`` is set the Kuzu graph is written under
        ``<persist_dir>/graph`` with a content-hash marker and reopened on the
        next build while the source is unchanged (no re-extraction); otherwise an
        in-memory graph is built.
        """
        import shutil

        from llama_index.core import PropertyGraphIndex
        from llama_index.core.indices.property_graph import SimpleLLMPathExtractor

        from ragmem.llm import make_extraction_llm
        from ragmem.persistence import content_hash

        chunks = list(chunks)
        chunk_by_id = {c.id: c for c in chunks}
        if not chunks:
            return cls(chunk_by_id, None)

        llm = llm or make_extraction_llm()
        embed_model = embed_model or _default_embed_model()

        graph_root = Path(persist_dir) / "graph" if persist_dir else None
        db_path = graph_root / "kuzu.db" if graph_root else None
        hash_file = Path(persist_dir) / "graph.hash" if persist_dir else None
        current_hash = content_hash(chunks)
        reuse = (
            use_cache
            and db_path is not None
            and db_path.exists()
            and hash_file is not None
            and hash_file.exists()
            and hash_file.read_text(encoding="utf-8") == current_hash
        )

        if reuse:
            store = graph_store or _kuzu_store(db_path)
            index = PropertyGraphIndex.from_existing(
                property_graph_store=store, llm=llm, embed_model=embed_model, embed_kg_nodes=False
            )
        else:
            if graph_root is not None and graph_root.exists():
                shutil.rmtree(graph_root)
            store = graph_store or _kuzu_store(db_path)
            index = PropertyGraphIndex(
                nodes=_chunks_to_nodes(chunks),
                llm=llm,
                embed_model=embed_model,
                kg_extractors=[SimpleLLMPathExtractor(llm=llm)],
                property_graph_store=store,
                embed_kg_nodes=False,
                use_async=False,
                show_progress=show_progress,
            )
            if hash_file is not None:
                hash_file.parent.mkdir(parents=True, exist_ok=True)
                hash_file.write_text(current_hash, encoding="utf-8")

        retriever = index.as_retriever(include_text=True, similarity_top_k=similarity_top_k)
        return cls(chunk_by_id, retriever)

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        """Retrieve source chunks via the knowledge graph.

        The PropertyGraphIndex retriever returns matched graph relationships
        ("subject -> relation -> object"). We keep the triplets whose entities
        relate to the query, attribute each to the source chunks whose text
        mentions those entities, and rank chunks by total matched score. If the
        retriever instead returns source nodes directly (carrying a chunk_id),
        those are mapped straight back.
        """
        if self._retriever is None or not self._chunk_by_id or k <= 0:
            return []

        query_lower = query.lower()
        query_tokens = _content_tokens(query)

        scores: dict[str, float] = {}
        for node in self._retriever.retrieve(query):
            score = float(node.score) if getattr(node, "score", None) is not None else 0.0
            direct = _node_chunk_id(node)
            if direct in self._chunk_by_id:
                scores[direct] = scores.get(direct, 0.0) + score
                continue
            text = _node_text(node)
            if "->" not in text:  # not a source chunk and not a triplet
                continue
            entities = _triplet_entities(text)
            if not _entities_match_query(entities, query_lower, query_tokens):
                continue
            for cid, lowered in self._chunk_texts_lower.items():
                if any(entity in lowered for entity in entities):
                    scores[cid] = scores.get(cid, 0.0) + score

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [
            SearchResult(chunk=self._chunk_by_id[cid], score=score, source="graphrag")
            for cid, score in ranked[:k]
        ]


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


def _node_text(node: Any) -> str:
    inner = getattr(node, "node", None)
    return getattr(inner, "text", "") or ""


def _triplet_entities(triplet_text: str) -> set[str]:
    """The subject and object (lower-cased) of a 'subj -> rel -> obj' triplet."""
    parts = [part.strip().lower() for part in triplet_text.split("->")]
    parts = [part for part in parts if part]
    if not parts:
        return set()
    return {parts[0], parts[-1]}


def _content_tokens(text: str) -> set[str]:
    """Significant lower-cased word tokens of *text* (drops short words/stopwords)."""
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 3 and token not in _STOPWORDS
    }


def _entities_match_query(entities: set[str], query_lower: str, query_tokens: set[str]) -> bool:
    """True if any triplet entity is mentioned by the query (phrase or word overlap)."""
    for entity in entities:
        if entity and entity in query_lower:
            return True
        if set(re.findall(r"[a-z0-9]+", entity)) & query_tokens:
            return True
    return False


def _default_embed_model() -> Any:
    from llama_index.embeddings.openai import OpenAIEmbedding

    return OpenAIEmbedding(model="text-embedding-3-small")


def _kuzu_store(db_path: Path | None) -> Any:
    import kuzu
    from llama_index.graph_stores.kuzu import KuzuPropertyGraphStore

    if db_path is None:
        db = kuzu.Database()  # in-memory
    else:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        db = kuzu.Database(str(db_path))
    # v0.1: pure entity-graph retrieval (no Kuzu vector index); retrieval is
    # driven by LLM-extracted entities + synonym matching over the graph.
    return KuzuPropertyGraphStore(db, use_vector_index=False)
