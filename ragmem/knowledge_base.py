"""KnowledgeBase: the public API tying loader, chunker, and the three retrievers.

Point ``from_directory`` at a folder of Markdown files and call
``semantic_search`` / ``bm25_search`` / ``graphrag_search``. Every external client
(the embedder, the graph LLM, the LanceDB connection) is injectable; defaults are
built from the environment / ``persist_dir``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from ragmem.chunker import chunk_document
from ragmem.embeddings import make_embedder
from ragmem.index.bm25 import Bm25Index
from ragmem.index.graphrag import GraphRagIndex
from ragmem.index.lance import LanceSemanticIndex
from ragmem.loader import load_markdown
from ragmem.persistence import manifest_fresh, save_manifest
from ragmem.types import Chunk, SearchResult


class KnowledgeBase:
    def __init__(
        self,
        chunks: list[Chunk],
        *,
        semantic: Any | None = None,
        bm25: Any | None = None,
        graphrag: Any | None = None,
    ):
        self._chunks = list(chunks)
        self._semantic = semantic
        self._bm25 = bm25
        self._graphrag = graphrag

    @classmethod
    def from_directory(
        cls,
        path: str | Path,
        *,
        embedder: Any | None = None,
        graph_llm: Any | None = None,
        graph_embed_model: Any | None = None,
        persist_dir: str | Path | None = None,
        use_cache: bool = True,
        build_graphrag: bool = True,
        vector_db: Any | None = None,
    ) -> "KnowledgeBase":
        """Load + chunk every ``.md`` file under *path* and build the indexes.

        ``build_graphrag=True`` (default) runs LLM extraction over the corpus and
        requires ``OPENROUTER_API_KEY`` (for both extraction and embeddings). With
        ``use_cache`` the embedding vectors live in a LanceDB table under
        ``persist_dir`` (default ``<path>/.ragmem``) and are reused while the source
        is unchanged. ``vector_db`` injects a LanceDB connection (defaults to one
        opened at ``persist_dir``).
        """
        path = Path(path)
        if persist_dir is None:
            persist_dir = path / ".ragmem"
        chunks = [chunk for doc in load_markdown(path) for chunk in chunk_document(doc)]
        embedder = embedder or make_embedder()
        vector_db = vector_db or _connect_vector_db(persist_dir)

        semantic = cls._build_or_load_semantic(chunks, embedder, vector_db, persist_dir, use_cache)
        bm25 = Bm25Index.build(chunks)
        graphrag = (
            GraphRagIndex.build(
                chunks,
                llm=graph_llm,
                embed_model=graph_embed_model,
                persist_dir=persist_dir,
                use_cache=use_cache,
            )
            if build_graphrag
            else None
        )
        return cls(chunks, semantic=semantic, bm25=bm25, graphrag=graphrag)

    @staticmethod
    def _build_or_load_semantic(
        chunks: list[Chunk], embedder: Any, vector_db: Any, persist_dir: str | Path, use_cache: bool
    ) -> LanceSemanticIndex:
        model = getattr(embedder, "model", "unknown")
        if use_cache and manifest_fresh(persist_dir, chunks, model):
            index = LanceSemanticIndex.open(vector_db, embedder)
            # Reuse the persisted table unless it's gone missing for a non-empty corpus.
            if not chunks or not index.is_empty:
                return index
        semantic = LanceSemanticIndex.build(chunks, embedder, vector_db)
        if use_cache:
            save_manifest(persist_dir, chunks, model)
        return semantic

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def semantic_search(
        self, query: str, k: int = 5, tags: Sequence[str] | None = None
    ) -> list[SearchResult]:
        return self._require(self._semantic, "semantic").search(query, k, tags=tags)

    def bm25_search(self, query: str, k: int = 5) -> list[SearchResult]:
        return self._require(self._bm25, "bm25").search(query, k)

    def graphrag_search(self, query: str, k: int = 5) -> list[SearchResult]:
        return self._require(self._graphrag, "graphrag").search(query, k)

    @staticmethod
    def _require(index: Any | None, name: str) -> Any:
        if index is None:
            raise RuntimeError(
                f"The {name!r} index is not built. "
                "Rebuild the KnowledgeBase with that index enabled."
            )
        return index


def _connect_vector_db(persist_dir: str | Path) -> Any:
    import lancedb  # local: keep ``import ragmem`` from pulling in lancedb until first use

    return lancedb.connect(str(persist_dir))
