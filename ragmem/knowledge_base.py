"""KnowledgeBase: the public API tying loader, chunker, and the three retrievers.

Point ``from_directory`` at a folder of Markdown files and call
``semantic_search`` / ``bm25_search`` / ``graphrag_search``. Every external
client is injectable; defaults are built from the environment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ragmem.chunker import chunk_document
from ragmem.embeddings import make_embedder
from ragmem.index.bm25 import Bm25Index
from ragmem.index.graphrag import GraphRagIndex
from ragmem.index.semantic import SemanticIndex
from ragmem.loader import load_markdown
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
        build_graphrag: bool = True,
    ) -> "KnowledgeBase":
        """Load + chunk every ``.md`` file under *path* and build the indexes.

        ``build_graphrag=True`` (default) runs LLM extraction over the corpus and
        requires ``ANTHROPIC_API_KEY`` (and ``OPENAI_API_KEY`` for embeddings).
        """
        chunks = [chunk for doc in load_markdown(path) for chunk in chunk_document(doc)]
        embedder = embedder or make_embedder()

        semantic = SemanticIndex.build(chunks, embedder)
        bm25 = Bm25Index.build(chunks)
        graphrag = (
            GraphRagIndex.build(
                chunks,
                llm=graph_llm,
                embed_model=graph_embed_model,
                persist_dir=persist_dir,
            )
            if build_graphrag
            else None
        )
        return cls(chunks, semantic=semantic, bm25=bm25, graphrag=graphrag)

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def semantic_search(self, query: str, k: int = 5) -> list[SearchResult]:
        return self._require(self._semantic, "semantic").search(query, k)

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
