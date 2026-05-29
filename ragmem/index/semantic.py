"""Dense semantic retrieval: embed chunks, rank by cosine similarity.

Vectors are L2-normalized at build time so a query's cosine similarity is a
single matrix-vector dot product. Embeddings come from a DI'd embedder (see
``ragmem.embeddings``); the embedder is held on the index so queries can be
embedded on demand, but it is never serialized (see ``ragmem.persistence``).
"""

from __future__ import annotations

import numpy as np

from ragmem.embeddings import OpenAIEmbedder
from ragmem.types import Chunk, SearchResult


class SemanticIndex:
    def __init__(self, chunks: list[Chunk], matrix: np.ndarray, embedder: OpenAIEmbedder):
        self._chunks = chunks
        self._matrix = matrix  # (n, dim) float32, rows L2-normalized
        self._embedder = embedder

    @classmethod
    def build(cls, chunks: list[Chunk], embedder: OpenAIEmbedder) -> "SemanticIndex":
        chunks = list(chunks)
        if not chunks:
            return cls(chunks, np.zeros((0, 0), dtype=np.float32), embedder)
        vectors = embedder.embed_texts([c.embedding_text for c in chunks])
        return cls(chunks, _normalize_rows(np.asarray(vectors, dtype=np.float32)), embedder)

    @classmethod
    def from_matrix(
        cls, chunks: list[Chunk], matrix: np.ndarray, embedder: OpenAIEmbedder
    ) -> "SemanticIndex":
        """Wrap an already-normalized matrix (loaded from cache)."""
        return cls(list(chunks), np.asarray(matrix, dtype=np.float32), embedder)

    @property
    def matrix(self) -> np.ndarray:
        """The L2-normalized embedding matrix (what persistence caches)."""
        return self._matrix

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        if not self._chunks or k <= 0:
            return []
        q = _normalize_vec(self._embedder.embed_query(query))
        sims = self._matrix @ q  # cosine similarity, shape (n,)
        k = min(k, len(self._chunks))
        top = np.argsort(-sims)[:k]
        return [
            SearchResult(chunk=self._chunks[i], score=float(sims[i]), source="semantic")
            for i in top
        ]


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _normalize_vec(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector
