"""Lexical (BM25) retrieval over chunks, backed by the ``bm25s`` library.

BM25 is local and deterministic (no network), so this index is exercised against
the real library in tests rather than mocked.
"""

from __future__ import annotations

import bm25s

from ragmem.types import Chunk, SearchResult

_STOPWORDS = "en"


class Bm25Index:
    """An in-memory BM25 index over a fixed list of chunks."""

    def __init__(self, chunks: list[Chunk], retriever: bm25s.BM25 | None):
        self._chunks = chunks
        self._retriever = retriever

    @classmethod
    def build(cls, chunks: list[Chunk]) -> "Bm25Index":
        chunks = list(chunks)
        if not chunks:
            return cls(chunks, None)
        corpus = [c.embedding_text for c in chunks]
        tokens = bm25s.tokenize(corpus, stopwords=_STOPWORDS, show_progress=False)
        retriever = bm25s.BM25()
        retriever.index(tokens, show_progress=False)
        return cls(chunks, retriever)

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        if self._retriever is None or not self._chunks or k <= 0:
            return []
        query_tokens = bm25s.tokenize(query, stopwords=_STOPWORDS, show_progress=False)
        if not query_tokens.ids or not query_tokens.ids[0]:
            return []  # query reduced to nothing (empty or all stopwords)

        k = min(k, len(self._chunks))
        indices, scores = self._retriever.retrieve(query_tokens, k=k, show_progress=False)

        results: list[SearchResult] = []
        for doc_idx, score in zip(indices[0].tolist(), scores[0].tolist()):
            if score <= 0:  # no query term matched this chunk
                continue
            results.append(
                SearchResult(chunk=self._chunks[doc_idx], score=float(score), source="bm25")
            )
        return results
