"""OpenAI text embeddings, dependency-injected for testability.

``make_embedder`` defaults to building an ``openai.OpenAI`` client from the
environment, but tests pass a fake client shaped like the OpenAI SDK
(``client.embeddings.create(model=, input=).data[i].embedding``).
"""

from __future__ import annotations

from typing import Any, Iterator, Sequence

import numpy as np

DEFAULT_MODEL = "text-embedding-3-small"
_BATCH_SIZE = 100


class OpenAIEmbedder:
    def __init__(self, client: Any, model: str = DEFAULT_MODEL):
        self._client = client
        self._model = model

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Embed many texts -> float32 matrix of shape (len(texts), dim)."""
        texts = list(texts)
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        vectors: list[list[float]] = []
        for batch in _batched(texts, _BATCH_SIZE):
            resp = self._client.embeddings.create(model=self._model, input=batch)
            vectors.extend(item.embedding for item in resp.data)
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query -> 1-D float32 vector of shape (dim,)."""
        resp = self._client.embeddings.create(model=self._model, input=[text])
        return np.asarray(resp.data[0].embedding, dtype=np.float32)


def make_embedder(client: Any | None = None, model: str = DEFAULT_MODEL) -> OpenAIEmbedder:
    if client is None:
        from openai import OpenAI

        client = OpenAI()  # reads OPENAI_API_KEY from the environment
    return OpenAIEmbedder(client, model)


def _batched(items: list[str], size: int) -> Iterator[list[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]
