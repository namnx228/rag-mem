"""On-disk cache for the (expensive) semantic embedding matrix.

Building embeddings costs OpenAI calls, so the normalized matrix is cached under
``<kb>/.ragmem`` next to a manifest. The cache is keyed by a content hash of the
chunks, the embedding model, and the exact chunk-id ordering, so any change to
the source Markdown, the model, or the chunking invalidates it.

BM25 is cheap to rebuild and the Kuzu graph persists itself, so only the
embedding matrix is cached here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ragmem.types import Chunk

_MANIFEST = "manifest.json"
_EMBEDDINGS = "embeddings.npy"


def content_hash(chunks: list[Chunk]) -> str:
    """A deterministic SHA-256 over the chunks' ids and text (order-sensitive)."""
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk.id.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(chunk.text.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def save_semantic(
    persist_dir: str | Path, chunks: list[Chunk], matrix: np.ndarray, embedding_model: str
) -> None:
    directory = Path(persist_dir)
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / _EMBEDDINGS, matrix)
    manifest = {
        "content_hash": content_hash(chunks),
        "embedding_model": embedding_model,
        "chunk_ids": [c.id for c in chunks],
    }
    (directory / _MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")


def load_semantic(
    persist_dir: str | Path, chunks: list[Chunk], embedding_model: str
) -> np.ndarray | None:
    """Return the cached matrix iff it is fresh for *chunks* + *embedding_model*."""
    directory = Path(persist_dir)
    manifest_path = directory / _MANIFEST
    embeddings_path = directory / _EMBEDDINGS
    if not manifest_path.exists() or not embeddings_path.exists():
        return None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("embedding_model") != embedding_model:
        return None
    if manifest.get("content_hash") != content_hash(chunks):
        return None
    if manifest.get("chunk_ids") != [c.id for c in chunks]:
        return None
    return np.load(embeddings_path)
