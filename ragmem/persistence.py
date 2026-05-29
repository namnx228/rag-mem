"""Freshness manifest for the cached semantic vectors.

LanceDB persists the vectors themselves (under ``<kb>/.ragmem/vectors.lance``); this
module only records *whether* that table is current for the present chunks + embedding
model, so a rebuild re-embeds (an OpenAI cost) only when the source Markdown, the
chunking, or the model changes. BM25 is cheap to rebuild and the Kuzu graph persists
itself, so neither is tracked here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ragmem.types import Chunk

_MANIFEST = "manifest.json"


def content_hash(chunks: list[Chunk]) -> str:
    """A deterministic SHA-256 over the chunks' ids and text (order-sensitive)."""
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk.id.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(chunk.text.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def save_manifest(persist_dir: str | Path, chunks: list[Chunk], embedding_model: str) -> None:
    """Record what the on-disk vector table was built from, for the freshness check."""
    directory = Path(persist_dir)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "content_hash": content_hash(chunks),
        "embedding_model": embedding_model,
        "chunk_ids": [c.id for c in chunks],
    }
    (directory / _MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")


def manifest_fresh(persist_dir: str | Path, chunks: list[Chunk], embedding_model: str) -> bool:
    """True iff a saved manifest matches *chunks* + *embedding_model* exactly."""
    manifest_path = Path(persist_dir) / _MANIFEST
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return (
        manifest.get("embedding_model") == embedding_model
        and manifest.get("content_hash") == content_hash(chunks)
        and manifest.get("chunk_ids") == [c.id for c in chunks]
    )
